import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from eyeq_benchmark.cache import append_jsonl, load_cache, make_cache_key
from eyeq_benchmark.data import load_eyeq_from_hf
from eyeq_benchmark.derivations import load_derivations
from eyeq_benchmark.hints import generate_hint
from eyeq_benchmark.prompts import get_system_prompt
from eyeq_benchmark.utils import clean_json_response, normalize_answer
from eyeq_benchmark.models.google import GoogleAPI
from eyeq_benchmark.models.grok import GrokAPI
from eyeq_benchmark.models.llama import LlamaAPI
from eyeq_benchmark.models.openai import OpenaiAPI
from eyeq_benchmark.models.qwen import QwenAPI


_LOG = logging.getLogger("eyeq")
_CACHE_LOCK = threading.Lock()


def _model_role(model_name: str) -> str:
    return "model" if "gemini" in (model_name or "").lower() else "assistant"


def _prepare_data_split(
    lang: str,
    dataset: List[Dict[str, Any]],
    num_examples: int,
    derivations: Dict[str, Dict[int, Dict[str, Any]]],
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    dmap = derivations.get(lang, {})
    idxs = list(dmap.keys())
    if num_examples:
        idxs = idxs[: max(0, int(num_examples))]

    id_to_row = {int(r.get("id")): r for r in dataset if "id" in r}

    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_ids = set()

    for ex_id in idxs:
        row = None
        if ex_id in id_to_row:
            row = id_to_row[ex_id]
        elif 0 <= ex_id < len(dataset):
            row = dataset[ex_id]
        elif 1 <= ex_id <= len(dataset):
            row = dataset[ex_id - 1]

        if row is None:
            continue

        examples.append((row, dmap[ex_id]))
        used_ids.add(int(row["id"]))

    test_set = [row for row in dataset if int(row.get("id", -1)) not in used_ids]
    return examples, test_set


def _build_messages(
    sample: Dict[str, Any],
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    use_context: bool,
    hint_type: Optional[str],
    role: str,
) -> List[Dict[str, Any]]:
    lang = sample["language"]
    sys_prompt = get_system_prompt(lang)
    hint_str = generate_hint(sample["id"], lang, sample["answer"], hint_type) if hint_type else ""

    messages: List[Dict[str, Any]] = []

    if use_context and examples:
        ex0, deriv0 = examples[0]
        messages.append(
            {
                "role": "user",
                "text": f"{sys_prompt}\n\nHere is an example puzzle:\n",
                "image_path": ex0["image_path"],
            }
        )
        messages.append(
            {
                "role": role,
                "text": json.dumps(
                    {
                        "primary_clues": deriv0.get("primary_clues", []),
                        "candidates": deriv0.get("candidates", []),
                        "final_answer": ex0["answer"],
                    },
                    ensure_ascii=False,
                ),
            }
        )

        for ex, deriv in examples[1:]:
            messages.append({"role": "user", "text": "Here is another example:", "image_path": ex["image_path"]})
            messages.append(
                {
                    "role": role,
                    "text": json.dumps(
                        {
                            "primary_clues": deriv.get("primary_clues", []),
                            "candidates": deriv.get("candidates", []),
                            "final_answer": ex["answer"],
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "text": f"Now solve this new puzzle. Provide the JSON output.{hint_str}",
                "image_path": sample["image_path"],
            }
        )
        return messages

    messages.append(
        {
            "role": "user",
            "text": f"{sys_prompt}\n\nAnalyze the image and provide the JSON solution.{hint_str}",
            "image_path": sample["image_path"],
        }
    )
    return messages


def _process_sample(
    *,
    sample: Dict[str, Any],
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    model: Any,
    use_context: bool,
    hint_type: Optional[str],
    pass_at_enabled: bool,
    num_pass: int,
    temperature: Optional[float],
    cache_file: str,
) -> None:
    lang = sample["language"]
    sid = int(sample["id"])
    model_name = getattr(model, "name", None) or getattr(model, "model_name", None) or str(model)
    role = _model_role(model_name)

    messages = _build_messages(sample, examples, use_context, hint_type, role)

    max_loops = int(num_pass) if pass_at_enabled else 1
    attempts: List[Dict[str, Any]] = []
    final_json: Optional[Dict[str, Any]] = None
    solved = False

    current_history = list(messages)
    gold = sample["answer"].strip()

    model_ans = ""
    for i in range(max_loops):
        _LOG.info("Processing %s-%s | Attempt %d", lang, sid, i + 1)
        response = model.generate_chat(current_history, temperature=temperature)
        parsed = clean_json_response(response.raw_text)
        model_ans = str(parsed.get("final_answer", "") or "").strip()

        is_right = normalize_answer(model_ans) == normalize_answer(gold)
        attempts.append(
            {
                "attempt_idx": i,
                "response": response.raw_text,
                "parsed": model_ans,
                "correct": is_right,
            }
        )

        if is_right:
            solved = True
            final_json = parsed
            break

        if i < max_loops - 1:
            current_history.append({"role": role, "text": response.raw_text})
            current_history.append(
                {
                    "role": "user",
                    "text": (
                        f"The answer '{model_ans}' is incorrect. Please Step-by-Step:\n"
                        "1. List the primary clues you see in the image again.\n"
                        "2. Consider alternative interpretations or puns for these elements.\n"
                        "3. Provide a NEW answer in the correct JSON format."
                    ),
                }
            )

    record: Dict[str, Any] = {
        "id": sid,
        "language": lang,
        "model_name": model_name,
        "ground_truth": sample["answer"],
        "model_ans": model_ans,
        "hint_type": hint_type,
        "use_context": use_context,
        "pass_at_enabled": pass_at_enabled,
        "num_pass": num_pass,
        "temperature": temperature,
        "solved": solved,
        "attempts": attempts,
        "final_response": final_json or (attempts[-1]["parsed"] if attempts else None),
        "message_history": current_history,
    }

    with _CACHE_LOCK:
        append_jsonl(cache_file, record)


def _build_models(model_names: List[str], temperature: Optional[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in model_names:
        n = name.lower().strip()
        if n == "openai":
            out[n] = OpenaiAPI(temperature=temperature)
        elif n == "qwen":
            out[n] = QwenAPI(temperature=temperature)
        elif n == "llama":
            out[n] = LlamaAPI(temperature=temperature)
        elif n == "grok":
            out[n] = GrokAPI(temperature=temperature)
        elif n in {"google", "gemini"}:
            out[n] = GoogleAPI(temperature=temperature)
        else:
            raise ValueError(f"Unknown model: {name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="llm-lab/Eye-Q")
    p.add_argument("--config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--languages", default="en,pe,cross,ar")
    p.add_argument("--models", default="openai")
    p.add_argument("--use-context", action="store_true")
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--hint-type", default="none", choices=["none", "char_count", "shuffle_chars"])
    p.add_argument("--pass-at", action="store_true")
    p.add_argument("--num-pass", type=int, default=3)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--cache-file", default="results_cache.jsonl")
    p.add_argument("--image-cache-dir", default=".cache/eyeq_images")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    model_list = [x.strip() for x in args.models.split(",") if x.strip()]
    lang_list = [x.strip() for x in args.languages.split(",") if x.strip()]

    derivations = load_derivations()
    models = _build_models(model_list, temperature=args.temperature)

    cache = load_cache(args.cache_file)
    samples, errors = load_eyeq_from_hf(
        repo_id=args.repo_id,
        config=args.config,
        split=args.split,
        image_cache_dir=args.image_cache_dir,
        languages=lang_list,
        limit=args.max_samples if args.max_samples and args.max_samples > 0 else None,
    )

    if errors:
        for e in errors[:5]:
            _LOG.warning("Data warning: %s", e)

    from eyeq_benchmark.data import canonical_lang

    canon_langs = [canonical_lang(l) for l in lang_list]

    grouped: Dict[str, List[Dict[str, Any]]] = {l: [] for l in canon_langs}
    for s in samples:
        grouped[s.language].append(
            {"id": int(s.id), "language": s.language, "answer": s.answer, "image_path": s.image_path}
        )

    for l in grouped.keys():
        grouped[l].sort(key=lambda x: x["id"])

    tasks = []
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        for model in models.values():
            model_name = getattr(model, "name", None) or getattr(model, "model_name", None) or str(model)

            for lang in canon_langs:
                dataset = grouped.get(lang, [])
                if not dataset:
                    continue

                examples, test_set = _prepare_data_split(lang, dataset, args.num_examples, derivations)

                for row in test_set:
                    key = make_cache_key(
                        model_name=model_name,
                        language=lang,
                        sample_id=int(row["id"]),
                        use_context=bool(args.use_context),
                        hint_type=None if args.hint_type == "none" else args.hint_type,
                        pass_at_enabled=bool(args.pass_at),
                        num_pass=int(args.num_pass),
                        temperature=float(args.temperature) if args.temperature is not None else None,
                    )

                    if key in cache:
                        continue

                    tasks.append(
                        ex.submit(
                            _process_sample,
                            sample=row,
                            examples=examples,
                            model=model,
                            use_context=bool(args.use_context),
                            hint_type=None if args.hint_type == "none" else args.hint_type,
                            pass_at_enabled=bool(args.pass_at),
                            num_pass=int(args.num_pass),
                            temperature=float(args.temperature) if args.temperature is not None else None,
                            cache_file=args.cache_file,
                        )
                    )

        for fut in as_completed(tasks):
            fut.result()


if __name__ == "__main__":
    main()
