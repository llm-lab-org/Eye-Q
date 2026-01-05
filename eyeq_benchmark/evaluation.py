from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import unicodedata
import json
import re


@dataclass
class ParsedModelOutput:
    reasoning: Optional[str]
    final_answer: Optional[str]


def parse_model_response(raw_text: str) -> ParsedModelOutput:
    text = (raw_text or "").strip()

    obj: Optional[Dict[str, Any]] = None

    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            obj = candidate
    except Exception:
        obj = None

    if obj is None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                candidate = json.loads(json_str)
                if isinstance(candidate, dict):
                    obj = candidate
            except Exception:
                obj = None

    reasoning: Optional[str] = None
    final_answer: Optional[str] = None

    if obj is not None:
        reasoning = obj.get("reasoning") or obj.get("thoughts") or obj.get("explanation")
        final_answer = obj.get("final_answer") or obj.get("answer")

    if final_answer is None:
        m = re.search(
            r"final_answer\s*[:=-]\s*['\"“”]?(.+?)['\"“”]?(?:$|\n)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            final_answer = m.group(1).strip()

    if reasoning is None:
        reasoning = text

    return ParsedModelOutput(reasoning=reasoning, final_answer=final_answer)


PERSIAN_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")


def normalize_answer(ans: str, language: Optional[str] = None) -> str:
    if ans is None:
        return ""

    s = unicodedata.normalize("NFKC", ans)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()

    lang = (language or "").lower()

    if (
        lang.startswith("fa")
        or "persian" in lang
        or "farsi" in lang
        or lang.startswith("pe")
    ):
        s = PERSIAN_DIACRITICS_RE.sub("", s)
        s = s.replace("\u064a", "\u06cc")
        s = s.replace("\u0643", "\u06a9")
        s = s.replace("\u200c", "")

    s = re.sub(r"[\s\-]+", "", s)

    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            cur_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = cur_row
    return prev_row[-1]


def _fuzzy_equal(a: str, b: str, max_edits: int) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_edits:
        return False
    return _levenshtein(a, b) <= max_edits


def is_correct(
    predicted: Optional[str],
    gold: str,
    language: Optional[str] = None,
    alt_answers: Optional[Sequence[str]] = None,
    max_edit_distance: int = 0,
) -> bool:
    if not predicted:
        return False

    n_pred = normalize_answer(predicted, language)
    if not n_pred:
        return False

    n_gold = normalize_answer(gold, language)
    if n_pred == n_gold:
        return True
    if _fuzzy_equal(n_pred, n_gold, max_edit_distance):
        return True
    if alt_answers:
        for alt in alt_answers:
            n_alt = normalize_answer(alt, language)
            if n_pred == n_alt or _fuzzy_equal(n_pred, n_alt, max_edit_distance):
                return True

    return False

@dataclass
class SampleEvaluation:
    sample_id: str
    experiment_name: str
    model_name: str
    correct: bool
    attempts_used: int


def summarize_accuracy(records: List[SampleEvaluation]) -> Dict[str, Any]:
    if not records:
        return {"accuracy": 0.0, "n": 0}

    n = len(records)
    correct = sum(1 for r in records if r.correct)
    return {
        "accuracy": correct / n,
        "n": n,
    }

