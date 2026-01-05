from __future__ import annotations

import hashlib
import random


def _stable_seed(sample_id: int | str, language: str) -> int:
    s = f"{language}:{sample_id}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def _shuffle_pattern(answer: str, seed: int, reveal_ratio: float = 0.25) -> str:
    clean_answer = (answer or "").strip()
    if not clean_answer:
        return ""

    char_indices = [i for i, c in enumerate(clean_answer) if c != " "]
    n = len(char_indices)
    if n == 0:
        return clean_answer

    reveal_count = round(reveal_ratio * n)
    rng = random.Random(seed)
    reveal_indices = set(rng.sample(char_indices, min(reveal_count, n)))

    out = []
    for i, ch in enumerate(clean_answer):
        if ch == " ":
            out.append(" ")
        elif i in reveal_indices:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def generate_hint(sample_id: int | str, language: str, answer: str, hint_type: str | None) -> str:
    if not hint_type or hint_type == "none" or not answer:
        return ""

    clean = answer.strip()

    if hint_type in {"char_count", "answer_length", "answer_length_hint"}:
        n = len(clean.replace(" ", ""))
        return f"\nHINT: The answer has {n} characters (excluding spaces)."

    if hint_type in {"shuffle_chars", "partial_character_reveal", "partial_reveal"}:
        pattern = _shuffle_pattern(clean, seed=_stable_seed(sample_id, language))
        return (
            "\nHINT: The pattern of the answer is '"
            + pattern
            + "'. In this pattern, '_' represents a hidden character and spaces represent actual spaces in the answer."
        )

    raise ValueError(f"Unknown hint_type: {hint_type}")
