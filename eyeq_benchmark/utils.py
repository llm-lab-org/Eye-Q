import json
import re
import string
from typing import Any, Dict


_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652]")


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.replace(" ", "")


def clean_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"error": "Invalid JSON", "raw": text}
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {"error": "Invalid JSON", "raw": text}
            except Exception:
                pass
        return {"error": "Invalid JSON", "raw": text}
