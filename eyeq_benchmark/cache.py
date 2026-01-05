from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


_LOCK = threading.Lock()


def make_cache_key(
    model_name: str,
    language: str,
    sample_id: int,
    use_context: bool,
    hint_type: Optional[str],
    pass_at_enabled: bool,
    num_pass: int,
    temperature: Optional[float],
) -> str:
    hint_str = str(hint_type) if hint_type else "None"
    temp_str = "None" if temperature is None else str(temperature)
    return (
        f"{model_name}_{language}_{sample_id}_{str(use_context)}_{hint_str}_"
        f"{str(pass_at_enabled)}_{str(num_pass)}_{temp_str}"
    )


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}

    cache: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            model_name = data.get("model_name", "unknown")
            language = data.get("language", "unknown")
            sid = data.get("id")
            if sid is None:
                continue
            try:
                sid_int = int(sid)
            except Exception:
                continue

            key = make_cache_key(
                model_name=model_name,
                language=language,
                sample_id=sid_int,
                use_context=bool(data.get("use_context")),
                hint_type=data.get("hint_type"),
                pass_at_enabled=bool(data.get("pass_at_enabled")),
                num_pass=int(data.get("num_pass", 1)),
                temperature=data.get("temperature"),
            )

            if data.get("final_response") is not None:
                cache[key] = data

    return cache


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
