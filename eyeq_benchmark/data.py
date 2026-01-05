from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import datasets
from PIL import Image as PILImage


_LANG_MAP = {
    "english": "en",
    "en": "en",
    "persian": "pe",
    "farsi": "pe",
    "fa": "pe",
    "pe": "pe",
    "arabic": "ar",
    "ar": "ar",
    "cross_lingual": "cross",
    "cross-lingual": "cross",
    "crosslingual": "cross",
    "cross": "cross",
}


@dataclass(frozen=True)
class Sample:
    id: int
    language: str
    answer: str
    image_path: str


def canonical_lang(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    return _LANG_MAP.get(key, key)


def _atomic_save_image(img: PILImage.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=out_path.suffix, dir=str(out_path.parent), delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        img.save(tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)




def cache_image(img: PILImage.Image, out_path: Path) -> str:
    if out_path.exists():
        return str(out_path)
    if img.mode not in {"RGB", "RGBA"}:
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    _atomic_save_image(img, out_path)
    return str(out_path)


def load_eyeq_from_hf(
    repo_id: str,
    config: Optional[str] = None,
    split: str = "train",
    languages: Optional[Sequence[str]] = None,
    image_cache_dir: str = ".cache/eyeq_images",
    limit: Optional[int] = None,
) -> Tuple[List[Sample], List[str]]:
    ds = datasets.load_dataset(repo_id, config, split=split) if config else datasets.load_dataset(repo_id, split=split)

    langs = None
    if languages:
        langs = {canonical_lang(l) for l in languages}

    cache_root = Path(image_cache_dir)
    samples: List[Sample] = []
    errors: List[str] = []

    for row in ds:
        try:
            sid = int(row["id"]) if "id" in row else len(samples)
            lang = canonical_lang(row.get("lang") or row.get("language") or "")
            if langs and lang not in langs:
                continue
            ans = str(row.get("answer", ""))
            pil: PILImage.Image = row["image"]
            image_path = cache_image(pil, cache_root / lang / f"{sid}.jpg")
            samples.append(Sample(id=sid, language=lang, answer=ans, image_path=image_path))
            if limit is not None and len(samples) >= limit:
                break
        except Exception as e:
            errors.append(str(e))

    samples.sort(key=lambda s: s.id)
    return samples, errors
