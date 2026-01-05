from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base import ModelResponse, VisionLanguageModel


def _read_image_as_inline_data(image_path_or_url: str) -> Dict[str, Any]:
    if image_path_or_url.startswith(("http://", "https://")):
        return {"file_uri": image_path_or_url}
    path = Path(image_path_or_url)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    return {"inline_data": {"mime_type": mime, "data": b64}}


class GoogleAPI(VisionLanguageModel):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        self.model_name = model or os.getenv("GOOGLE_MODEL") or "gemini-1.5-flash"
        self.base_url = (base_url or os.getenv("GOOGLE_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        return f"google-{self.model_name}"

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "model":
                role = "model"
            if role == "assistant":
                role = "model"
            if role == "system":
                role = "user"

            parts: List[Dict[str, Any]] = []
            if msg.get("text"):
                parts.append({"text": msg["text"]})
            if msg.get("image_path"):
                parts.append(_read_image_as_inline_data(msg["image_path"]))
            contents.append({"role": role, "parts": parts})

        payload: Dict[str, Any] = {"contents": contents}

        t = self.temperature if temperature is None else temperature
        if t is not None:
            payload.setdefault("generationConfig", {})["temperature"] = float(t)

        out = self.max_output_tokens if max_tokens is None else max_tokens
        if out is not None:
            payload.setdefault("generationConfig", {})["maxOutputTokens"] = int(out)

        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Google Gemini error {resp.status_code}: {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raw_text = ""
        else:
            parts = candidates[0].get("content", {}).get("parts", [])
            raw_text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)])

        return ModelResponse(
            raw_text=raw_text,
            provider_payload={"endpoint": url, "model": self.model_name, "raw_response": data, "extra_metadata": extra_metadata or {}},
        )
