from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base import ModelResponse, VisionLanguageModel


def _image_to_data_url(image_path_or_url: str) -> str:
    if image_path_or_url.startswith(("http://", "https://", "data:")):
        return image_path_or_url
    path = Path(image_path_or_url)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    return f"data:{mime};base64,{b64}"


class GrokAPI(VisionLanguageModel):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        image_detail: Optional[str] = "high",
    ):
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GROK_API_KEY (or OPENROUTER_API_KEY)")
        self.model_name = model or os.getenv("GROK_MODEL") or "x-ai/grok-4.1-fast"
        self.base_url = (base_url or os.getenv("GROK_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.image_detail = image_detail
        self.endpoint = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @property
    def name(self) -> str:
        return f"grok-{self.model_name}"

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"

            if role == "system":
                openai_messages.append({"role": "system", "content": msg.get("text", "")})
                continue

            parts: List[Dict[str, Any]] = []
            if msg.get("text"):
                parts.append({"type": "text", "text": msg["text"]})
            if msg.get("image_path"):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(msg["image_path"]), "detail": self.image_detail},
                    }
                )
            openai_messages.append({"role": role, "content": parts})

        payload: Dict[str, Any] = {"model": self.model_name, "messages": openai_messages}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        temp = self.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = float(temp)

        resp = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Grok/OpenRouter error {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            raw_text = content
        else:
            raw_text = "\n".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])

        return ModelResponse(
            raw_text=raw_text,
            provider_payload={"endpoint": self.endpoint, "model": self.model_name, "raw_response": data, "extra_metadata": extra_metadata or {}},
        )
