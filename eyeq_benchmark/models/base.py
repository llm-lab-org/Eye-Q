from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ModelResponse:
    raw_text: str
    provider_payload: Dict[str, Any]


class VisionLanguageModel(Protocol):
    @property
    def name(self) -> str:
        ...

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        ...
