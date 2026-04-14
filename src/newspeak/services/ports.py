from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class ChatResult:
    text: str
    transcription: str | None = None
    mode: str = "chat"


class ChatClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        history: Sequence[dict[str, Any]],
        user_message: dict[str, Any],
        use_transcription_tool: bool,
        roleplay_system: str | None = None,
    ) -> ChatResult: ...


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_b64: str) -> str: ...


class TTSBackend(Protocol):
    sample_rate: int

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray: ...
