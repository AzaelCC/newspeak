import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ClientMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["interrupt", "set_mode", "deep_dive"] | None = None
    text: str | None = None
    audio: str | None = None
    image: str | None = None
    mode_id: str | None = None
    turn_id: str | None = None

    @property
    def is_interrupt(self) -> bool:
        return self.type == "interrupt"

    @property
    def is_set_mode(self) -> bool:
        return self.type == "set_mode"

    @property
    def is_deep_dive(self) -> bool:
        return self.type == "deep_dive"


class WebSocketEvent(BaseModel):
    type: str

    def to_json(self) -> str:
        return json.dumps(self.model_dump(exclude_none=True))


class TextEvent(WebSocketEvent):
    type: Literal["text"] = "text"
    text: str
    llm_time: float
    audio_pipeline: str | None = None
    asr_time: float | None = None
    transcription: str | None = None
    turn_id: str | None = None


class AudioStartEvent(WebSocketEvent):
    type: Literal["audio_start"] = "audio_start"
    sample_rate: int
    sentence_count: int


class AudioChunkEvent(WebSocketEvent):
    type: Literal["audio_chunk"] = "audio_chunk"
    audio: str
    index: int


class AudioEndEvent(WebSocketEvent):
    type: Literal["audio_end"] = "audio_end"
    tts_time: float


class ErrorEvent(WebSocketEvent):
    type: Literal["error"] = "error"
    code: str
    message: str


class CoachEvent(WebSocketEvent):
    type: Literal["coach"] = "coach"
    turn_id: str
    mode_id: str
    deep: bool
    note: str
    skipped: bool


class ModeChangedEvent(WebSocketEvent):
    type: Literal["mode_changed"] = "mode_changed"
    active_mode_id: str


class ModesListEvent(WebSocketEvent):
    type: Literal["modes_list"] = "modes_list"
    modes: list[dict]
    active_mode_id: str
