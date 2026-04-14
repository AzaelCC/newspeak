import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from newspeak.api.websocket import WebSocketSession
from newspeak.config import AppConfig, AudioConfig
from newspeak.schemas import (
    AudioChunkEvent,
    AudioEndEvent,
    AudioStartEvent,
    ClientMessage,
    ModeChangedEvent,
    TextEvent,
)
from newspeak.services.coach import CoachNote
from newspeak.services.conversation import ConversationService, ConversationTurn
from newspeak.services.modes import Mode, ModeRegistry
from newspeak.services.ports import ChatResult
from newspeak.services.tts_streaming import TTSStreamer


class FakeChatClient:
    def __init__(self, result: ChatResult):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        history: Sequence[dict[str, Any]],
        user_message: dict[str, Any],
        use_transcription_tool: bool,
        roleplay_system: str | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "model": model,
                "history": history,
                "user_message": user_message,
                "use_transcription_tool": use_transcription_tool,
                "roleplay_system": roleplay_system,
            }
        )
        return self.result


class FakeTranscriber:
    def transcribe(self, audio_b64: str) -> str:
        return f"heard {audio_b64}"


class FakeTTSBackend:
    sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        return np.array([0.0, 0.25], dtype=np.float32)


def test_text_event_serializes_current_wire_shape() -> None:
    event = TextEvent(
        text="hello",
        llm_time=1.23,
        audio_pipeline="direct",
        transcription="hi",
    )

    payload = json.loads(event.to_json())

    assert payload == {
        "type": "text",
        "text": "hello",
        "llm_time": 1.23,
        "audio_pipeline": "direct",
        "transcription": "hi",
    }


def test_mode_changed_event_serializes_current_wire_shape() -> None:
    event = ModeChangedEvent(active_mode_id="language_en")

    payload = json.loads(event.to_json())

    assert payload == {"type": "mode_changed", "active_mode_id": "language_en"}


def test_direct_audio_conversation_uses_transcription_tool() -> None:
    chat = FakeChatClient(ChatResult(text="answer", transcription="hello", mode="tool"))
    service = ConversationService(AppConfig(_env_file=None), chat)

    turn = asyncio.run(service.process(message_from_audio("wav"), []))

    assert chat.calls[0]["use_transcription_tool"] is True
    assert turn.transcription == "hello"
    assert turn.audio_pipeline == "direct"
    assert turn.assistant_history_message == {"role": "assistant", "content": "answer"}


def test_whisperx_conversation_uses_transcriber_before_chat() -> None:
    chat = FakeChatClient(ChatResult(text="answer"))
    settings = AppConfig(audio=AudioConfig(pipeline="whisperx"), _env_file=None)
    service = ConversationService(settings, chat, FakeTranscriber())

    turn = asyncio.run(service.process(message_from_audio("wav"), []))

    assert chat.calls[0]["use_transcription_tool"] is False
    assert "heard wav" in chat.calls[0]["user_message"]["content"][0]["text"]
    assert turn.transcription == "heard wav"
    assert turn.asr_time is not None


def test_tts_streamer_emits_audio_events() -> None:
    streamer = TTSStreamer(FakeTTSBackend())

    async def collect() -> list[Any]:
        return [event async for event in streamer.stream("Hello. Bye.", should_stop=lambda: False)]

    events = asyncio.run(collect())

    assert isinstance(events[0], AudioStartEvent)
    assert events[0].sentence_count == 2
    assert isinstance(events[1], AudioChunkEvent)
    assert isinstance(events[2], AudioChunkEvent)
    assert isinstance(events[3], AudioEndEvent)


def test_set_mode_resets_session_state() -> None:
    session = make_session()
    turn = make_turn("old-turn")
    session._history.extend([{"role": "user", "content": "old"}])
    session._coach_notes.append(
        CoachNote(
            turn_id="old-turn",
            mode_id="interview",
            note="old note",
            skipped=False,
            deep=False,
        )
    )
    session._turn_cache["old-turn"] = (turn, [])
    session._turn_cache_order.append("old-turn")
    session._last_turn_id = "old-turn"

    asyncio.run(session._handle_set_mode("language_en"))

    assert session._resolved_mode_id() == "language_en"
    assert session._history == []
    assert list(session._coach_notes) == []
    assert session._turn_cache == {}
    assert list(session._turn_cache_order) == []
    assert session._last_turn_id is None
    assert session._interrupted.is_set()
    assert json.loads(session._ws.sent[-1]) == {
        "type": "mode_changed",
        "active_mode_id": "language_en",
    }


def test_set_mode_clears_history_before_next_turn() -> None:
    conversation = FakeConversation()
    session = make_session(conversation=conversation)
    session._history.extend([{"role": "user", "content": "old"}])

    asyncio.run(session._handle_set_mode("language_en"))
    session._interrupted.clear()
    asyncio.run(session._handle_message(ClientMessage(text="hello")))

    assert conversation.calls[0]["history"] == []
    assert conversation.calls[0]["roleplay_system"] == "roleplay:language_en"


def test_invalid_set_mode_keeps_existing_session() -> None:
    session = make_session()
    session._mode_id = "interview"
    session._history.append({"role": "user", "content": "old"})
    session._last_turn_id = "old-turn"

    asyncio.run(session._handle_set_mode("missing"))

    assert session._resolved_mode_id() == "interview"
    assert session._history == [{"role": "user", "content": "old"}]
    assert session._last_turn_id == "old-turn"
    assert json.loads(session._ws.sent[-1]) == {
        "type": "error",
        "code": "invalid_mode",
        "message": "Unknown practice mode: missing",
    }


def test_deep_dive_after_mode_reset_cannot_use_previous_turn() -> None:
    session = make_session(coach_service=object())
    turn = make_turn("old-turn")
    session._turn_cache["old-turn"] = (turn, [])
    session._turn_cache_order.append("old-turn")
    session._last_turn_id = "old-turn"

    asyncio.run(session._handle_set_mode("language_en"))
    asyncio.run(session._handle_deep_dive("old-turn"))

    assert json.loads(session._ws.sent[-1]) == {
        "type": "error",
        "code": "turn_not_found",
        "message": "Turn not found in cache",
    }


def message_from_audio(audio: str):
    from newspeak.schemas import ClientMessage

    return ClientMessage(audio=audio)


class DummyWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


class FakeConversation:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def process(
        self,
        message: ClientMessage,
        history: list[dict[str, Any]],
        roleplay_system: str | None = None,
    ) -> ConversationTurn:
        self.calls.append(
            {
                "message": message,
                "history": list(history),
                "roleplay_system": roleplay_system,
            }
        )
        return make_turn("new-turn")


class FakeEmptyTTSStreamer:
    async def stream(self, text: str, should_stop):
        if False:
            yield None


def make_session(
    *,
    conversation: FakeConversation | None = None,
    coach_service: object | None = None,
) -> WebSocketSession:
    registry = ModeRegistry()
    for mode_id in ("interview", "language_en"):
        registry.register(
            Mode(
                id=mode_id,
                name=mode_id,
                roleplay_system=f"roleplay:{mode_id}",
                coach_system_auto="auto",
                coach_system_deep="deep",
            )
        )

    fake_conversation = conversation or FakeConversation()
    container = SimpleNamespace(
        settings=AppConfig(_env_file=None),
        mode_registry=registry,
        coach_service=coach_service,
        require_conversation=lambda: fake_conversation,
        require_tts_streamer=lambda: FakeEmptyTTSStreamer(),
    )
    return WebSocketSession(DummyWebSocket(), container)


def make_turn(turn_id: str) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        text_response="answer",
        transcription="hello",
        llm_time=0.1,
        user_history_message={"role": "user", "content": "hello"},
        assistant_history_message={"role": "assistant", "content": "answer"},
    )
