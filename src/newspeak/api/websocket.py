import asyncio
import json
import logging
from collections import deque
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from newspeak.schemas import (
    ClientMessage,
    CoachEvent,
    ErrorEvent,
    ModeChangedEvent,
    ModesListEvent,
    TextEvent,
    WebSocketEvent,
)
from newspeak.services.coach import CoachNote
from newspeak.services.container import ServiceContainer
from newspeak.services.conversation import ConversationTurn

logger = logging.getLogger(__name__)
router = APIRouter()

# Maximum number of turns to cache for deep-dive requests
_TURN_CACHE_SIZE = 20


class WebSocketSession:
    def __init__(self, ws: WebSocket, container: ServiceContainer):
        self._ws = ws
        self._container = container
        self._history: list[dict[str, Any]] = []
        self._interrupted = asyncio.Event()
        self._queue: asyncio.Queue[ClientMessage | ErrorEvent | None] = asyncio.Queue()
        self._send_lock = asyncio.Lock()

        # Coach state
        self._coach_notes: deque[CoachNote] = deque(maxlen=6)
        self._turn_cache: dict[str, tuple[ConversationTurn, list[dict[str, Any]]]] = {}
        self._turn_cache_order: deque[str] = deque(maxlen=_TURN_CACHE_SIZE)
        self._last_turn_id: str | None = None

        # Active mode — resolved on first use
        self._mode_id: str | None = None

    async def run(self) -> None:
        await self._ws.accept()

        # Send mode list on connect if coach is enabled
        if self._container.mode_registry is not None:
            registry = self._container.mode_registry
            active_id = self._resolved_mode_id()
            await self._send_event(
                ModesListEvent(
                    modes=registry.list_dicts(),
                    active_mode_id=active_id,
                )
            )

        receiver_task = asyncio.create_task(self._receive_messages())

        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                if isinstance(item, ErrorEvent):
                    await self._send_event(item)
                    continue

                self._interrupted.clear()
                await self._handle_message(item)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        finally:
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task

    def _resolved_mode_id(self) -> str:
        if self._mode_id:
            return self._mode_id
        if self._container.settings.coach.enabled:
            return self._container.settings.coach.default_mode_id
        return ""

    async def _receive_messages(self) -> None:
        try:
            while True:
                raw = await self._ws.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await self._queue.put(
                        ErrorEvent(code="invalid_json", message="Message must be valid JSON")
                    )
                    continue

                try:
                    message = ClientMessage.model_validate(payload)
                except ValidationError as exc:
                    await self._queue.put(
                        ErrorEvent(code="invalid_payload", message=exc.errors()[0]["msg"])
                    )
                    continue

                if message.is_interrupt:
                    self._interrupted.set()
                    logger.info("Client interrupted")
                    continue

                if message.is_set_mode:
                    await self._handle_set_mode(message.mode_id)
                    continue

                if message.is_deep_dive:
                    await self._queue.put(message)
                    continue

                await self._queue.put(message)
        except WebSocketDisconnect:
            await self._queue.put(None)

    async def _handle_message(self, message: ClientMessage) -> None:
        # Deep-dive request — run coach with deep=True against cached turn
        if message.is_deep_dive:
            await self._handle_deep_dive(message.turn_id)
            return

        conversation = self._container.require_conversation()
        tts_streamer = self._container.require_tts_streamer()

        # Resolve active mode for roleplay system prompt
        active_mode = None
        roleplay_system = None
        if self._container.mode_registry is not None:
            active_mode = self._container.mode_registry.get(self._resolved_mode_id())
            if active_mode:
                roleplay_system = active_mode.roleplay_system

        try:
            turn = await conversation.process(
                message, self._history, roleplay_system=roleplay_system
            )
        except ValueError as exc:
            logger.warning("Client payload failed validation: %s", exc)
            await self._send_event(ErrorEvent(code="invalid_payload", message=str(exc)))
            return
        except Exception as exc:
            logger.exception("Failed to process WebSocket message")
            await self._send_event(ErrorEvent(code="server_error", message=str(exc)))
            return

        if self._interrupted.is_set():
            logger.info("Interrupted after LLM; skipping response")
            return

        self._history.append(turn.user_history_message)
        self._history.append(turn.assistant_history_message)

        # Cache turn for deep-dive (before any awaits so we never lose it)
        self._cache_turn(turn)
        self._last_turn_id = turn.turn_id

        await self._send_event(
            TextEvent(
                text=turn.text_response,
                llm_time=round(turn.llm_time, 2),
                audio_pipeline=turn.audio_pipeline,
                asr_time=round(turn.asr_time, 2) if turn.asr_time is not None else None,
                transcription=turn.transcription,
                turn_id=turn.turn_id,
            )
        )

        if self._interrupted.is_set():
            logger.info("Interrupted before TTS; skipping audio")
            return

        # Fire-and-forget coach task — must start AFTER TTS dispatch to preserve immersion
        if (
            self._container.settings.coach.enabled
            and self._container.coach_service is not None
            and active_mode is not None
            and turn.transcription
        ):
            asyncio.create_task(
                self._run_coach(turn, active_mode, deep=False),
                name=f"coach-{turn.turn_id}",
            )

        async for event in tts_streamer.stream(
            turn.text_response,
            should_stop=self._interrupted.is_set,
        ):
            if self._interrupted.is_set():
                return
            await self._send_event(event)

    async def _handle_deep_dive(self, turn_id: str | None) -> None:
        if not self._container.settings.coach.enabled or self._container.coach_service is None:
            return

        if turn_id is None:
            turn_id = self._last_turn_id

        if turn_id is None:
            await self._send_event(
                ErrorEvent(code="no_turn", message="No turn available for deep dive")
            )
            return

        cached = self._turn_cache.get(turn_id)
        if cached is None:
            await self._send_event(
                ErrorEvent(code="turn_not_found", message="Turn not found in cache")
            )
            return

        turn, history_snapshot = cached
        registry = self._container.mode_registry
        active_mode = registry.get(self._resolved_mode_id()) if registry else None
        if active_mode is None:
            return

        await self._run_coach(turn, active_mode, deep=True, history_override=history_snapshot)

    async def _run_coach(
        self,
        turn: ConversationTurn,
        mode: Any,
        *,
        deep: bool,
        history_override: list[dict[str, Any]] | None = None,
    ) -> None:
        coach = self._container.coach_service
        if coach is None:
            return
        try:
            history = history_override if history_override is not None else list(self._history)
            note = await coach.analyze(
                turn_id=turn.turn_id,
                transcription=turn.transcription or "",
                assistant_text=turn.text_response,
                roleplay_history=history,
                prior_notes=list(self._coach_notes),
                mode=mode,
                deep=deep,
            )
            if not deep:
                self._coach_notes.append(note)
            await self._send_event(
                CoachEvent(
                    turn_id=note.turn_id,
                    mode_id=note.mode_id,
                    deep=note.deep,
                    note=note.note,
                    skipped=note.skipped,
                )
            )
        except Exception:
            logger.exception("Coach analysis failed for turn %s", turn.turn_id)

    async def _handle_set_mode(self, mode_id: str | None) -> None:
        registry = self._container.mode_registry
        if registry is None:
            await self._send_event(
                ErrorEvent(code="modes_unavailable", message="Practice modes are not available")
            )
            return
        if not mode_id:
            await self._send_event(
                ErrorEvent(code="invalid_payload", message="mode_id is required")
            )
            return
        if registry.get(mode_id) is None:
            await self._send_event(
                ErrorEvent(code="invalid_mode", message=f"Unknown practice mode: {mode_id}")
            )
            return

        self._interrupted.set()
        self._mode_id = mode_id
        self._reset_session_state()
        logger.info("Mode switched with fresh session: %r", self._mode_id)
        await self._send_event(ModeChangedEvent(active_mode_id=mode_id))

    def _reset_session_state(self) -> None:
        self._history.clear()
        self._coach_notes.clear()
        self._turn_cache.clear()
        self._turn_cache_order.clear()
        self._last_turn_id = None

    def _cache_turn(self, turn: ConversationTurn) -> None:
        # Evict oldest if at capacity
        if len(self._turn_cache_order) >= _TURN_CACHE_SIZE:
            oldest = self._turn_cache_order[0]
            self._turn_cache.pop(oldest, None)
        history_snapshot = list(self._history)
        self._turn_cache[turn.turn_id] = (turn, history_snapshot)
        self._turn_cache_order.append(turn.turn_id)

    async def _send_event(self, event: WebSocketEvent) -> None:
        async with self._send_lock:
            await self._ws.send_text(event.to_json())


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    container = ws.app.state.container
    session = WebSocketSession(ws, container)
    await session.run()
