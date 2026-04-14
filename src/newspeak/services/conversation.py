import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from newspeak.config import AppConfig
from newspeak.schemas import ClientMessage
from newspeak.services.ports import ChatClient, SpeechTranscriber
from newspeak.services.prompts import build_history_user_message, build_user_content

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversationTurn:
    turn_id: str
    text_response: str
    user_history_message: dict[str, Any]
    assistant_history_message: dict[str, Any]
    llm_time: float
    transcription: str | None = None
    asr_time: float | None = None
    audio_pipeline: str | None = None


class ConversationService:
    def __init__(
        self,
        settings: AppConfig,
        chat_client: ChatClient,
        transcriber: SpeechTranscriber | None = None,
    ):
        self._settings = settings
        self._chat_client = chat_client
        self._transcriber = transcriber

    async def process(
        self,
        message: ClientMessage,
        history: list[dict[str, Any]],
        roleplay_system: str | None = None,
    ) -> ConversationTurn:
        has_audio = bool(message.audio)
        audio_pipeline = self._settings.audio.pipeline if has_audio else None
        transcription = None
        asr_time = None

        if has_audio and self._settings.audio.pipeline == "whisperx":
            if self._transcriber is None:
                raise RuntimeError(
                    "WhisperX pipeline is enabled, but the transcriber is not loaded"
                )

            asr_start = time.perf_counter()
            transcription = await asyncio.to_thread(
                self._transcriber.transcribe, message.audio or ""
            )
            asr_time = time.perf_counter() - asr_start
            logger.info("WhisperX heard transcription=%r asr_time=%.2fs", transcription, asr_time)

        content = build_user_content(
            message,
            audio_pipeline=audio_pipeline,
            transcription=transcription,
        )
        user_message = {"role": "user", "content": content}
        use_transcription_tool = has_audio and self._settings.audio.pipeline == "direct"

        llm_start = time.perf_counter()
        chat_result = await self._chat_client.complete(
            model=self._settings.server.llama_model,
            history=history,
            user_message=user_message,
            use_transcription_tool=use_transcription_tool,
            roleplay_system=roleplay_system,
        )
        llm_time = time.perf_counter() - llm_start

        if transcription is None:
            transcription = chat_result.transcription

        if chat_result.mode == "tool":
            logger.info(
                "LLM tool response llm_time=%.2fs transcription=%r text=%r",
                llm_time,
                transcription,
                chat_result.text,
            )
        else:
            logger.info("LLM chat response llm_time=%.2fs text=%r", llm_time, chat_result.text)

        return ConversationTurn(
            turn_id=str(uuid.uuid4()),
            text_response=chat_result.text,
            transcription=transcription,
            asr_time=asr_time,
            audio_pipeline=audio_pipeline,
            llm_time=llm_time,
            user_history_message=build_history_user_message(
                content,
                has_audio=has_audio,
                transcription=transcription,
            ),
            assistant_history_message={"role": "assistant", "content": chat_result.text},
        )
