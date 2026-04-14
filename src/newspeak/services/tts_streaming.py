import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

from newspeak.schemas import AudioChunkEvent, AudioEndEvent, AudioStartEvent, WebSocketEvent
from newspeak.services.audio import encode_pcm_float_to_base64_int16
from newspeak.services.ports import TTSBackend
from newspeak.services.text import sentences_or_original

logger = logging.getLogger(__name__)


class TTSStreamer:
    def __init__(self, backend: TTSBackend):
        self._backend = backend

    async def stream(
        self,
        text: str,
        should_stop: Callable[[], bool],
    ) -> AsyncIterator[WebSocketEvent]:
        sentences = sentences_or_original(text)
        yield AudioStartEvent(
            sample_rate=self._backend.sample_rate,
            sentence_count=len(sentences),
        )

        tts_start = time.perf_counter()
        for index, sentence in enumerate(sentences):
            if should_stop():
                logger.info("TTS interrupted before sentence %s/%s", index + 1, len(sentences))
                return

            pcm = await asyncio.to_thread(self._backend.generate, sentence)
            if should_stop():
                logger.info("TTS interrupted after sentence %s/%s", index + 1, len(sentences))
                return

            yield AudioChunkEvent(
                audio=encode_pcm_float_to_base64_int16(pcm),
                index=index,
            )

        tts_time = time.perf_counter() - tts_start
        logger.info("TTS completed tts_time=%.2fs sentence_count=%s", tts_time, len(sentences))
        if not should_stop():
            yield AudioEndEvent(tts_time=round(tts_time, 2))
