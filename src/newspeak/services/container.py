import logging
from dataclasses import dataclass

from newspeak.adapters.llm import OpenAIChatClient
from newspeak.adapters.tts import load_tts_backend
from newspeak.adapters.whisperx import WhisperXTranscriber
from newspeak.config import AppConfig
from newspeak.services.coach import CoachService
from newspeak.services.container_modes import build_mode_registry
from newspeak.services.conversation import ConversationService
from newspeak.services.modes import ModeRegistry
from newspeak.services.ports import ChatClient, SpeechTranscriber, TTSBackend
from newspeak.services.tts_streaming import TTSStreamer

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    settings: AppConfig
    chat_client: ChatClient | None = None
    transcriber: SpeechTranscriber | None = None
    tts_backend: TTSBackend | None = None
    conversation: ConversationService | None = None
    tts_streamer: TTSStreamer | None = None
    coach_client: OpenAIChatClient | None = None
    coach_service: CoachService | None = None
    mode_registry: ModeRegistry | None = None

    def load(self) -> None:
        self.chat_client = OpenAIChatClient(self.settings.server.llama_server_url)

        if self.settings.audio.pipeline == "whisperx":
            transcriber = WhisperXTranscriber(self.settings.whisperx)
            transcriber.load()
            self.transcriber = transcriber

        self.tts_backend = load_tts_backend()
        self.conversation = ConversationService(
            self.settings,
            self.chat_client,
            self.transcriber,
        )
        self.tts_streamer = TTSStreamer(self.tts_backend)

        # Coach setup
        coach_cfg = self.settings.coach
        if coach_cfg.enabled:
            coach_base_url = coach_cfg.base_url or self.settings.server.llama_server_url
            coach_api_key = coach_cfg.api_key or "not-needed"
            if coach_base_url != self.settings.server.llama_server_url:
                self.coach_client = OpenAIChatClient(coach_base_url, coach_api_key)
            else:
                # Share the primary client to avoid duplicate connections
                self.coach_client = self.chat_client  # type: ignore[assignment]
                if coach_cfg.base_url is None:
                    logger.info("coach.base_url not set — sharing primary LLM endpoint for coach")

            if coach_cfg.model is None:
                logger.info(
                    "coach.model not set — will use server.llama_model (%s) for coach",
                    self.settings.server.llama_model,
                )
                # Fall back to primary model for coach
                coach_cfg.model = self.settings.server.llama_model

            self.coach_service = CoachService(self.coach_client, coach_cfg)
            self.mode_registry = build_mode_registry(coach_cfg)

    def require_conversation(self) -> ConversationService:
        if self.conversation is None:
            raise RuntimeError("Conversation service has not been loaded")
        return self.conversation

    def require_tts_streamer(self) -> TTSStreamer:
        if self.tts_streamer is None:
            raise RuntimeError("TTS streamer has not been loaded")
        return self.tts_streamer

    def require_mode_registry(self) -> ModeRegistry:
        if self.mode_registry is None:
            raise RuntimeError("Mode registry has not been loaded (coach may be disabled)")
        return self.mode_registry
