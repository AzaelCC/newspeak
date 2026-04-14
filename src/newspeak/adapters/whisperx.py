import logging
from typing import Any

from newspeak.config import WhisperXConfig
from newspeak.services.audio import decode_wav_base64

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    def __init__(self, config: WhisperXConfig):
        self.config = config
        self.device = ""
        self.compute_type = ""
        self.model: Any = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _resolve_runtime(self) -> tuple[str, str]:
        requested_device = self.config.device
        cuda_available = self._cuda_available()

        if requested_device == "cpu":
            device = "cpu"
        elif requested_device in {"gpu", "cuda"}:
            if not cuda_available:
                raise RuntimeError(
                    "WhisperX config requested CUDA/GPU, but CUDA is not available. "
                    "Set WHISPERX__DEVICE=cpu or install a CUDA-enabled PyTorch runtime."
                )
            device = "cuda"
        elif cuda_available:
            device = "cuda"
        else:
            device = "cpu"

        if self.config.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.config.compute_type

        return device, compute_type

    def load(self) -> None:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "AUDIO__PIPELINE is 'whisperx', but WhisperX is not installed. "
                "Install it with `uv sync --extra whisper`."
            ) from exc

        self.device, self.compute_type = self._resolve_runtime()
        kwargs: dict[str, Any] = {"compute_type": self.compute_type}
        if self.config.language:
            kwargs["language"] = self.config.language
        if self.config.download_root:
            kwargs["download_root"] = self.config.download_root

        logger.info(
            "Loading WhisperX model=%r device=%r compute_type=%r",
            self.config.model,
            self.device,
            self.compute_type,
        )
        self.model = whisperx.load_model(self.config.model, self.device, **kwargs)
        logger.info("WhisperX loaded")

    def transcribe(self, audio_b64: str) -> str:
        if self.model is None:
            self.load()

        audio = decode_wav_base64(audio_b64)
        result = self.model.transcribe(audio, batch_size=self.config.batch_size)
        segments = result.get("segments", [])
        text = " ".join(segment.get("text", "").strip() for segment in segments)
        return " ".join(text.split())
