import logging
import os
import platform
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class BaseTTSBackend:
    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(BaseTTSBackend):
    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(result.audio) for result in results])


class ONNXBackend(BaseTTSBackend):
    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sample_rate = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load_tts_backend() -> BaseTTSBackend:
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            logger.info("TTS backend loaded: mlx-audio sample_rate=%s", backend.sample_rate)
            return backend
        except ImportError:
            logger.info("mlx-audio is not installed; falling back to kokoro-onnx")

    backend = ONNXBackend()
    logger.info("TTS backend loaded: kokoro-onnx sample_rate=%s", backend.sample_rate)
    return backend
