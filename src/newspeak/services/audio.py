import base64
import io
import wave

import numpy as np


def decode_wav_base64(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64, validate=True)
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV audio, got {sample_width * 8}-bit")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz WAV audio, got {sample_rate} Hz")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def encode_pcm_float_to_base64_int16(pcm: np.ndarray) -> str:
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    return base64.b64encode(pcm_int16.tobytes()).decode()
