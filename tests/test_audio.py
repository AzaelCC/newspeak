import base64
import io
import wave

import numpy as np
import pytest

from newspeak.services.audio import decode_wav_base64, encode_pcm_float_to_base64_int16


def make_wav_b64(
    samples: np.ndarray,
    *,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        if sample_width == 2:
            frames = (samples * 32767).astype("<i2").tobytes()
        else:
            frames = (samples * 127).astype("i1").tobytes()
        wav.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_wav_base64_mono_16khz() -> None:
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    decoded = decode_wav_base64(make_wav_b64(samples))

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, [0.0, 0.49996948, -0.49996948])


def test_decode_wav_base64_downmixes_stereo() -> None:
    stereo = np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float32)

    decoded = decode_wav_base64(make_wav_b64(stereo, channels=2))

    np.testing.assert_allclose(decoded, [0.0, 0.49996948])


def test_decode_wav_base64_rejects_non_16_bit_audio() -> None:
    samples = np.array([0.0, 0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 16-bit"):
        decode_wav_base64(make_wav_b64(samples, sample_width=1))


def test_decode_wav_base64_rejects_non_16khz_audio() -> None:
    samples = np.array([0.0, 0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 16 kHz"):
        decode_wav_base64(make_wav_b64(samples, sample_rate=8000))


def test_encode_pcm_float_to_base64_int16_clips() -> None:
    encoded = encode_pcm_float_to_base64_int16(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    )

    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.int16)

    np.testing.assert_array_equal(decoded, [-32768, -32767, 0, 32767, 32767])
