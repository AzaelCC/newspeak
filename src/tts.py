"""Compatibility wrapper for the TTS backend factory."""

from newspeak.adapters.tts import BaseTTSBackend, load_tts_backend

TTSBackend = BaseTTSBackend


def load() -> BaseTTSBackend:
    return load_tts_backend()
