import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from newspeak.paths import ROOT_DIR


class ServerConfig(BaseModel):
    port: int = 8000
    llama_server_url: str = "http://localhost:8080"
    llama_model: str = "gemma-4-E2B-it"
    log_level: str = "INFO"


class AudioConfig(BaseModel):
    pipeline: Literal["direct", "whisperx"] = "direct"


class WhisperXConfig(BaseModel):
    model: str = "small"
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"
    compute_type: Literal["auto", "int8", "float16", "float32"] = "auto"
    batch_size: int = 8
    language: str = ""
    download_root: str = ""


class CoachConfig(BaseModel):
    enabled: bool = True
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    auto_max_tokens: int = 180
    deep_max_tokens: int = 600
    history_window: int = 6
    default_mode_id: str = "interview"
    custom_modes_dir: Path = ROOT_DIR / "modes" / "custom"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=str(ROOT_DIR / ".env"),
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)
    coach: CoachConfig = Field(default_factory=CoachConfig)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
