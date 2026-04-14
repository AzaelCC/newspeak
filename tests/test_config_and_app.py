from newspeak.app import create_app
from newspeak.config import AppConfig, AudioConfig, ServerConfig
from newspeak.services.container import ServiceContainer


def test_config_defaults_ignore_env_file() -> None:
    settings = AppConfig(_env_file=None)

    assert settings.server.port == 8000
    assert settings.server.llama_server_url == "http://localhost:8080"
    assert settings.server.llama_model == "gemma-4-E2B-it"
    assert settings.server.log_level == "INFO"
    assert settings.audio.pipeline == "direct"


def test_config_reads_nested_environment(monkeypatch) -> None:
    monkeypatch.setenv("SERVER__PORT", "9001")
    monkeypatch.setenv("SERVER__LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AUDIO__PIPELINE", "whisperx")

    settings = AppConfig(_env_file=None)

    assert settings.server.port == 9001
    assert settings.server.log_level == "DEBUG"
    assert settings.audio.pipeline == "whisperx"


def test_create_app_registers_routes_without_loading_services() -> None:
    settings = AppConfig(
        server=ServerConfig(port=9000),
        audio=AudioConfig(pipeline="direct"),
        _env_file=None,
    )

    app = create_app(settings, container=ServiceContainer(settings))
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/ws" in paths
