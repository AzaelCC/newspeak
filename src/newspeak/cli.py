import uvicorn

from newspeak.app import create_app
from newspeak.config import AppConfig


def main() -> None:
    settings = AppConfig()
    uvicorn.run(create_app(settings), host="0.0.0.0", port=settings.server.port)
