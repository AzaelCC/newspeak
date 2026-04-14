import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from newspeak.api.http import router as http_router
from newspeak.api.websocket import router as websocket_router
from newspeak.config import AppConfig, configure_logging
from newspeak.paths import STATIC_DIR
from newspeak.services.container import ServiceContainer


def create_app(
    settings: AppConfig | None = None,
    container: ServiceContainer | None = None,
) -> FastAPI:
    app_settings = settings or AppConfig()
    configure_logging(app_settings.server.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        app.state.container = container or ServiceContainer(app_settings)
        if container is None:
            await asyncio.to_thread(app.state.container.load)
        yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(http_router)
    app.include_router(websocket_router)
    return app
