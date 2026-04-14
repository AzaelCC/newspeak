from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from newspeak.paths import STATIC_DIR

router = APIRouter()


@router.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@router.get("/modes")
async def list_modes(request: Request) -> JSONResponse:
    container = request.app.state.container
    if container.mode_registry is None:
        return JSONResponse({"modes": [], "active_mode_id": ""})
    registry = container.mode_registry
    active_id = container.settings.coach.default_mode_id
    return JSONResponse({"modes": registry.list_dicts(), "active_mode_id": active_id})
