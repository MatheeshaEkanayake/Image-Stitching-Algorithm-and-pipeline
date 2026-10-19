"""FastAPI application factory and shared dependencies."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from stitcher.server.db import Database
from stitcher.server.jobs import JobRunner, JobStore
from stitcher.server.security import LoginThrottle, token_hash

logger = logging.getLogger("PanoramaStitcher")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@dataclass
class ServerSettings:
    data_dir: str = "server_data"
    allow_registration: bool = True
    session_days: float = 30.0
    max_photos: int = 60
    max_file_mb: float = 50.0
    max_canvas_megapixels: float = 150.0
    max_crop_megapixels: float = 16.0
    max_post_chars: int = 5000
    max_comment_chars: int = 2000

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            data_dir=os.environ.get("STITCH_DATA_DIR", "server_data"),
            allow_registration=os.environ.get("STITCH_ALLOW_REGISTRATION", "1") != "0",
        )


def current_user(request: Request) -> Dict[str, Any]:
    """Dependency: the user owning the ``Authorization: Bearer`` token, else 401."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required.")
    db: Database = request.app.state.db
    with db.connect() as conn:
        row = conn.execute(
            "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires > ?",
            (token_hash(header[7:].strip()), time.time()),
        ).fetchone()
    if row is None:
        raise HTTPException(401, "Session expired or invalid; sign in again.")
    return {"id": row["id"], "username": row["username"]}


def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    from stitcher.server import routes_auth, routes_captures, routes_community

    settings = settings or ServerSettings.from_env()
    os.makedirs(settings.data_dir, exist_ok=True)
    store = JobStore(settings.data_dir)
    interrupted = store.fail_interrupted()
    if interrupted:
        logger.warning("Marked %d interrupted capture(s) as failed.", interrupted)

    app = FastAPI(title="Image Stitch", version="0.4.0")
    app.state.settings = settings
    app.state.store = store
    app.state.runner = JobRunner(store, settings.max_canvas_megapixels)
    app.state.db = Database(os.path.join(settings.data_dir, "app.db"))
    app.state.throttle = LoginThrottle()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.include_router(routes_auth.router)
    app.include_router(routes_captures.router)
    app.include_router(routes_community.router)
    return app
