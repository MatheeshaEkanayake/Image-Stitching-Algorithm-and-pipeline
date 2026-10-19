"""FastAPI application."""

from __future__ import annotations

import hmac
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from stitcher.image_io import SUPPORTED_EXTENSIONS
from stitcher.server.jobs import JobRunner, JobStore, parse_options

logger = logging.getLogger("PanoramaStitcher")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TILE_NAME = re.compile(r"^\d+_\d+\.(jpg|png)$")
_CHUNK = 1 << 20


@dataclass
class ServerSettings:
    data_dir: str = "server_data"
    api_token: Optional[str] = None  # when set, every job route needs it
    max_files: int = 60
    max_file_mb: float = 50.0
    max_total_mb: float = 500.0
    max_canvas_megapixels: float = 150.0
    max_crop_megapixels: float = 16.0


def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    settings = settings or ServerSettings(
        data_dir=os.environ.get("STITCH_DATA_DIR", "server_data"),
        api_token=os.environ.get("STITCH_API_TOKEN") or None,
    )
    store = JobStore(settings.data_dir)
    runner = JobRunner(store, settings.max_canvas_megapixels)
    if not settings.api_token:
        logger.warning("No STITCH_API_TOKEN set: anyone who can reach this server can upload.")

    def require_token(request: Request) -> None:
        if not settings.api_token:
            return
        supplied = request.query_params.get("token", "")
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            supplied = header[7:]
        if not hmac.compare_digest(supplied.encode(), settings.api_token.encode()):
            raise HTTPException(status_code=401, detail="Missing or invalid token.")

    app = FastAPI(title="Image Stitch", version="0.3.0")
    app.state.store, app.state.runner, app.state.settings = store, runner, settings
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def job_or_404(job_id: str) -> Dict:
        if not store.exists(job_id):
            raise HTTPException(status_code=404, detail="Unknown job.")
        return store.read_status(job_id)

    def done_job(job_id: str) -> Dict:
        status = job_or_404(job_id)
        if status.get("state") != "done":
            raise HTTPException(status_code=409, detail=f"Job is {status.get('state')}.")
        return status

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", status_code=202, dependencies=[Depends(require_token)])
    async def create_job(
        files: List[UploadFile] = File(...),  # noqa: B008
        max_image_dimension: Optional[str] = Form(None),
        projection: Optional[str] = Form(None),
        blend_mode: Optional[str] = Form(None),
        detector: Optional[str] = Form(None),
        tile_size: Optional[str] = Form(None),
    ) -> Dict:
        """Upload the photos in left-to-right order (upload order is the stitch order)."""
        if not 2 <= len(files) <= settings.max_files:
            raise HTTPException(400, f"Send between 2 and {settings.max_files} images.")
        try:
            options = parse_options({
                "max_image_dimension": max_image_dimension, "projection": projection,
                "blend_mode": blend_mode, "detector": detector, "tile_size": tile_size,
            })
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        job_id = store.new_id()
        inputs = store.path(job_id, "inputs")
        os.makedirs(inputs)
        total = 0
        try:
            for index, upload in enumerate(files):
                extension = os.path.splitext(upload.filename or "")[1].lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(400, "Only jpg, jpeg and png images are accepted.")
                size = 0
                # The client's file name is never used on disk; the index keeps upload order.
                with open(os.path.join(inputs, f"{index:04d}{extension}"), "wb") as out:
                    while chunk := await upload.read(_CHUNK):
                        size += len(chunk)
                        total += len(chunk)
                        if size > settings.max_file_mb * 1e6 or total > settings.max_total_mb * 1e6:
                            raise HTTPException(413, "Upload is too large.")
                        out.write(chunk)
        except HTTPException:
            store.delete(job_id)
            raise
        status = store.write_status(
            job_id, state="queued", created=time.time(), image_count=len(files),
            options=vars(options),
        )
        runner.submit(job_id, options)
        return status

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
    def job_status(job_id: str) -> Dict:
        return job_or_404(job_id)

    @app.delete("/jobs/{job_id}", status_code=204, dependencies=[Depends(require_token)])
    def delete_job(job_id: str) -> Response:
        job_or_404(job_id)
        store.delete(job_id)
        return Response(status_code=204)

    @app.get("/jobs/{job_id}/report.json", dependencies=[Depends(require_token)])
    def report(job_id: str) -> FileResponse:
        done_job(job_id)
        return FileResponse(store.path(job_id, "report.json"), media_type="application/json")

    @app.get("/jobs/{job_id}/manifest.json", dependencies=[Depends(require_token)])
    def manifest(job_id: str) -> FileResponse:
        done_job(job_id)
        return FileResponse(store.path(job_id, "tiles", "manifest.json"),
                            media_type="application/json")

    @app.get("/jobs/{job_id}/tiles/{level}/{name}", dependencies=[Depends(require_token)])
    def tile(job_id: str, level: int, name: str) -> FileResponse:
        done_job(job_id)
        if not TILE_NAME.match(name) or not 0 <= level <= 32:
            raise HTTPException(404, "No such tile.")
        path = store.path(job_id, "tiles", str(level), name)
        if not os.path.isfile(path):
            raise HTTPException(404, "No such tile.")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/jobs/{job_id}/panorama.tif", dependencies=[Depends(require_token)])
    def download(job_id: str) -> FileResponse:
        done_job(job_id)
        return FileResponse(store.path(job_id, "panorama.tif"), media_type="image/tiff",
                            filename="panorama.tif")

    @app.get("/jobs/{job_id}/crop", dependencies=[Depends(require_token)])
    def crop(job_id: str, x: int, y: int, w: int, h: int, scale: float = 1.0) -> Response:
        """A region of the full-resolution panorama as PNG (scale 1.0 = native pixels)."""
        status = done_job(job_id)
        width, height = status["width"], status["height"]
        if not (0 < scale <= 1.0) or w <= 0 or h <= 0 or x < 0 or y < 0:
            raise HTTPException(400, "Invalid region.")
        if x >= width or y >= height:
            raise HTTPException(400, "Region is outside the image.")
        w, h = min(w, width - x), min(h, height - y)
        if w * h * scale * scale > settings.max_crop_megapixels * 1e6:
            raise HTTPException(400, f"Region is larger than {settings.max_crop_megapixels:g} MP.")
        image = cv2.imread(store.path(job_id, "panorama.tif"), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(500, "Panorama is unreadable.")
        region = image[y:y + h, x:x + w]
        if scale < 1.0:
            region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", region)
        if not ok:
            raise HTTPException(500, "Could not encode the region.")
        return Response(encoded.tobytes(), media_type="image/png")

    @app.get("/jobs/{job_id}/view", dependencies=[Depends(require_token)])
    def view(job_id: str) -> FileResponse:
        done_job(job_id)
        return FileResponse(os.path.join(STATIC_DIR, "viewer.html"), media_type="text/html")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app
