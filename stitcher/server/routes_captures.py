"""Captures: create, upload photos one by one, finish (stitch), library, sharing and viewing."""

from __future__ import annotations

import glob
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stitcher.image_io import SUPPORTED_EXTENSIONS
from stitcher.server.app import STATIC_DIR, current_user
from stitcher.server.jobs import JobStore, parse_options

router = APIRouter(prefix="/captures", tags=["captures"])
TILE_NAME = re.compile(r"^\d+_\d+\.(jpg|png)$")
_CHUNK = 1 << 20
User = Dict[str, Any]


class CaptureCreate(BaseModel):
    title: str = ""


class CaptureFinish(BaseModel):
    max_image_dimension: Optional[int] = None
    projection: Optional[str] = None
    blend_mode: Optional[str] = None


class CaptureUpdate(BaseModel):
    title: Optional[str] = None
    public: Optional[bool] = None


def capture_json(row: Any, status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "owner": {"id": row["owner_id"], "username": row["username"]},
        "created": row["created"],
        "is_public": bool(row["is_public"]),
        "state": status.get("state", "unknown"),
        "photo_count": status.get("photo_count", 0),
        "width": status.get("width"),
        "height": status.get("height"),
        "error": status.get("error"),
    }


def _row(request: Request, capture_id: str) -> Any:
    with request.app.state.db.connect() as conn:
        return conn.execute(
            "SELECT c.*, u.username FROM captures c JOIN users u ON u.id = c.owner_id "
            "WHERE c.id = ?",
            (capture_id,),
        ).fetchone()


def load_capture(
    request: Request, capture_id: str, user: User, owner: bool = False, done: bool = False
) -> Dict[str, Any]:
    """Fetch a capture the user may see (own, or published). 404 hides others' private ones."""
    store: JobStore = request.app.state.store
    row = _row(request, capture_id) if store.exists(capture_id) else None
    if row is None or (row["owner_id"] != user["id"] and (owner or not row["is_public"])):
        raise HTTPException(404, "Capture not found.")
    capture = capture_json(row, store.read_status(capture_id))
    if done and capture["state"] != "done":
        raise HTTPException(409, f"Capture is {capture['state']}.")
    return capture


def _photo_files(store: JobStore, capture_id: str) -> List[str]:
    return sorted(glob.glob(os.path.join(store.path(capture_id, "inputs"), "[0-9]" * 4 + ".*")))


@router.post("", status_code=201)
def create_capture(
    body: CaptureCreate, request: Request, user: User = Depends(current_user)  # noqa: B008
) -> Dict[str, Any]:
    store: JobStore = request.app.state.store
    capture_id = store.new_id()
    os.makedirs(store.path(capture_id, "inputs"))
    store.write_status(capture_id, state="capturing", photo_count=0, created=time.time())
    with request.app.state.db.write() as conn:
        conn.execute(
            "INSERT INTO captures (id, owner_id, title, created) VALUES (?, ?, ?, ?)",
            (capture_id, user["id"], body.title.strip()[:120], time.time()),
        )
    return load_capture(request, capture_id, user)


@router.post("/{capture_id}/photos")
async def upload_photo(
    capture_id: str,
    request: Request,
    index: int = Form(...),
    file: UploadFile = File(...),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> Dict[str, Any]:
    """Upload one photo right after it is taken. ``index`` is its left-to-right position;
    re-sending an index replaces that photo (safe to retry after a network error)."""
    settings = request.app.state.settings
    store: JobStore = request.app.state.store
    capture = load_capture(request, capture_id, user, owner=True)
    if capture["state"] != "capturing":
        raise HTTPException(409, "This capture is already finished.")
    if not 0 <= index < settings.max_photos:
        raise HTTPException(400, f"index must be between 0 and {settings.max_photos - 1}.")
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Only jpg, jpeg and png images are accepted.")
    inputs = store.path(capture_id, "inputs")
    target = os.path.join(inputs, f"{index:04d}{extension}")
    partial = target + ".part"
    size = 0
    with open(partial, "wb") as out:
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > settings.max_file_mb * 1e6:
                out.close()
                os.remove(partial)
                raise HTTPException(413, "Photo is too large.")
            out.write(chunk)
    for old in glob.glob(os.path.join(inputs, f"{index:04d}.*")):
        if not old.endswith(".part"):
            os.remove(old)
    os.replace(partial, target)
    count = len(_photo_files(store, capture_id))
    store.write_status(capture_id, photo_count=count)
    return {"index": index, "photo_count": count}


@router.post("/{capture_id}/finish", status_code=202)
def finish_capture(
    capture_id: str,
    request: Request,
    body: Optional[CaptureFinish] = None,
    user: User = Depends(current_user),  # noqa: B008
) -> Dict[str, Any]:
    store: JobStore = request.app.state.store
    capture = load_capture(request, capture_id, user, owner=True)
    if capture["state"] != "capturing":
        raise HTTPException(409, f"Capture is already {capture['state']}.")
    indices = [int(os.path.basename(p)[:4]) for p in _photo_files(store, capture_id)]
    if len(indices) < 2:
        raise HTTPException(400, "Take at least 2 overlapping photos.")
    missing = sorted(set(range(max(indices) + 1)) - set(indices))
    if missing:
        raise HTTPException(400, f"Photos {missing} have not finished uploading.")
    body = body or CaptureFinish()
    try:
        options = parse_options({
            "max_image_dimension": None if body.max_image_dimension is None
            else str(body.max_image_dimension),
            "projection": body.projection, "blend_mode": body.blend_mode,
        })
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.write_status(capture_id, state="queued", options=vars(options))
    request.app.state.runner.submit(capture_id, options)
    return load_capture(request, capture_id, user)


@router.get("")
def my_library(request: Request, user: User = Depends(current_user)) -> List[Dict[str, Any]]:  # noqa: B008
    """The signed-in user's captures, newest first."""
    store: JobStore = request.app.state.store
    with request.app.state.db.connect() as conn:
        rows = conn.execute(
            "SELECT c.*, u.username FROM captures c JOIN users u ON u.id = c.owner_id "
            "WHERE c.owner_id = ? ORDER BY c.created DESC",
            (user["id"],),
        ).fetchall()
    return [capture_json(r, store.read_status(r["id"])) for r in rows if store.exists(r["id"])]


@router.get("/{capture_id}")
def get_capture(
    capture_id: str, request: Request, user: User = Depends(current_user)  # noqa: B008
) -> Dict[str, Any]:
    return load_capture(request, capture_id, user)


@router.patch("/{capture_id}")
def update_capture(
    capture_id: str, body: CaptureUpdate, request: Request,
    user: User = Depends(current_user),  # noqa: B008
) -> Dict[str, Any]:
    capture = load_capture(request, capture_id, user, owner=True)
    with request.app.state.db.write() as conn:
        if body.title is not None:
            conn.execute("UPDATE captures SET title = ? WHERE id = ?",
                         (body.title.strip()[:120], capture_id))
        if body.public is not None:
            if body.public and capture["state"] != "done":
                raise HTTPException(409, "Only finished panoramas can be shared.")
            conn.execute(
                "UPDATE captures SET is_public = ?, published_at = ? WHERE id = ?",
                (int(body.public), time.time() if body.public else None, capture_id),
            )
    return load_capture(request, capture_id, user)


@router.delete("/{capture_id}", status_code=204)
def delete_capture(
    capture_id: str, request: Request, user: User = Depends(current_user)  # noqa: B008
) -> Response:
    load_capture(request, capture_id, user, owner=True)
    with request.app.state.db.write() as conn:
        conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
    request.app.state.store.delete(capture_id)
    return Response(status_code=204)


def _file(request: Request, capture_id: str, *parts: str, media_type: str,
          cache: bool = True) -> FileResponse:
    path = request.app.state.store.path(capture_id, *parts)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found.")
    headers = {"Cache-Control": "private, max-age=86400"} if cache else {}
    return FileResponse(path, media_type=media_type, headers=headers)


@router.get("/{capture_id}/thumb.jpg")
def thumbnail(capture_id: str, request: Request,
              user: User = Depends(current_user)) -> FileResponse:  # noqa: B008
    load_capture(request, capture_id, user, done=True)
    return _file(request, capture_id, "thumb.jpg", media_type="image/jpeg")


@router.get("/{capture_id}/manifest.json")
def manifest(capture_id: str, request: Request,
             user: User = Depends(current_user)) -> FileResponse:  # noqa: B008
    load_capture(request, capture_id, user, done=True)
    return _file(request, capture_id, "tiles", "manifest.json", media_type="application/json")


@router.get("/{capture_id}/tiles/{level}/{name}")
def tile(capture_id: str, level: int, name: str, request: Request,
         user: User = Depends(current_user)) -> FileResponse:  # noqa: B008
    load_capture(request, capture_id, user, done=True)
    if not TILE_NAME.match(name) or not 0 <= level <= 32:
        raise HTTPException(404, "No such tile.")
    media = "image/png" if name.endswith(".png") else "image/jpeg"
    return _file(request, capture_id, "tiles", str(level), name, media_type=media)


@router.get("/{capture_id}/report.json")
def report(capture_id: str, request: Request,
           user: User = Depends(current_user)) -> FileResponse:  # noqa: B008
    load_capture(request, capture_id, user, done=True)
    return _file(request, capture_id, "report.json", media_type="application/json")


@router.get("/{capture_id}/panorama.tif")
def download(capture_id: str, request: Request,
             user: User = Depends(current_user)) -> FileResponse:  # noqa: B008
    load_capture(request, capture_id, user, done=True)
    return _file(request, capture_id, "panorama.tif", media_type="image/tiff", cache=False)


@router.get("/{capture_id}/crop")
def crop(capture_id: str, x: int, y: int, w: int, h: int, request: Request,
         scale: float = 1.0, user: User = Depends(current_user)) -> Response:  # noqa: B008
    """A region of the full-resolution panorama as PNG (scale 1.0 = native pixels)."""
    capture = load_capture(request, capture_id, user, done=True)
    width, height = capture["width"], capture["height"]
    limit = request.app.state.settings.max_crop_megapixels
    if not (0 < scale <= 1.0) or w <= 0 or h <= 0 or x < 0 or y < 0:
        raise HTTPException(400, "Invalid region.")
    if x >= width or y >= height:
        raise HTTPException(400, "Region is outside the image.")
    w, h = min(w, width - x), min(h, height - y)
    if w * h * scale * scale > limit * 1e6:
        raise HTTPException(400, f"Region is larger than {limit:g} MP.")
    image = cv2.imread(request.app.state.store.path(capture_id, "panorama.tif"), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(500, "Panorama is unreadable.")
    region = image[y:y + h, x:x + w]
    if scale < 1.0:
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", region)
    if not ok:
        raise HTTPException(500, "Could not encode the region.")
    return Response(encoded.tobytes(), media_type="image/png")


@router.get("/{capture_id}/view")
def view(capture_id: str) -> FileResponse:
    """Static viewer page. It holds no data: it reads the token from the URL fragment
    (never sent to the server) and fetches everything with an Authorization header."""
    return FileResponse(os.path.join(STATIC_DIR, "viewer.html"), media_type="text/html")
