"""Job storage and the background worker."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from stitcher.models import StitchConfig, StitchingError
from stitcher.panorama_stitcher import PanoramaStitcher
from stitcher.tiles import build_tile_pyramid

logger = logging.getLogger("PanoramaStitcher")
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class JobOptions:
    """Subset of StitchConfig that clients may set."""

    max_image_dimension: int = 0
    projection: str = "auto"
    blend_mode: str = "multiband"
    detector: str = "SIFT"
    tile_size: int = 512


class JobStore:
    """Jobs live in ``<data_dir>/jobs/<id>/`` with a ``status.json`` that survives restarts."""

    def __init__(self, data_dir: str) -> None:
        self.root = os.path.join(data_dir, "jobs")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def path(self, job_id: str, *parts: str) -> str:
        if not JOB_ID_PATTERN.match(job_id):
            raise KeyError(job_id)
        return os.path.join(self.root, job_id, *parts)

    def exists(self, job_id: str) -> bool:
        if not JOB_ID_PATTERN.match(job_id):
            return False
        return os.path.isfile(self.path(job_id, "status.json"))

    def write_status(self, job_id: str, **changes: Any) -> Dict[str, Any]:
        with self._lock:
            status = self.read_status(job_id) if self.exists(job_id) else {"id": job_id}
            status.update(changes, updated=time.time())
            target = self.path(job_id, "status.json")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temp = target + ".tmp"
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(status, handle)
            os.replace(temp, target)
            return status

    def read_status(self, job_id: str) -> Dict[str, Any]:
        with open(self.path(job_id, "status.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.path(job_id), ignore_errors=True)

    def fail_interrupted(self) -> int:
        """Mark jobs left queued/running by a previous server process as failed."""
        count = 0
        for job_id in os.listdir(self.root):
            if self.exists(job_id) and self.read_status(job_id).get("state") in (
                "queued", "running"
            ):
                self.write_status(job_id, state="failed",
                                  error="The server restarted while this was processing.")
                count += 1
        return count


class JobRunner:
    """Runs one stitch at a time: a stitch needs gigabytes of RAM, so jobs are queued."""

    def __init__(self, store: JobStore, max_canvas_megapixels: float) -> None:
        self.store = store
        self.max_canvas_megapixels = max_canvas_megapixels
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stitch")

    def submit(self, job_id: str, options: JobOptions) -> None:
        self._pool.submit(self._run, job_id, options)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, options: JobOptions) -> None:
        store = self.store
        store.write_status(job_id, state="running", started=time.time())
        try:
            config = StitchConfig(
                feature_detector=options.detector,
                projection=options.projection,
                blend_mode=options.blend_mode,
                max_image_dimension=options.max_image_dimension,
                max_canvas_megapixels=self.max_canvas_megapixels,
            )
            panorama, report = PanoramaStitcher(config).stitch_with_report(
                store.path(job_id, "inputs")
            )
            if not cv2.imwrite(store.path(job_id, "panorama.tif"), panorama):
                raise StitchingError("Failed to write the panorama.")
            with open(store.path(job_id, "report.json"), "w", encoding="utf-8") as handle:
                handle.write(report.to_json())
            write_thumbnail(panorama, store.path(job_id, "thumb.jpg"))
            manifest = build_tile_pyramid(
                panorama, store.path(job_id, "tiles"), tile_size=options.tile_size
            )
            store.write_status(
                job_id, state="done", width=manifest.width, height=manifest.height,
                finished=time.time(),
            )
        except (StitchingError, ValueError, FileNotFoundError) as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            store.write_status(job_id, state="failed", error=str(exc), finished=time.time())
        except Exception:  # noqa: BLE001 - never let a job crash the worker thread
            logger.exception("Job %s crashed", job_id)
            store.write_status(
                job_id, state="failed", error="Internal error while stitching.",
                finished=time.time(),
            )


def write_thumbnail(panorama: np.ndarray, path: str, width: int = 1024) -> None:
    """Small JPEG preview for library and feed cards."""
    height, full_width = panorama.shape[:2]
    scale = min(1.0, width / full_width)
    thumb = cv2.resize(panorama, (max(1, round(full_width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])


def parse_options(raw: Dict[str, Optional[str]]) -> JobOptions:
    """Validate client-supplied options; raises ValueError with a client-safe message."""
    options = JobOptions()
    if raw.get("max_image_dimension") not in (None, ""):
        options.max_image_dimension = int(raw["max_image_dimension"])  # type: ignore[arg-type]
        if not 0 <= options.max_image_dimension <= 20000:
            raise ValueError("max_image_dimension must be between 0 and 20000.")
    for name in ("projection", "blend_mode", "detector"):
        if raw.get(name):
            setattr(options, name, str(raw[name]))
    if raw.get("tile_size") not in (None, ""):
        options.tile_size = int(raw["tile_size"])  # type: ignore[arg-type]
        if not 128 <= options.tile_size <= 2048:
            raise ValueError("tile_size must be between 128 and 2048.")
    StitchConfig(  # validates projection / blend_mode / detector names
        feature_detector=options.detector, projection=options.projection,
        blend_mode=options.blend_mode,
    )
    return options
