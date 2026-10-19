"""Multi-resolution tile pyramid (Deep Zoom numbering) for zoomable viewers.

Level ``max_level`` is the full-resolution image; every lower level is half the size of the
next one up. Tiles are written to ``<out_dir>/<level>/<column>_<row>.<ext>``, and a
``manifest.json`` describes the pyramid. The lossless panorama stays the master; tiles are
for viewing (JPEG by default).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import cv2
import numpy as np

from stitcher.models import StitchingError

TILE_FORMATS = ("jpg", "png")


@dataclass
class TileManifest:
    width: int
    height: int
    tile_size: int
    format: str
    min_level: int  # smallest level whose longer side still fits one tile
    max_level: int  # full resolution
    tile_count: int

    def level_size(self, level: int) -> Tuple[int, int]:
        """(width, height) in pixels of ``level``."""
        scale = 2.0 ** (level - self.max_level)
        return max(1, math.ceil(self.width * scale)), max(1, math.ceil(self.height * scale))

    def grid(self, level: int) -> Tuple[int, int]:
        """(columns, rows) of tiles at ``level``."""
        width, height = self.level_size(level)
        return math.ceil(width / self.tile_size), math.ceil(height / self.tile_size)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def build_tile_pyramid(
    image: np.ndarray,
    out_dir: str,
    tile_size: int = 512,
    fmt: str = "jpg",
    quality: int = 90,
) -> TileManifest:
    """Write the tile pyramid of a BGR ``image`` to ``out_dir`` and return its manifest."""
    if fmt not in TILE_FORMATS:
        raise ValueError(f"Unknown tile format '{fmt}'; use one of {TILE_FORMATS}.")
    if tile_size < 64:
        raise ValueError("tile_size must be at least 64.")
    height, width = image.shape[:2]
    max_level = max(0, math.ceil(math.log2(max(width, height))))
    # the smallest level whose longer side fits in a single tile
    levels_to_one_tile = math.ceil(math.log2(max(1.0, max(width, height) / tile_size)))
    min_level = max(0, max_level - levels_to_one_tile)
    manifest = TileManifest(width, height, tile_size, fmt, min_level, max_level, 0)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []

    count = 0
    current = image
    for level in range(max_level, min_level - 1, -1):
        level_w, level_h = manifest.level_size(level)
        if current.shape[1] != level_w or current.shape[0] != level_h:
            current = cv2.resize(current, (level_w, level_h), interpolation=cv2.INTER_AREA)
        level_dir = os.path.join(out_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)
        columns, rows = manifest.grid(level)
        for row in range(rows):
            for column in range(columns):
                x0, y0 = column * tile_size, row * tile_size
                tile = current[y0:y0 + tile_size, x0:x0 + tile_size]
                path = os.path.join(level_dir, f"{column}_{row}.{fmt}")
                if not cv2.imwrite(path, tile, params):
                    raise StitchingError(f"Failed to write tile '{path}'.")
                count += 1
    manifest.tile_count = count
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        handle.write(manifest.to_json())
    return manifest
