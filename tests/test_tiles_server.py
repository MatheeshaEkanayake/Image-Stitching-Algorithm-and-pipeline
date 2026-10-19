from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from stitcher.tiles import TileManifest, build_tile_pyramid


def test_tile_pyramid_levels_and_lossless_top_level(tmp_path):
    rng = np.random.default_rng(0)
    image = cv2.GaussianBlur(rng.integers(0, 255, (700, 1300, 3), dtype=np.uint8), (0, 0), 2)
    manifest = build_tile_pyramid(image, str(tmp_path), tile_size=256, fmt="png")
    assert (manifest.width, manifest.height) == (1300, 700)
    assert manifest.max_level == 11  # ceil(log2(1300))
    assert manifest.level_size(manifest.max_level) == (1300, 700)
    assert manifest.level_size(manifest.min_level)[0] <= 256 * 1  # fits one tile
    assert manifest.grid(manifest.max_level) == (6, 3)
    # PNG tiles at full resolution reassemble to the exact original
    columns, rows = manifest.grid(manifest.max_level)
    canvas = np.zeros_like(image)
    for r in range(rows):
        for c in range(columns):
            tile = cv2.imread(os.path.join(tmp_path, str(manifest.max_level), f"{c}_{r}.png"))
            canvas[r * 256:r * 256 + tile.shape[0], c * 256:c * 256 + tile.shape[1]] = tile
    assert np.array_equal(canvas, image)
    count = sum(len(f) for _, _, f in os.walk(tmp_path) if f) - 1  # minus manifest.json
    assert manifest.tile_count == count
    reloaded = TileManifest(**json.loads((tmp_path / "manifest.json").read_text()))
    assert reloaded == manifest


def test_tile_pyramid_small_image_is_one_level(tmp_path):
    manifest = build_tile_pyramid(np.zeros((100, 200, 3), np.uint8), str(tmp_path), tile_size=512)
    assert manifest.min_level == manifest.max_level and manifest.tile_count == 1


def test_tile_pyramid_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_tile_pyramid(np.zeros((10, 10, 3), np.uint8), str(tmp_path), fmt="gif")
    with pytest.raises(ValueError):
        build_tile_pyramid(np.zeros((10, 10, 3), np.uint8), str(tmp_path), tile_size=8)


