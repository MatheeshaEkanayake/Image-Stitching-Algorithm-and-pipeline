"""Region-of-interest warping, feather blending and coverage-based cropping."""

from __future__ import annotations

import logging
import os
from typing import Iterator, List, NamedTuple, Tuple

import cv2
import numpy as np

from stitcher.geometry import image_corners, project_points
from stitcher.models import ImageData, StitchConfig

logger = logging.getLogger("PanoramaStitcher")

# Zero-valued margin (pixels) kept around each warped image so that the distance
# transform sees the same background it would see on the full canvas.
_ROI_PAD = 2


class WarpedImage(NamedTuple):
    """An image warped into its own bounding box of the canvas."""

    image: np.ndarray  # (h, w, 3) uint8
    mask: np.ndarray  # (h, w) uint8, 255 where the image has data
    origin: Tuple[int, int]  # (x, y) of the box's top-left corner on the canvas


def warp_images(
    images: List[ImageData],
    transforms: List[np.ndarray],
    translation: np.ndarray,
    canvas_size: Tuple[int, int],
    config: StitchConfig,
) -> Iterator[WarpedImage]:
    """Warp each image only into the part of the canvas it covers."""
    canvas_width, canvas_height = canvas_size
    for index, data in enumerate(images):
        height, width = data.image.shape[:2]
        final_transform = translation @ transforms[index]
        projected, _ = project_points(image_corners(width, height), final_transform)
        x0 = max(0, int(np.floor(projected[:, 0].min())) - _ROI_PAD)
        y0 = max(0, int(np.floor(projected[:, 1].min())) - _ROI_PAD)
        x1 = min(canvas_width, int(np.ceil(projected[:, 0].max())) + _ROI_PAD)
        y1 = min(canvas_height, int(np.ceil(projected[:, 1].max())) + _ROI_PAD)
        if x1 <= x0 or y1 <= y0:
            logger.warning("Image %d lies outside the canvas; skipped.", index)
            continue
        roi_size = (x1 - x0, y1 - y0)
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        roi_transform = shift @ final_transform
        warped = cv2.warpPerspective(data.image, roi_transform, roi_size)
        source_mask = np.full((height, width), 255, dtype=np.uint8)
        mask = cv2.warpPerspective(source_mask, roi_transform, roi_size, flags=cv2.INTER_NEAREST)
        if config.save_debug_images:
            os.makedirs(config.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(config.debug_dir, f"warped_{index:02d}.jpg"), warped)
        yield WarpedImage(warped, mask, (x0, y0))


def blend_images(
    warped: Iterator[WarpedImage], canvas_size: Tuple[int, int], config: StitchConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Feather-blend warped images into a canvas.

    Returns the uint8 panorama and the uint8 coverage mask (255 wherever at least one
    image contributed, i.e. the union of the warped image masks).
    """
    width, height = canvas_size
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    for image, mask, (x0, y0) in warped:
        h, w = mask.shape
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        if config.feather_width > 0:
            weight = np.clip(weight, 0, config.feather_width)
        accumulator[y0:y0 + h, x0:x0 + w] += image.astype(np.float32) * weight[..., None]
        weight_sum[y0:y0 + h, x0:x0 + w] += weight
        coverage[y0:y0 + h, x0:x0 + w][mask > 0] = 255
    # Pixels nobody covers have an all-zero accumulator, so dividing by 1 leaves them black.
    np.divide(accumulator, np.where(weight_sum == 0, 1.0, weight_sum)[..., None], out=accumulator)
    np.clip(accumulator, 0, 255, out=accumulator)
    return accumulator.astype(np.uint8), coverage


def crop_panorama(panorama: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of the coverage mask.

    Dark but genuine image content is kept because only the mask, never the pixel
    values, decides what is empty canvas.
    """
    rows = np.flatnonzero(coverage.any(axis=1))
    cols = np.flatnonzero(coverage.any(axis=0))
    if rows.size == 0:
        return panorama
    return panorama[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
