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


_SEAM_WORK_MEGAPIXELS = 0.2  # resolution the seam search runs at
_BLEND_STRENGTH = 5.0  # multi-band overlap width, percent of sqrt(canvas area)


def blend_multiband(
    warped: Iterator[WarpedImage], canvas_size: Tuple[int, int], config: StitchConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Exposure-compensate, find graph-cut seams and blend with a Laplacian pyramid.

    Every pixel is taken (almost) entirely from one image, so moving objects and parallax no
    longer double up; the pyramid hides the brightness step at the seam. Returns the uint8
    panorama and the coverage mask, like :func:`blend_images`.
    """
    items = list(warped)
    width, height = canvas_size
    corners = [item.origin for item in items]
    images = [item.image for item in items]
    masks = [item.mask for item in items]

    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN_BLOCKS
    )
    compensator.feed(corners, images, [cv2.UMat(m) for m in masks])
    for index, (corner, image, mask) in enumerate(zip(corners, images, masks)):
        compensator.apply(index, corner, image, mask)

    scale = min(1.0, float(np.sqrt(_SEAM_WORK_MEGAPIXELS * 1e6 / (width * height))))
    small_images = [
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        for image in images
    ]
    small_masks = [
        cv2.resize(mask, (si.shape[1], si.shape[0]), interpolation=cv2.INTER_NEAREST)
        for mask, si in zip(masks, small_images)
    ]
    small_corners = [(int(round(x * scale)), int(round(y * scale))) for x, y in corners]
    seam_masks = [cv2.UMat(m) for m in small_masks]
    cv2.detail_GraphCutSeamFinder("COST_COLOR").find(small_images, small_corners, seam_masks)

    bands = int(np.clip(np.ceil(np.log2(np.sqrt(width * height) * _BLEND_STRENGTH / 100)) - 1,
                        1, 8))
    blender = cv2.detail_MultiBandBlender(0, bands)
    blender.prepare((0, 0, width, height))
    coverage = np.zeros((height, width), dtype=np.uint8)
    for corner, image, mask, seam in zip(corners, images, masks, seam_masks):
        seam_full = cv2.resize(cv2.dilate(seam.get(), None), (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        seam_full = cv2.bitwise_and(seam_full, mask)
        blender.feed(image.astype(np.int16), seam_full, corner)
        x0, y0 = corner
        h, w = mask.shape
        coverage[y0:y0 + h, x0:x0 + w][mask > 0] = 255
    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8), coverage
