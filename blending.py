from __future__ import annotations

import cv2
import numpy as np
from typing import Iterator, Tuple

from models import StitchConfig, StitchingError

WarpedImage = Tuple[np.ndarray, np.ndarray]


def warp_images(images, transforms, translation, canvas_size, config: StitchConfig) -> Iterator[WarpedImage]:
    width, height = canvas_size
    for index, data in enumerate(images):
        final_transform = translation @ transforms[index]
        warped = cv2.warpPerspective(data.image, final_transform, (width, height))
        source_mask = np.full(data.image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(source_mask, final_transform, (width, height), flags=cv2.INTER_NEAREST)
        if config.save_debug_images:
            import os
            os.makedirs(config.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(config.debug_dir, f"warped_{index:02d}.jpg"), warped)
        yield warped, mask


def blend_images(warped: Iterator[WarpedImage], canvas_size, config: StitchConfig) -> np.ndarray:
    if config.blend_mode == "multiband":
        raise NotImplementedError("multiband_blend is not implemented; use --blend-mode feather.")
    width, height = canvas_size
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for image, mask in warped:
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        if config.feather_width > 0:
            weight = np.clip(weight, 0, config.feather_width)
        accumulator += image.astype(np.float32) * weight[..., None]
        weight_sum += weight
    safe_weights = np.where(weight_sum == 0, 1.0, weight_sum)
    result = accumulator / safe_weights[..., None]
    result[weight_sum == 0] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_panorama(panorama: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    nonzero = cv2.findNonZero(gray)
    if nonzero is None:
        return panorama
    x, y, width, height = cv2.boundingRect(nonzero)
    return panorama[y:y + height, x:x + width]
