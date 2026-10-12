from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from blending import blend_images, crop_panorama, warp_images
from features import FeatureMatcher
from geometry import canvas_geometry, estimate_homographies, global_transforms
from image_io import load_images
from models import StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")


class PanoramaStitcher:
    """Orchestrates the panorama pipeline without owning its individual algorithms."""

    def __init__(self, config: StitchConfig):
        self.config = config
        self.matcher = FeatureMatcher(config)
        self.images = []
        self.canvas_size: Tuple[int, int] = (0, 0)
        self.translation = np.eye(3, dtype=np.float64)

    def stitch(self, input_folder: str) -> np.ndarray:
        self.images = load_images(input_folder, self.config)
        if len(self.images) == 1:
            return self.images[0].image
        self.matcher.detect(self.images)
        matches = self.matcher.match_adjacent(self.images)
        homographies = estimate_homographies(self.images, matches, self.config)
        transforms, reference = global_transforms(homographies, len(self.images))
        logger.info("Reference image: [%d]", reference)
        self.canvas_size, self.translation = canvas_geometry(self.images, transforms, self.config)
        try:
            result = blend_images(warp_images(self.images, transforms, self.translation, self.canvas_size, self.config), self.canvas_size, self.config)
        except MemoryError as exc:
            raise StitchingError("Ran out of memory; try --max-image-dimension 2000.") from exc
        return crop_panorama(result)
