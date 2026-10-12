"""Image discovery, natural-order sorting and loading."""

from __future__ import annotations

import logging
import os
import re
from typing import List, Tuple, Union

import cv2

from stitcher.models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def natural_sort_key(filename: str) -> List[Union[int, str]]:
    """Sort key so that ``img2.jpg`` comes before ``img10.jpg``."""
    parts = re.split(r"(\d+)", filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_images(folder: str, config: StitchConfig) -> List[ImageData]:
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Input folder not found: {folder}")

    filenames = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not filenames:
        raise FileNotFoundError(f"No jpg/jpeg/png images found in: {folder}")

    images: List[ImageData] = []
    for filename in sorted(filenames, key=natural_sort_key):
        path = os.path.join(folder, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not read image, skipping: %s", path)
            continue

        max_dim = config.max_image_dimension
        if max_dim > 0:
            height, width = image.shape[:2]
            scale = max_dim / max(height, width)
            if scale < 1.0:
                image = cv2.resize(
                    image,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        images.append(ImageData(path=path, image=image))

    if not images:
        raise StitchingError("No valid images could be loaded.")

    logger.info("Loaded %d images from '%s'", len(images), folder)
    for index, data in enumerate(images):
        height, width = data.image.shape[:2]
        logger.info("  [%d] %-30s %dx%d", index, os.path.basename(data.path), width, height)
    return images
