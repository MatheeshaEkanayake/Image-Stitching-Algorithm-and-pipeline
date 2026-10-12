from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class StitchConfig:
    feature_detector: str = "SIFT"
    orb_n_features: int = 5000
    ratio_threshold: float = 0.75
    ransac_reproj_threshold: float = 4.0
    min_matches: int = 10
    blend_mode: str = "feather"
    feather_width: int = 50
    max_canvas_megapixels: float = 400.0
    max_image_dimension: int = 0
    save_debug_images: bool = False
    debug_dir: str = "debug"


class StitchingError(Exception):
    """Raised when the pipeline cannot proceed reliably."""


@dataclass
class ImageData:
    path: str
    image: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
