"""Configuration, data containers and the run report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BLEND_MODES: Tuple[str, ...] = ("multiband", "feather")
DETECTORS: Tuple[str, ...] = ("SIFT", "ORB")
PROJECTIONS: Tuple[str, ...] = ("auto", "cylindrical", "planar")
MATCHINGS: Tuple[str, ...] = ("auto", "all", "adjacent")


class StitchingError(Exception):
    """Raised when the pipeline cannot proceed reliably."""


@dataclass
class StitchConfig:
    feature_detector: str = "SIFT"
    orb_n_features: int = 5000
    ratio_threshold: float = 0.75
    ransac_reproj_threshold: float = 4.0
    matching: str = "auto"
    max_all_pairs_images: int = 30
    match_window: int = 5
    min_matches: int = 10
    min_inliers: int = 15
    max_area_ratio: float = 10.0
    projection: str = "auto"
    max_rotation_fit_px: float = 5.0
    focal_length: float = 0.0
    bundle_adjust: bool = True
    blend_mode: str = "multiband"
    multiband_max_megapixels: float = 40.0
    feather_width: int = 50
    max_canvas_megapixels: float = 400.0
    max_image_dimension: int = 0
    seed: int = 0
    save_debug_images: bool = False
    debug_dir: str = "debug"

    def __post_init__(self) -> None:
        self.feature_detector = self.feature_detector.upper()
        if self.feature_detector not in DETECTORS:
            raise ValueError(f"Unknown feature detector '{self.feature_detector}'.")
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f"Unknown blend mode '{self.blend_mode}'; use one of {BLEND_MODES}.")
        if self.projection not in PROJECTIONS:
            raise ValueError(f"Unknown projection '{self.projection}'; use one of {PROJECTIONS}.")
        if self.matching not in MATCHINGS:
            raise ValueError(f"Unknown matching mode '{self.matching}'; use one of {MATCHINGS}.")
        if self.match_window < 1:
            raise ValueError("match_window must be at least 1.")
        if self.max_area_ratio <= 1.0:
            raise ValueError("max_area_ratio must be greater than 1.")


@dataclass
class ImageData:
    path: str
    image: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None


@dataclass
class PairStats:
    """Matching statistics for an overlapping pair of images (0-based indices a < b)."""

    a: int
    b: int
    matches: int
    inliers: int
    inlier_ratio: float


@dataclass
class StitchReport:
    """Machine-readable summary of one run (JSON-serialisable via ``to_json``)."""

    image_names: List[str] = field(default_factory=list)
    image_sizes: List[Tuple[int, int]] = field(default_factory=list)  # (width, height)
    detector: str = ""
    keypoints: List[int] = field(default_factory=list)
    pairs_tried: int = 0
    pairs: List[PairStats] = field(default_factory=list)
    projection: str = ""  # model actually used ("cylindrical" or "planar")
    reference_index: int = 0
    focal_length: Optional[float] = None  # pixels; cylindrical projection only
    canvas_size: Tuple[int, int] = (0, 0)  # (width, height)
    canvas_megapixels: float = 0.0
    output_size: Tuple[int, int] = (0, 0)  # (width, height) after cropping
    timings_s: Dict[str, float] = field(default_factory=dict)
    peak_memory_mb: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
