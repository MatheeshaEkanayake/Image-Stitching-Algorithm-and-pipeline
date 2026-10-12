from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")


class FeatureMatcher:
    """Detect features and match only adjacent images."""

    def __init__(self, config: StitchConfig):
        self.config = config
        self.detector: Optional[cv2.Feature2D] = None
        self.detector_name = ""
        self.norm_type = cv2.NORM_L2

    def _init_detector(self) -> None:
        if self.config.feature_detector.upper() == "SIFT":
            try:
                self.detector = cv2.SIFT_create()
                self.detector_name = "SIFT"
                return
            except (AttributeError, cv2.error) as exc:
                logger.warning("SIFT unavailable (%s); falling back to ORB", exc)
        self.detector = cv2.ORB_create(nfeatures=self.config.orb_n_features)
        self.detector_name = "ORB"
        self.norm_type = cv2.NORM_HAMMING

    def detect(self, images: List[ImageData]) -> None:
        if self.detector is None:
            self._init_detector()
        for index, data in enumerate(images):
            gray = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is None or not keypoints:
                raise StitchingError(f"No features detected in '{data.path}'.")
            data.keypoints = keypoints
            data.descriptors = descriptors
            logger.info("[%d] %s: %d keypoints (%s)", index, os.path.basename(data.path), len(keypoints), self.detector_name)

    def match_adjacent(self, images: List[ImageData]) -> Dict[int, List[cv2.DMatch]]:
        matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)
        matches_by_pair: Dict[int, List[cv2.DMatch]] = {}
        for index in range(len(images) - 1):
            knn_matches = matcher.knnMatch(images[index].descriptors, images[index + 1].descriptors, k=2)
            good_matches = [pair[0] for pair in knn_matches if len(pair) == 2 and pair[0].distance < self.config.ratio_threshold * pair[1].distance]
            matches_by_pair[index] = good_matches
            logger.info("Pair (%d, %d): %d/%d good matches", index, index + 1, len(good_matches), len(knn_matches))
            if self.config.save_debug_images:
                self._save_debug_image(images, index, good_matches)
        return matches_by_pair

    def _save_debug_image(self, images: List[ImageData], index: int, matches: List[cv2.DMatch]) -> None:
        os.makedirs(self.config.debug_dir, exist_ok=True)
        image = cv2.drawMatches(images[index].image, images[index].keypoints, images[index + 1].image, images[index + 1].keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(self.config.debug_dir, f"matches_{index:02d}_{index + 1:02d}.jpg"), image)
