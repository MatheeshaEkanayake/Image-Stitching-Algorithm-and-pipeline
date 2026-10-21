"""Feature detection and adjacent-pair matching."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2

from stitcher.models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")


def candidate_pairs(count: int, config: StitchConfig) -> List[Tuple[int, int]]:
    """Image pairs ``(i, j)``, ``i < j``, to try to match.

    ``all`` tries every pair, so the capture order does not matter (rows, snake patterns,
    right-to-left). ``adjacent`` tries only neighbours within ``config.match_window`` positions
    in capture order, which is much cheaper for long sequences. ``auto`` uses ``all`` up to
    ``config.max_all_pairs_images`` photos and ``adjacent`` beyond that.
    """
    mode = config.matching
    if mode == "auto":
        mode = "all" if count <= config.max_all_pairs_images else "adjacent"
    window = count if mode == "all" else config.match_window
    return [(i, j) for i in range(count) for j in range(i + 1, min(count, i + 1 + window))]


class FeatureMatcher:
    """Detect features and match only adjacent images."""

    def __init__(self, config: StitchConfig) -> None:
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
        assert self.detector is not None
        for index, data in enumerate(images):
            gray = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is None or not keypoints:
                raise StitchingError(f"No features detected in '{data.path}'.")
            data.keypoints = list(keypoints)
            data.descriptors = descriptors
            logger.info(
                "[%d] %s: %d keypoints (%s)",
                index,
                os.path.basename(data.path),
                len(keypoints),
                self.detector_name,
            )

    def match(self, images: List[ImageData]) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        """Lowe-ratio-filtered matches for each candidate pair ``(i, j)`` with ``i < j``.

        Which pairs are tried depends on ``config.matching`` (see :func:`candidate_pairs`);
        the photos may have been taken in any order.
        """
        matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)
        ratio = self.config.ratio_threshold
        matches_by_pair: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        for i, j in candidate_pairs(len(images), self.config):
            knn_matches = matcher.knnMatch(images[i].descriptors, images[j].descriptors, k=2)
            good_matches = [
                pair[0]
                for pair in knn_matches
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
            ]
            matches_by_pair[(i, j)] = good_matches
            logger.info("Pair (%d, %d): %d/%d good matches", i, j, len(good_matches),
                        len(knn_matches))
            if self.config.save_debug_images:
                self._save_debug_image(images, i, j, good_matches)
        return matches_by_pair

    def _save_debug_image(
        self, images: List[ImageData], i: int, j: int, matches: List[cv2.DMatch]
    ) -> None:
        os.makedirs(self.config.debug_dir, exist_ok=True)
        image = cv2.drawMatches(
            images[i].image,
            images[i].keypoints,
            images[j].image,
            images[j].keypoints,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        name = f"matches_{i:02d}_{j:02d}.jpg"
        cv2.imwrite(os.path.join(self.config.debug_dir, name), image)
