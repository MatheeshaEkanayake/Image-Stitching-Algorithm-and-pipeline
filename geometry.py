from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")


def estimate_homographies(images: List[ImageData], matches_by_pair: Dict[int, List[cv2.DMatch]], config: StitchConfig) -> Dict[int, np.ndarray]:
    homographies: Dict[int, np.ndarray] = {}
    for index in range(len(images) - 1):
        matches = matches_by_pair.get(index, [])
        if len(matches) < config.min_matches:
            raise StitchingError(f"Insufficient matches between image {index} and {index + 1}: got {len(matches)}, need >= {config.min_matches}.")
        points_src = np.float32([images[index + 1].keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_dst = np.float32([images[index].keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(points_src, points_dst, cv2.RANSAC, config.ransac_reproj_threshold)
        if homography is None:
            raise StitchingError(f"cv2.findHomography failed for pair ({index}, {index + 1}).")
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        logger.info("Pair (%d, %d): RANSAC inliers %d/%d (%.1f%%)", index, index + 1, inliers, len(matches), 100 * inliers / max(len(matches), 1))
        homographies[index + 1] = homography
    return homographies


def global_transforms(homographies: Dict[int, np.ndarray], image_count: int) -> Tuple[List[np.ndarray], int]:
    reference = image_count // 2
    transforms = [np.eye(3, dtype=np.float64) for _ in range(image_count)]
    for index in range(reference + 1, image_count):
        transforms[index] = transforms[index - 1] @ homographies[index]
    for index in range(reference - 1, -1, -1):
        transforms[index] = transforms[index + 1] @ np.linalg.inv(homographies[index + 1])
    return transforms, reference


def canvas_geometry(images: List[ImageData], transforms: List[np.ndarray], config: StitchConfig) -> Tuple[Tuple[int, int], np.ndarray]:
    corners = []
    for image_data, transform in zip(images, transforms):
        height, width = image_data.image.shape[:2]
        source_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(source_corners, transform))
    all_corners = np.concatenate(corners).reshape(-1, 2)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    size = (int(x_max - x_min), int(y_max - y_min))
    megapixels = size[0] * size[1] / 1e6
    if megapixels > config.max_canvas_megapixels:
        raise StitchingError(f"Computed canvas is {megapixels:.1f} MP, exceeding the {config.max_canvas_megapixels:.0f} MP safety cap.")
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    logger.info("Canvas size: %d x %d (%.1f MP)", size[0], size[1], megapixels)
    return size, translation
