"""Pairwise homography estimation, global transform chaining and canvas geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from stitcher.models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")

# Smallest homogeneous w accepted when projecting a corner. Anything at or below this is
# behind (or on) the plane at infinity of the reference view and cannot be drawn.
_MIN_W = 1e-6
# |det| of a homography normalised to H[2, 2] == 1 below which it is considered singular.
_MIN_ABS_DET = 1e-6


@dataclass
class PairResult:
    """Validated registration of image ``index + 1`` onto image ``index``."""

    index: int
    homography: np.ndarray
    matches: int
    inliers: int
    points_src: Optional[np.ndarray] = None  # (inliers, 2) pixels in image index + 1
    points_dst: Optional[np.ndarray] = None  # (inliers, 2) matching pixels in image index

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


def image_corners(width: int, height: int) -> np.ndarray:
    """Corners of a ``width`` x ``height`` image as a (4, 2) float64 array, clockwise."""
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def project_points(points: np.ndarray, homography: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a homography to (N, 2) points. Returns the (N, 2) result and the (N,) w values.

    Unlike ``cv2.perspectiveTransform`` the sign of w is kept, so points behind the
    projection plane can be detected instead of silently mirrored.
    """
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    w = homogeneous[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous[:, :2] / w[:, None]
    return projected, w


def _polygon_area_and_convexity(quad: np.ndarray) -> Tuple[float, bool]:
    """Signed (shoelace) area and convexity of a 4-point polygon."""
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    edges = np.roll(quad, -1, axis=0) - quad
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    convex = bool(np.all(crosses > 0) or np.all(crosses < 0))
    return area, convex


def validate_homography(
    homography: np.ndarray, width: int, height: int, config: StitchConfig
) -> str:
    """Check a pair homography (source image ``width`` x ``height`` -> destination).

    Returns an empty string when it is plausible, otherwise a description of the problem.
    """
    if not np.all(np.isfinite(homography)):
        return "homography contains non-finite values"
    if abs(homography[2, 2]) < 1e-12:
        return "homography is degenerate (H[2,2] is zero)"
    normalised = homography / homography[2, 2]
    determinant = float(np.linalg.det(normalised))
    if abs(determinant) < _MIN_ABS_DET:
        return f"homography is near-singular (det={determinant:.2e})"
    quad, w = project_points(image_corners(width, height), normalised)
    if np.any(w <= _MIN_W) or not np.all(np.isfinite(quad)):
        return "a corner projects to the plane at infinity or behind the camera (w <= 0)"
    area, convex = _polygon_area_and_convexity(quad)
    if area <= 0:
        return "homography mirrors or collapses the image"
    if not convex:
        return "image projects to a non-convex quadrilateral"
    ratio = area / (width * height)
    limit = config.max_area_ratio
    if ratio > limit or ratio < 1.0 / limit:
        return f"implausible scale change (area ratio {ratio:.3g}, allowed 1/{limit:g}..{limit:g})"
    return ""


def estimate_homographies(
    images: List[ImageData],
    matches_by_pair: Dict[int, List[cv2.DMatch]],
    config: StitchConfig,
) -> List[PairResult]:
    """Estimate and validate the homography mapping image i+1 onto image i for every pair."""
    results: List[PairResult] = []
    for index in range(len(images) - 1):
        pair = f"images {index} and {index + 1}"
        matches = matches_by_pair.get(index, [])
        if len(matches) < config.min_matches:
            raise StitchingError(
                f"Insufficient matches between {pair}: got {len(matches)}, "
                f"need >= {config.min_matches}."
            )
        kp_src, kp_dst = images[index + 1].keypoints, images[index].keypoints
        assert kp_src is not None and kp_dst is not None
        points_src = np.float32([kp_src[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_dst = np.float32([kp_dst[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            points_src, points_dst, cv2.RANSAC, config.ransac_reproj_threshold
        )
        if homography is None:
            raise StitchingError(f"cv2.findHomography failed for {pair}.")
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        logger.info(
            "Pair (%d, %d): RANSAC inliers %d/%d (%.1f%%)",
            index,
            index + 1,
            inliers,
            len(matches),
            100 * inliers / max(len(matches), 1),
        )
        if inliers < config.min_inliers:
            raise StitchingError(
                f"Too few RANSAC inliers between {pair}: got {inliers} of {len(matches)} "
                f"matches, need >= {config.min_inliers}."
            )
        height, width = images[index + 1].image.shape[:2]
        problem = validate_homography(homography, width, height, config)
        if problem:
            raise StitchingError(f"Implausible homography between {pair}: {problem}.")
        keep = inlier_mask.ravel().astype(bool)
        results.append(
            PairResult(
                index,
                homography,
                len(matches),
                inliers,
                points_src[keep].reshape(-1, 2).astype(np.float64),
                points_dst[keep].reshape(-1, 2).astype(np.float64),
            )
        )
    return results


def global_transforms(
    homographies: Sequence[np.ndarray], image_count: int
) -> Tuple[List[np.ndarray], int]:
    """Chain pairwise homographies into transforms onto the middle (reference) image.

    ``homographies[i]`` maps image i+1 onto image i. Returns one 3x3 matrix per image
    (image i -> reference frame) and the reference index.
    """
    reference = image_count // 2
    transforms = [np.eye(3, dtype=np.float64) for _ in range(image_count)]
    for index in range(reference + 1, image_count):
        transforms[index] = transforms[index - 1] @ homographies[index - 1]
    for index in range(reference - 1, -1, -1):
        transforms[index] = transforms[index + 1] @ np.linalg.inv(homographies[index])
    return transforms, reference


def canvas_geometry(
    images: List[ImageData], transforms: List[np.ndarray], config: StitchConfig
) -> Tuple[Tuple[int, int], np.ndarray]:
    """Compute the output canvas ``(width, height)`` and the translation that fits it.

    Raises StitchingError, before anything is allocated, if the transforms cannot be drawn
    on a plane or would need a canvas above ``config.max_canvas_megapixels``.
    """
    all_corners = []
    for index, (image_data, transform) in enumerate(zip(images, transforms)):
        height, width = image_data.image.shape[:2]
        projected, w = project_points(image_corners(width, height), transform)
        if np.any(w <= _MIN_W) or not np.all(np.isfinite(projected)):
            raise StitchingError(
                f"Image {index} projects outside the reference plane (a corner has w <= 0 or a "
                "non-finite position). A planar homography chain cannot represent this "
                "image set, which typically happens for a wide sweep of a rotating camera; "
                "use fewer images that cover a narrower field of view."
            )
        all_corners.append(projected)
    stacked = np.concatenate(all_corners)
    x_min, y_min = np.floor(stacked.min(axis=0))
    x_max, y_max = np.ceil(stacked.max(axis=0))
    megapixels = (x_max - x_min) * (y_max - y_min) / 1e6
    if megapixels > config.max_canvas_megapixels:
        raise StitchingError(
            f"Computed canvas is {megapixels:.1f} MP, exceeding the "
            f"{config.max_canvas_megapixels:.0f} MP safety cap. The chained planar "
            "homographies are diverging (typical for a wide sweep of a rotating camera); "
            "use fewer or more tightly framed images, or raise --max-canvas-mp if this "
            "size is really intended."
        )
    size = (int(x_max - x_min), int(y_max - y_min))
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    logger.info("Canvas size: %d x %d (%.1f MP)", size[0], size[1], megapixels)
    return size, translation
