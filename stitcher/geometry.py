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
    """Validated registration of image ``j`` onto image ``i`` (0-based, ``i < j``)."""

    i: int
    j: int
    homography: np.ndarray  # maps pixels of image j to pixels of image i
    matches: int
    inliers: int
    points_src: Optional[np.ndarray] = None  # (inliers, 2) pixels in image j
    points_dst: Optional[np.ndarray] = None  # (inliers, 2) matching pixels in image i

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
    matches_by_pair: Dict[Tuple[int, int], List[cv2.DMatch]],
    config: StitchConfig,
) -> List[PairResult]:
    """Register every candidate pair that really overlaps.

    A pair that has too few matches or inliers, or whose homography is implausible, is simply
    not an overlapping pair and is skipped (most pairs of a grid capture do not overlap).
    Whether the images are connected as a whole is checked by :func:`spanning_tree`.
    """
    results: List[PairResult] = []
    for (i, j), matches in sorted(matches_by_pair.items()):
        if len(matches) < config.min_matches:
            continue
        kp_src, kp_dst = images[j].keypoints, images[i].keypoints
        assert kp_src is not None and kp_dst is not None
        points_src = np.float32([kp_src[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_dst = np.float32([kp_dst[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            points_src, points_dst, cv2.RANSAC, config.ransac_reproj_threshold
        )
        if homography is None or inlier_mask is None:
            continue
        inliers = int(inlier_mask.sum())
        # Brown & Lowe's test: genuine overlaps have clearly more inliers than chance.
        if inliers < config.min_inliers or inliers < 5.9 + 0.22 * len(matches):
            logger.info("Pair (%d, %d): %d inliers of %d matches; not overlapping.",
                        i, j, inliers, len(matches))
            continue
        height, width = images[j].image.shape[:2]
        problem = validate_homography(homography, width, height, config)
        if problem:
            logger.warning("Pair (%d, %d) ignored: implausible homography (%s).", i, j, problem)
            continue
        logger.info("Pair (%d, %d): RANSAC inliers %d/%d (%.1f%%)", i, j, inliers,
                    len(matches), 100 * inliers / len(matches))
        keep = inlier_mask.ravel().astype(bool)
        results.append(
            PairResult(
                i, j, homography, len(matches), inliers,
                points_src[keep].reshape(-1, 2).astype(np.float64),
                points_dst[keep].reshape(-1, 2).astype(np.float64),
            )
        )
    return results


def spanning_tree(pairs: Sequence[PairResult], image_count: int) -> List[PairResult]:
    """Maximum spanning tree of the overlap graph (edge weight = RANSAC inliers).

    Raises StitchingError, naming the photos (1-based, as in the app), if some images cannot
    be connected to the others through overlapping pairs.
    """
    parent = list(range(image_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree: List[PairResult] = []
    for pair in sorted(pairs, key=lambda p: -p.inliers):
        root_i, root_j = find(pair.i), find(pair.j)
        if root_i != root_j:
            parent[root_i] = root_j
            tree.append(pair)
    if len(tree) != image_count - 1:
        groups: Dict[int, List[int]] = {}
        for node in range(image_count):
            groups.setdefault(find(node), []).append(node + 1)
        parts = " | ".join(", ".join(map(str, g)) for g in groups.values())
        raise StitchingError(
            f"The photos split into groups that do not overlap each other (photo numbers: "
            f"{parts}). Each photo needs about a third of overlap with at least one other "
            "photo, and the whole set must be joined by overlaps; retake the missing shots."
        )
    return tree


def chain_transforms(
    edges: Sequence[Tuple[int, int, np.ndarray]], image_count: int
) -> Tuple[List[np.ndarray], int]:
    """Transform of every image onto a reference image, chained along tree ``edges``.

    Each edge ``(a, b, M)`` has ``M`` mapping image b onto image a (a homography or a
    rotation). The reference is the centre of the tree (fewest hops to the farthest image),
    which keeps chained error and distortion lowest whatever order the photos were taken in.
    Returns one 3x3 matrix per image and the reference index.
    """
    neighbours: Dict[int, List[Tuple[int, np.ndarray]]] = {n: [] for n in range(image_count)}
    for a, b, matrix in edges:
        neighbours[a].append((b, matrix))  # reaching b from a: T[b] = T[a] @ M
        neighbours[b].append((a, np.linalg.inv(matrix)))

    def hops(start: int) -> Dict[int, int]:
        distance = {start: 0}
        queue = [start]
        for node in queue:
            for other, _ in neighbours[node]:
                if other not in distance:
                    distance[other] = distance[node] + 1
                    queue.append(other)
        return distance

    middle = (image_count - 1) / 2
    reference = min(
        range(image_count),
        key=lambda n: (max(hops(n).values()), sum(hops(n).values()), abs(n - middle)),
    )
    transforms: List[Optional[np.ndarray]] = [None] * image_count
    transforms[reference] = np.eye(3, dtype=np.float64)
    queue = [reference]
    for node in queue:
        for other, matrix in neighbours[node]:
            if transforms[other] is None:
                transforms[other] = transforms[node] @ matrix
                queue.append(other)
    return [t for t in transforms if t is not None], reference


def global_transforms(
    pairs: Sequence[PairResult], image_count: int
) -> Tuple[List[np.ndarray], int]:
    """Homography of every image onto the reference image (see :func:`chain_transforms`)."""
    tree = spanning_tree(pairs, image_count)
    return chain_transforms([(p.i, p.j, p.homography) for p in tree], image_count)


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
