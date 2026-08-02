#!/usr/bin/env python3
"""
Artwork Mosaic Stitcher
========================

Stitches an UNORDERED set of overlapping high-resolution photographs (e.g.
tiled close-up shots of a mural or painting) into a single, maximally
faithful digital map -- optimized for detail preservation and geometric
accuracy, not for "panorama" aesthetics. Designed to feed a future deep-zoom
viewer (OpenSeadragon/IIIF-style), so a non-rectangular or non-perfectly-flat
output is entirely acceptable; blurring brushwork or shifting the true
position of a crack, inscription, or pigment boundary is not.

Pipeline
--------
  1. Detect features on every ORIGINAL image exactly once (SIFT, with
     optional sub-pixel keypoint refinement for maximum geometric accuracy).
  2. Match EVERY pair of images (not just neighbors -- the input has no
     inherent order), keep only pairs whose homography passes a geometric
     confidence test (Brown & Lowe's AutoStitch heuristic), and record each
     accepted pair as an edge in a match graph.
  3. Build a maximum-confidence spanning tree over that graph and propagate
     one homography per image, all relative to a single fixed reference
     image (the best-connected node) -- this replaces the old "chain through
     adjacent files" logic with something that works for an arbitrary,
     unordered overlap graph, including 2D grids and loop closures.
  4. Globally refine ALL image homographies simultaneously (bundle
     adjustment) by minimizing total reprojection error across every
     accepted pair's inlier correspondences -- not just the spanning tree --
     via sparse nonlinear least squares. This is what actually reduces drift
     versus naive sequential composition, especially where the overlap graph
     has loops (redundant constraints).
  5. Warp every ORIGINAL image exactly once, each into its own tight
     bounding box (not the full canvas) -- memory-efficient, and exactly the
     (image, mask, corner) form the quality blending stage below needs.
     Planar homography by default: a photographed flat artwork genuinely
     IS a plane, so this is the geometrically exact model (unlike a wide
     rotating-camera panorama of a 3D scene, where cylindrical projection
     is needed -- that mode is kept, but OFF by default here).
  6. Blend: per-image exposure/gain compensation (corrects lighting
     differences between shots) -> graph-cut seam finding (routes seams
     through low-difference, low-detail regions instead of blindly through
     the middle of the overlap) -> multi-band (Laplacian pyramid) blending
     (blends low frequencies over a wide band, high-frequency detail over a
     narrow one -- unlike feather blending, this doesn't soften brushwork
     near every seam). A fast legacy feather-blend mode is kept for
     quick previews.

Processing time is treated as unbounded by default: exhaustive all-pairs
matching, dense bundle adjustment, and full-resolution multi-band blending
are all significantly more expensive than the original chain-based
pipeline, in exchange for fidelity.

Usage:
    python stitch.py input_folder output.jpg
    python stitch.py ./tiles map.jpg --match-confidence 1.0 --debug

Requirements:
    pip install opencv-contrib-python numpy scipy
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

# OpenCV opportunistically routes many operations (including cv2.detail's
# exposure compensators, seam finders, and blenders) through OpenCL/UMat
# whenever ANY OpenCL platform is detected -- even a weak integrated GPU or
# a limited software ICD. For large, high-resolution artwork tiles this can
# try to allocate a multi-gigabyte buffer on a device that doesn't have that
# much usable memory, failing with something like:
#   OpenCL error CL_MEM_OBJECT_ALLOCATION_FAILURE ... in function 'upload'
# Since this pipeline already treats CPU time/RAM as the resource to spend
# for quality (see StitchConfig docstring), we force the plain CPU path
# everywhere: deterministic, and bounded only by system RAM (which
# max_canvas_megapixels / --max-image-dimension already guard), not by
# whatever a given machine's OpenCL device happens to expose.
cv2.ocl.setUseOpenCL(False)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class StitchConfig:
    """All tunable parameters for the stitching pipeline live here."""

    # --- Feature detection ---
    feature_detector: str = "SIFT"       # "SIFT" or "ORB" (auto-falls back to ORB if SIFT unavailable)
    orb_n_features: int = 5000           # Only used if ORB is selected/used as fallback
    subpixel_refine: bool = True         # Refine keypoint locations to sub-pixel accuracy (cv2.cornerSubPix)
    subpixel_win_size: int = 5

    # --- Matching (unordered graph, not a fixed chain) ---
    ratio_threshold: float = 0.75        # Lowe's ratio test threshold
    min_matches: int = 10                # Minimum good matches even attempt a homography for a candidate pair
    # Brown & Lowe (2007) geometric confidence: inliers / (8 + 0.3 * n_matches).
    # A pair below this is treated as coincidental feature overlap, not a
    # real spatial adjacency -- this is how the algorithm figures out which
    # images actually neighbor each other without any filename ordering.
    match_confidence_threshold: float = 1.0
    max_correspondences_per_edge: int = 300  # cap correspondences kept per edge, bounds bundle-adjustment cost

    # --- Homography estimation ---
    ransac_reproj_threshold: float = 4.0  # RANSAC reprojection error threshold (pixels)

    # --- Cylindrical projection (rotating-camera, wide-FOV panoramas ONLY) ---
    # A flat, translating-camera capture of an artwork is exactly a planar
    # mosaic -- cylindrical projection would introduce curvature that isn't
    # physically there, so this defaults OFF here (unlike an outdoor
    # rotating-camera panorama, where it's the correct model).
    use_cylindrical_projection: bool = False
    cylindrical_focal_length: float = 0.0  # 0 = auto-estimate from match-graph homographies

    # --- Global bundle adjustment ---
    bundle_adjustment: bool = True
    bundle_adjustment_loss: str = "soft_l1"   # robust loss, downweights any remaining bad correspondences
    bundle_adjustment_max_iterations: int = 200

    # --- Warping ---
    # Quality over speed: Lanczos for the one-time final resample of every
    # original pixel into the mosaic frame.
    final_warp_interpolation: int = cv2.INTER_LANCZOS4

    # --- Blending / seam finding / exposure compensation ---
    blend_mode: str = "multiband_graphcut"   # "multiband_graphcut" (quality, default) | "feather" (fast legacy)
    feather_width: int = 50                  # only used by blend_mode="feather"
    exposure_compensation: bool = True
    exposure_compensator: str = "blocks_gain"  # "gain" (one gain/image) | "blocks_gain" (spatially-varying) | "none"
    seam_finder: str = "graphcut"              # "graphcut" | "voronoi" | "none"
    # Seam LOCATIONS are searched for on downscaled copies (graph-cut cost
    # grows fast with pixel count); blending itself always happens at full
    # resolution, so no detail is lost -- only the (smooth, low-frequency)
    # seam boundary is decided using a coarser image.
    seam_estimation_max_dimension: int = 1600
    multiband_num_bands: int = 5

    # --- Memory safety ---
    max_canvas_megapixels: float = 400.0
    max_image_dimension: int = 0          # 0 = keep original resolution (default)

    # --- Debugging / output ---
    save_debug_images: bool = False
    debug_dir: str = "debug"


# Module-level logger. Configured in main(); library code just uses it.
logger = logging.getLogger("MosaicStitcher")

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


class StitchingError(Exception):
    """Raised when the pipeline cannot proceed reliably."""


# ==============================================================================
# Data containers
# ==============================================================================


@dataclass
class ImageData:
    """Holds one original image plus its (lazily filled in) feature data."""

    path: str
    image: np.ndarray                                   # Original-resolution BGR image
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    # Per-pixel validity mask for `image` (255 = real pixel, 0 = no data).
    # None means "every pixel is valid". Set to a real mask after cylindrical
    # warping, which introduces curved black regions at top/bottom.
    valid_mask: Optional[np.ndarray] = None


@dataclass
class MatchEdge:
    """One accepted (geometrically confident) pairwise match, i < j."""

    i: int
    j: int
    H: np.ndarray             # maps image j's coordinates -> image i's frame
    confidence: float
    pts_i: np.ndarray         # inlier correspondence points in image i, Nx2 (float64)
    pts_j: np.ndarray         # inlier correspondence points in image j, Nx2, same order


# ==============================================================================
# Helpers
# ==============================================================================


def natural_sort_key(filename: str) -> List:
    """Sort key for deterministic, human-friendly image indexing/logging (ordering has no spatial meaning here)."""
    parts = re.split(r"(\d+)", filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


class _UnionFind:
    """Minimal union-find for connected-component detection and Kruskal's spanning tree."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


# ==============================================================================
# Main pipeline class
# ==============================================================================


class PanoramaStitcher:
    """
    Unordered multi-image mosaic stitcher built around a match graph, a
    max-confidence spanning tree for initial alignment, and global bundle
    adjustment for final geometric accuracy.

    Typical usage:
        stitcher = PanoramaStitcher(StitchConfig())
        mosaic = stitcher.stitch("./tiles")
        cv2.imwrite("mosaic.jpg", mosaic)
    """

    def __init__(self, config: StitchConfig):
        self.config = config

        self.images: List[ImageData] = []

        self.detector: Optional[cv2.Feature2D] = None
        self.detector_name: str = ""
        self.norm_type: int = cv2.NORM_L2

        # Every accepted pairwise match, from exhaustive all-pairs matching.
        self.match_edges: List[MatchEdge] = []

        # global_homographies[i] maps image i's coordinates directly into the
        # single shared reference frame (the mosaic frame, before translation).
        self.global_homographies: List[np.ndarray] = []
        self.reference_index: int = 0

        # Canvas geometry.
        self.canvas_size: Tuple[int, int] = (0, 0)     # (width, height)
        self.translation: np.ndarray = np.eye(3, dtype=np.float64)

        self.panorama: Optional[np.ndarray] = None

    # --------------------------------------------------------------------
    # Step 1: Load images
    # --------------------------------------------------------------------
    def load_images(self, folder: str) -> None:
        """Load every jpg/jpeg/png in `folder`. Order is for indexing/logging only -- not spatial."""
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Input folder not found: {folder}")

        filenames = [
            f for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        if not filenames:
            raise FileNotFoundError(f"No jpg/jpeg/png images found in: {folder}")

        filenames.sort(key=natural_sort_key)

        self.images = []
        for fname in filenames:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Could not read image, skipping: %s", path)
                continue

            max_dim = self.config.max_image_dimension
            if max_dim > 0:
                h, w = img.shape[:2]
                scale = max_dim / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(
                        img, (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )

            self.images.append(ImageData(path=path, image=img))

        if not self.images:
            raise StitchingError("No valid images could be loaded.")

        logger.info("Loaded %d images from '%s':", len(self.images), folder)
        for i, data in enumerate(self.images):
            h, w = data.image.shape[:2]
            logger.info("  [%d] %-30s %dx%d", i, os.path.basename(data.path), w, h)

    # --------------------------------------------------------------------
    # Step 2: Feature detection
    # --------------------------------------------------------------------
    def _init_detector(self) -> None:
        """Initialize SIFT if available, otherwise fall back to ORB."""
        requested = self.config.feature_detector.upper()

        if requested == "SIFT":
            try:
                self.detector = cv2.SIFT_create()
                self.detector_name = "SIFT"
                self.norm_type = cv2.NORM_L2
                logger.info("Feature detector: SIFT")
                return
            except (AttributeError, cv2.error) as exc:
                logger.warning(
                    "SIFT unavailable in this OpenCV build (%s). Falling back to ORB.",
                    exc,
                )

        self.detector = cv2.ORB_create(nfeatures=self.config.orb_n_features)
        self.detector_name = "ORB"
        self.norm_type = cv2.NORM_HAMMING
        logger.info("Feature detector: ORB (nfeatures=%d)", self.config.orb_n_features)

    def _refine_keypoints_subpixel(self, gray: np.ndarray, keypoints: List[cv2.KeyPoint]) -> None:
        """
        Refine every keypoint location to sub-pixel accuracy in place, via
        cv2.cornerSubPix. Directly serves the "accurate spatial
        representation" goal: integer-pixel keypoint locations otherwise
        put an avoidable ~0.5px floor on alignment accuracy everywhere.
        """
        if not keypoints:
            return
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        win = self.config.subpixel_win_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        try:
            cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
        except cv2.error as exc:
            logger.warning("Sub-pixel keypoint refinement failed (%s); using integer-precision keypoints.", exc)
            return
        for kp, p in zip(keypoints, pts.reshape(-1, 2)):
            kp.pt = (float(p[0]), float(p[1]))

    def detect_features(self) -> None:
        """Compute keypoints + descriptors once for every original image."""
        if self.detector is None:
            self._init_detector()

        for idx, data in enumerate(self.images):
            gray = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) == 0:
                raise StitchingError(
                    f"No features detected in '{data.path}'. "
                    "Image may be blank, low-texture, or corrupted."
                )

            if self.config.subpixel_refine:
                self._refine_keypoints_subpixel(gray, keypoints)

            data.keypoints = keypoints
            data.descriptors = descriptors
            logger.info(
                "[%d] %s: %d keypoints (%s)",
                idx, os.path.basename(data.path), len(keypoints), self.detector_name,
            )

    # --------------------------------------------------------------------
    # Step 3: Match EVERY pair (unordered input -- no adjacency assumption)
    # --------------------------------------------------------------------
    def match_all_pairs(self) -> None:
        """
        Exhaustively match every unordered pair of images (i, j), i < j.
        For each candidate pair: BFMatcher + Lowe's ratio test, then RANSAC
        homography fit, then a geometric confidence score (Brown & Lowe,
        "Automatic Panoramic Image Stitching", IJCV 2007):

            confidence = n_inliers / (8 + 0.3 * n_good_matches)

        Pairs below match_confidence_threshold are discarded as coincidental
        feature overlap rather than genuine spatial adjacency. Accepted
        pairs become edges in self.match_edges, used both to build an
        initial spanning-tree alignment and as constraints for bundle
        adjustment.
        """
        n = len(self.images)
        total_pairs = n * (n - 1) // 2
        if total_pairs > 300:
            logger.warning(
                "%d images -> %d candidate pairs to match exhaustively (O(N^2)); "
                "this is expected (and can take a while) for an unordered set.",
                n, total_pairs,
            )

        matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)
        self.match_edges = []

        for i, j in itertools.combinations(range(n), 2):
            desc_a = self.images[i].descriptors
            desc_b = self.images[j].descriptors
            if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
                continue

            knn_matches = matcher.knnMatch(desc_a, desc_b, k=2)
            good = []
            for pair in knn_matches:
                if len(pair) != 2:
                    continue
                m, n_ = pair
                if m.distance < self.config.ratio_threshold * n_.distance:
                    good.append(m)

            if len(good) < self.config.min_matches:
                continue

            kp_a = self.images[i].keypoints
            kp_b = self.images[j].keypoints
            pts_src = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # image j
            pts_dst = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # image i

            H, inlier_mask = cv2.findHomography(
                pts_src, pts_dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
            )
            if H is None or inlier_mask is None:
                continue

            n_inliers = int(inlier_mask.sum())
            confidence = n_inliers / (8.0 + 0.3 * len(good))

            if confidence < self.config.match_confidence_threshold:
                logger.debug(
                    "Pair (%d, %d): confidence %.2f < %.2f -- discarded (likely coincidental overlap).",
                    i, j, confidence, self.config.match_confidence_threshold,
                )
                continue

            inlier_flat = inlier_mask.ravel().astype(bool)
            pts_i_in = pts_dst.reshape(-1, 2)[inlier_flat]
            pts_j_in = pts_src.reshape(-1, 2)[inlier_flat]

            cap = self.config.max_correspondences_per_edge
            if cap > 0 and len(pts_i_in) > cap:
                sel = np.random.default_rng(42).choice(len(pts_i_in), size=cap, replace=False)
                pts_i_in = pts_i_in[sel]
                pts_j_in = pts_j_in[sel]

            self.match_edges.append(MatchEdge(
                i=i, j=j, H=H, confidence=confidence,
                pts_i=pts_i_in.astype(np.float64), pts_j=pts_j_in.astype(np.float64),
            ))
            logger.info(
                "Pair (%d, %d): %d good matches, %d inliers, confidence %.2f -- accepted",
                i, j, len(good), n_inliers, confidence,
            )

            if self.config.save_debug_images:
                self._save_match_debug_image(i, j, good)

        logger.info(
            "Matching complete: %d/%d candidate pairs accepted as real overlaps.",
            len(self.match_edges), total_pairs,
        )

    def _save_match_debug_image(self, i: int, j: int, good_matches: List[cv2.DMatch]) -> None:
        os.makedirs(self.config.debug_dir, exist_ok=True)
        vis = cv2.drawMatches(
            self.images[i].image, self.images[i].keypoints,
            self.images[j].image, self.images[j].keypoints,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        out_path = os.path.join(self.config.debug_dir, f"matches_{i:02d}_{j:02d}.jpg")
        cv2.imwrite(out_path, vis)

    # --------------------------------------------------------------------
    # Step 4: Spanning-tree initial alignment
    # --------------------------------------------------------------------
    def build_spanning_tree_and_propagate(self) -> None:
        """
        From the accepted match graph: verify every image belongs to one
        connected mosaic, pick the best-connected image as reference, build
        a maximum-confidence spanning tree (Kruskal), and propagate one
        homography per image (relative to the reference) via BFS over that
        tree. No image is warped here -- pure matrix composition.
        """
        n = len(self.images)
        if not self.match_edges:
            raise StitchingError(
                "No image pairs passed the geometric confidence check -- none of the "
                "supplied images could be confidently linked to any other. Check that "
                "the photos genuinely overlap, or lower --match-confidence / --min-matches."
            )

        uf = _UnionFind(n)
        for e in self.match_edges:
            uf.union(e.i, e.j)

        components: Dict[int, List[int]] = {}
        for idx in range(n):
            components.setdefault(uf.find(idx), []).append(idx)

        if len(components) > 1:
            lines = []
            for members in components.values():
                names = [os.path.basename(self.images[m].path) for m in members]
                lines.append(f"  - {names}")
            raise StitchingError(
                f"Images form {len(components)} disconnected group(s) -- not every photo "
                "overlaps with the rest, so they can't be placed on one consistent map:\n"
                + "\n".join(lines)
                + "\nEither these are genuinely separate artworks (stitch each group in "
                "its own folder) or some connecting photos are missing."
            )

        # Reference = best-connected image (most accepted edges) -- the most
        # stable place to anchor the whole mosaic.
        degree = [0] * n
        for e in self.match_edges:
            degree[e.i] += 1
            degree[e.j] += 1
        self.reference_index = int(np.argmax(degree))
        logger.info(
            "Reference image: [%d] %s (degree %d)",
            self.reference_index, os.path.basename(self.images[self.reference_index].path),
            degree[self.reference_index],
        )

        # Maximum spanning tree by confidence: the initial placement of any
        # image is dictated only by its most reliable edge. Weaker edges
        # still contribute later, in bundle adjustment, but don't get to
        # dominate the starting point.
        uf_tree = _UnionFind(n)
        tree_adj: Dict[int, List[Tuple[int, np.ndarray]]] = {idx: [] for idx in range(n)}
        for e in sorted(self.match_edges, key=lambda e: -e.confidence):
            if uf_tree.union(e.i, e.j):
                tree_adj[e.i].append((e.j, e.H))                    # H maps j -> i
                tree_adj[e.j].append((e.i, np.linalg.inv(e.H)))     # inverse maps i -> j

        self.global_homographies = [np.eye(3, dtype=np.float64) for _ in range(n)]
        visited = {self.reference_index}
        queue = [self.reference_index]
        while queue:
            cur = queue.pop(0)
            for neighbor, H_neighbor_to_cur in tree_adj[cur]:
                if neighbor in visited:
                    continue
                self.global_homographies[neighbor] = self.global_homographies[cur] @ H_neighbor_to_cur
                visited.add(neighbor)
                queue.append(neighbor)

        logger.info("Initial alignment propagated across spanning tree (%d images).", len(visited))

    # --------------------------------------------------------------------
    # Step 5: Global bundle adjustment
    # --------------------------------------------------------------------
    @staticmethod
    def _homography_to_params(H: np.ndarray) -> np.ndarray:
        Hn = H / H[2, 2]
        return Hn.flatten()[:8].astype(np.float64)

    @staticmethod
    def _params_to_homography(p: np.ndarray) -> np.ndarray:
        return np.array(
            [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1.0], dtype=np.float64
        ).reshape(3, 3)

    @staticmethod
    def _to_homog(pts: np.ndarray) -> np.ndarray:
        return np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])

    def bundle_adjust(self) -> None:
        """
        Jointly refine every non-reference image's homography by minimizing
        total reprojection error across ALL accepted pairwise
        correspondences simultaneously (not just spanning-tree edges) --
        this is what a sequential chain composition can never do, and is
        exactly where redundant constraints (loop closures in the overlap
        graph, e.g. a 2D grid) pay off: error gets distributed and
        cancelled across the whole mosaic instead of accumulating in one
        direction from the reference image.

        Uses sparse nonlinear least squares (scipy) with a robust loss, and
        an explicit sparsity pattern (each correspondence's residual only
        depends on its two images' 8 parameters) so this stays tractable
        even with many images/correspondences.
        """
        n = len(self.images)
        ref = self.reference_index
        other_indices = [i for i in range(n) if i != ref]
        if not other_indices or not self.match_edges:
            logger.info("Bundle adjustment skipped (nothing to optimize).")
            return

        index_pos = {idx: k for k, idx in enumerate(other_indices)}
        x0 = np.concatenate([
            self._homography_to_params(self.global_homographies[idx]) for idx in other_indices
        ])

        edges = self.match_edges
        pts_i_h = [self._to_homog(e.pts_i) for e in edges]
        pts_j_h = [self._to_homog(e.pts_j) for e in edges]
        n_res_per_edge = [len(e.pts_i) for e in edges]

        def unpack(x: np.ndarray) -> Dict[int, np.ndarray]:
            H = {ref: np.eye(3, dtype=np.float64)}
            for idx in other_indices:
                k = index_pos[idx]
                H[idx] = self._params_to_homography(x[k * 8:(k + 1) * 8])
            return H

        def residual_fn(x: np.ndarray) -> np.ndarray:
            H = unpack(x)
            chunks = []
            for e, pih, pjh in zip(edges, pts_i_h, pts_j_h):
                proj_i = pih @ H[e.i].T
                proj_i = proj_i[:, :2] / proj_i[:, 2:3]
                proj_j = pjh @ H[e.j].T
                proj_j = proj_j[:, :2] / proj_j[:, 2:3]
                chunks.append((proj_i - proj_j).ravel())
            return np.concatenate(chunks)

        total_res = sum(n_res_per_edge) * 2
        total_params = len(other_indices) * 8
        jac_sparsity = sparse.lil_matrix((total_res, total_params), dtype=bool)
        row = 0
        for e, nres in zip(edges, n_res_per_edge):
            block_rows = nres * 2
            for img_idx in (e.i, e.j):
                if img_idx in index_pos:
                    col = index_pos[img_idx] * 8
                    jac_sparsity[row:row + block_rows, col:col + 8] = True
            row += block_rows

        logger.info(
            "Running global bundle adjustment: %d images, %d edges, %d correspondences, %d parameters...",
            n, len(edges), total_res // 2, total_params,
        )

        rms_before = float(np.sqrt(np.mean(residual_fn(x0) ** 2)))

        result = least_squares(
            residual_fn, x0,
            jac_sparsity=jac_sparsity,
            loss=self.config.bundle_adjustment_loss,
            max_nfev=self.config.bundle_adjustment_max_iterations * max(total_params, 1),
            method="trf",
            verbose=0,
        )

        rms_after = float(np.sqrt(np.mean(result.fun ** 2)))
        logger.info(
            "Bundle adjustment done: RMS reprojection error %.3f px -> %.3f px (status=%d)",
            rms_before, rms_after, result.status,
        )

        H_refined = unpack(result.x)
        for idx in other_indices:
            self.global_homographies[idx] = H_refined[idx]

    # --------------------------------------------------------------------
    # Step 5b: Cylindrical projection (rotating-camera panoramas only)
    # --------------------------------------------------------------------
    def estimate_focal_length(self) -> float:
        """Estimate a global focal length (pixels) from the match graph's homographies."""
        focals: List[float] = []
        for e in self.match_edges:
            f0, f1 = np.array([0.0]), np.array([0.0])
            f0_ok, f1_ok = np.array([False]), np.array([False])
            cv2.detail.focalsFromHomography(e.H, f0, f1, f0_ok, f1_ok)
            if f0_ok[0] and f0[0] > 0:
                focals.append(float(f0[0]))
            if f1_ok[0] and f1[0] > 0:
                focals.append(float(f1[0]))

        if not focals:
            w = self.images[len(self.images) // 2].image.shape[1]
            fallback = w / (2.0 * np.tan(np.radians(30.0)))
            logger.warning(
                "Could not estimate focal length from the match graph; falling back to a "
                "60-degree-FOV guess (%.1f px). Pass --focal-length explicitly for accuracy.",
                fallback,
            )
            return fallback

        focal = float(np.median(focals))
        logger.info("Estimated cylindrical focal length: %.1f px (median of %d samples)", focal, len(focals))
        return focal

    @staticmethod
    def _cylindrical_warp(image: np.ndarray, focal: float) -> Tuple[np.ndarray, np.ndarray]:
        """Reproject `image` onto a cylinder with the given focal length. Returns (warped_image, valid_mask)."""
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        x_dst, y_dst = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        theta = (x_dst - cx) / focal
        h_norm = (y_dst - cy) / focal

        X, Y, Z = np.sin(theta), h_norm, np.cos(theta)
        map_x = (focal * (X / Z) + cx).astype(np.float32)
        map_y = (focal * (Y / Z) + cy).astype(np.float32)

        warped = cv2.remap(
            image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        valid_mask = cv2.remap(
            mask_src, map_x, map_y, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped, valid_mask

    def cylindrical_warp_images(self, focal: float) -> None:
        """
        Replace every image with its cylindrical projection and invalidate
        feature data -- detect_features()/match_all_pairs()/spanning-tree/
        bundle-adjustment must all be re-run afterward on the new images.
        """
        for data in self.images:
            warped, mask = self._cylindrical_warp(data.image, focal)
            data.image = warped
            data.valid_mask = mask
            data.keypoints = None
            data.descriptors = None

        self.match_edges = []

        if self.config.save_debug_images:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            for idx, data in enumerate(self.images):
                cv2.imwrite(os.path.join(self.config.debug_dir, f"cylindrical_{idx:02d}.jpg"), data.image)

    # --------------------------------------------------------------------
    # Step 6: Compute mosaic canvas bounds
    # --------------------------------------------------------------------
    def compute_canvas(self) -> None:
        """Project every image's corners through its global homography to find the overall bounding box."""
        all_corners = []
        for i, data in enumerate(self.images):
            h, w = data.image.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, self.global_homographies[i])
            all_corners.append(warped_corners)

        all_corners = np.concatenate(all_corners, axis=0).reshape(-1, 2)
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        self.translation = np.array(
            [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64,
        )
        self.canvas_size = (int(x_max - x_min), int(y_max - y_min))
        n_pixels = self.canvas_size[0] * self.canvas_size[1]
        megapixels = n_pixels / 1e6
        est_gb = (n_pixels * 20) / (1024 ** 3)

        logger.info(
            "Canvas size: %d x %d (%.1f MP, translation offset: (%d, %d), ~%.2f GB est. peak memory)",
            self.canvas_size[0], self.canvas_size[1], megapixels, -x_min, -y_min, est_gb,
        )

        if megapixels > self.config.max_canvas_megapixels:
            raise StitchingError(
                f"Computed canvas is {megapixels:.1f} MP, exceeding the "
                f"{self.config.max_canvas_megapixels:.0f} MP safety cap. This is almost always "
                "caused by one or more unreliable pairwise homographies -- check the log above "
                "for low-confidence or low-inlier warnings. Try a stricter --match-confidence or "
                "--ratio, or raise --max-canvas-mp if this size is genuinely correct. "
                "--max-image-dimension shrinks inputs while you debug."
            )

    # --------------------------------------------------------------------
    # Step 7: Warp every ORIGINAL image exactly once, into its own tight ROI
    # --------------------------------------------------------------------
    def compute_per_image_rois(self) -> List[Tuple[int, int, int, int]]:
        rois = []
        for i, data in enumerate(self.images):
            h, w = data.image.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            H_final = self.translation @ self.global_homographies[i]
            warped_corners = cv2.perspectiveTransform(corners, H_final).reshape(-1, 2)
            x_min, y_min = np.floor(warped_corners.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(warped_corners.max(axis=0)).astype(int)
            rois.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        return rois

    def warp_images_to_rois(self) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        """
        Warp each original image exactly once into its own tight bounding
        box (not the full canvas) -- far more memory-efficient for a large
        mosaic, and exactly the (image, mask, corner) form the exposure
        compensation / seam finding / multi-band blending stages expect.
        """
        rois = self.compute_per_image_rois()
        if self.config.save_debug_images:
            os.makedirs(self.config.debug_dir, exist_ok=True)

        w_canvas, h_canvas = self.canvas_size
        for i, data in enumerate(self.images):
            x_min, y_min, x_max, y_max = rois[i]
            roi_w = max(min(x_max, w_canvas) - max(x_min, 0), 1)
            roi_h = max(min(y_max, h_canvas) - max(y_min, 0), 1)
            x_min, y_min = max(x_min, 0), max(y_min, 0)

            T_local = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
            H_final = T_local @ self.translation @ self.global_homographies[i]

            warped_img = cv2.warpPerspective(
                data.image, H_final, (roi_w, roi_h),
                flags=self.config.final_warp_interpolation,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            h, w = data.image.shape[:2]
            mask_src = data.valid_mask if data.valid_mask is not None else np.full((h, w), 255, dtype=np.uint8)
            warped_mask = cv2.warpPerspective(
                mask_src, H_final, (roi_w, roi_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            logger.info("Warped image %d into its ROI (%dx%d at (%d,%d)).", i, roi_w, roi_h, x_min, y_min)
            if self.config.save_debug_images:
                cv2.imwrite(os.path.join(self.config.debug_dir, f"warped_{i:02d}.jpg"), warped_img)

            yield warped_img, warped_mask, (x_min, y_min)

    # --------------------------------------------------------------------
    # Step 8: Blend (exposure compensation -> seam finding -> multi-band)
    # --------------------------------------------------------------------
    def blend_images(self, warped: Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]) -> np.ndarray:
        materialized = list(warped)
        if self.config.blend_mode == "feather":
            return self._feather_blend(materialized)
        elif self.config.blend_mode == "multiband_graphcut":
            return self._quality_blend(materialized)
        raise ValueError(f"Unknown blend_mode: {self.config.blend_mode!r}")

    def _feather_blend(self, materialized: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]) -> np.ndarray:
        """Fast legacy distance-weighted blending, kept for quick previews."""
        w_canvas, h_canvas = self.canvas_size
        accumulator = np.zeros((h_canvas, w_canvas, 3), dtype=np.float32)
        weight_sum = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        for img, mask, (cx, cy) in materialized:
            h, w = img.shape[:2]
            weight = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            if self.config.feather_width > 0:
                weight = np.clip(weight, 0, self.config.feather_width)
            cy2, cx2 = min(cy + h, h_canvas), min(cx + w, w_canvas)
            accumulator[cy:cy2, cx:cx2] += img[:cy2 - cy, :cx2 - cx].astype(np.float32) * weight[:cy2 - cy, :cx2 - cx, None]
            weight_sum[cy:cy2, cx:cx2] += weight[:cy2 - cy, :cx2 - cx]

        safe_weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        blended = accumulator / safe_weight_sum[..., None]
        blended[weight_sum == 0] = 0
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _find_seams(
        self, images: List[np.ndarray], masks: List[np.ndarray], corners: List[Tuple[int, int]],
    ) -> List[np.ndarray]:
        """
        Graph-cut (or Voronoi) seam finding, run on downscaled copies for
        tractable runtime -- the resulting masks are upscaled back to full
        resolution before blending, so no detail is lost; only the smooth
        seam boundary is decided at reduced scale.
        """
        max_dim = self.config.seam_estimation_max_dimension
        small_images, small_masks, small_corners, scales = [], [], [], []
        for img, mask, corner in zip(images, masks, corners):
            h, w = img.shape[:2]
            scale = min(1.0, max_dim / max(h, w)) if max_dim > 0 else 1.0
            scales.append(scale)
            if scale < 1.0:
                small_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                small_masks.append(cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
            else:
                small_images.append(img)
                small_masks.append(mask)
            small_corners.append((int(round(corner[0] * scale)), int(round(corner[1] * scale))))

        if self.config.seam_finder == "graphcut":
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        elif self.config.seam_finder == "voronoi":
            seam_finder = cv2.detail_VoronoiSeamFinder()
        else:
            return masks

        images_f = [im.astype(np.float32) for im in small_images]
        refined_small = seam_finder.find(images_f, small_corners, small_masks)
        refined_small = [m.get() if isinstance(m, cv2.UMat) else m for m in refined_small]

        refined_masks = []
        for i, (mask, scale) in enumerate(zip(refined_small, scales)):
            full_h, full_w = images[i].shape[:2]
            mask_up = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask
            refined_masks.append(mask_up)

        logger.info("Seam finding complete (%s, %d images).", self.config.seam_finder, len(images))
        return refined_masks

    def _quality_blend(self, materialized: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]) -> np.ndarray:
        images = [m[0] for m in materialized]
        masks = [m[1] for m in materialized]
        corners = [m[2] for m in materialized]

        if self.config.exposure_compensation and self.config.exposure_compensator != "none":
            comp = (
                cv2.detail_BlocksGainCompensator() if self.config.exposure_compensator == "blocks_gain"
                else cv2.detail_GainCompensator()
            )
            comp.feed(corners, images, masks)
            images = [comp.apply(idx, corners[idx], images[idx], masks[idx]) for idx in range(len(images))]
            logger.info("Exposure compensation applied (%s).", self.config.exposure_compensator)

        if self.config.seam_finder != "none":
            masks = self._find_seams(images, masks, corners)

        w_canvas, h_canvas = self.canvas_size
        blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=self.config.multiband_num_bands)
        blender.prepare((0, 0, w_canvas, h_canvas))
        for img, mask, corner in zip(images, masks, corners):
            blender.feed(img.astype(np.int16), mask, corner)
        result, _ = blender.blend(None, None)
        result = np.clip(result, 0, 255).astype(np.uint8)
        logger.info("Multi-band blend complete (%d bands).", self.config.multiband_num_bands)
        return result

    # --------------------------------------------------------------------
    # Step 9: Crop empty canvas margins
    # --------------------------------------------------------------------
    def crop_panorama(self, panorama: np.ndarray) -> np.ndarray:
        """Trim only the empty (black) canvas margin -- never touches valid image content."""
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        nonzero_coords = cv2.findNonZero(gray)

        if nonzero_coords is None:
            logger.warning("Mosaic is entirely black; skipping crop.")
            return panorama

        x, y, w, h = cv2.boundingRect(nonzero_coords)
        cropped = panorama[y:y + h, x:x + w]
        logger.info("Cropped mosaic: %dx%d -> %dx%d", panorama.shape[1], panorama.shape[0], w, h)
        return cropped

    # --------------------------------------------------------------------
    # Orchestration
    # --------------------------------------------------------------------
    def stitch(self, input_folder: str) -> np.ndarray:
        """Run the full pipeline end-to-end and return the final mosaic."""
        self.load_images(input_folder)

        if len(self.images) == 1:
            logger.info("Only one image found; returning it unchanged.")
            self.panorama = self.images[0].image
            return self.panorama

        self.detect_features()
        self.match_all_pairs()
        self.build_spanning_tree_and_propagate()
        if self.config.bundle_adjustment:
            self.bundle_adjust()

        if self.config.use_cylindrical_projection:
            focal = self.config.cylindrical_focal_length or self.estimate_focal_length()
            self.cylindrical_warp_images(focal)
            self.detect_features()
            self.match_all_pairs()
            self.build_spanning_tree_and_propagate()
            if self.config.bundle_adjustment:
                self.bundle_adjust()

        self.compute_canvas()

        try:
            warped = self.warp_images_to_rois()
            blended = self.blend_images(warped)
        except MemoryError as exc:
            raise StitchingError(
                "Ran out of memory during warping/blending. Try re-running with "
                "--max-image-dimension 2000 (or smaller) to downscale inputs, "
                "or check --max-canvas-mp isn't larger than expected."
            ) from exc

        self.panorama = self.crop_panorama(blended)
        return self.panorama


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch an unordered set of overlapping images into a single high-fidelity mosaic.",
    )
    parser.add_argument("input_folder", help="Folder containing input images (jpg/jpeg/png).")
    parser.add_argument("output", help="Output mosaic path, e.g. mosaic.jpg")

    parser.add_argument("--detector", choices=["sift", "orb"], default="sift",
                         help="Feature detector (default: sift; auto-falls back to orb if unavailable).")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold (default: 0.75).")
    parser.add_argument("--ransac-thresh", type=float, default=4.0,
                         help="RANSAC reprojection threshold in pixels (default: 4.0).")
    parser.add_argument("--min-matches", type=int, default=10,
                         help="Minimum good matches to even attempt a candidate pair (default: 10).")
    parser.add_argument("--match-confidence", type=float, default=1.0,
                         help="Geometric confidence threshold to accept a pair as a real overlap "
                              "(Brown & Lowe heuristic; default: 1.0). Lower to accept weaker pairs.")
    parser.add_argument("--no-subpixel", action="store_true", help="Disable sub-pixel keypoint refinement.")
    parser.add_argument("--no-bundle-adjustment", action="store_true",
                         help="Skip global bundle adjustment and use the spanning-tree alignment as-is.")

    parser.add_argument("--cylindrical", action="store_true",
                         help="Enable cylindrical projection (rotating-camera, wide-FOV panoramas only -- "
                              "OFF by default, since a flat artwork is exactly a planar mosaic).")
    parser.add_argument("--focal-length", type=float, default=0.0,
                         help="Focal length in pixels for cylindrical projection (0 = auto-estimate).")

    parser.add_argument("--blend-mode", choices=["multiband_graphcut", "feather"], default="multiband_graphcut",
                         help="Blending strategy (default: multiband_graphcut; 'feather' is a faster legacy mode).")
    parser.add_argument("--exposure-compensator", choices=["blocks_gain", "gain", "none"], default="blocks_gain",
                         help="Exposure/lighting compensation across tiles (default: blocks_gain).")
    parser.add_argument("--seam-finder", choices=["graphcut", "voronoi", "none"], default="graphcut",
                         help="Seam placement strategy (default: graphcut).")
    parser.add_argument("--seam-max-dim", type=int, default=1600,
                         help="Max dimension for seam-finding working resolution (default: 1600). "
                              "Blending always happens at full resolution regardless.")
    parser.add_argument("--multiband-bands", type=int, default=5, help="Number of pyramid bands (default: 5).")
    parser.add_argument("--feather-width", type=int, default=50,
                         help="Feather blend distance clip in pixels; only used with --blend-mode feather.")

    parser.add_argument("--max-image-dimension", type=int, default=0,
                         help="Downscale inputs so their longest side is at most this many pixels "
                              "(0 = keep original resolution).")
    parser.add_argument("--max-canvas-mp", type=float, default=400.0,
                         help="Safety cap on canvas size in megapixels (default: 400.0).")
    parser.add_argument("--enable-opencl", action="store_true",
                         help="Allow OpenCV to use OpenCL/GPU acceleration (default: disabled). Only "
                              "enable this if you know your GPU/OpenCL device has enough memory for "
                              "your full-resolution tiles -- otherwise large images can fail with a "
                              "CL_MEM_OBJECT_ALLOCATION_FAILURE error instead of just running on CPU.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images.")
    parser.add_argument("--debug-dir", default="debug", help="Directory for debug images (default: ./debug).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG-level) logging.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.enable_opencl:
        cv2.ocl.setUseOpenCL(True)
        logger.warning(
            "OpenCL/GPU acceleration enabled by request. If large tiles fail with "
            "CL_MEM_OBJECT_ALLOCATION_FAILURE, drop --enable-opencl to force the CPU path."
        )
    else:
        logger.info("OpenCL disabled (default) -- running on CPU. Pass --enable-opencl to opt in.")

    config = StitchConfig(
        feature_detector=args.detector.upper(),
        ratio_threshold=args.ratio,
        ransac_reproj_threshold=args.ransac_thresh,
        min_matches=args.min_matches,
        match_confidence_threshold=args.match_confidence,
        subpixel_refine=not args.no_subpixel,
        bundle_adjustment=not args.no_bundle_adjustment,
        use_cylindrical_projection=args.cylindrical,
        cylindrical_focal_length=args.focal_length,
        blend_mode=args.blend_mode,
        exposure_compensator=args.exposure_compensator,
        seam_finder=args.seam_finder,
        seam_estimation_max_dimension=args.seam_max_dim,
        multiband_num_bands=args.multiband_bands,
        feather_width=args.feather_width,
        max_image_dimension=args.max_image_dimension,
        max_canvas_megapixels=args.max_canvas_mp,
        save_debug_images=args.debug,
        debug_dir=args.debug_dir,
    )

    stitcher = PanoramaStitcher(config)

    try:
        panorama = stitcher.stitch(args.input_folder)
    except (StitchingError, FileNotFoundError, NotADirectoryError) as exc:
        logger.error("Stitching failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("Unexpected error during stitching: %s", exc)
        raise

    success = cv2.imwrite(args.output, panorama)
    if not success:
        logger.error("Failed to write output image to '%s'.", args.output)
        sys.exit(1)

    logger.info("Mosaic saved to '%s' (%dx%d).", args.output, panorama.shape[1], panorama.shape[0])


if __name__ == "__main__":
    main()
