#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class StitchConfig:
    """All tunable parameters for the stitching pipeline live here."""

    # --- Feature detection ---
    feature_detector: str = "SIFT"       # "SIFT" or "ORB" (auto-falls back to ORB if SIFT unavailable)
    orb_n_features: int = 5000           # Only used if ORB is selected/used as fallback

    # --- Matching ---
    ratio_threshold: float = 0.75        # Lowe's ratio test threshold

    # --- Homography estimation ---
    ransac_reproj_threshold: float = 4.0  # RANSAC reprojection error threshold (pixels)
    min_matches: int = 10                 # Minimum good matches required to accept a pair

    # --- Blending ---
    blend_mode: str = "feather"           # "feather" (implemented) or "multiband" (stub, see below)
    feather_width: int = 50               # Max distance-transform clip radius; 0 = unclipped

    # --- Memory safety ---
    # Hard cap on canvas size. A single bad pairwise homography (weak/wrong
    # match) can make the composed canvas explode to an absurd size, which
    # otherwise silently eats all available RAM and kills the process (often
    # showing up as "the IDE/terminal crashed" with no clean Python error).
    max_canvas_megapixels: float = 400.0
    # If >0, downscale loaded images so max(width, height) <= this value
    # before feature detection/warping. Great for quickly testing pipelines
    # with many images before committing to full-resolution output.
    max_image_dimension: int = 0          # 0 = keep original resolution (default)

    # --- Debugging / output ---
    save_debug_images: bool = False
    debug_dir: str = "debug"


# Module-level logger. Configured in main(); library code just uses it.
logger = logging.getLogger("PanoramaStitcher")

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


class StitchingError(Exception):
    """Raised when the pipeline cannot proceed reliably (e.g. broken homography chain)."""


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


# ==============================================================================
# Helpers
# ==============================================================================


def natural_sort_key(filename: str) -> List:
    """
    Build a sort key so filenames sort in "human" order
    (img2.jpg before img10.jpg, not after).
    """
    parts = re.split(r"(\d+)", filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


# ==============================================================================
# Main pipeline class
# ==============================================================================


class PanoramaStitcher:
    """
    Multi-image panorama stitcher built around a global homography chain.

    Typical usage:
        stitcher = PanoramaStitcher(StitchConfig())
        panorama = stitcher.stitch("./images")
        cv2.imwrite("panorama.jpg", panorama)
    """

    def __init__(self, config: StitchConfig):
        self.config = config

        self.images: List[ImageData] = []

        self.detector: Optional[cv2.Feature2D] = None
        self.detector_name: str = ""
        self.norm_type: int = cv2.NORM_L2

        self.pairwise_homographies: Dict[int, np.ndarray] = {}


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

        # Either ORB was explicitly requested, or SIFT init failed above.
        self.detector = cv2.ORB_create(nfeatures=self.config.orb_n_features)
        self.detector_name = "ORB"
        self.norm_type = cv2.NORM_HAMMING
        logger.info("Feature detector: ORB (nfeatures=%d)", self.config.orb_n_features)

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

            data.keypoints = keypoints
            data.descriptors = descriptors
            logger.info(
                "[%d] %s: %d keypoints (%s)",
                idx, os.path.basename(data.path), len(keypoints), self.detector_name,
            )

    # --------------------------------------------------------------------
    # Step 3: Match neighboring images only
    # --------------------------------------------------------------------
    def match_features(self) -> Dict[int, List[cv2.DMatch]]:
        """
        Match image i against image i+1 for every adjacent pair, using
        BFMatcher + Lowe's ratio test. Returns {i: [good matches]}.
        """
        matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)
        matches_by_pair: Dict[int, List[cv2.DMatch]] = {}

        n = len(self.images)
        for i in range(n - 1):
            desc_a = self.images[i].descriptors
            desc_b = self.images[i + 1].descriptors

            knn_matches = matcher.knnMatch(desc_a, desc_b, k=2)

            good_matches = []
            for pair in knn_matches:
                if len(pair) != 2:
                    continue
                m, n_ = pair
                if m.distance < self.config.ratio_threshold * n_.distance:
                    good_matches.append(m)

            matches_by_pair[i] = good_matches
            logger.info(
                "Pair (%d, %d): %d/%d good matches survived ratio test (threshold=%.2f)",
                i, i + 1, len(good_matches), len(knn_matches), self.config.ratio_threshold,
            )

            if self.config.save_debug_images:
                self._save_match_debug_image(i, good_matches)

        return matches_by_pair

    def _save_match_debug_image(self, i: int, good_matches: List[cv2.DMatch]) -> None:
        """Save a visualization of matches between image i and i+1."""
        os.makedirs(self.config.debug_dir, exist_ok=True)
        vis = cv2.drawMatches(
            self.images[i].image, self.images[i].keypoints,
            self.images[i + 1].image, self.images[i + 1].keypoints,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        out_path = os.path.join(self.config.debug_dir, f"matches_{i:02d}_{i+1:02d}.jpg")
        cv2.imwrite(out_path, vis)

    # --------------------------------------------------------------------
    # Step 4: Estimate pairwise homographies
    # --------------------------------------------------------------------
    def estimate_homographies(self, matches_by_pair: Dict[int, List[cv2.DMatch]]) -> None:
        n = len(self.images)

        for i in range(n - 1):
            good_matches = matches_by_pair.get(i, [])

            if len(good_matches) < self.config.min_matches:
                raise StitchingError(
                    f"Insufficient matches between image {i} and {i + 1}: "
                    f"got {len(good_matches)}, need >= {self.config.min_matches}. "
                    "Check that images are ordered left-to-right with real overlap, "
                    "or lower --min-matches / raise --ratio."
                )

            kp_a = self.images[i].keypoints
            kp_b = self.images[i + 1].keypoints


            pts_src = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts_dst = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, inlier_mask = cv2.findHomography(
                pts_src, pts_dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
            )

            if H is None:
                raise StitchingError(
                    f"cv2.findHomography failed for pair ({i}, {i + 1}); "
                    "matches may be degenerate (e.g. all collinear)."
                )

            if abs(np.linalg.det(H)) < 1e-10:
                logger.warning(
                    "Pair (%d, %d): homography looks near-singular (det~0); "
                    "result may be unstable.", i, i + 1,
                )


            h_b, w_b = self.images[i + 1].image.shape[:2]
            src_area = float(h_b * w_b)
            corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
            warped_corners_b = cv2.perspectiveTransform(corners_b, H)
            dst_area = abs(cv2.contourArea(warped_corners_b))
            area_ratio = dst_area / src_area if src_area > 0 else float("inf")
            if area_ratio > 20.0 or area_ratio < 1.0 / 20.0:
                logger.warning(
                    "Pair (%d, %d): homography implies a %.1fx area scale change -- "
                    "this pair is the likely culprit if the canvas ends up huge. "
                    "Check image order/overlap for this pair, or inspect "
                    "matches_%02d_%02d.jpg with --debug.",
                    i, i + 1, area_ratio, i, i + 1,
                )

            n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            inlier_pct = 100.0 * n_inliers / max(len(good_matches), 1)
            log_fn = logger.warning if inlier_pct < 20.0 else logger.info
            log_fn(
                "Pair (%d, %d): RANSAC inliers %d/%d (%.1f%%)%s",
                i, i + 1, n_inliers, len(good_matches), inlier_pct,
                "  <- low inlier rate, homography may be unreliable" if inlier_pct < 20.0 else "",
            )

            self.pairwise_homographies[i + 1] = H

    # --------------------------------------------------------------------
    # Step 5: Compose global homographies (the core "no iterative warping" idea)
    # --------------------------------------------------------------------
    def compute_global_transforms(self) -> None:
        n = len(self.images)
        self.reference_index = n // 2
        self.global_homographies = [np.eye(3, dtype=np.float64) for _ in range(n)]

        ref = self.reference_index
        logger.info(
            "Reference image: [%d] %s",
            ref, os.path.basename(self.images[ref].path),
        )

        for i in range(ref + 1, n):
            H_local = self.pairwise_homographies[i]  # maps i -> i-1
            self.global_homographies[i] = self.global_homographies[i - 1] @ H_local

        for i in range(ref - 1, -1, -1):
            H_local = self.pairwise_homographies[i + 1]        # maps i+1 -> i
            H_local_inv = np.linalg.inv(H_local)                # maps i -> i+1
            self.global_homographies[i] = self.global_homographies[i + 1] @ H_local_inv

    # --------------------------------------------------------------------
    # Step 6: Compute panorama canvas bounds
    # --------------------------------------------------------------------
    def compute_canvas(self) -> None:
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
            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]],
            dtype=np.float64,
        )
        self.canvas_size = (int(x_max - x_min), int(y_max - y_min))
        n_pixels = self.canvas_size[0] * self.canvas_size[1]
        megapixels = n_pixels / 1e6

 
        est_gb = (n_pixels * 20) / (1024 ** 3)

        logger.info(
            "Canvas size: %d x %d (%.1f MP, translation offset: (%d, %d), "
            "~%.2f GB estimated peak memory for blending)",
            self.canvas_size[0], self.canvas_size[1], megapixels, -x_min, -y_min, est_gb,
        )

        if megapixels > self.config.max_canvas_megapixels:
            raise StitchingError(
                f"Computed canvas is {megapixels:.1f} MP, exceeding the "
                f"{self.config.max_canvas_megapixels:.0f} MP safety cap. This is almost "
                "always caused by one or more unreliable pairwise homographies -- check "
                "the log above for 'low inlier rate' or 'implies a NNx area scale change' "
                "warnings. With many images, even several mediocre (not outright wrong) "
                "pairs can compound across the chain into a huge canvas. Try a stricter "
                "--ratio (e.g. 0.6) or --ransac-thresh, or reshoot/reorder the flagged "
                "pairs. If this canvas size is genuinely correct, raise --max-canvas-mp. "
                "--max-image-dimension shrinks inputs while you debug."
            )

    # --------------------------------------------------------------------
    # Step 7: Warp every ORIGINAL image exactly once
    # --------------------------------------------------------------------
    def warp_images(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        w_canvas, h_canvas = self.canvas_size

        if self.config.save_debug_images:
            os.makedirs(self.config.debug_dir, exist_ok=True)

        for i, data in enumerate(self.images):
            H_final = self.translation @ self.global_homographies[i]

            warped_img = cv2.warpPerspective(
                data.image, H_final, (w_canvas, h_canvas),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            h, w = data.image.shape[:2]
            mask_src = np.full((h, w), 255, dtype=np.uint8)
            warped_mask = cv2.warpPerspective(
                mask_src, H_final, (w_canvas, h_canvas),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            logger.info("Warped image %d into panorama canvas (once).", i)

            if self.config.save_debug_images:
                cv2.imwrite(
                    os.path.join(self.config.debug_dir, f"warped_{i:02d}.jpg"), warped_img
                )

            yield warped_img, warped_mask

    # --------------------------------------------------------------------
    # Step 8: Blend
    # --------------------------------------------------------------------
    def blend_images(self, warped: Iterator[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Dispatch to the configured blending strategy."""
        if self.config.blend_mode == "feather":
            return self.feather_blend(warped)
        elif self.config.blend_mode == "multiband":
            return self.multiband_blend(warped)
        raise ValueError(f"Unknown blend_mode: {self.config.blend_mode!r}")

    def feather_blend(self, warped: Iterator[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Simple feather (distance-weighted) blending.

        Each warped image gets a per-pixel weight equal to its distance
        (via cv2.distanceTransform) from the nearest invalid/edge pixel of
        its own mask -- i.e. pixels near the center of an image are trusted
        more than pixels near its warped edge. Overlapping regions are then
        a weighted average, which removes hard seams.

        `warped` is consumed lazily (one image at a time) so only ONE
        warped image is ever in memory alongside the running accumulator --
        not all N of them. float32 (not float64) is used for the running
        totals since panorama pixel values never need double precision,
        halving this stage's memory footprint for free.
        """
        w_canvas, h_canvas = self.canvas_size
        accumulator = np.zeros((h_canvas, w_canvas, 3), dtype=np.float32)
        weight_sum = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        for img, mask in warped:
            weight = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            if self.config.feather_width > 0:
                weight = np.clip(weight, 0, self.config.feather_width)

            accumulator += img.astype(np.float32) * weight[..., None]
            weight_sum += weight

        safe_weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        blended = accumulator / safe_weight_sum[..., None]
        blended[weight_sum == 0] = 0
        return np.clip(blended, 0, 255).astype(np.uint8)

    def multiband_blend(self, warped: Iterator[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Placeholder for multiband (Laplacian pyramid) blending.

        Not implemented in this prototype. To implement:
          1. For each warped image/mask, build a Gaussian pyramid of the
             mask and a Laplacian pyramid of the image.
          2. Blend each pyramid level across all images using that level's
             (smoothed) mask weights.
          3. Collapse the blended Laplacian pyramid back into one image.
        This removes low-frequency seams (feather blending only fixes
        pixel-level jumps, not brightness/color mismatches across images).
        """
        raise NotImplementedError(
            "multiband_blend is a stub -- see docstring for the implementation plan. "
            "Use blend_mode='feather' for now."
        )

    # --------------------------------------------------------------------
    # Step 9: Crop black borders
    # --------------------------------------------------------------------
    def crop_panorama(self, panorama: np.ndarray) -> np.ndarray:
        """Trim the panorama to its tightest non-black bounding box."""
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        nonzero_coords = cv2.findNonZero(gray)

        if nonzero_coords is None:
            logger.warning("Panorama is entirely black; skipping crop.")
            return panorama

        x, y, w, h = cv2.boundingRect(nonzero_coords)
        cropped = panorama[y:y + h, x:x + w]
        logger.info(
            "Cropped panorama: %dx%d -> %dx%d",
            panorama.shape[1], panorama.shape[0], w, h,
        )
        return cropped

    # --------------------------------------------------------------------
    # Orchestration
    # --------------------------------------------------------------------
    def stitch(self, input_folder: str) -> np.ndarray:
        """Run the full pipeline end-to-end and return the final panorama."""
        self.load_images(input_folder)

        if len(self.images) == 1:
            logger.info("Only one image found; returning it unchanged.")
            self.panorama = self.images[0].image
            return self.panorama

        self.detect_features()
        matches_by_pair = self.match_features()
        self.estimate_homographies(matches_by_pair)
        self.compute_global_transforms()
        self.compute_canvas()

        try:
            warped = self.warp_images()
            blended = self.blend_images(warped)
        except MemoryError as exc:
            raise StitchingError(
                "Ran out of memory during warping/blending. Try re-running with "
                "--max-image-dimension 2000 (or smaller) to downscale inputs, "
                "or --max-canvas-mp to double-check the canvas isn't larger than "
                "expected."
            ) from exc

        self.panorama = self.crop_panorama(blended)
        return self.panorama


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch an arbitrary number of ordered, overlapping images "
                    "into one panorama using global homography composition.",
    )
    parser.add_argument("input_folder", help="Folder containing input images (jpg/jpeg/png).")
    parser.add_argument("output", help="Output panorama path, e.g. panorama.jpg")

    parser.add_argument("--detector", choices=["sift", "orb"], default="sift",
                         help="Feature detector (default: sift; auto-falls back to orb if unavailable).")
    parser.add_argument("--ratio", type=float, default=0.75,
                         help="Lowe's ratio test threshold (default: 0.75).")
    parser.add_argument("--ransac-thresh", type=float, default=4.0,
                         help="RANSAC reprojection threshold in pixels (default: 4.0).")
    parser.add_argument("--min-matches", type=int, default=10,
                         help="Minimum good matches required between neighbors (default: 10).")
    parser.add_argument("--feather-width", type=int, default=50,
                         help="Feather blend distance clip in pixels, 0 = unclipped (default: 50).")
    parser.add_argument("--blend-mode", choices=["feather", "multiband"], default="feather",
                         help="Blending strategy (default: feather; multiband is a stub).")
    parser.add_argument("--max-image-dimension", type=int, default=0,
                         help="Downscale inputs so their longest side is at most this many "
                              "pixels (0 = keep original resolution). Useful to bound memory "
                              "when testing with many images.")
    parser.add_argument("--max-canvas-mp", type=float, default=400.0,
                         help="Safety cap on canvas size in megapixels; aborts with a clear "
                              "error instead of trying to allocate an absurd canvas "
                              "(default: 400.0).")
    parser.add_argument("--debug", action="store_true",
                         help="Save intermediate debug images (matches, per-image warps).")
    parser.add_argument("--debug-dir", default="debug",
                         help="Directory for debug images (default: ./debug).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG-level) logging.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    config = StitchConfig(
        feature_detector=args.detector.upper(),
        ratio_threshold=args.ratio,
        ransac_reproj_threshold=args.ransac_thresh,
        min_matches=args.min_matches,
        blend_mode=args.blend_mode,
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

    logger.info("Panorama saved to '%s' (%dx%d).", args.output, panorama.shape[1], panorama.shape[0])


if __name__ == "__main__":
    main()
