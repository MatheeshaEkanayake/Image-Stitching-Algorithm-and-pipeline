#!/usr/bin/env python3
"""
Multi-Image Panorama Stitcher (Cylindrical Projection)
========================================================

Stitches an arbitrary number of ordered (left-to-right), overlapping images
into a single panorama using a *global* homography chain instead of
iterative pairwise stitching.

Why not iterative stitching (stitch pano so far + next image, repeat)?
  - Every iteration re-warps an already-warped (interpolated, resampled)
    image, compounding resampling blur and geometric error.
  - Homography error from early pairs propagates and is amplified at every
    subsequent step.
  - The "reference frame" silently drifts to whatever the running panorama
    happens to be, rather than a single, stable coordinate system.

Why cylindrical projection (and why BEFORE feature detection)?
  A single planar homography per pair is only an exact model of two photos
  of the same scene taken from the same optical center. For a panorama
  swept across a wide angle, that model breaks down: straight lines bow,
  and content near the edges gets stretched far more than content near the
  center, because a flat plane can only "cover" a limited field of view
  before the projection math itself becomes extreme. Projecting each photo
  onto a cylinder first re-expresses the images in a coordinate system
  where a pure camera rotation becomes an (approximately) constant
  horizontal shift -- which is what makes wide, multi-image panoramas from
  commercial tools look rectilinear-correct instead of "bow-tied." This
  must happen before feature detection/matching/homography estimation,
  because those steps are what discover the alignment between images --
  if they run on the un-projected originals, the alignment they find (and
  therefore everything built on top of it) is still in the distorted,
  planar space. See the longer discussion in the chat response for detail.

Pipeline order:
  1. Load images.
  2. Estimate focal length (EXIF, explicit override, or heuristic).
  3. Cylindrically project every image (before any feature work).
  4. Detect features on the projected images.
  5. Match adjacent pairs.
  6. Estimate one homography per pair via RANSAC.
  7. Compose a global homography per image, straighten, compute canvas,
     and warp every projected image exactly once into the canvas.
  8. Blend overlapping regions (feather; pluggable for multiband).
  9. Crop black borders.
  10. (Folded into step 7's composition -- see straighten_panorama().)
  11. Final cleanup (free per-image buffers once the panorama is built).

Usage:
    python stitch.py input_folder output.jpg
    python stitch.py ./images panorama.jpg --detector sift --ratio 0.75
    python stitch.py ./images panorama.jpg --focal-length-px 1800

Requirements:
    pip install opencv-contrib-python numpy
    pip install Pillow   # optional: enables EXIF-based focal length lookup
"""

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

    # --- Cylindrical projection (NEW) ---
    # "cylindrical" projects every image onto a cylinder before feature
    # detection (see module docstring for why). "planar" keeps the old
    # behavior (raw images fed straight to feature detection) -- useful
    # for comparison/fallback, and kept as a first-class mode rather than
    # deleted code so the two approaches can be A/B'd on the same images.
    projection_mode: str = "cylindrical"  # "cylindrical" or "planar"
    # Explicit focal length in PIXELS. If set, this wins over EXIF/heuristic
    # and is used for every image (a panorama sequence should share one
    # focal length -- see estimate_focal_length()). None = auto-detect.
    focal_length_px: Optional[float] = None
    # Fallback used only when focal_length_px is None AND no image has a
    # usable EXIF focal length. Approximates a "normal" consumer lens.
    assumed_horizontal_fov_deg: float = 60.0

    # --- Straightening (NEW) ---
    # Corrects the systematic vertical "wave"/tilt that accumulates across
    # a long sequence (small per-pair rotation errors compounding). See
    # straighten_panorama().
    straighten_panorama: bool = True

    # --- Exposure compensation (NEW, optional) ---
    # Off by default: this is a simple mean-brightness gain-matching model,
    # not a substitute for a proper multi-band/graph-cut exposure
    # compensator. See compute_exposure_gains().
    exposure_compensation: bool = False

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
    image: np.ndarray                                   # Working image: raw until cylindrical
                                                          # projection runs, then the projected image.
                                                          # Every downstream step (features, matching,
                                                          # homography, final warp) uses THIS field.
    original_image: Optional[np.ndarray] = None          # Untouched raw photo (kept for reference/EXIF;
                                                          # None if projection_mode == "planar").
    valid_mask: Optional[np.ndarray] = None               # 0/255 uint8: which pixels of `image` are real
                                                          # content vs. the cylindrical warp's empty
                                                          # corners. None => whole image is valid.
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

        # pairwise_homographies[k] maps image k's coordinates into image (k-1)'s
        # coordinate frame. Populated for k = 1 .. N-1.
        self.pairwise_homographies: Dict[int, np.ndarray] = {}

        # global_homographies[i] maps image i's coordinates directly into the
        # single shared reference frame (the panorama frame, before translation).
        self.global_homographies: List[np.ndarray] = []
        self.reference_index: int = 0

        # Canvas geometry.
        self.canvas_size: Tuple[int, int] = (0, 0)     # (width, height)
        self.translation: np.ndarray = np.eye(3, dtype=np.float64)

        # Shared focal length (pixels) used for cylindrical projection of
        # every image -- see estimate_focal_length().
        self.focal_px: Optional[float] = None

        # Per-image brightness multiplier from optional exposure
        # compensation; index-aligned with self.images. Defaults to all
        # 1.0 (no-op) even when the feature is disabled, so downstream
        # code never needs an extra "is this enabled" branch.
        self.exposure_gains: List[float] = []

        self.panorama: Optional[np.ndarray] = None

    # --------------------------------------------------------------------
    # Step 1: Load images
    # --------------------------------------------------------------------
    def load_images(self, folder: str) -> None:
        """
        Load every jpg/jpeg/png in `folder`, sorted in natural filename order.
        Images are kept at their original resolution.
        """
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
    # Step 2: Estimate focal length (EXIF, explicit override, or heuristic)
    # --------------------------------------------------------------------
    def _read_exif_focal_length_px(self, path: str, image_width_px: int) -> Optional[float]:
        """
        Try to recover a focal length in PIXELS from EXIF, using the
        35mm-equivalent focal length tag when present. That tag is
        sensor-independent (a 35mm-format frame is a fixed 36mm wide), so
        f_px = f_35mm_equiv * (image_width_px / 36mm) is reliable across
        different cameras -- unlike the raw FocalLength tag, which needs
        the physical sensor width (rarely present in EXIF) to convert to
        pixels. Returns None if Pillow isn't installed or the tag is
        missing, so callers must have a fallback.
        """
        try:
            from PIL import Image, ExifTags
        except ImportError:
            logger.debug("Pillow not installed; skipping EXIF focal length lookup.")
            return None

        try:
            with Image.open(path) as pil_img:
                exif_raw = pil_img.getexif()
                # FocalLengthIn35mmFilm (and most camera-settings tags)
                # live in the "Exif" sub-IFD, not the top-level IFD that
                # getexif() returns directly -- it has to be fetched
                # explicitly via get_ifd().
                exif_sub_ifd = exif_raw.get_ifd(ExifTags.IFD.Exif) if exif_raw else {}
        except Exception as exc:
            logger.debug("Could not read EXIF from %s (%s)", path, exc)
            return None

        if not exif_sub_ifd:
            return None

        tag_name_by_id = {v: k for k, v in ExifTags.TAGS.items()}
        focal_35mm_tag = tag_name_by_id.get("FocalLengthIn35mmFilm")
        focal_35mm = exif_sub_ifd.get(focal_35mm_tag) if focal_35mm_tag else None

        if not focal_35mm:
            return None

        return float(focal_35mm) * (image_width_px / 36.0)

    def estimate_focal_length(self) -> float:
        """
        Determine ONE shared focal length (pixels) used to cylindrically
        project every image. A single shared value is used deliberately:
        cylindrical projection assumes one consistent camera/lens/zoom
        across the whole sweep, which is the normal way panoramas are
        shot (and mixing per-image focal lengths would misalign the
        cylinder radius image-to-image, causing vertical seams).

        Priority: explicit config override > EXIF (median across images,
        with a warning if they disagree a lot) > heuristic from image
        width and the configured assumed FOV.
        """
        if self.config.focal_length_px is not None:
            logger.info("Focal length: %.1f px (explicit override).", self.config.focal_length_px)
            return self.config.focal_length_px

        exif_estimates = []
        for data in self.images:
            w = data.image.shape[1]
            f = self._read_exif_focal_length_px(data.path, w)
            if f is not None:
                exif_estimates.append(f)

        if exif_estimates:
            f_final = float(np.median(exif_estimates))
            spread = (max(exif_estimates) - min(exif_estimates)) / f_final if f_final else 0.0
            logger.info(
                "Focal length: %.1f px (EXIF median across %d/%d images).",
                f_final, len(exif_estimates), len(self.images),
            )
            if spread > 0.15:
                logger.warning(
                    "EXIF focal lengths vary by %.0f%% across images -- if zoom changed "
                    "mid-sequence, cylindrical projection will be geometrically "
                    "inconsistent. Consider --focal-length-px to force one value.",
                    spread * 100,
                )
            return f_final

        w = self.images[0].image.shape[1]
        fov_rad = np.radians(self.config.assumed_horizontal_fov_deg)
        f_heuristic = (w / 2.0) / np.tan(fov_rad / 2.0)
        logger.info(
            "Focal length: %.1f px (no EXIF found; heuristic from image width assuming "
            "a %.0f-degree horizontal FOV -- pass --focal-length-px if you know the "
            "real value for better accuracy).",
            f_heuristic, self.config.assumed_horizontal_fov_deg,
        )
        return f_heuristic

    # --------------------------------------------------------------------
    # Step 3: Cylindrical projection (runs BEFORE feature detection)
    # --------------------------------------------------------------------
    def cylindrical_project(
        self, image: np.ndarray, focal_px: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp `image` onto a cylinder of radius `focal_px`, using the
        projection equations:
            theta = atan((x - cx) / f)      h = (y - cy) / sqrt((x-cx)^2 + f^2)
            x'    = f * theta + cx          y' = f * h + cy

        Implemented via the INVERSE mapping (solve for source (x, y) given
        destination (x', y')) and cv2.remap, which is what keeps this
        artifact-free: every destination pixel gets a value bilinearly
        interpolated from real source pixels, so there are no holes to
        patch up afterward. Solving the equations above for (x, y):
            theta = (x' - cx) / f            h = (y' - cy) / f
            x = cx + f * tan(theta)          y = cy + h * f / cos(theta)

        Returns (projected_image, valid_mask). valid_mask is 0/255 uint8,
        255 where the source coordinate actually fell inside the original
        image (the rectangular output canvas has unavoidable empty
        corners/edges for a cylindrical warp -- this mask is what lets
        every later stage, from feature detection through blending,
        correctly ignore them instead of treating them as real black
        content).
        """
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        x_dst, y_dst = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )

        theta = (x_dst - cx) / focal_px
        h_norm = (y_dst - cy) / focal_px
        # cos(theta) only approaches 0 (a singularity) at a +/-90 degree
        # field of view, far beyond any realistic single photo -- normal
        # FOVs (< ~100 deg total) keep |theta| well under that.
        map_x = (focal_px * np.tan(theta) + cx).astype(np.float32)
        map_y = (h_norm * focal_px / np.cos(theta) + cy).astype(np.float32)

        projected = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Validity is computed from the SOURCE coordinates, not by
        # thresholding the output -- that way genuinely dark/black source
        # pixels near the frame edge aren't mistaken for "no data."
        valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
        valid_mask = (valid.astype(np.uint8)) * 255

        return projected, valid_mask

    def apply_cylindrical_projection(self, focal_px: float) -> None:
        """
        Project every loaded image onto the cylinder in place. This
        replaces `data.image` (which every downstream step reads) with the
        projected version, stashes the untouched original in
        `data.original_image`, and records the per-pixel validity mask.
        No-op if config.projection_mode == "planar".
        """
        if self.config.projection_mode != "cylindrical":
            logger.info("projection_mode='planar': skipping cylindrical projection.")
            return

        if self.config.save_debug_images:
            os.makedirs(self.config.debug_dir, exist_ok=True)

        for i, data in enumerate(self.images):
            projected, valid_mask = self.cylindrical_project(data.image, focal_px)
            data.original_image = data.image
            data.image = projected
            data.valid_mask = valid_mask
            logger.info("Cylindrically projected image %d (f=%.1f px).", i, focal_px)

            if self.config.save_debug_images:
                cv2.imwrite(
                    os.path.join(self.config.debug_dir, f"cyl_{i:02d}.jpg"), projected
                )

    # --------------------------------------------------------------------
    # Step 4: Feature detection
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
            # Restrict detection to real content: after cylindrical
            # projection, the rectangular frame has empty corners/edges
            # that would otherwise seed spurious keypoints and matches.
            keypoints, descriptors = self.detector.detectAndCompute(gray, data.valid_mask)

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
    # Step 5: Match neighboring images only
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
    # Step 6: Estimate pairwise homographies via RANSAC
    # --------------------------------------------------------------------
    def estimate_homographies(self, matches_by_pair: Dict[int, List[cv2.DMatch]]) -> None:
        """
        For every adjacent pair (i, i+1), estimate a homography mapping
        image (i+1)'s coordinates into image i's coordinate frame, via
        RANSAC. Pairs with too few matches abort the pipeline, since a
        broken link makes the rest of the chain unrecoverable.
        """
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

            # src = points in image i+1, dst = points in image i.
            # H will map image (i+1) -> image i's frame.
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

            # Sanity check: how much does this homography scale image (i+1)'s
            # area? A weak/wrong match set can produce a technically-valid
            # homography that implies a wildly implausible scale change --
            # this is the #1 cause of a canvas silently exploding in size
            # with many images. We warn here (not fail), since real
            # wide-baseline pairs can have moderate scale changes.
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
    # Optional post-processing: exposure compensation
    #
    # Placed here (not later, near blending) so it can reuse the pairwise
    # homographies and cylindrical images we already have in memory --
    # computing brightness ratios in each pair's overlap this way needs no
    # extra full-canvas warps and fits the same memory-safe, streaming
    # design as the rest of the pipeline. It's still conceptually a
    # "post-processing" concern (off by default; corrects exposure, not
    # geometry) -- it's just computed early for architectural reuse.
    # --------------------------------------------------------------------
    def _pairwise_exposure_ratio(self, idx_a: int, idx_b: int, H_b_to_a: np.ndarray) -> float:
        """
        Mean-brightness ratio (A / B) over the two images' overlap region,
        computed by warping B into A's frame with the pairwise homography
        we already estimated -- no new geometry, just reused.
        """
        img_a = self.images[idx_a].image
        img_b = self.images[idx_b].image
        h_a, w_a = img_a.shape[:2]

        warped_b = cv2.warpPerspective(img_b, H_b_to_a, (w_a, h_a))
        mask_b_src = self.images[idx_b].valid_mask
        if mask_b_src is None:
            mask_b_src = np.full(img_b.shape[:2], 255, dtype=np.uint8)
        warped_mask_b = cv2.warpPerspective(
            mask_b_src, H_b_to_a, (w_a, h_a), flags=cv2.INTER_NEAREST
        )

        mask_a = self.images[idx_a].valid_mask
        if mask_a is None:
            mask_a = np.full((h_a, w_a), 255, dtype=np.uint8)

        overlap = (mask_a > 0) & (warped_mask_b > 0)
        if overlap.sum() < 100:
            return 1.0  # Overlap too small to trust a brightness estimate.

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_b = cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mean_a = gray_a[overlap].mean()
        mean_b = gray_b[overlap].mean()
        return float(mean_a / mean_b) if mean_b > 1e-3 else 1.0

    def compute_exposure_gains(self) -> None:
        """
        Estimate a per-image brightness multiplier so overlapping regions
        match in exposure, chained outward from the reference image
        exactly like compute_global_transforms() chains geometry. This is
        a simple mean-matching model (not a full gain-compensation solve
        across all overlaps at once); it's meant as a reasonable default,
        not a professional-grade compensator.

        Always populates self.exposure_gains (with all 1.0 if disabled),
        so downstream code never needs to branch on whether this ran.
        """
        n = len(self.images)
        self.exposure_gains = [1.0] * n

        if not self.config.exposure_compensation:
            return

        ref = self.reference_index
        for i in range(ref + 1, n):
            ratio = self._pairwise_exposure_ratio(i - 1, i, self.pairwise_homographies[i])
            self.exposure_gains[i] = self.exposure_gains[i - 1] * ratio
        for i in range(ref - 1, -1, -1):
            ratio = self._pairwise_exposure_ratio(i, i + 1, self.pairwise_homographies[i + 1])
            self.exposure_gains[i] = self.exposure_gains[i + 1] / ratio if ratio else self.exposure_gains[i + 1]

        logger.info(
            "Exposure gains (reference=%d): %s",
            ref, ["%.2f" % g for g in self.exposure_gains],
        )

    # --------------------------------------------------------------------
    # Step 7a: Compose global homographies (the core "no iterative warping" idea)
    # --------------------------------------------------------------------
    def compute_global_transforms(self) -> None:
        """
        Compose the adjacent-pair homographies into one homography per
        image, all relative to a single fixed reference frame: the middle
        image of the sequence.

        For an image to the right of the reference, we chain forward:
            global[i] = global[i-1] @ pairwise[i]        (pairwise[i]: i -> i-1)

        For an image to the left of the reference, we chain the inverse:
            global[i] = global[i+1] @ inverse(pairwise[i+1])  (pairwise[i+1]: i+1 -> i)

        No image is ever warped at this stage -- these are pure matrix
        compositions. Warping happens exactly once, later, in warp_images().
        """
        n = len(self.images)
        self.reference_index = n // 2
        self.global_homographies = [np.eye(3, dtype=np.float64) for _ in range(n)]

        ref = self.reference_index
        logger.info(
            "Reference image: [%d] %s",
            ref, os.path.basename(self.images[ref].path),
        )

        # Chain forward (right of reference).
        for i in range(ref + 1, n):
            H_local = self.pairwise_homographies[i]  # maps i -> i-1
            self.global_homographies[i] = self.global_homographies[i - 1] @ H_local

        # Chain backward (left of reference), using inverses.
        for i in range(ref - 1, -1, -1):
            H_local = self.pairwise_homographies[i + 1]        # maps i+1 -> i
            H_local_inv = np.linalg.inv(H_local)                # maps i -> i+1
            self.global_homographies[i] = self.global_homographies[i + 1] @ H_local_inv

    # --------------------------------------------------------------------
    # Step 10: Straighten the panorama (executed here, not after cropping)
    #
    # Long sequences accumulate small per-pair rotation errors, which show
    # up as a systematic vertical "wave" or tilt across the panorama --
    # the classic symptom of naive multi-image stitching. We fold the
    # correction directly into the homography chain (a pure matrix
    # multiply) rather than rotating the finished raster panorama,
    # because rotating a finished raster would mean warping it a SECOND
    # time -- exactly the repeated-resampling quality loss this whole
    # pipeline is built to avoid (see module docstring). Every original
    # image still gets warped exactly once, straight into its final,
    # already-straightened position.
    # --------------------------------------------------------------------
    def straighten_panorama(self) -> None:
        """
        Fit a straight line through each image's projected center point
        (in the not-yet-translated global/reference frame) and rotate the
        whole frame, about the reference image's own center, so that line
        is level. No-op if disabled or if there are too few images for a
        trend line to be meaningful.
        """
        n = len(self.images)
        if not self.config.straighten_panorama or n < 3:
            return

        centers_x, centers_y = [], []
        for i, data in enumerate(self.images):
            h, w = data.image.shape[:2]
            center = np.float32([[w / 2.0, h / 2.0]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(center, self.global_homographies[i])
            cx, cy = projected.ravel()
            centers_x.append(float(cx))
            centers_y.append(float(cy))

        # Fit y = m*x + b through the panorama-space image centers.
        m, _b = np.polyfit(centers_x, centers_y, deg=1)
        angle = np.arctan(m)

        if abs(np.degrees(angle)) < 0.1:
            logger.info("Straightening: drift is negligible (%.2f deg); skipping.", np.degrees(angle))
            return

        logger.info("Straightening: correcting %.2f degree vertical drift.", np.degrees(angle))

        # Rotate by -angle about the reference image's own projected
        # center, so the reference image doesn't shift -- only the
        # sequence's overall "lean" relative to it is removed.
        px = centers_x[self.reference_index]
        py = centers_y[self.reference_index]
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R = np.array([
            [cos_a, -sin_a, px - cos_a * px + sin_a * py],
            [sin_a,  cos_a, py - sin_a * px - cos_a * py],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        for i in range(n):
            self.global_homographies[i] = R @ self.global_homographies[i]


    # --------------------------------------------------------------------
    # Step 7b: Compute panorama canvas bounds
    # --------------------------------------------------------------------
    def compute_canvas(self) -> None:
        """
        Project every image's corners through its global homography to find
        the bounding box of the full panorama, then build a translation
        matrix so every warped coordinate ends up non-negative.
        """
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

        # Rough peak-memory estimate for the warp+blend stage (streaming
        # accumulator in float32, plus one warped image/mask in flight).
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
    # Step 7c: Warp every PROJECTED image exactly once
    # --------------------------------------------------------------------
    def warp_images(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Warp each (cylindrically projected) image directly into the shared
        panorama canvas, using translation @ global_homography[i]. Also
        produces a binary validity mask per image (used for blending and
        cropping).

        Each image is warped exactly once -- never re-warped, and never
        composited into an intermediate panorama before this point.

        This is a GENERATOR, not a list-builder: it yields one (image, mask)
        pair at a time instead of holding every warped, full-canvas-sized
        image in memory simultaneously. For N images that's the difference
        between peak memory of ~1 canvas vs. ~N canvases -- the single
        biggest lever for stitching many images without running out of RAM.
        The caller (feather_blend) consumes each pair and lets it be
        garbage-collected before the next one is produced.
        """
        w_canvas, h_canvas = self.canvas_size

        if self.config.save_debug_images:
            os.makedirs(self.config.debug_dir, exist_ok=True)

        for i, data in enumerate(self.images):
            H_final = self.translation @ self.global_homographies[i]

            # Apply the (optional) exposure gain before warping, so
            # interpolation during the warp works on already-corrected
            # values rather than correcting after the fact.
            gain = self.exposure_gains[i] if self.exposure_gains else 1.0
            src_img = data.image if gain == 1.0 else np.clip(
                data.image.astype(np.float32) * gain, 0, 255
            ).astype(np.uint8)

            warped_img = cv2.warpPerspective(
                src_img, H_final, (w_canvas, h_canvas),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            # Use the cylindrical projection's own validity mask when
            # present (the projected frame has empty corners/edges that
            # must NOT be treated as real image content), falling back to
            # "whole rectangle is valid" for planar mode.
            h, w = data.image.shape[:2]
            mask_src = data.valid_mask if data.valid_mask is not None else \
                np.full((h, w), 255, dtype=np.uint8)
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
        # Step 1: load.
        self.load_images(input_folder)

        if len(self.images) == 1:
            logger.info("Only one image found; returning it unchanged.")
            self.panorama = self.images[0].image
            return self.panorama

        # Step 2: focal length (shared across the whole sequence).
        self.focal_px = self.estimate_focal_length()

        # Step 3: cylindrical projection -- BEFORE any feature work, so
        # everything from here on operates in cylindrical space.
        self.apply_cylindrical_projection(self.focal_px)

        # Step 4-6: detect, match, estimate homographies (all on the
        # cylindrical images now).
        self.detect_features()
        matches_by_pair = self.match_features()
        self.estimate_homographies(matches_by_pair)

        # Step 7: compose global transforms (this sets self.reference_index,
        # which exposure-gain chaining below also needs), then straighten
        # (step 10, folded in here -- see straighten_panorama docstring for
        # why), then compute the canvas and warp+stitch.
        self.compute_global_transforms()

        # Optional: exposure gains, reusing the pairwise homographies
        # from estimate_homographies() and the reference index just set
        # above (see compute_exposure_gains docstring).
        self.compute_exposure_gains()

        self.straighten_panorama()
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

        # Step 9: crop black borders.
        self.panorama = self.crop_panorama(blended)

        # Step 11: final cleanup. The panorama is built; drop the heavy
        # per-image buffers (descriptors, keypoints, cylindrical copies,
        # masks) in case the caller keeps this stitcher object alive --
        # otherwise they'd sit in memory needlessly for the rest of the
        # object's lifetime.
        self._release_intermediate_buffers()

        return self.panorama

    def _release_intermediate_buffers(self) -> None:
        """Step 11: free per-image working data once the panorama is final."""
        for data in self.images:
            data.keypoints = None
            data.descriptors = None
            data.original_image = None
            data.valid_mask = None
        logger.info("Released intermediate per-image buffers.")


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
    parser.add_argument("--projection", choices=["cylindrical", "planar"], default="cylindrical",
                         help="Project onto a cylinder before feature detection (default), or "
                              "use the original flat/planar behavior for comparison.")
    parser.add_argument("--focal-length-px", type=float, default=None,
                         help="Explicit focal length in PIXELS for cylindrical projection. "
                              "Overrides EXIF/heuristic. Omit to auto-detect.")
    parser.add_argument("--fov", type=float, default=60.0,
                         help="Assumed horizontal field of view in degrees, used ONLY as a "
                              "fallback focal-length heuristic when no EXIF and no "
                              "--focal-length-px is available (default: 60.0).")
    parser.add_argument("--no-straighten", action="store_true",
                         help="Disable automatic panorama straightening (wave/tilt correction).")
    parser.add_argument("--exposure-compensation", action="store_true",
                         help="Enable simple mean-brightness exposure matching between "
                              "overlapping images (default: off).")
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
        projection_mode=args.projection,
        focal_length_px=args.focal_length_px,
        assumed_horizontal_fov_deg=args.fov,
        straighten_panorama=not args.no_straighten,
        exposure_compensation=args.exposure_compensation,
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
