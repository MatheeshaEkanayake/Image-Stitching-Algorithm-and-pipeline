"""Command-line interface: ``python stitch.py <folder> <output>`` or ``python -m stitcher``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

import cv2

from stitcher.models import BLEND_MODES, PROJECTIONS, StitchConfig, StitchingError
from stitcher.panorama_stitcher import PanoramaStitcher

logger = logging.getLogger("PanoramaStitcher")


def build_parser() -> argparse.ArgumentParser:
    defaults = StitchConfig()
    parser = argparse.ArgumentParser(
        prog="stitch",
        description="Stitch ordered overlapping images into one panorama.",
    )
    parser.add_argument("input_folder", help="Folder containing input images (jpg/jpeg/png).")
    parser.add_argument("output", help="Output panorama path, e.g. panorama.jpg")
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift")
    parser.add_argument("--ratio", type=float, default=defaults.ratio_threshold,
                        help="Lowe ratio-test threshold.")
    parser.add_argument("--ransac-thresh", type=float, default=defaults.ransac_reproj_threshold,
                        help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--min-matches", type=int, default=defaults.min_matches,
                        help="Minimum good matches per adjacent pair.")
    parser.add_argument("--min-inliers", type=int, default=defaults.min_inliers,
                        help="Minimum RANSAC inliers per adjacent pair.")
    parser.add_argument("--max-area-ratio", type=float, default=defaults.max_area_ratio,
                        help="Reject pair homographies that scale an image's area by more "
                             "than this factor (or its inverse).")
    parser.add_argument("--feather-width", type=int, default=defaults.feather_width,
                        help="Feather ramp width in pixels (0 = unclipped).")
    parser.add_argument("--projection", choices=list(PROJECTIONS), default=defaults.projection,
                        help="cylindrical (rotating camera, default) or planar homography chain.")
    parser.add_argument("--focal", type=float, default=defaults.focal_length,
                        help="Focal length in pixels for cylindrical projection (0 = estimate).")
    parser.add_argument("--blend-mode", choices=list(BLEND_MODES), default=defaults.blend_mode)
    parser.add_argument("--max-image-dimension", type=int, default=0,
                        help="Downscale inputs so the longer side is at most this (0 = off).")
    parser.add_argument("--max-canvas-mp", type=float, default=defaults.max_canvas_megapixels,
                        help="Refuse canvases larger than this many megapixels.")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Seed for OpenCV's random number generator (RANSAC).")
    parser.add_argument("--report", metavar="FILE", help="Write a JSON run report to FILE.")
    parser.add_argument("--debug", action="store_true", help="Save match/warp debug images.")
    parser.add_argument("--debug-dir", default=defaults.debug_dir)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI. Returns the process exit code (0 on success, 1 on a stitching failure)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    try:
        config = StitchConfig(
            feature_detector=args.detector.upper(),
            ratio_threshold=args.ratio,
            ransac_reproj_threshold=args.ransac_thresh,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
            max_area_ratio=args.max_area_ratio,
            projection=args.projection,
            focal_length=args.focal,
            blend_mode=args.blend_mode,
            feather_width=args.feather_width,
            max_image_dimension=args.max_image_dimension,
            max_canvas_megapixels=args.max_canvas_mp,
            seed=args.seed,
            save_debug_images=args.debug,
            debug_dir=args.debug_dir,
        )
        panorama, report = PanoramaStitcher(config).stitch_with_report(args.input_folder)
        _ensure_parent_dir(args.output)
        if not cv2.imwrite(args.output, panorama):
            raise StitchingError(f"Failed to write output image to '{args.output}'.")
        logger.info(
            "Panorama saved to '%s' (%dx%d).", args.output, panorama.shape[1], panorama.shape[0]
        )
        if args.report:
            _ensure_parent_dir(args.report)
            with open(args.report, "w", encoding="utf-8") as handle:
                handle.write(report.to_json())
            logger.info("Report saved to '%s'.", args.report)
    except (StitchingError, FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        logger.error("Stitching failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
