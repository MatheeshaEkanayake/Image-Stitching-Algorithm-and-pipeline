#!/usr/bin/env python3
"""Command-line entry point for the panorama stitcher."""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from models import StitchConfig, StitchingError
from panorama_stitcher import PanoramaStitcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch ordered overlapping images into one panorama."
    )
    parser.add_argument("input_folder", help="Folder containing input images (jpg/jpeg/png).")
    parser.add_argument("output", help="Output panorama path, e.g. panorama.jpg")
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=4.0)
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--feather-width", type=int, default=50)
    parser.add_argument("--blend-mode", choices=["feather", "multiband"], default="multiband")
    parser.add_argument("--max-image-dimension", type=int, default=0)
    parser.add_argument("--max-canvas-mp", type=float, default=400.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("-v", "--verbose", action="store_true")
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
    logger = logging.getLogger("PanoramaStitcher")
    try:
        panorama = PanoramaStitcher(config).stitch(args.input_folder)
        if not cv2.imwrite(args.output, panorama):
            raise StitchingError(f"Failed to write output image to '{args.output}'.")
        logger.info(
            "Panorama saved to '%s' (%dx%d).",
            args.output,
            panorama.shape[1],
            panorama.shape[0],
        )
    except (StitchingError, FileNotFoundError, NotADirectoryError) as exc:
        logger.error("Stitching failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
