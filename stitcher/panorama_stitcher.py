"""Pipeline orchestration."""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

from stitcher.blending import blend_images, blend_multiband, crop_panorama, warp_images
from stitcher.features import FeatureMatcher
from stitcher.geometry import (
    canvas_geometry,
    estimate_homographies,
    global_transforms,
    spanning_tree,
)
from stitcher.image_io import load_images
from stitcher.models import PairStats, StitchConfig, StitchingError, StitchReport
from stitcher.projection import CylindricalProjection

logger = logging.getLogger("PanoramaStitcher")
_OUT_OF_MEMORY = "Ran out of memory; try --max-image-dimension 2000 (or a smaller value)."


def peak_memory_mb() -> Optional[float]:
    """Peak resident set size of this process in MB, or None where it cannot be read."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


class PanoramaStitcher:
    """Orchestrates the panorama pipeline without owning its individual algorithms."""

    def __init__(self, config: StitchConfig) -> None:
        self.config = config
        self.matcher = FeatureMatcher(config)

    def stitch(self, input_folder: str) -> np.ndarray:
        """Stitch the images in ``input_folder`` and return the cropped BGR panorama."""
        return self.stitch_with_report(input_folder)[0]

    def stitch_with_report(self, input_folder: str) -> Tuple[np.ndarray, StitchReport]:
        """Like :meth:`stitch`, but also return a machine-readable :class:`StitchReport`."""
        try:
            return self._run(input_folder)
        except MemoryError as exc:
            raise StitchingError(_OUT_OF_MEMORY) from exc
        except cv2.error as exc:
            if exc.code == cv2.Error.StsNoMem:
                raise StitchingError(_OUT_OF_MEMORY) from exc
            raise

    def _run(self, input_folder: str) -> Tuple[np.ndarray, StitchReport]:
        config = self.config
        cv2.setRNGSeed(config.seed)
        timings: Dict[str, float] = {}

        @contextmanager
        def stage(name: str) -> Iterator[None]:
            start = time.perf_counter()
            try:
                yield
            finally:
                timings[name] = round(time.perf_counter() - start, 4)

        total_start = time.perf_counter()
        with stage("load"):
            images = load_images(input_folder, config)
        report = StitchReport(
            image_names=[os.path.basename(data.path) for data in images],
            image_sizes=[(data.image.shape[1], data.image.shape[0]) for data in images],
            config=dataclasses.asdict(config),
        )

        if len(images) == 1:
            panorama = images[0].image
            report.canvas_size = report.output_size = report.image_sizes[0]
            report.canvas_megapixels = panorama.shape[0] * panorama.shape[1] / 1e6
        else:
            with stage("detect"):
                self.matcher.detect(images)
            report.detector = self.matcher.detector_name
            report.keypoints = [len(data.keypoints or []) for data in images]
            with stage("match"):
                matches = self.matcher.match(images)
            with stage("homography"):
                pairs, attempts = estimate_homographies(images, matches, config)
            report.pairs_tried = len(matches)
            report.pairs = [
                PairStats(p.i, p.j, p.matches, p.inliers, round(p.inlier_ratio, 4)) for p in pairs
            ]
            # Fail early, naming the unconnected photos and the closest rejected candidate
            # for each gap (see estimate_homographies for why a pair was rejected).
            spanning_tree(pairs, len(images), attempts=attempts)
            with stage("canvas"):
                projection = None
                if config.projection in ("auto", "cylindrical"):
                    projection = CylindricalProjection(images, pairs, config)
                    if (
                        config.projection == "auto"
                        and projection.fit_error is not None
                        and projection.fit_error > config.max_rotation_fit_px
                    ):
                        logger.warning(
                            "The photos do not fit a rotating camera (worst pair error %.1f px "
                            "> %.1f px), so the camera probably moved, as when photographing a "
                            "flat board or document. Using the planar model instead.",
                            projection.fit_error, config.max_rotation_fit_px,
                        )
                        projection = None
                if projection is not None:
                    reference, canvas_size = projection.reference, projection.canvas_size
                    report.focal_length = round(projection.focal, 2)
                else:
                    transforms, reference = global_transforms(pairs, len(images))
                    canvas_size, translation = canvas_geometry(images, transforms, config)
                report.projection = "cylindrical" if projection is not None else "planar"
                logger.info("Projection: %s", report.projection)
            report.reference_index = reference
            logger.info("Reference image: [%d]", reference)
            report.canvas_size = canvas_size
            report.canvas_megapixels = round(canvas_size[0] * canvas_size[1] / 1e6, 3)
            with stage("warp_blend"):
                if projection is not None:
                    warped = projection.warp_images(images, config)
                else:
                    warped = warp_images(images, transforms, translation, canvas_size, config)
                blend = blend_images
                if config.blend_mode == "multiband":
                    if report.canvas_megapixels <= config.multiband_max_megapixels:
                        blend = blend_multiband
                    else:
                        logger.warning(
                            "Canvas of %.1f MP is above the %.0f MP multiband limit; falling "
                            "back to feather blending (use --max-image-dimension to shrink it).",
                            report.canvas_megapixels, config.multiband_max_megapixels,
                        )
                blended, coverage = blend(warped, canvas_size, config)
            with stage("crop"):
                panorama = crop_panorama(blended, coverage)
            report.output_size = (panorama.shape[1], panorama.shape[0])

        timings["total"] = round(time.perf_counter() - total_start, 4)
        report.timings_s = timings
        report.peak_memory_mb = peak_memory_mb()
        if report.peak_memory_mb is not None:
            report.peak_memory_mb = round(report.peak_memory_mb, 1)
        return panorama, report
