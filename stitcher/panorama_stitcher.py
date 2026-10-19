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

from stitcher.blending import blend_images, crop_panorama, warp_images
from stitcher.features import FeatureMatcher
from stitcher.geometry import canvas_geometry, estimate_homographies, global_transforms
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
                matches = self.matcher.match_adjacent(images)
            with stage("homography"):
                pairs = estimate_homographies(images, matches, config)
            report.pairs = [
                PairStats(p.index, p.matches, p.inliers, round(p.inlier_ratio, 4)) for p in pairs
            ]
            homographies = [p.homography for p in pairs]
            with stage("canvas"):
                if config.projection == "cylindrical":
                    projection = CylindricalProjection(images, homographies, config)
                    reference, canvas_size = projection.reference, projection.canvas_size
                    report.focal_length = round(projection.focal, 2)
                else:
                    transforms, reference = global_transforms(homographies, len(images))
                    canvas_size, translation = canvas_geometry(images, transforms, config)
            report.reference_index = reference
            logger.info("Reference image: [%d]", reference)
            report.canvas_size = canvas_size
            report.canvas_megapixels = round(canvas_size[0] * canvas_size[1] / 1e6, 3)
            with stage("warp_blend"):
                if config.projection == "cylindrical":
                    warped = projection.warp_images(images, config)
                else:
                    warped = warp_images(images, transforms, translation, canvas_size, config)
                blended, coverage = blend_images(warped, canvas_size, config)
            with stage("crop"):
                panorama = crop_panorama(blended, coverage)
            report.output_size = (panorama.shape[1], panorama.shape[0])

        timings["total"] = round(time.perf_counter() - total_start, 4)
        report.timings_s = timings
        report.peak_memory_mb = peak_memory_mb()
        if report.peak_memory_mb is not None:
            report.peak_memory_mb = round(report.peak_memory_mb, 1)
        return panorama, report
