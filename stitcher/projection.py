"""Cylindrical projection from a rotating-camera model.

A planar homography chain stretches without bound on wide sweeps. Here every image is instead
treated as a pure camera rotation: the focal length is estimated from the pairwise
homographies (H ~ K R K^-1), the rotations are chained to the reference image, and each image
is resampled onto a cylinder around the reference camera's vertical axis.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from stitcher.blending import WarpedImage
from stitcher.geometry import PairResult, global_transforms
from stitcher.models import ImageData, StitchConfig, StitchingError

logger = logging.getLogger("PanoramaStitcher")

_EDGE_SAMPLES = 64
_BAND_ROWS = 256
_ROI_PAD = 2
WARP_INTERPOLATION = cv2.INTER_CUBIC  # sharper than bilinear when resampling
_ROBUST_SCALES = (300.0, 60.0, 15.0, 5.0)  # px; truncation thresholds for bundle adjustment
_FINAL_SCALE = _ROBUST_SCALES[-1]


def _intrinsics(f: float, width: int, height: int) -> np.ndarray:
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _rotation_residual(
    homography: np.ndarray, f: float, src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> float:
    """How far K_dst^-1 H K_src is from a rotation (scale-free)."""
    m = np.linalg.inv(_intrinsics(f, *dst_size)) @ homography @ _intrinsics(f, *src_size)
    m = m / np.cbrt(np.linalg.det(m))
    return float(np.sum((m.T @ m - np.eye(3)) ** 2))


def estimate_focal(
    homographies: Sequence[np.ndarray], sizes: Sequence[Tuple[int, int]]
) -> float:
    """Focal length in pixels that makes the pair homographies best explained by rotations.

    ``homographies[i]`` maps image i+1 onto image i; ``sizes`` are (width, height) per image.
    """
    longest = float(max(max(s) for s in sizes))

    def cost(f: float) -> float:
        return sum(
            _rotation_residual(h, f, sizes[i + 1], sizes[i]) for i, h in enumerate(homographies)
        )

    grid = np.geomspace(0.25 * longest, 8.0 * longest, 400)
    best = int(np.argmin([cost(f) for f in grid]))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    fine = np.linspace(low, high, 200)
    focal = float(fine[int(np.argmin([cost(f) for f in fine]))])
    if best in (0, len(grid) - 1):
        logger.warning(
            "Focal length estimate %.0f px is at the edge of the search range; the images may "
            "not be a pure camera rotation. Consider --focal.", focal
        )
    return focal


def pairwise_rotations(
    homographies: Sequence[np.ndarray], f: float, sizes: Sequence[Tuple[int, int]]
) -> List[np.ndarray]:
    """Nearest rotation to K_i^-1 H_i K_{i+1} for each pair (maps camera i+1 onto camera i)."""
    rotations = []
    for i, h in enumerate(homographies):
        m = np.linalg.inv(_intrinsics(f, *sizes[i])) @ h @ _intrinsics(f, *sizes[i + 1])
        m = m / np.cbrt(np.linalg.det(m))
        u, _, vt = np.linalg.svd(m)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        rotations.append(rotation)
    return rotations


def refine_cameras(
    pairs: Sequence[PairResult],
    rotations: Sequence[np.ndarray],
    reference: int,
    focal: float,
    sizes: Sequence[Tuple[int, int]],
    iterations: int = 40,
    fix_focal: bool = False,
) -> Tuple[List[np.ndarray], float, float]:
    """Jointly refine all rotations and one shared focal length (bundle adjustment).

    Minimises the pixel reprojection error of every RANSAC inlier between adjacent images
    with Levenberg-Marquardt. ``rotations[i]`` maps camera i onto the reference camera.
    ``fix_focal`` keeps the focal length constant. Returns the refined rotations, the focal
    length and the worst per-pair median error in pixels (``inf`` if refinement did not help):
    a large value means the images are not a pure camera rotation. If refinement does not
    improve the error or drives the focal length out of a plausible range, the inputs are
    returned instead.
    """
    count = len(rotations)
    free = [i for i in range(count) if i != reference]
    longest = float(max(max(s) for s in sizes))

    def unpack(x: np.ndarray) -> Tuple[List[np.ndarray], float]:
        rots = [np.eye(3) for _ in range(count)]
        for slot, i in enumerate(free):
            rots[i] = cv2.Rodrigues(x[1 + 3 * slot:4 + 3 * slot])[0]
        return rots, float(np.exp(x[0]))

    def residuals(x: np.ndarray) -> np.ndarray:
        rots, f = unpack(x)
        out = []
        for pair in pairs:
            i = pair.index
            (wi, hi), (wj, hj) = sizes[i], sizes[i + 1]
            src, dst = pair.points_src, pair.points_dst
            rays = np.stack([(src[:, 0] - wj / 2) / f, (src[:, 1] - hj / 2) / f,
                             np.ones(len(src))], axis=1) @ (rots[i].T @ rots[i + 1]).T
            with np.errstate(divide="ignore", invalid="ignore"):
                px = f * rays[:, 0] / rays[:, 2] + wi / 2
                py = f * rays[:, 1] / rays[:, 2] + hi / 2
            err = np.stack([px - dst[:, 0], py - dst[:, 1]], axis=1)
            err[~np.isfinite(err) | (rays[:, 2:3] <= 0).repeat(2, axis=1)] = 1e3
            out.append(err.ravel())
        return np.concatenate(out)

    x = np.zeros(1 + 3 * len(free))
    x[0] = np.log(focal)
    for slot, i in enumerate(free):
        x[1 + 3 * slot:4 + 3 * slot] = cv2.Rodrigues(np.asarray(rotations[i]))[0].ravel()

    def robust_cost(r: np.ndarray, scale: float) -> float:
        return float(np.sum(np.minimum(r * r, scale * scale)))  # truncated quadratic

    r = residuals(x)
    start_cost = robust_cost(r, _FINAL_SCALE)
    start_med = float(np.median(np.abs(r)))
    # Graduated truncation: a tight threshold from the start would give zero weight to every
    # pair whose initial guess is already off by more than the threshold, so the optimiser
    # would silently ignore it. Start wide and tighten.
    for scale in _ROBUST_SCALES:
        cost, damping = robust_cost(r, scale), 1e-3
        for _ in range(iterations):
            weights = np.where(np.abs(r) < scale, 1.0, 0.0)
            jac = np.empty((len(r), len(x)))
            for k in range(len(x)):
                step = np.zeros_like(x)
                step[k] = 1e-6
                jac[:, k] = (residuals(x + step) - r) / 1e-6
            if fix_focal:
                jac[:, 0] = 0.0
            jw = jac * weights[:, None]
            normal = jw.T @ jac
            gradient = jw.T @ r
            improved = False
            for _ in range(8):
                try:
                    delta = np.linalg.solve(
                        normal + damping * np.diag(np.diag(normal) + 1e-9), -gradient
                    )
                except np.linalg.LinAlgError:
                    damping *= 10
                    continue
                r_new = residuals(x + delta)
                cost_new = robust_cost(r_new, scale)
                if cost_new < cost:
                    x, r, cost = x + delta, r_new, cost_new
                    damping = max(damping / 3, 1e-9)
                    improved = True
                    break
                damping *= 4
            if not improved or np.linalg.norm(delta) < 1e-9:
                break
    rots, f = unpack(x)
    end_med = float(np.median(np.abs(r)))
    if not (0.25 * longest <= f <= 8 * longest) or robust_cost(r, _FINAL_SCALE) >= start_cost:
        logger.warning("Bundle adjustment did not help (median error %.2f -> %.2f px); keeping "
                       "the initial cameras.", start_med, end_med)
        return list(rotations), focal, float("inf")
    logger.info("Bundle adjustment: median error %.2f -> %.2f px, focal %.1f -> %.1f px",
                start_med, end_med, focal, f)
    offset = 0
    worst = 0.0
    for pair in pairs:
        count = 2 * len(pair.points_src)
        pair_med = float(np.median(np.abs(r[offset:offset + count])))
        offset += count
        worst = max(worst, pair_med)
        if pair_med > 5.0:
            logger.warning(
                "Images %d and %d do not fit a rotating-camera model (median error %.1f px "
                "after adjustment); expect a visible misalignment there, usually caused by "
                "camera translation/parallax or a different tilt.",
                pair.index, pair.index + 1, pair_med,
            )
    return rots, f, worst


class CylindricalProjection:
    """Maps every image onto a cylinder of radius ``focal`` around the reference camera."""

    def __init__(
        self,
        images: List[ImageData],
        pairs: Sequence[PairResult],
        config: StitchConfig,
    ) -> None:
        homographies = [p.homography for p in pairs]
        self.sizes = [(d.image.shape[1], d.image.shape[0]) for d in images]
        if config.focal_length > 0:
            self.focal = float(config.focal_length)
        else:
            self.focal = estimate_focal(homographies, self.sizes)
        logger.info("Focal length: %.1f px (%s)", self.focal,
                    "given" if config.focal_length > 0 else "estimated")
        self.fit_error: Optional[float] = None  # worst per-pair error after adjustment (px)
        pair_rotations = pairwise_rotations(homographies, self.focal, self.sizes)
        self.rotations, self.reference = global_transforms(pair_rotations, len(images))
        if config.bundle_adjust:
            self.rotations, refined, self.fit_error = refine_cameras(
                pairs, self.rotations, self.reference, self.focal, self.sizes,
                fix_focal=config.focal_length > 0,
            )
            self.focal = refined
        self.offset = (0.0, 0.0)
        self.canvas_size = (0, 0)
        self._compute_canvas(config)

    def project(self, index: int, points: np.ndarray) -> np.ndarray:
        """Pixels of image ``index`` -> cylinder coordinates (u, v) in pixels."""
        w, h = self.sizes[index]
        f = self.focal
        rays = np.stack([(points[:, 0] - w / 2) / f, (points[:, 1] - h / 2) / f,
                         np.ones(len(points))], axis=1) @ self.rotations[index].T
        theta = np.arctan2(rays[:, 0], rays[:, 2])
        height = rays[:, 1] / np.hypot(rays[:, 0], rays[:, 2])
        return np.stack([f * theta, f * height], axis=1)

    def _border(self, index: int) -> np.ndarray:
        w, h = self.sizes[index]
        t = np.linspace(0, 1, _EDGE_SAMPLES)
        return np.concatenate([
            np.stack([t * w, np.zeros_like(t)], axis=1),
            np.stack([t * w, np.full_like(t, h)], axis=1),
            np.stack([np.zeros_like(t), t * h], axis=1),
            np.stack([np.full_like(t, w), t * h], axis=1),
        ])

    def _compute_canvas(self, config: StitchConfig) -> None:
        extents = []
        for index in range(len(self.sizes)):
            border = self.project(index, self._border(index))
            if np.ptp(border[:, 0]) > np.pi * self.focal:
                raise StitchingError(
                    f"Image {index} reaches the +/-180 degree seam of the cylinder around the "
                    "reference image. The sweep is about 360 degrees or more, and closed "
                    "360-degree panoramas are not supported; drop some images from one end."
                )
            extents.append(border)
        stacked = np.concatenate(extents)
        if not np.all(np.isfinite(stacked)):
            raise StitchingError("Cylindrical projection produced non-finite coordinates.")
        lo = np.floor(stacked.min(axis=0))
        hi = np.ceil(stacked.max(axis=0))
        megapixels = (hi[0] - lo[0]) * (hi[1] - lo[1]) / 1e6
        if megapixels > config.max_canvas_megapixels:
            raise StitchingError(
                f"Computed canvas is {megapixels:.1f} MP, exceeding the "
                f"{config.max_canvas_megapixels:.0f} MP safety cap (focal length "
                f"{self.focal:.0f} px); check --focal or raise --max-canvas-mp."
            )
        self.offset = (float(lo[0]), float(lo[1]))
        self.canvas_size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
        logger.info("Canvas size: %d x %d (%.1f MP)", *self.canvas_size, megapixels)

    def warp_images(self, images: List[ImageData], config: StitchConfig) -> Iterator[WarpedImage]:
        """Resample each image into its own bounding box of the canvas."""
        import os

        canvas_w, canvas_h = self.canvas_size
        f = self.focal
        for index, data in enumerate(images):
            border = self.project(index, self._border(index)) - np.array(self.offset)
            x0 = max(0, int(np.floor(border[:, 0].min())) - _ROI_PAD)
            y0 = max(0, int(np.floor(border[:, 1].min())) - _ROI_PAD)
            x1 = min(canvas_w, int(np.ceil(border[:, 0].max())) + _ROI_PAD)
            y1 = min(canvas_h, int(np.ceil(border[:, 1].max())) + _ROI_PAD)
            if x1 <= x0 or y1 <= y0:
                logger.warning("Image %d lies outside the canvas; skipped.", index)
                continue
            src_h, src_w = data.image.shape[:2]
            rot_t = self.rotations[index].T
            warped = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cols = (np.arange(x0, x1) + self.offset[0]) / f
            sin_t, cos_t = np.sin(cols), np.cos(cols)
            for ya in range(y0, y1, _BAND_ROWS):
                yb = min(ya + _BAND_ROWS, y1)
                height = ((np.arange(ya, yb) + self.offset[1]) / f)[:, None]
                rx = sin_t[None, :] * np.ones_like(height)
                rz = cos_t[None, :] * np.ones_like(height)
                ry = height * np.ones_like(rx)
                ix = rot_t[0, 0] * rx + rot_t[0, 1] * ry + rot_t[0, 2] * rz
                iy = rot_t[1, 0] * rx + rot_t[1, 1] * ry + rot_t[1, 2] * rz
                iz = rot_t[2, 0] * rx + rot_t[2, 1] * ry + rot_t[2, 2] * rz
                with np.errstate(divide="ignore", invalid="ignore"):
                    map_x = (f * ix / iz + src_w / 2.0).astype(np.float32)
                    map_y = (f * iy / iz + src_h / 2.0).astype(np.float32)
                valid = (iz > 1e-6) & (map_x >= 0) & (map_x <= src_w - 1) \
                    & (map_y >= 0) & (map_y <= src_h - 1)
                map_x[~valid] = -1
                map_y[~valid] = -1
                warped[ya - y0:yb - y0] = cv2.remap(
                    data.image, map_x, map_y, WARP_INTERPOLATION, borderMode=cv2.BORDER_CONSTANT
                )
                mask[ya - y0:yb - y0][valid] = 255
            if config.save_debug_images:
                os.makedirs(config.debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(config.debug_dir, f"warped_{index:02d}.jpg"), warped)
            yield WarpedImage(warped, mask, (x0, y0))
