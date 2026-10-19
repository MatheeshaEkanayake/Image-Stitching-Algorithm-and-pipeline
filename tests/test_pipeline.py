from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List

import cv2
import numpy as np
import pytest

from stitcher import ImageData, PanoramaStitcher, StitchConfig, StitchingError
from stitcher.blending import blend_images, crop_panorama, warp_images
from stitcher.cli import main
from stitcher.features import FeatureMatcher
from stitcher.geometry import (
    canvas_geometry,
    estimate_homographies,
    global_transforms,
    image_corners,
    project_points,
)
from stitcher.image_io import natural_sort_key

from .conftest import FRAME_H, FRAME_W

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corner_error(estimated: np.ndarray, truth: np.ndarray, w: int = FRAME_W, h: int = FRAME_H):
    """Max distance (px) between where two homographies send the frame corners."""
    a, _ = project_points(image_corners(w, h), estimated)
    b, _ = project_points(image_corners(w, h), truth)
    return float(np.linalg.norm(a - b, axis=1).max())


def images_from(frames: List[np.ndarray]) -> List[ImageData]:
    return [ImageData(path=f"frame{i}.png", image=f) for i, f in enumerate(frames)]


def test_natural_sort_order():
    names = ["img10.jpg", "img2.jpg", "img1.jpg", "Img3.jpg"]
    assert sorted(names, key=natural_sort_key) == ["img1.jpg", "img2.jpg", "Img3.jpg", "img10.jpg"]


def test_adjacent_pair_homography_recovery(frames_and_truth):
    frames, scene_from_frame = frames_and_truth
    images = images_from(frames)
    config = StitchConfig()
    matcher = FeatureMatcher(config)
    matcher.detect(images)
    pairs = estimate_homographies(images, matcher.match_adjacent(images), config)
    assert len(pairs) == len(frames) - 1
    for pair in pairs:
        # H maps frame i+1 -> frame i; truth = inv(G_i) @ G_{i+1}
        truth = np.linalg.inv(scene_from_frame[pair.index]) @ scene_from_frame[pair.index + 1]
        assert corner_error(pair.homography, truth) < 2.0
        assert pair.inliers >= config.min_inliers


def test_global_chain_composition(frames_and_truth):
    _, scene_from_frame = frames_and_truth
    count = len(scene_from_frame)
    pairwise = [
        np.linalg.inv(scene_from_frame[i]) @ scene_from_frame[i + 1] for i in range(count - 1)
    ]
    transforms, reference = global_transforms(pairwise, count)
    assert reference == count // 2
    assert np.allclose(transforms[reference], np.eye(3))
    for i in range(count):
        truth = np.linalg.inv(scene_from_frame[reference]) @ scene_from_frame[i]
        assert corner_error(transforms[i], truth) < 1e-6


def test_canvas_geometry_bounds_all_corners():
    images = images_from([np.zeros((100, 200, 3), np.uint8)] * 2)
    transforms = [np.eye(3), np.array([[1, 0, 150], [0, 1, -30], [0, 0, 1.0]])]
    size, translation = canvas_geometry(images, transforms, StitchConfig())
    assert size == (350, 130)
    # The top-left-most corner (image 1's top edge sits at y=-30) lands on y=0
    assert translation[0, 2] == 0 and translation[1, 2] == 30


def test_degenerate_canvas_raises_before_allocating():
    images = images_from([np.zeros((100, 200, 3), np.uint8)] * 2)
    # Strong perspective term: the corner at x=200 maps to w = 1 - 0.01*200 < 0
    behind = np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1.0]])
    with pytest.raises(StitchingError, match="Image 1 projects outside"):
        canvas_geometry(images, [np.eye(3), behind], StitchConfig())
    nan_transform = np.full((3, 3), np.nan)
    with pytest.raises(StitchingError, match="Image 1"):
        canvas_geometry(images, [np.eye(3), nan_transform], StitchConfig())


def test_canvas_size_cap_raises():
    images = images_from([np.zeros((100, 200, 3), np.uint8)] * 2)
    far = np.array([[1, 0, 100000], [0, 1, 100000], [0, 0, 1.0]])
    with pytest.raises(StitchingError, match="safety cap"):
        canvas_geometry(images, [np.eye(3), far], StitchConfig(max_canvas_megapixels=50))


def full_canvas_reference(images, transforms, translation, canvas_size, config):
    """The original (pre-Phase-1) algorithm: warp everything onto the full canvas."""
    width, height = canvas_size
    acc = np.zeros((height, width, 3), np.float32)
    weights = np.zeros((height, width), np.float32)
    for index, data in enumerate(images):
        final = translation @ transforms[index]
        warped = cv2.warpPerspective(data.image, final, (width, height))
        mask = cv2.warpPerspective(
            np.full(data.image.shape[:2], 255, np.uint8), final, (width, height),
            flags=cv2.INTER_NEAREST,
        )
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        weight = np.clip(weight, 0, config.feather_width)
        acc += warped.astype(np.float32) * weight[..., None]
        weights += weight
    out = acc / np.where(weights == 0, 1.0, weights)[..., None]
    out[weights == 0] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def test_roi_warp_matches_full_canvas(frames_and_truth):
    frames, scene_from_frame = frames_and_truth
    small = [cv2.resize(f, None, fx=0.4, fy=0.4) for f in frames]
    images = images_from(small)
    scale = np.diag([0.4, 0.4, 1.0])
    reference_index = len(frames) // 2
    to_ref = np.linalg.inv(scene_from_frame[reference_index])
    transforms = [scale @ to_ref @ g @ np.linalg.inv(scale) for g in scene_from_frame]
    config = StitchConfig()
    size, translation = canvas_geometry(images, transforms, config)
    expected = full_canvas_reference(images, transforms, translation, size, config)
    blended, coverage = blend_images(
        warp_images(images, transforms, translation, size, config), size, config
    )
    diff = np.abs(blended.astype(int) - expected.astype(int))
    assert diff.max() <= 1
    assert coverage.shape == blended.shape[:2]


def test_roi_warp_only_allocates_bounding_boxes(frames_and_truth):
    frames, _ = frames_and_truth
    images = images_from(frames[:2])
    transforms = [np.eye(3), np.array([[1, 0, 2000], [0, 1, 0], [0, 0, 1.0]])]
    size, translation = canvas_geometry(images, transforms, StitchConfig())
    for warped in warp_images(images, transforms, translation, size, StitchConfig()):
        assert warped.image.shape[1] < size[0] / 2  # nowhere near full canvas width


def test_coverage_mask_crop_keeps_dark_content():
    panorama = np.zeros((100, 200, 3), np.uint8)
    coverage = np.zeros((100, 200), np.uint8)
    coverage[20:80, 30:170] = 255  # image content, but it is pure black (0,0,0) at the edges
    panorama[40:60, 60:100] = 200
    cropped = crop_panorama(panorama, coverage)
    assert cropped.shape[:2] == (60, 140)
    # a pixel-value based crop would have shrunk to the bright 20x40 patch
    assert cropped[0, 0].tolist() == [0, 0, 0]


def test_end_to_end_with_report(frame_folder, frames_and_truth):
    _, scene_from_frame = frames_and_truth
    config = StitchConfig(projection="planar")
    panorama, report = PanoramaStitcher(config).stitch_with_report(frame_folder)
    assert panorama.shape[1] > FRAME_W * 2
    assert report.reference_index == 2
    assert len(report.pairs) == 3
    assert [p.index for p in report.pairs] == [0, 1, 2]
    expected_width = scene_from_frame[-1][0, 2] - scene_from_frame[0][0, 2] + FRAME_W
    assert abs(report.canvas_size[0] - expected_width) < 0.15 * expected_width
    # stitch() returns the same pixels
    again = PanoramaStitcher(config).stitch(frame_folder)
    assert np.array_equal(again, panorama)


def test_report_schema(frame_folder):
    _, report = PanoramaStitcher(StitchConfig(seed=3, projection="planar")).stitch_with_report(
        frame_folder
    )
    data = json.loads(report.to_json())
    expected_keys = {
        "image_names", "image_sizes", "detector", "keypoints", "pairs", "reference_index",
        "focal_length", "canvas_size", "canvas_megapixels", "output_size", "timings_s",
        "peak_memory_mb", "config",
    }
    assert set(data) == expected_keys
    assert data["image_names"] == ["frame1.png", "frame2.png", "frame3.png", "frame4.png"]
    assert data["image_sizes"] == [[FRAME_W, FRAME_H]] * 4
    assert len(data["keypoints"]) == 4 and all(k > 0 for k in data["keypoints"])
    assert set(data["pairs"][0]) == {"index", "matches", "inliers", "inlier_ratio"}
    assert 0 < data["pairs"][0]["inlier_ratio"] <= 1
    stages = {"load", "detect", "match", "homography", "canvas", "warp_blend", "crop", "total"}
    assert stages <= set(data["timings_s"])
    assert data["config"]["seed"] == 3 and data["config"]["blend_mode"] == "multiband"
    assert data["canvas_megapixels"] > 0


def test_cli_smoke(frame_folder, tmp_path):
    out = tmp_path / "sub" / "pano.jpg"
    report = tmp_path / "report.json"
    code = main([frame_folder, str(out), "--report", str(report), "--seed", "1"])
    assert code == 0
    assert cv2.imread(str(out)) is not None
    assert json.loads(report.read_text())["config"]["seed"] == 1


def test_cli_entry_points_and_failure_exit_code(frame_folder, tmp_path):
    out = tmp_path / "pano.jpg"
    for command in (
        [sys.executable, "stitch.py", frame_folder, str(out)],
        [sys.executable, "-m", "stitcher", frame_folder, str(out)],
    ):
        out.unlink(missing_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True)
        assert out.exists()
    missing = subprocess.run(
        [sys.executable, "stitch.py", str(tmp_path / "nope"), str(out)],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert missing.returncode == 1


def test_unknown_blend_mode_rejected(frame_folder, tmp_path):
    with pytest.raises(SystemExit):
        main([frame_folder, str(tmp_path / "x.jpg"), "--blend-mode", "poisson"])
    with pytest.raises(ValueError):
        StitchConfig(blend_mode="poisson")


def test_unrelated_images_give_clear_pair_error(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(2):
        noise = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
        cv2.imwrite(str(tmp_path / f"p{i}.png"), noise)
    with pytest.raises(StitchingError, match=r"images 0 and 1"):
        PanoramaStitcher(StitchConfig()).stitch(str(tmp_path))


def test_homography_validation_names_the_pair():
    from stitcher.geometry import validate_homography

    config = StitchConfig()
    assert validate_homography(np.eye(3), 100, 100, config) == ""
    assert "implausible scale" in validate_homography(np.diag([5.0, 5.0, 1.0]), 100, 100, config)
    assert "near-singular" in validate_homography(np.diag([1.0, 1e-9, 1.0]), 100, 100, config)
    assert "w <= 0" in validate_homography(
        np.array([[1, 0, 0], [0, 1, 0], [-0.02, 0, 1.0]]), 100, 100, config
    )
    assert "mirrors" in validate_homography(np.diag([-1.0, 1.0, 1.0]), 100, 100, config)


def test_seed_makes_runs_identical(frame_folder):
    a, _ = PanoramaStitcher(StitchConfig(seed=5)).stitch_with_report(frame_folder)
    b, _ = PanoramaStitcher(StitchConfig(seed=5)).stitch_with_report(frame_folder)
    assert np.array_equal(a, b)




def test_cylindrical_focal_and_canvas_on_rotation_frames(rotation_folder):
    config = StitchConfig()  # cylindrical is the default
    panorama, report = PanoramaStitcher(config).stitch_with_report(rotation_folder)
    from .conftest import ROT_FOCAL, ROT_STEP_DEG

    assert abs(report.focal_length - ROT_FOCAL) / ROT_FOCAL < 0.08
    half_fov = np.arctan(FRAME_W / 2 / ROT_FOCAL)
    expected_width = ROT_FOCAL * (np.deg2rad(3 * ROT_STEP_DEG) + 2 * half_fov)
    assert abs(report.canvas_size[0] - expected_width) / expected_width < 0.08
    assert report.reference_index == 2
    assert panorama.shape[0] > 0.8 * FRAME_H


def test_cylindrical_beats_planar_on_wide_rotation(rotation_folder):
    cyl = PanoramaStitcher(StitchConfig()).stitch_with_report(rotation_folder)[1]
    planar = PanoramaStitcher(StitchConfig(projection="planar")).stitch_with_report(
        rotation_folder
    )[1]
    assert cyl.canvas_megapixels < planar.canvas_megapixels


def test_given_focal_is_used(rotation_folder):
    _, report = PanoramaStitcher(StitchConfig(focal_length=620.0)).stitch_with_report(
        rotation_folder
    )
    assert report.focal_length == 620.0


def test_unknown_projection_rejected():
    with pytest.raises(ValueError):
        StitchConfig(projection="spherical")


def test_bundle_adjustment_reduces_reprojection_error(rotation_folder):
    from stitcher.image_io import load_images
    from stitcher.projection import CylindricalProjection

    images = load_images(rotation_folder, StitchConfig())
    config = StitchConfig()
    matcher = FeatureMatcher(config)
    matcher.detect(images)
    pairs = estimate_homographies(images, matcher.match_adjacent(images), config)
    refined = CylindricalProjection(images, pairs, config)
    plain = CylindricalProjection(images, pairs, StitchConfig(bundle_adjust=False))
    from .conftest import ROT_FOCAL

    assert abs(refined.focal - ROT_FOCAL) <= abs(plain.focal - ROT_FOCAL) + 1.0
    assert abs(refined.focal - ROT_FOCAL) / ROT_FOCAL < 0.03


def test_multiband_blend_matches_coverage_and_has_no_double_image(rotation_folder):
    multi = PanoramaStitcher(StitchConfig(blend_mode="multiband"))
    feather = PanoramaStitcher(StitchConfig(blend_mode="feather"))
    pano_m, rep_m = multi.stitch_with_report(rotation_folder)
    pano_f, rep_f = feather.stitch_with_report(rotation_folder)
    assert pano_m.shape == pano_f.shape
    assert rep_m.canvas_size == rep_f.canvas_size
    assert pano_m.dtype == np.uint8 and pano_m.std() > 10


def test_multiband_falls_back_to_feather_on_huge_canvas(rotation_folder, caplog):
    config = StitchConfig(multiband_max_megapixels=0.01)
    with caplog.at_level("WARNING"):
        pano, _ = PanoramaStitcher(config).stitch_with_report(rotation_folder)
    assert "falling back to feather" in caplog.text
    assert pano.size > 0


def test_multiband_removes_moving_object_ghost():
    """A bright blob present in only one of two overlapping images must not be half-blended."""
    from stitcher.blending import WarpedImage, blend_multiband

    base = np.full((120, 200, 3), 90, np.uint8)
    left, right = base.copy(), base.copy()
    cv2.circle(right, (60, 60), 12, (255, 255, 255), -1)  # in the overlap, right image only
    mask = np.full((120, 200), 255, np.uint8)
    warped = [WarpedImage(left, mask.copy(), (0, 0)), WarpedImage(right, mask.copy(), (100, 0))]
    out, coverage = blend_multiband(iter(warped), (300, 120), StitchConfig())
    assert coverage.all()
    # Feather would leave a half-intensity ghost (~170) in the overlap; a seam gives ~90 or 255.
    ghost = out[60, 160].mean()
    assert ghost < 120 or ghost > 230
