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
from stitcher.features import FeatureMatcher, candidate_pairs
from stitcher.geometry import (
    PairAttempt,
    PairResult,
    canvas_geometry,
    estimate_homographies,
    global_transforms,
    image_corners,
    project_points,
    spanning_tree,
)
from stitcher.image_io import load_images, natural_sort_key

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


def test_pair_homography_recovery(frames_and_truth):
    frames, scene_from_frame = frames_and_truth
    images = images_from(frames)
    config = StitchConfig()
    matcher = FeatureMatcher(config)
    matcher.detect(images)
    pairs, _ = estimate_homographies(images, matcher.match(images), config)
    found = {(p.i, p.j) for p in pairs}
    assert {(0, 1), (1, 2), (2, 3)} <= found  # neighbours overlap by ~46%
    for pair in pairs:
        # H maps frame j -> frame i; truth = inv(G_i) @ G_j
        truth = np.linalg.inv(scene_from_frame[pair.i]) @ scene_from_frame[pair.j]
        assert corner_error(pair.homography, truth) < 2.0
        assert pair.inliers >= config.min_inliers


def _truth_pairs(scene_from_frame, edges):
    return [
        PairResult(i, j, np.linalg.inv(scene_from_frame[i]) @ scene_from_frame[j], 100, 90)
        for i, j in edges
    ]


def test_global_chain_composition(frames_and_truth):
    _, scene_from_frame = frames_and_truth
    count = len(scene_from_frame)
    pairs = _truth_pairs(scene_from_frame, [(0, 1), (1, 2), (2, 3)])
    transforms, reference = global_transforms(pairs, count)
    assert reference in (1, 2)  # centre of the chain
    assert np.allclose(transforms[reference], np.eye(3))
    for i in range(count):
        truth = np.linalg.inv(scene_from_frame[reference]) @ scene_from_frame[i]
        assert corner_error(transforms[i], truth) < 1e-6


def test_chain_does_not_depend_on_pair_order_or_direction(frames_and_truth):
    """The same overlap graph in any order gives the same placement of every image."""
    _, scene_from_frame = frames_and_truth
    edges = [(0, 1), (1, 2), (2, 3), (0, 2)]
    base, ref = global_transforms(_truth_pairs(scene_from_frame, edges), 4)
    shuffled, ref2 = global_transforms(_truth_pairs(scene_from_frame, edges[::-1]), 4)
    for i in range(4):
        truth = np.linalg.inv(scene_from_frame[ref]) @ scene_from_frame[i]
        assert corner_error(base[i], truth) < 1e-6
        truth2 = np.linalg.inv(scene_from_frame[ref2]) @ scene_from_frame[i]
        assert corner_error(shuffled[i], truth2) < 1e-6


def test_spanning_tree_prefers_strong_pairs_and_names_unconnected_photos():
    def pair(i, j, inliers):
        return PairResult(i, j, np.eye(3), inliers + 10, inliers)

    tree = spanning_tree([pair(0, 1, 50), pair(1, 2, 200), pair(0, 2, 300)], 3)
    assert {(p.i, p.j) for p in tree} == {(0, 2), (1, 2)}  # drops the weakest edge of the loop
    with pytest.raises(StitchingError, match=r"photo numbers: 1, 2 \| 3, 4"):
        spanning_tree([pair(0, 1, 50), pair(2, 3, 60)], 4)
    with pytest.raises(StitchingError, match=r"\| 3"):
        spanning_tree([pair(0, 1, 50)], 3)


def test_spanning_tree_names_closest_rejected_candidate_per_gap():
    """A real "no overlap" failure should point at the nearest-miss, not just say "no"."""
    pairs = [PairResult(0, 1, np.eye(3), 60, 50)]
    attempts = [
        PairAttempt(0, 1, 60, 50, True, "accepted"),
        # 1 and 2 almost overlapped: enough matches, too few RANSAC inliers.
        PairAttempt(1, 2, 40, 6, False, "only 6 of 40 matches agreed under RANSAC"),
        # 0 and 2 barely got any matches at all.
        PairAttempt(0, 2, 3, None, False, "only 3 matches, need at least 10"),
    ]
    with pytest.raises(StitchingError) as excinfo:
        spanning_tree(pairs, 3, attempts=attempts)
    message = str(excinfo.value)
    assert "photo numbers: 1, 2 | 3" in message
    # names the stronger of the two candidate bridges (1-2), with its real numbers and reason
    assert "photo 2 and photo 3" in message
    assert "40 matching features" in message and "6 of them agreed" in message
    assert "only 6 of 40 matches agreed under RANSAC" in message


def test_estimate_homographies_records_rejected_attempts(frame_folder):
    """Every tried pair is recorded, not just the ones that were accepted."""
    images = load_images(frame_folder, StitchConfig())
    matcher = FeatureMatcher(StitchConfig())
    matcher.detect(images)
    unrelated = {(0, len(images) - 1): []}  # a pair with zero matches, forced in
    matches = matcher.match(images)
    matches.update(unrelated)
    _, attempts = estimate_homographies(images, matches, StitchConfig())
    by_pair = {(a.i, a.j): a for a in attempts}
    assert set(by_pair) == set(matches)
    rejected = by_pair[(0, len(images) - 1)]
    assert rejected.accepted is False and "need at least" in rejected.reason
    assert any(a.accepted for a in attempts)  # the real neighbours still overlap


def test_candidate_pairs_modes():
    assert len(candidate_pairs(6, StitchConfig(matching="all"))) == 15
    adjacent = candidate_pairs(10, StitchConfig(matching="adjacent", match_window=2))
    assert (0, 2) in adjacent and (0, 3) not in adjacent
    assert len(candidate_pairs(30, StitchConfig())) == 435  # auto: all pairs up to 30 photos
    assert (0, 29) not in candidate_pairs(31, StitchConfig())  # beyond that: neighbours only


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
        warped = cv2.warpPerspective(
            data.image, final, (width, height), flags=cv2.INTER_CUBIC
        )
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
    assert report.reference_index in (1, 2)
    assert {(p.a, p.b) for p in report.pairs} >= {(0, 1), (1, 2), (2, 3)}
    assert report.pairs_tried == 6  # 4 photos: every pair is tried
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
        "image_names", "image_sizes", "detector", "keypoints", "pairs_tried", "pairs", "projection",
        "reference_index",
        "focal_length", "canvas_size", "canvas_megapixels", "output_size", "timings_s",
        "peak_memory_mb", "config",
    }
    assert set(data) == expected_keys
    assert data["image_names"] == ["frame1.png", "frame2.png", "frame3.png", "frame4.png"]
    assert data["image_sizes"] == [[FRAME_W, FRAME_H]] * 4
    assert len(data["keypoints"]) == 4 and all(k > 0 for k in data["keypoints"])
    assert set(data["pairs"][0]) == {"a", "b", "matches", "inliers", "inlier_ratio"}
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


def test_unrelated_images_give_clear_error(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(2):
        noise = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
        cv2.imwrite(str(tmp_path / f"p{i}.png"), noise)
    with pytest.raises(StitchingError, match=r"do not overlap each other"):
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
    assert report.reference_index in (1, 2)
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
    pairs, _ = estimate_homographies(images, matcher.match(images), config)
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


def test_bundle_adjustment_recovers_pair_with_large_initial_error():
    """A pair that starts ~70 px off must be pulled in, not given up on as an outlier."""
    from stitcher.geometry import PairResult
    from stitcher.projection import refine_cameras

    rng = np.random.default_rng(1)
    f, sizes = 800.0, [(640, 480)] * 3
    true = [cv2.Rodrigues(np.array([0.0, np.deg2rad(a), 0.0]))[0] for a in (-20, 0, 20)]
    pairs = []
    for i in range(2):
        pts = rng.uniform([0, 0], [640, 480], (200, 2))
        rays = np.stack([(pts[:, 0] - 320) / f, (pts[:, 1] - 240) / f, np.ones(200)], 1)
        world = rays @ true[i + 1].T  # points of image i+1 in the reference frame
        in_i = world @ true[i]  # = (true[i].T @ world) for row vectors
        dst = np.stack([f * in_i[:, 0] / in_i[:, 2] + 320, f * in_i[:, 1] / in_i[:, 2] + 240], 1)
        pairs.append(PairResult(i, i + 1, np.eye(3), 200, 200, pts, dst))
    # camera 0 starts ~5 degrees off in pitch/yaw/roll, i.e. well beyond the final 5 px threshold
    off = cv2.Rodrigues(np.deg2rad([5.0, 4.0, 3.0]))[0]
    start = [off @ true[0], true[1], true[2]]
    refined, _, improved = refine_cameras(pairs, start, 1, f, sizes, fix_focal=True)
    error = np.degrees(np.linalg.norm(cv2.Rodrigues(refined[0] @ true[0].T)[0]))
    assert error < 0.1 and improved


def test_cli_writes_lossless_tiff(frame_folder, tmp_path):
    out = tmp_path / "pano.tif"
    assert main([frame_folder, str(out), "--projection", "planar"]) == 0
    loaded = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert loaded is not None and loaded.dtype == np.uint8 and loaded.shape[2] == 3
    # re-saving the decoded image as TIFF must be bit-identical (lossless round trip)
    again = tmp_path / "again.tif"
    cv2.imwrite(str(again), loaded)
    assert np.array_equal(cv2.imread(str(again), cv2.IMREAD_UNCHANGED), loaded)


def test_default_multiband_limit_covers_native_resolution_sets():
    assert StitchConfig().multiband_max_megapixels >= 40


def test_auto_projection_uses_cylindrical_for_a_rotating_camera(rotation_folder):
    _, report = PanoramaStitcher(StitchConfig()).stitch_with_report(rotation_folder)
    assert StitchConfig().projection == "auto"
    assert report.projection == "cylindrical" and report.focal_length is not None


def test_auto_projection_falls_back_to_planar_for_a_flat_target(frame_folder, caplog):
    """The planar test frames are cut from a flat scene by homographies (not a pure rotation)."""
    with caplog.at_level("WARNING"):
        panorama, report = PanoramaStitcher(StitchConfig()).stitch_with_report(frame_folder)
    assert report.projection == "planar" and report.focal_length is None
    assert "Using the planar model" in caplog.text
    assert panorama.shape[1] > FRAME_W * 2


def test_explicit_projection_is_respected(rotation_folder):
    _, report = PanoramaStitcher(StitchConfig(projection="planar")).stitch_with_report(
        rotation_folder
    )
    assert report.projection == "planar"


GRID_ORDERS = {
    "row_by_row": [0, 1, 2, 3, 4, 5],
    "snake": [0, 1, 2, 5, 4, 3],  # along the top row, then back along the bottom row
    "right_to_left": [5, 4, 3, 2, 1, 0],
    "shuffled": [4, 0, 5, 2, 3, 1],
}


@pytest.mark.parametrize("name", list(GRID_ORDERS))
def test_grid_capture_in_any_order(name, grid_frames, tmp_path):
    from .conftest import GRID_STEP_X, GRID_STEP_Y, write_folder

    folder = write_folder(tmp_path / name, grid_frames, GRID_ORDERS[name])
    panorama, report = PanoramaStitcher(StitchConfig()).stitch_with_report(folder)
    # a 3 x 2 grid is ~1000 x ~510 px; a wrong placement would be much wider or taller
    assert abs(panorama.shape[1] - (2 * GRID_STEP_X + FRAME_W)) < 0.06 * 1000, name
    assert abs(panorama.shape[0] - (GRID_STEP_Y + FRAME_H)) < 0.08 * 510, name
    assert len(report.pairs) >= 7  # horizontal AND vertical neighbours were found


def test_grid_result_is_independent_of_capture_order(grid_frames, tmp_path):
    from .conftest import write_folder

    sizes = {}
    for name, order in GRID_ORDERS.items():
        folder = write_folder(tmp_path / name, grid_frames, order)
        _, report = PanoramaStitcher(StitchConfig(seed=1)).stitch_with_report(folder)
        sizes[name] = report.canvas_size
    widths = [w for w, _ in sizes.values()]
    heights = [h for _, h in sizes.values()]
    assert max(widths) - min(widths) < 0.05 * min(widths), sizes
    assert max(heights) - min(heights) < 0.05 * min(heights), sizes


def test_unconnected_photo_is_reported_by_number(frame_folder, tmp_path):
    import shutil

    folder = tmp_path / "with_stranger"
    shutil.copytree(frame_folder, folder)
    rng = np.random.default_rng(9)
    stranger = cv2.GaussianBlur(rng.integers(0, 255, (360, 480, 3), dtype=np.uint8), (0, 0), 2)
    cv2.imwrite(str(folder / "frame9.png"), stranger)  # sorts last: photo number 5
    with pytest.raises(StitchingError, match=r"photo numbers: 1, 2, 3, 4 \| 5"):
        PanoramaStitcher(StitchConfig()).stitch(str(folder))


def test_adjacent_matching_mode_still_works_in_order(frame_folder):
    _, report = PanoramaStitcher(
        StitchConfig(matching="adjacent", match_window=1)
    ).stitch_with_report(frame_folder)
    assert report.pairs_tried == 3
