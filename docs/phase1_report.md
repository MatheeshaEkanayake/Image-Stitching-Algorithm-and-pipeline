# Phase 1 report: cleanup and bug fixes

Scope: fix bugs, clean the code, add tests, docs, packaging and a run report. No new
algorithms. Reference numbers come from [baseline.md](baseline.md) (commit `ceebf74`).

## Bugs found and fixed

| # | Problem | Fix |
|---|---------|-----|
| 1 | CLI default `--blend-mode multiband` raised `NotImplementedError` | Default is `feather`; `multiband` removed from CLI, config (now validated) and docs |
| 2 | `warp_images` warped every image onto the full canvas | Each image is warped into its own padded bounding box and accumulated into the canvas slice; `blend_images` no longer keeps full-canvas warps. Test: ROI result vs the old full-canvas algorithm, max abs diff <= 1 |
| 3 | `crop_panorama` cropped on non-black pixels, so dark content was treated as empty | Crop uses the coverage mask (union of warped image masks) returned by `blend_images`; test with pure-black content |
| 4 | No homography validation | `estimate_homographies` checks min matches, min RANSAC inliers (`--min-inliers`), finite/non-singular H, w > 0 at the corners, positive area, convexity and area ratio (`--max-area-ratio`); errors name the pair ("images 3 and 4") |
| 5 | Degenerate canvas | `canvas_geometry` projects corners keeping the sign of w (`cv2.perspectiveTransform` hid it) and raises a clear `StitchingError` on w <= 0 or non-finite values before allocating anything; the size-cap message now explains the planar-model cause |
| 6 | Not reproducible | `cv2.setRNGSeed(config.seed)` at the start of a run, `--seed` flag; test that two runs give identical pixels |
| 7 | Dead/misleading code | Removed unused imports (`List`, `Optional`, `Tuple`, `cv2` where unused), the inline `import os` in the warp loop, and the stale instance state (`images`, `canvas_size`, `translation`) on `PanoramaStitcher`; logging only (the original had no `print` calls, and no unused config fields besides the removed multiband mode). OpenCV allocation failures (`cv2.error` StsNoMem) are now reported as `StitchingError` instead of a traceback |
| 8 | Packaging/docs | `requirement.txt` replaced by `requirements.txt` (numpy, opencv-contrib-python only; the pins installed and imported in a fresh venv, so they were kept); `pyproject.toml`; `.gitignore`; `setup.md` replaced by `README.md` |

`stitch2.py` and `stitch3.py` did not exist in the repository (they were only listed in the old
`.gitignore`), so there was nothing to delete.

Also added: package `stitcher/`, `python -m stitcher`, `StitchReport` and `--report`,
`tests/` (17 tests, about 1.5 s), `scripts/run_all_sets.sh`, `scripts/compare_to_baseline.py`,
ruff configuration (line length 100; `ruff check` is clean).

## Baseline vs after

All runs use `--blend-mode feather --max-image-dimension 1200`, one run each on the same
machine (12 cores, 7 GB RAM with about 1.4 GB free, so the timings are indicative only). Runtime
and memory from `/usr/bin/time -v` (wall clock, max RSS).

| Set | Result before | Result after | Canvas before | Canvas after | Runtime before / after | Peak memory before / after |
|-----|---------------|--------------|---------------|--------------|-----------------------:|---------------------------:|
| boat | ok | ok | 10283 x 3603 | 10283 x 3603 | 28.7 s / 8.6 s | 2157 MB / 1177 MB |
| cafe | ok | ok | 4522 x 1454 | 4522 x 1454 | 2.1 s / 2.0 s | 617 MB / 415 MB |
| class | ok | ok | 4216 x 2357 | 4216 x 2357 | 2.3 s / 1.8 s | 812 MB / 629 MB |
| images | fail (507 MP > 400 MP cap) | fail (clear error: image 0 projects outside the reference plane) | refused | refused | 2.5 s / 3.2 s | 332 MB / 347 MB |

RANSAC inliers per pair are identical to the baseline for all three successful sets.

Per-set mean absolute pixel difference of the new renders to `baseline/`
(`scripts/compare_to_baseline.py`):

| Set | Mean abs diff | Max | Size before / after |
|-----|--------------:|----:|---------------------|
| boat | 0.0019 | 50 | 10281x3601 / 10281x3601 |
| cafe | 0.0020 | 6 | 4521x1453 / 4521x1453 |
| class | 0.0009 | 4 | 4216x2356 / 4216x2356 |

Explanation: about 0.2% of pixels differ (hundreds of pixels differ by more than 3, scattered
along image borders). I did not trace these pixel by pixel; the likely cause is that OpenCV's
fixed-point interpolation in `warpPerspective` rounds slightly differently when the output
window is shifted to the image's bounding box. The synthetic ROI-vs-full-canvas test stays
within 1 grey level. The crop fix did not change the border on these sets (their image content
is not dark at the border), so the expected crop-related difference did not appear.
The outputs were inspected visually and look like the baseline renders.

## Remaining known issues

- Peak memory for large sets is still dominated by the float32 accumulator (16 bytes per canvas
  pixel). Full-resolution boat (10 MP photos) could not be run on this machine: a run capped at
  4.5 GB virtual memory ended with the clear out-of-memory error, and an uncapped run was not
  attempted because only about 1.4 GB was free. The CLI command from the brief
  (`python stitch.py Images/boat Results/boat-panorama.jpg`) was verified only with
  `--max-image-dimension 1200`; `Images/cafe` was also run with no options at all and succeeds.
- The planar model fails on wide rotating-camera sweeps (`Images/images`), and boat/cafe/class
  are visibly stretched at the ends. Needs a cylindrical/spherical model (later phase).
- Parallax and moving objects ghost under feather blending (see cafe).
- No exposure compensation; the crop is the bounding box, so black canvas wedges remain.
- `peak_memory_mb` in the report is the process-wide peak RSS, so it includes anything the host
  process did earlier; it is unavailable (`null`) on Windows.
- The pinned versions in `requirements.txt` were verified only on Linux with Python 3.14 (and
  the unpinned lower bounds in `pyproject.toml` were not tested against older numpy/OpenCV).
- `ruff` and `pytest` were only run on Linux.
- I deleted the stale `Stitch/` virtualenv directory (it contained no OpenCV or NumPy, was
  git-ignored and is referenced as a mess in the brief). The `.venv/` directory was left alone.

## Follow-up: cylindrical projection (added after the Phase 1 report was written)

The planar chain visibly distorted wide sets (Images/Rainier: a 517 px wide frame became a
wedge about 3000 px wide). On request, a rotating-camera model was added as an option and made
the default: `--projection cylindrical` (`stitcher/projection.py`). `--projection planar`
reproduces the original behaviour, so the baseline comparison above should be repeated with
`--projection planar`.

Canvas sizes at `--max-image-dimension 1200`, planar vs cylindrical: boat 10283x3603 ->
3340x971, cafe 4522x1454 -> 2503x798, class 4216x2357 -> 2183x958; Rainier 5592x2590 ->
1100x771. `Images/PXL` renders (1427x1440). `Images/images` is still refused, now because
its 16 frames span about 350 degrees and reach the cylinder seam. Not done: bundle adjustment,
360-degree wrap, horizon straightening, exposure compensation. The estimated focal lengths
were checked only on synthetic frames (within 8% of the true value) and by eye on the photos.

## Follow-up: alignment and seams (bundle adjustment, multiband blending)

- **Bundle adjustment** (`refine_cameras`): Levenberg-Marquardt over all rotations and one shared
  focal length on the RANSAC inliers. Median reprojection error at `--max-image-dimension 1200`:
  boat 8.6 -> 0.6 px, cafe 25.8 -> 0.9 px, class 10.5 -> 1.9 px, PXL 9.3 -> 1.1 px, Rainier
  1.4 -> 0.2 px. The initial focal estimate was off by up to about 9% (cafe 1174 -> 1073 px).
  It is skipped (with a warning) if it does not lower the robust error.
- **Multiband blending** is now the default (`--blend-mode multiband`; `feather` is still
  available): gain-block exposure compensation, graph-cut seams, Laplacian-pyramid blend (OpenCV
  `cv2.detail`). Ghosting of moving people in class and the cafe table is gone in the renders.
  Canvases above 20 MP fall back to feather with a warning.
- Remaining: boat's right-hand join still shows a skyline break (parallax between the shots),
  there is no 360-degree wrap, and the full-resolution boat canvas (34 MP in cylindrical) uses feather because
  of the memory limit.
