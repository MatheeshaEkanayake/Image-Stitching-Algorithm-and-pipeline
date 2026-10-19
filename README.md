# Image Stitch

Builds one panorama from a folder of ordered, overlapping photos (SIFT/ORB features, RANSAC
homographies, feather blending). Final-year project, BSc Software Engineering, KDU.

## Install

Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # installs the package and its dependencies
# or, for the exact versions this project was verified with:
pip install -r requirements.txt
```

Add `pip install -e ".[dev]"` for pytest and ruff.

## Usage

Images are read in natural filename order (`img2.jpg` before `img10.jpg`), so name them from
left to right.

```bash
python stitch.py Images/boat Results/boat-panorama.jpg
python -m stitcher Images/boat Results/boat-panorama.jpg      # same thing
python stitch.py Images/boat out/boat.jpg --max-image-dimension 1200 --report out/boat.json
scripts/run_all_sets.sh --max-image-dimension 1200            # every set under Images/ -> out/
```

Large photos need a lot of memory (the canvas is held as float32 while blending); use
`--max-image-dimension 1200` or `2000` on small machines.

From Python:

```python
from stitcher import PanoramaStitcher, StitchConfig

panorama = PanoramaStitcher(StitchConfig()).stitch("Images/boat")                # np.ndarray (BGR)
panorama, report = PanoramaStitcher(StitchConfig()).stitch_with_report("Images/boat")
print(report.to_json())
```

### Options

| Option | Default | Meaning |
|--------|---------|---------|
| `input_folder`, `output` | required | Folder of jpg/jpeg/png images; output image path |
| `--detector {sift,orb}` | `sift` | Feature detector (SIFT falls back to ORB if unavailable) |
| `--ratio` | `0.75` | Lowe ratio-test threshold |
| `--ransac-thresh` | `4.0` | RANSAC reprojection threshold (px) |
| `--min-matches` | `10` | Minimum good matches per adjacent pair |
| `--min-inliers` | `15` | Minimum RANSAC inliers per adjacent pair |
| `--max-area-ratio` | `10.0` | Reject a pair homography that scales an image's area by more than this factor (or its inverse) |
| `--projection {cylindrical,planar}` | `cylindrical` | `cylindrical` treats the photos as a rotating camera and removes the stretching of the planar chain; `planar` is the original homography chain (kept for baseline comparison and flat scenes) |
| `--focal` | `0` (estimate) | Focal length in pixels for the cylindrical projection |
| `--blend-mode {multiband,feather}` | `multiband` | `multiband`: exposure compensation, graph-cut seams and Laplacian-pyramid blending (removes ghosting). `feather`: distance-weighted averaging. Canvases above 20 MP fall back to feather automatically |
| `--feather-width` | `50` | Feather ramp width in px (`0` = unclipped distance weighting) |
| `--max-image-dimension` | `0` (off) | Downscale inputs so the longer side is at most this |
| `--max-canvas-mp` | `400` | Refuse canvases larger than this many megapixels |
| `--seed` | `0` | Seed for OpenCV's RNG, for reproducible RANSAC |
| `--report FILE` | off | Write a JSON run report |
| `--debug`, `--debug-dir` | off, `debug` | Save match visualisations and warped images |
| `-v` | off | Debug logging |

Exit code is 0 on success and 1 on a stitching error (message on stderr via `logging`).

### Run report

`--report` / `stitch_with_report` produce a `StitchReport`: image names and sizes (after any
`--max-image-dimension` downscale), detector used, keypoints per image, per pair
matches/inliers/inlier ratio, reference index, canvas size and megapixels, output size,
per-stage timings in seconds, peak memory in MB (process-wide peak RSS, `null` on Windows) and
the config used.

## Pipeline

1. **Load** images in natural filename order (optional downscale).
2. **Detect** SIFT (or ORB) keypoints and descriptors.
3. **Match** adjacent pairs only, with BFMatcher and the Lowe ratio test.
4. **Estimate** a RANSAC homography per pair and validate it (inlier count, determinant, area
   ratio, convexity, corners with w > 0). Failures name the offending pair.
5. **Chain** to the middle image (the reference). Cylindrical (default): estimate the focal
   length from the pair homographies, turn them into rotations, chain them, then refine all
   rotations and one shared focal length jointly (bundle adjustment on the RANSAC inliers).
   Planar: chain the homographies directly.
6. **Canvas**: project all image borders (onto the cylinder, or onto the plane checking w > 0),
   check the size cap, and compute the offset. This happens before any big allocation.
7. **Warp and blend**: each image is resampled into its own bounding box. Multiband (default):
   gain-block exposure compensation, graph-cut seams at low resolution, then pyramid blending.
   Feather: distance-transform weights. Both track a coverage mask.
8. **Crop** to the bounding box of the coverage mask.

## Layout

```
stitch.py                  thin CLI entry point
stitcher/
  __main__.py              python -m stitcher
  cli.py                   argument parsing, writing output and report
  models.py                StitchConfig, ImageData, StitchingError, StitchReport
  image_io.py              file discovery, natural sort, loading
  features.py              FeatureMatcher (detection + adjacent matching)
  geometry.py              homography estimation/validation, chaining, canvas geometry
  blending.py              ROI warping, feather blending, coverage crop
  panorama_stitcher.py     PanoramaStitcher orchestration and report
tests/                     pytest suite on synthetic data (about 2 s)
scripts/                   run_all_sets.sh, compare_to_baseline.py
docs/                      baseline.md (original code), phase1_report.md
Images/                    sample input sets (ordered left to right)
```

## Known limitations

- **Strong parallax can still show.** Multiband blending picks one image per pixel along a seam,
  so moving people and most parallax no longer ghost, but a seam that must cross a misaligned
  structure (for example the far skyline in `Images/boat`) can still show a break. With
  `--blend-mode feather` moving objects and parallax double up.
- **Multiband memory.** It holds all warped images and pyramids, so it is limited to canvases of
  20 MP or less (then it falls back to feather); use `--max-image-dimension` for large photos.
- **`--projection planar` fails on wide rotating-camera sweeps.** Chaining planar homographies
  stretches the outer images without bound, so cylindrical is the default.
- **Cylindrical assumes a camera rotating about its centre with one focal length.** The focal
  length is estimated without bundle adjustment, so small errors accumulate over long sweeps
  and the horizon can curve if the camera was tilted.
- **No closed 360-degree panoramas.** `Images/images` (16 frames, about 350 degrees) is refused
  with a clear `StitchingError` because it reaches the cylinder seam.
- Adjacent pairs only: the images must be in order and each must overlap its neighbour.
- No exposure compensation, so brightness steps can show at seams.
- The output keeps the black wedges of the canvas inside its bounding box (no
  largest-inscribed-rectangle crop).
- Run `pytest` and `ruff check .` for the test and lint status.
