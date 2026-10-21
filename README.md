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
python stitch.py Images/boat Results/boat-panorama.tif        # lossless, recommended
python stitch.py Images/boat Results/boat-panorama.jpg        # smaller but lossy
python -m stitcher Images/boat Results/boat-panorama.jpg      # same thing
python stitch.py Images/boat out/boat.jpg --max-image-dimension 1200 --report out/boat.json
scripts/run_all_sets.sh --max-image-dimension 1200            # every set under Images/ -> out/
```

## Keeping maximum detail

- Leave `--max-image-dimension` at its default (`0`, no downscaling).
- Write `.tif` (or `.png`): the output format is chosen by the file extension. JPEG is lossy
  and is limited to 65535 px per side.
- Resampling is bicubic, and multiband blending is used up to 40 MP canvases
  (`--multiband-max-mp`); beyond that it falls back to feather blending.

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
| `--projection {auto,cylindrical,planar}` | `auto` | `auto` tries the rotating-camera model and falls back to planar when the photos do not fit it (worst pair error above 5 px, e.g. a moved camera). `cylindrical`: wide panoramas from a camera turning on the spot. `planar`: flat targets such as whiteboards, posters and documents |
| `--focal` | `0` (estimate) | Focal length in pixels for the cylindrical projection |
| `--blend-mode {multiband,feather}` | `multiband` | `multiband`: exposure compensation, graph-cut seams and Laplacian-pyramid blending (removes ghosting). `feather`: distance-weighted averaging. Canvases above 40 MP fall back to feather automatically |
| `--feather-width` | `50` | Feather ramp width in px (`0` = unclipped distance weighting) |
| `--max-image-dimension` | `0` (off) | Downscale inputs so the longer side is at most this |
| `--multiband-max-mp` | `40` | Canvases above this many megapixels fall back to feather blending |
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

## Zoomable tiles, server and mobile app

`--tiles DIR` writes a Deep Zoom style tile pyramid (JPEG tiles, 512 px by default, plus
`manifest.json`) next to the normal output, so a viewer only downloads the tiles on screen.
The lossless TIFF stays the master.

The server (`pip install -e ".[server]"`, then `python -m stitcher.server --host 0.0.0.0`)
adds accounts, per-photo uploads, a private library, sharing and a community. The React Native
app in [`mobile/`](mobile/README.md) is its client.

| Route | Purpose |
|-------|---------|
| `POST /auth/register`, `/auth/login`, `/auth/logout`, `GET /auth/me` | Accounts (scrypt-hashed passwords, bearer session tokens, login throttling) |
| `POST /captures` | Start a capture |
| `POST /captures/{id}/photos` | Upload one photo (`index` = left-to-right position, `file`); re-sending an index replaces it |
| `POST /captures/{id}/finish` | All photos uploaded: queue the stitch |
| `GET /captures`, `GET /captures/{id}` | Your library; one capture with `state` (`capturing`, `queued`, `running`, `done`, `failed`) |
| `PATCH /captures/{id}` | Rename or share (`{"public": true}`) |
| `GET /captures/{id}/view` | Zoomable viewer page (token passed as `#token=` fragment) |
| `GET /captures/{id}/manifest.json`, `/tiles/{level}/{col}_{row}.jpg`, `/thumb.jpg` | Tiles and thumbnail |
| `GET /captures/{id}/crop?x=&y=&w=&h=&scale=` | A region at native resolution as PNG (up to 16 MP) |
| `GET /captures/{id}/panorama.tif`, `/report.json` | Lossless master and run report |
| `GET/POST /posts`, `GET/DELETE /posts/{id}`, `PUT/DELETE /posts/{id}/like` | Community feed and posts |
| `GET/POST /posts/{id}/comments`, `DELETE /comments/{id}` | Comments |
| `GET /users/{username}` | A user's shared panoramas and posts |

Captures are private to their owner until shared (attaching one to a post shares it). Stitches
run one at a time. Data lives under `STITCH_DATA_DIR` (default `server_data/`, SQLite database
`app.db`). Set `STITCH_ALLOW_REGISTRATION=0` to close sign-ups. The server speaks plain HTTP:
put it behind HTTPS before exposing it beyond a trusted network. There is no automatic clean-up
of old captures yet. `openseadragon.min.js` is vendored from OpenSeadragon 4.1.1 (BSD licence).

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
   per-image gain exposure compensation, graph-cut seams at low resolution, then pyramid blending.
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
  projection.py            cylindrical model, focal estimation, bundle adjustment, warping
  tiles.py                 tile pyramid generator
  server/                  FastAPI app (accounts, captures, community), job runner, viewer
mobile/                    React Native (Expo) app
tests/                     pytest suite on synthetic data (about 2 s)
scripts/                   run_all_sets.sh, compare_to_baseline.py
docs/                      baseline.md (original code), phase1_report.md
Images/                    sample input sets (ordered left to right)
```

## Known limitations

- **Strong parallax can still show.** Multiband blending picks one image per pixel along a seam,
  so moving people and most parallax no longer ghost, but a seam that must cross a misaligned
  structure can still show a break (the run warns when a pair does not fit the rotating-camera
  model). With `--blend-mode feather` moving objects and parallax double up.
- **Multiband memory.** It needs about 100 bytes per canvas pixel (29.5 MP full-resolution boat peaks at 2.9 GB), so canvases above 40 MP (`--multiband-max-mp`) fall back to feather; use `--max-image-dimension` for larger photos.
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
