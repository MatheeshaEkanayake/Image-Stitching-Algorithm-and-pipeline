# Image Stitching Project

This project creates a panorama from an ordered set of overlapping images using SIFT or ORB feature matching, homographies, global transforms, and feather blending.

The supported application is `stitch.py`. The files `stitch2.py` and `stitch3.py` are excluded from this setup and usage guide.

## Requirements

- Python 3.12 or compatible Python 3 version
- OpenCV with SIFT support
- NumPy
- Linux, macOS, or Windows

The required packages are listed in [requirement.txt](requirement.txt).

## Project Structure

```text
Image-stitch/
├── stitch.py              # Command-line entry point
├── models.py              # Configuration and data models
├── image_io.py            # Image loading and filename sorting
├── features.py            # SIFT/ORB detection and feature matching
├── geometry.py            # Homographies, transforms, and canvas geometry
├── blending.py            # Warping, feather blending, and cropping
├── panorama_stitcher.py   # Pipeline orchestration
├── requirement.txt        # Python dependencies
├── Images/                # Input image folders
└── Results/               # Generated panoramas
```

## Setup

From the project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

The repository also contains an existing environment in `Stitch/`. It can be used directly instead:

```bash
./Stitch/bin/python -m pip install -r requirement.txt
```

On Windows, use the corresponding commands:

```powershell
.venv\Scripts\activate
python -m pip install -r requirement.txt
```

## Input Images

Place images in a folder under `Images/`. Supported formats are:

- `.jpg`
- `.jpeg`
- `.png`

Images are loaded in natural filename order. Use names such as `boat1.jpg`, `boat2.jpg`, and `boat3.jpg` for a left-to-right sequence. Neighboring images must have enough visual overlap for feature matching.

## Run the Stitcher

The command requires an input folder and an output image path:

```bash
python stitch.py Images/boat Results/boat-panorama.jpg
```

Using the repository environment:

```bash
./Stitch/bin/python stitch.py Images/boat Results/boat-panorama.jpg
```

Other examples:

```bash
python stitch.py Images/cafe Results/cafe-panorama.jpg
python stitch.py Images/class Results/class-panorama.jpg
python stitch.py Images/images Results/images-panorama.jpg
```

The output panorama is written to the path provided as the second argument.

## Useful Options

```text
--detector {sift,orb}       Feature detector; default: sift
--ratio RATIO              Lowe ratio-test threshold; default: 0.75
--ransac-thresh VALUE      Homography RANSAC threshold; default: 4.0
--min-matches VALUE        Minimum matches per adjacent pair; default: 10
--feather-width VALUE      Feather blending width; default: 50
--blend-mode {feather,multiband}
                           Use feather; multiband is not implemented
--max-image-dimension VALUE
                           Downscale large input images for testing
--max-canvas-mp VALUE      Maximum panorama canvas size; default: 400
--debug                    Save match and warped-image debug files
--debug-dir DIRECTORY      Debug output directory; default: debug
-v, --verbose              Enable detailed logging
```

For a smaller and safer test run:

```bash
./Stitch/bin/python stitch.py \
    Images/boat Results/boat-test.jpg \
    --max-image-dimension 1000 --max-canvas-mp 100 --verbose
```

## Data Flow

1. `stitch.py` parses command-line arguments and builds `StitchConfig`.
2. `image_io.py` loads and sorts the input images.
3. `features.py` detects features and matches adjacent image pairs.
4. `geometry.py` estimates pairwise homographies and computes global transforms.
5. `geometry.py` calculates the panorama canvas and translation matrix.
6. `blending.py` warps each source image into the shared canvas.
7. `blending.py` feather-blends the warped images and crops black borders.
8. `stitch.py` saves the final panorama with OpenCV.

## Troubleshooting

### Missing positional arguments

This command is incomplete:

```bash
python stitch.py
```

Provide both the input folder and output path:

```bash
python stitch.py Images/boat Results/boat-panorama.jpg
```

### OpenCV is not installed

Use the project virtual environment or activate `.venv` before installing dependencies. Verify the installation with:

```bash
./Stitch/bin/python -c "import cv2, numpy; print(cv2.__version__)"
```

### Insufficient matches

Check that:

- Images are ordered from left to right.
- Neighboring images overlap substantially.
- Images contain texture and are not blank or blurred.

You can try ORB or a less strict ratio threshold:

```bash
./Stitch/bin/python stitch.py Images/boat Results/boat-panorama.jpg \
    --detector orb --ratio 0.8
```

### High memory use or a very large canvas

Downscale images while testing:

```bash
./Stitch/bin/python stitch.py Images/boat Results/boat-panorama.jpg \
    --max-image-dimension 1500
```

A large or unstable canvas can indicate incorrect image ordering or unreliable matches. Use `--debug` to inspect match images and warped images.

### Checking whether the process is running

In another terminal, run:

```bash
pgrep -af '(^|/)stitch\.py([[:space:]]|$)'
```

No output means that `stitch.py` is not currently running. An exit code of `0` means the command completed successfully; exit code `130` usually means it was interrupted with `Ctrl+C`.
