# Process: what changed, issues faced and how they were resolved

This document records how the image-stitching system developed from the original `stitch.py`
script into a stitching server with a mobile app: the problems found, their causes, the
fixes, and the measurements that confirmed them. Details of individual changes are in the git
history; the original code's measurements are in [docs/baseline.md](docs/baseline.md) and
[docs/phase1_report.md](docs/phase1_report.md).

Unless stated otherwise, figures were measured on the development machine (Linux, 12 cores,
7 GB RAM, often only 1–4 GB free) with the sample sets in `Images/`.

## Where the project stands

| Area | State |
|------|-------|
| Stitching engine | Python package `stitcher/`: SIFT features, all-pairs matching, overlap graph, RANSAC homographies, automatic choice of cylindrical (rotating camera) or planar (flat target) model, bundle adjustment, exposure compensation, graph-cut seams, multiband blending |
| Output | Lossless TIFF master, Deep Zoom tile pyramid (JPEG), thumbnail, JSON run report |
| Server | FastAPI + SQLite: accounts, photo-by-photo upload, background stitching, private library, sharing, community (posts, comments, likes, profiles), zoom viewer |
| Mobile app | React Native (Expo SDK 57): capture with live upload, grid and level, library, deep-zoom viewer, community; blue theme |
| Checks | 53 automated tests and `ruff` for the Python code; type check, lint, Expo doctor and an Android bundle build for the app |

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-10-12 | Original code measured (baseline); restructured into a package; Phase 1 bug fixes, tests, README |
| 2026-10-19 | Cylindrical projection; bundle adjustment and multiband blending; detail and memory work; TIFF output; tile pyramid and server; accounts and community; React Native app |
| 2026-10-21 | First real phone tests: Expo SDK fix, upload fix, automatic projection choice, any-order capture, grid and level |

---

## 1. Starting point and Phase 1 clean-up

**Situation.** A working prototype that stitched photos taken strictly left to right. Before
changing anything, every sample set was run and measured to give a regression reference
([docs/baseline.md](docs/baseline.md)).

**Problems found and fixed**

| Problem | Fix |
|---------|-----|
| The default blend mode (`multiband`) was not implemented, so the program crashed by default | Made `feather` the default and removed the dead option (multiband was later implemented properly, see section 4) |
| Every image was warped onto the full canvas, so memory and time grew with canvas size × image count | Warp each image only into its own bounding box. Boat: 28.7 s → 8.6 s, 2157 MB → 1177 MB peak |
| Cropping looked for non-black pixels, so genuinely dark content was cut off | Crop with a coverage mask of where images actually landed |
| No checks on homographies; bad pairs gave garbage | Minimum inlier count, rejection of singular or implausible transforms, errors naming the pair |
| Wide sweeps produced absurd canvases (507 MP for `Images/images`) | Detect points behind the camera and refuse with a clear message before allocating memory |
| Results were not reproducible | Seeded OpenCV's random number generator (`--seed`) |
| Unused code, misleading packaging and docs | Package layout, `pyproject.toml`, accurate `requirements.txt`, README, tests |

---

## 2. Crash on full-resolution photos

**Symptom.** `python stitch.py Images/boat ...` crashed several times on full-size photos.

**Cause.** Memory, not a code error. With the original planar model the full-resolution boat
canvas was 31955 × 11938 px (381.5 MP). Blending holds about 16 bytes per canvas pixel, so the
run needed about 6 GB (measured peak 5.9 GB, 2 min 18 s). It succeeded when about 4 GB was
free and failed when only about 1.4 GB was free. The 400 MP safety limit was too high to help.

**Resolution.** Out-of-memory errors (including OpenCV's own allocation errors) now produce a
clear message instead of a traceback, and `--max-image-dimension` is documented for small
machines. The underlying cause, the over-stretched canvas, was removed by the cylindrical
model in section 3: the same photos now give a 10786 × 2733 px (29.5 MP) canvas.

---

## 3. Distortion on wide panoramas

**Symptom.** The Rainier set stitched without errors but one side became a huge stretched
wedge. A 517 px wide photo was smeared across about 3000 px.

**Cause.** The original method chains flat-plane transforms (homographies) from photo to
photo. For a camera turning on the spot, each step stretches the outer photos further, without
limit.

**Resolution.** A cylindrical projection: estimate the camera's focal length from the pair
transforms, convert them to camera rotations, and map every photo onto a cylinder around the
viewer.

| Set (1200 px) | Planar canvas | Cylindrical canvas |
|---------------|---------------|--------------------|
| Rainier | 5592 × 2590 (stretched wedge) | 1100 × 771 |
| boat | 10283 × 3603 | 3340 × 971 |
| cafe | 4522 × 1454 | 2503 × 798 |
| class | 4216 × 2357 | 2183 × 958 |

---

## 4. Ghosting and visible seams

**Symptom.** Doubled people in the classroom set, a doubled drink and chair in the cafe set,
and a duplicated skyline in the boat set.

**Causes.**
1. Small alignment errors accumulated along the chain of photos.
2. Feather blending averages overlapping pixels, so anything that moved between shots, or sits
   at a different distance (parallax), appears twice.

**Resolution.**
1. **Bundle adjustment.** All camera rotations and one shared focal length are refined
   together to minimise the error of every matched point (Levenberg–Marquardt). Median
   reprojection error: boat 8.6 → 0.6 px, cafe 25.8 → 0.9 px, class 10.5 → 1.9 px.
2. **Seams and multiband blending.** Exposure compensation evens out brightness. A graph-cut
   seam then takes each pixel from one photo instead of averaging, and Laplacian-pyramid
   (multiband) blending hides the seam. The doubled people and objects disappeared.

---

## 5. Boat panorama in the wrong place (a bug in the fix)

**Symptom.** After section 4 the boat panorama had its right half at a different height, with
a duplicated skyline. The user reported it as "wrong order".

**Investigation.** The file order was correct. Per-pair errors showed two pairs still about
90–160 px out after the "adjustment".

**Cause.** The adjustment ignored any point more than 5 px out (a robust loss). Those two
pairs started about 90 px out, so all their points were ignored and the optimiser never
corrected them.

**Resolution.** Start with a wide threshold and tighten it (300 → 60 → 15 → 5 px), and warn
when any pair still fits badly. Boat: median error 0.37 px, one continuous horizon. A
regression test was added. The first version of that test also passed on the buggy code, so it
was rewritten until it failed on the old code and passed on the new.

---

## 6. Keeping maximum detail

**Goal.** "As much detail as possible", meaning native resolution and a lossless file, not
just pixel count.

| Issue | Cause | Fix and result |
|-------|-------|----------------|
| Multiband blending needed 4.4 GB and 59 s on a 7.8 MP canvas | OpenCV's block-based exposure compensator (`GAIN_BLOCKS`) | Per-image gain compensation: 0.2 s, no measurable extra memory. Full-resolution boat (29.5 MP) now blends with multiband in about 20 s at 2.8 GB |
| Large canvases ran out of memory | The limit was set while the compensator above was still in use | Multiband limit 20 → 40 MP (`--multiband-max-mp`); larger canvases fall back to feather blending |
| Slight softness | Bilinear resampling while warping | Bicubic. Sharpness measure (variance of the Laplacian) 117.6 → 170.3 on boat at 2000 px |
| JPEG output is lossy | Format choice | TIFF recommended; the output format follows the file extension |

### Output format: TIFF or AVIF?

Measured on the full-resolution boat panorama (10784 × 2732 px):

| Format | Size | Write time | Exact pixels |
|--------|-----:|-----------:|:------------:|
| TIFF | 21.4 MB | 0.9 s | yes |
| AVIF lossless | 21.0 MB | 7.0 s | yes |
| AVIF quality 90 | 2.5 MB | 1.6 s | no |
| PNG | 21.2 MB | 5.7 s | yes |

**Decision.** TIFF stays the master. Lossless AVIF saves only 2 %, takes about 8× longer and
is poorly supported by editing tools. Lossy AVIF loses the detail the project is about. AVIF is
attractive for the viewing tiles (one sky tile: 4 KB AVIF, 12 KB WebP, 26 KB JPEG) but is not
implemented yet.

---

## 7. Viewing very large images on a phone

**Problem.** A 30 MP panorama is quick to download as one JPEG (2.7 MB), but decoding it needs
about 118 MB of phone memory, and zooming becomes slow.

**Resolution.** A Deep Zoom tile pyramid (512 px JPEG tiles at every zoom level) and a viewer
(OpenSeadragon) that loads only the tiles on screen. The full-resolution boat makes 183 tiles
(4.0 MB in total). Opening it fetches one 9 KB overview tile, and a full-detail phone screen
needs about six tiles. A "Full detail" button fetches the visible region at native pixels from
the lossless TIFF.

---

## 8. Server, accounts and community

**Requirements from the user.** Capture photos in the app, upload them as they are taken,
stitch on a server, browse a personal library with the author's name, zoom into detail, and
share ideas in a community.

**Decisions** (each chosen by the user from the options offered):
- **Own login:** usernames and passwords on our own server, with passwords hashed (scrypt) and
  session tokens stored only as hashes. No external services.
- **Live upload:** each photo uploads right after it is taken. A capture is created, photos
  arrive with their sequence number (retries are safe), and "Stitch" queues the job.
- **Private until shared:** each library is private. Attaching a panorama to a post shares it.
- **Community:** posts with an optional panorama, comments, likes, and user profiles.

**Security measures.**
- **Brute-force protection:** five failed logins lock that username for a minute.
- **Private captures:** another user's private capture returns "not found", not "forbidden".
- **Safe file storage:** uploaded file names are never used on disk.
- **Upload limits:** uploads are capped in size and count.
- **Viewer token:** the viewer receives the session token in the URL fragment, which is never
  sent to the server or written to its logs.

---

## 9. First tests on a real phone

### 9.1 "Version error" in Expo Go
- **Cause:** the app was built for Expo SDK 58, but the Expo Go app on the test phone
  supported up to SDK 57. The development server's log gave the exact message.
- **Resolution:** moved the whole app to SDK 57 with Expo's version alignment tool, then
  re-ran the type check, lint, Expo doctor and an Android bundle build. No code changes were
  needed.

### 9.2 "No connection to the server" when taking photos
- **Investigation:** the server log showed the phone registering, opening the library and
  creating captures, so the network was fine. No photo upload request ever arrived.
- **Cause:** the upload used an older React Native trick for sending files that current Expo
  versions do not support reliably. The upload failed on the phone before anything was sent,
  and a generic error message hid the real reason.
- **Resolution:** switched to the upload method documented by Expo (`expo-file-system`'s
  `File.upload`), which streams the photo from disk, and made the app show the real error. The
  next phone test uploaded and stitched successfully.

### 9.3 An unreadable whiteboard
- **Symptom:** four photos of a whiteboard stitched into a distorted, unreadable image.
- **Cause:** a whiteboard is flat, and the user moved the phone across it rather than turning
  on the spot. The cylindrical model (section 3) assumes a turning camera, so it estimated an
  impossible focal length (about 15,000 px, where about 3,000 is normal) and misplaced the
  pieces.
- **Resolution:** an **auto** projection, now the default. It first tries the rotating-camera
  model and measures how well it fits. If any key pair is more than 5 px out, it switches to
  the planar model, which is correct for flat targets. All rotating-camera sample sets fit
  within about 2 px, and the whiteboard measured 8–15 px. The whiteboard became fully readable
  at 5497 × 5408 px, with the handwriting sharp at 100 % zoom.

### 9.4 "What about multiple rows?"
- **Problem:** the app told users to shoot left to right, and the stitcher only compared each
  photo with the next one in capture order. Real users shoot boards in rows, in a snake, or
  right to left, and the whiteboard photos were in fact a zig-zag.
- **Resolution:**
  1. Compare every pair of photos, and keep only pairs that genuinely overlap (Brown & Lowe's
     test: inliers > 5.9 + 0.22 × matches).
  2. Build a graph of which photos overlap, and align along its strongest connections (a
     maximum spanning tree) to the photo at the centre of that graph.
  3. If some photos connect to nothing, fail with their photo numbers so the user knows what
     to retake.
- **A problem this created:** extra overlapping pairs made some results worse. One cafe pair
  disagreed with the rest by 126 px (a false match or parallax), and in the classroom set extra
  pairs pulled a key pair to 11.7 px out. **Fix:** align on the tree first, then add extra
  pairs only if they agree within 5 px, and judge the auto-projection choice on the tree pairs
  only.
- **Verification:**
  - **Boat:** the same result (about 3329 × 843 px) captured left to right, right to left or
    shuffled.
  - **Whiteboard:** correct in snake and reversed order.
  - **Old approach fails:** on a synthetic 3 × 2 grid, the old neighbour-only matching fails
    for row-by-row and shuffled capture, and the new method succeeds in every order.
- **Cost:** matching all 6 pairs of the full-resolution board took 0.2 s. For 30 photos (435
  pairs) the estimate is about 12 s. Above 30 photos only photos close in capture order are
  compared.
- **App:** the capture guidance now says "any order". An arrow picker chooses which side of
  the preview shows the overlap guide, a faint copy of the previous photo's edge.

### 9.5 Alignment aid: grid and level
- **Request:** an iPhone-style grid to help keep shots aligned.
- **Resolution:** a 3 × 3 grid overlay and a level badge driven by the accelerometer. The
  badge turns green when the phone is level, either upright (aimed at a scene or wall) or flat
  (shooting straight down at a table). The overlap guide is now exactly one third, so its edge
  sits on a grid line. Expo's documentation does not state the accelerometer's sign convention
  per platform, so the level uses only magnitudes. Its calculation was checked on eight
  orientations, including an inverted axis.

---

## How changes were verified

- **Run, then look:** every change to the stitching was run on all sample sets and the outputs
  were inspected by eye, not just checked for "no error". Two bugs (sections 5 and 9.4) were
  found only by looking.
- **Regression reference:** the original code's results (docs/baseline.md) and a comparison
  script (`scripts/compare_to_baseline.py`).
- **Automated tests:** 53 tests use synthetic images with known geometry. They cover pair
  recovery, chaining, canvas, cropping, cylindrical and planar models, bundle adjustment, any
  capture order, memory fall-backs, the tile pyramid, and the server's accounts, uploads,
  privacy and community rules.
- **Fail-first tests:** a test written for a bug is checked to fail on the old code.
- **App checks:** type check, lint, Expo doctor, Android bundle build, and an end-to-end run of
  the app's own API client against a real server process (27 checks).

## Known issues and next steps

| Issue | Notes |
|-------|-------|
| Grid, level and direction arrows not yet tried on a phone | Built and type-checked; they need a real-device check |
| Closed 360° panoramas are not supported | `Images/images` (16 photos, about 350°) is refused with a clear message |
| Flat targets can come out slightly keystoned | The planar result takes the perspective of the centre photo; a "straighten document" step would fix it |
| Small seams remain in some sets | e.g. a faint step in Rainier's sky |
| Plain HTTP only | Needs HTTPS before use outside a trusted network |
| No automatic clean-up of old captures | Stored under `server_data/` until deleted in the app |
| Memory for very large canvases | Multiband needs about 100 bytes per canvas pixel; above 40 MP it falls back to feather blending |
| AVIF tiles not implemented | Would cut tile size substantially (section 6) |
| One stitch job at a time; login lockout kept in memory | Fine for a demo server; would need changing for many users |
