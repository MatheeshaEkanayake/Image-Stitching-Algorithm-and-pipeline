# Baseline (original code, before Phase 1 changes)

Commit `ceebf74` (original code), run as
`python -u stitch.py Images/<set> baseline/<set>.jpg --blend-mode feather --max-image-dimension 1200`
(`--blend-mode feather` is required because the default `multiband` crashes, see bug 1).
Runtime is wall clock and peak memory is max resident set size, both from `/usr/bin/time -v`.
Machine: 12 cores, 7 GB RAM (about 1.4 GB free during the run), Python 3.14, numpy 2.5.1,
opencv-contrib-python 5.0.0. Raw logs are in `baseline/*.log` (git-ignored, regenerate by
checking out `ceebf74`).

| Set | Result | Images | Canvas (px) | Output (px) | Runtime | Peak memory |
|-----|--------|-------:|-------------|-------------|--------:|------------:|
| boat | success | 6 | 10283 x 3603 (37.0 MP) | 10281 x 3601 | 28.7 s | 2157 MB |
| cafe | success | 5 | 4522 x 1454 (6.6 MP) | 4521 x 1453 | 2.1 s | 617 MB |
| class | success | 4 | 4216 x 2357 (9.9 MP) | 4216 x 2356 | 2.3 s | 812 MB |
| images | **failure** | 16 | 507.1 MP (refused) | none | 2.5 s | 332 MB |

`images` stops with `Computed canvas is 507.1 MP, exceeding the 400 MP safety cap.`
(no per-pair error; every pair registered fine but the chained homographies blow up).

## RANSAC inliers per pair (inliers / good matches)

| Pair | boat | cafe | class | images |
|------|------|------|-------|--------|
| 0-1 | 695/776 | 114/331 | 131/413 | 447/481 |
| 1-2 | 452/581 | 150/439 | 456/691 | 219/272 |
| 2-3 | 311/357 | 443/833 | 252/511 | 290/407 |
| 3-4 | 619/673 | 212/399 | | 354/433 |
| 4-5 | 891/998 | | | 264/291 |
| 5-6 | | | | 449/483 |
| 6-7 | | | | 331/377 |
| 7-8 | | | | 237/283 |
| 8-9 | | | | 101/156 |
| 9-10 | | | | 222/256 |
| 10-11 | | | | 211/241 |
| 11-12 | | | | 129/163 |
| 12-13 | | | | 451/522 |
| 13-14 | | | | 335/456 |
| 14-15 | | | | 400/464 |

Reference image: boat 3, cafe 2, class 2, images 8.
