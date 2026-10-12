#!/usr/bin/env python3
"""Compare rendered panoramas in out/ against the reference renders in baseline/.

Reports, per set, the image sizes and the mean absolute pixel difference. When the sizes
differ (for example because the coverage-mask crop moved the border) the images are aligned
on their top-left corner by cropping both to the common size; the "size" column shows both.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np


def compare(baseline_path: str, new_path: str) -> str:
    old = cv2.imread(baseline_path, cv2.IMREAD_COLOR)
    new = cv2.imread(new_path, cv2.IMREAD_COLOR)
    if old is None or new is None:
        return "missing"
    height, width = min(old.shape[0], new.shape[0]), min(old.shape[1], new.shape[1])
    diff = np.abs(old[:height, :width].astype(np.int16) - new[:height, :width].astype(np.int16))
    old_size = f"{old.shape[1]}x{old.shape[0]}"
    new_size = f"{new.shape[1]}x{new.shape[0]}"
    return (
        f"mean abs diff {diff.mean():8.4f}  max {int(diff.max()):3d}  "
        f"baseline {old_size}  new {new_size}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="baseline", help="Folder with reference renders.")
    parser.add_argument("--new", default="out", help="Folder with new renders.")
    args = parser.parse_args(argv)

    names = sorted(f for f in os.listdir(args.baseline) if f.lower().endswith((".jpg", ".png")))
    if not names:
        print(f"No reference images found in {args.baseline}/", file=sys.stderr)
        return 1
    for name in names:
        stem = os.path.splitext(name)[0]
        new_path = os.path.join(args.new, name)
        if not os.path.exists(new_path):
            print(f"{stem:8s} no new render (the set may now fail on purpose)")
            continue
        print(f"{stem:8s} {compare(os.path.join(args.baseline, name), new_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
