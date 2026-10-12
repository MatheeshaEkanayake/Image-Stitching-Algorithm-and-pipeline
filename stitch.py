#!/usr/bin/env python3
"""Thin entry point: ``python stitch.py Images/boat Results/boat-panorama.jpg``."""

import sys

from stitcher.cli import main

if __name__ == "__main__":
    sys.exit(main())
