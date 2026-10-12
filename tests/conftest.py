"""Synthetic test data: one textured scene cut into frames by known homographies."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
import pytest

FRAME_W, FRAME_H = 480, 360
STEP = 260  # horizontal shift between consecutive frames (about 46% overlap)


def make_scene(width: int = 1500, height: int = 520, seed: int = 7) -> np.ndarray:
    """A feature-rich BGR image: blurred noise plus random coloured shapes."""
    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(rng.integers(60, 200, (height, width, 3), dtype=np.uint8), (0, 0), 1.5)
    for _ in range(900):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(scene, centre, int(rng.integers(3, 18)), colour, -1)
        else:
            size = int(rng.integers(6, 30))
            cv2.rectangle(scene, centre, (centre[0] + size, centre[1] + size), colour, -1)
    return scene


def frame_to_scene(k: int) -> np.ndarray:
    """Known homography mapping frame k's pixels onto scene pixels (mild tilt + perspective)."""
    angle = np.deg2rad(1.5 * (k % 2) - 0.5)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    perspective = np.array([[1, 0, 0], [0, 1, 0], [2e-5 * (k - 1), 1e-5, 1]])
    shift = np.array([[1, 0, 40 + STEP * k], [0, 1, 70 + 12 * k], [0, 0, 1]], dtype=np.float64)
    return shift @ rotation @ perspective


def make_frames(count: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    scene = make_scene()
    matrices = [frame_to_scene(k) for k in range(count)]
    frames = [
        cv2.warpPerspective(scene, np.linalg.inv(m), (FRAME_W, FRAME_H)) for m in matrices
    ]
    return frames, matrices


@pytest.fixture(scope="session")
def frames_and_truth() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return make_frames()


@pytest.fixture(scope="session")
def frame_folder(tmp_path_factory: pytest.TempPathFactory, frames_and_truth) -> str:
    folder = tmp_path_factory.mktemp("frames")
    for index, frame in enumerate(frames_and_truth[0]):
        cv2.imwrite(os.path.join(str(folder), f"frame{index + 1}.png"), frame)
    return str(folder)
