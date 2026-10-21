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


ROT_FOCAL = 600.0
ROT_STEP_DEG = 18.0


def make_rotation_frames(count: int = 4) -> List[np.ndarray]:
    """Pinhole views (known focal length) of a cylindrical scene from a camera panning in yaw."""
    scene = make_scene(width=2000, height=700, seed=11)
    scene_h, scene_w = scene.shape[:2]
    xs, ys = np.meshgrid(np.arange(FRAME_W) - FRAME_W / 2, np.arange(FRAME_H) - FRAME_H / 2)
    rays = np.stack([xs, ys, np.full_like(xs, ROT_FOCAL)], axis=-1).astype(np.float64)
    frames = []
    for k in range(count):
        yaw = np.deg2rad((k - (count - 1) / 2) * ROT_STEP_DEG)
        rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        world = rays @ rot.T
        theta = np.arctan2(world[..., 0], world[..., 2])
        height = world[..., 1] / np.hypot(world[..., 0], world[..., 2])
        map_x = (ROT_FOCAL * theta + scene_w / 2).astype(np.float32)
        map_y = (ROT_FOCAL * height + scene_h / 2).astype(np.float32)
        frames.append(cv2.remap(scene, map_x, map_y, cv2.INTER_LINEAR))
    return frames


@pytest.fixture(scope="session")
def rotation_folder(tmp_path_factory: pytest.TempPathFactory) -> str:
    folder = tmp_path_factory.mktemp("rotation")
    for index, frame in enumerate(make_rotation_frames()):
        cv2.imwrite(os.path.join(str(folder), f"r{index + 1}.png"), frame)
    return str(folder)


GRID_COLUMNS, GRID_ROWS = 3, 2
GRID_STEP_X, GRID_STEP_Y = 260, 150


def make_grid_frames() -> List[np.ndarray]:
    """A 3 x 2 grid of overlapping views of one flat scene, in row-major order."""
    scene = make_scene(width=1500, height=720)
    frames = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLUMNS):
            angle = np.deg2rad(0.8 * ((row + col) % 2) - 0.3)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            shift = np.array([[1, 0, 40 + GRID_STEP_X * col], [0, 1, 40 + GRID_STEP_Y * row],
                              [0, 0, 1]], dtype=np.float64)
            frames.append(cv2.warpPerspective(scene, np.linalg.inv(shift @ rotation),
                                              (FRAME_W, FRAME_H)))
    return frames


def write_folder(folder, frames: List[np.ndarray], order: List[int]) -> str:
    """Save ``frames[order[0]], frames[order[1]], ...`` as capture-order files 0000, 0001, ..."""
    folder.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(order):
        cv2.imwrite(str(folder / f"{position:04d}.png"), frames[index])
    return str(folder)


@pytest.fixture(scope="session")
def grid_frames() -> List[np.ndarray]:
    return make_grid_frames()
