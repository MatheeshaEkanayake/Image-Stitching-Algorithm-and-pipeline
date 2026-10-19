from __future__ import annotations

import json
import os
import time
from typing import List

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stitcher.server import ServerSettings, create_app
from stitcher.tiles import TileManifest, build_tile_pyramid


def test_tile_pyramid_levels_and_lossless_top_level(tmp_path):
    rng = np.random.default_rng(0)
    image = cv2.GaussianBlur(rng.integers(0, 255, (700, 1300, 3), dtype=np.uint8), (0, 0), 2)
    manifest = build_tile_pyramid(image, str(tmp_path), tile_size=256, fmt="png")
    assert (manifest.width, manifest.height) == (1300, 700)
    assert manifest.max_level == 11  # ceil(log2(1300))
    assert manifest.level_size(manifest.max_level) == (1300, 700)
    assert manifest.level_size(manifest.min_level)[0] <= 256 * 1  # fits one tile
    assert manifest.grid(manifest.max_level) == (6, 3)
    # PNG tiles at full resolution reassemble to the exact original
    columns, rows = manifest.grid(manifest.max_level)
    canvas = np.zeros_like(image)
    for r in range(rows):
        for c in range(columns):
            tile = cv2.imread(os.path.join(tmp_path, str(manifest.max_level), f"{c}_{r}.png"))
            canvas[r * 256:r * 256 + tile.shape[0], c * 256:c * 256 + tile.shape[1]] = tile
    assert np.array_equal(canvas, image)
    count = sum(len(f) for _, _, f in os.walk(tmp_path) if f) - 1  # minus manifest.json
    assert manifest.tile_count == count
    reloaded = TileManifest(**json.loads((tmp_path / "manifest.json").read_text()))
    assert reloaded == manifest


def test_tile_pyramid_small_image_is_one_level(tmp_path):
    manifest = build_tile_pyramid(np.zeros((100, 200, 3), np.uint8), str(tmp_path), tile_size=512)
    assert manifest.min_level == manifest.max_level and manifest.tile_count == 1


def test_tile_pyramid_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_tile_pyramid(np.zeros((10, 10, 3), np.uint8), str(tmp_path), fmt="gif")
    with pytest.raises(ValueError):
        build_tile_pyramid(np.zeros((10, 10, 3), np.uint8), str(tmp_path), tile_size=8)


@pytest.fixture()
def client(tmp_path):
    settings = ServerSettings(data_dir=str(tmp_path / "data"), max_files=6, max_file_mb=5)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.runner.shutdown()


def upload(client: TestClient, folder: str, count: int = 4, **fields):
    names = sorted(os.listdir(folder))[:count]
    files = [("files", (n, open(os.path.join(folder, n), "rb"), "image/png")) for n in names]
    try:
        return client.post("/jobs", files=files, data=fields)
    finally:
        for _, (_, handle, _) in files:
            handle.close()


def wait_done(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_server_end_to_end(client, rotation_folder):
    response = upload(client, rotation_folder)
    assert response.status_code == 202
    job_id = response.json()["id"]
    status = wait_done(client, job_id)
    assert status["state"] == "done", status
    assert status["width"] > 480

    manifest = client.get(f"/jobs/{job_id}/manifest.json").json()
    assert manifest["width"] == status["width"]
    top = manifest["max_level"]
    tile = client.get(f"/jobs/{job_id}/tiles/{top}/0_0.jpg")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(tile.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] <= manifest["tile_size"]

    # full-resolution crop is native pixels of the stored TIFF
    crop = client.get(f"/jobs/{job_id}/crop", params={"x": 10, "y": 20, "w": 100, "h": 80})
    assert crop.status_code == 200
    pixels = cv2.imdecode(np.frombuffer(crop.content, np.uint8), cv2.IMREAD_COLOR)
    assert pixels.shape[:2] == (80, 100)

    assert client.get(f"/jobs/{job_id}/report.json").json()["image_names"][0] == "0000.png"
    assert client.get(f"/jobs/{job_id}/panorama.tif").headers["content-type"] == "image/tiff"
    assert "OpenSeadragon" in client.get("/static/openseadragon.min.js").text
    assert "<html" in client.get(f"/jobs/{job_id}/view").text
    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_server_failure_is_reported_not_raised(client, tmp_path):
    rng = np.random.default_rng(3)
    folder = tmp_path / "noise"
    folder.mkdir()
    for i in range(2):
        noise = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
        cv2.imwrite(str(folder / f"n{i}.png"), noise)
    job_id = upload(client, str(folder), count=2).json()["id"]
    status = wait_done(client, job_id)
    assert status["state"] == "failed" and "images 0 and 1" in status["error"]
    assert client.get(f"/jobs/{job_id}/manifest.json").status_code == 409


def test_server_input_validation(client, rotation_folder, tmp_path):
    assert upload(client, rotation_folder, count=1).status_code == 400  # need >= 2 images
    assert upload(client, rotation_folder, projection="spherical").status_code == 400
    assert upload(client, rotation_folder, max_image_dimension="-5").status_code == 400
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    files = [("files", ("a.exe", bad.read_bytes(), "application/octet-stream")),
             ("files", ("b.png", b"x", "image/png"))]
    assert client.post("/jobs", files=files).status_code == 400
    big = [("files", (f"{i}.png", b"0" * 6_000_000, "image/png")) for i in range(2)]
    assert client.post("/jobs", files=big).status_code == 413
    too_many: List = [("files", (f"{i}.png", b"x", "image/png")) for i in range(7)]
    assert client.post("/jobs", files=too_many).status_code == 400
    # no job directory is left behind by rejected uploads
    assert os.listdir(os.path.join(client.app.state.settings.data_dir, "jobs")) == []


def test_server_rejects_bad_ids_and_tile_names(client, rotation_folder):
    assert client.get("/jobs/../../etc/passwd").status_code in (404, 422)
    assert client.get("/jobs/" + "z" * 32).status_code == 404
    job_id = upload(client, rotation_folder).json()["id"]
    assert wait_done(client, job_id)["state"] == "done"
    assert client.get(f"/jobs/{job_id}/tiles/5/..%2f..%2fstatus.json").status_code == 404
    assert client.get(f"/jobs/{job_id}/tiles/5/0_0.exe").status_code == 404
    assert client.get(f"/jobs/{job_id}/crop", params={"x": 0, "y": 0, "w": 0, "h": 5}
                      ).status_code == 400
    assert client.get(f"/jobs/{job_id}/crop", params={"x": 99999, "y": 0, "w": 5, "h": 5}
                      ).status_code == 400


def test_server_token_auth(tmp_path, rotation_folder):
    settings = ServerSettings(data_dir=str(tmp_path / "d"), api_token="s3cret")
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert upload(c, rotation_folder).status_code == 401
        names = sorted(os.listdir(rotation_folder))[:2]
        files = [("files", (n, open(os.path.join(rotation_folder, n), "rb"), "image/png"))
                 for n in names]
        ok = c.post("/jobs", files=files, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 202
        job_id = ok.json()["id"]
        assert c.get(f"/jobs/{job_id}").status_code == 401
        assert c.get(f"/jobs/{job_id}?token=s3cret").status_code == 200  # for WebView/browser
        assert c.get(f"/jobs/{job_id}?token=wrong").status_code == 401
    app.state.runner.shutdown()
