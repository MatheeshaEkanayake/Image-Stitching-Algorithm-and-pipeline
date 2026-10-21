"""Server API: accounts, per-photo capture upload, library, sharing, community."""

from __future__ import annotations

import os
import time
from typing import Dict

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stitcher.server import ServerSettings, create_app


@pytest.fixture()
def app(tmp_path):
    application = create_app(ServerSettings(data_dir=str(tmp_path / "data"), max_photos=8,
                                            max_file_mb=5))
    yield application
    application.state.runner.shutdown()


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, name: str, password: str = "correct horse") -> Dict[str, str]:
    response = client.post("/auth/register", json={"username": name, "password": password})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}


def capture_photos(client, auth, folder: str, count: int = 4, order=None) -> str:
    capture_id = client.post("/captures", json={"title": "Test"}, headers=auth).json()["id"]
    names = sorted(os.listdir(folder))[:count]
    for index in order or range(len(names)):
        with open(os.path.join(folder, names[index]), "rb") as handle:
            r = client.post(f"/captures/{capture_id}/photos", headers=auth,
                            data={"index": str(index)}, files={"file": (names[index], handle)})
        assert r.status_code == 200, r.text
    return capture_id


def wait_done(client, auth, capture_id: str, timeout: float = 60) -> Dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        capture = client.get(f"/captures/{capture_id}", headers=auth).json()
        if capture["state"] in ("done", "failed"):
            return capture
        time.sleep(0.2)
    raise AssertionError("capture did not finish")


def test_register_login_logout(client):
    auth = register(client, "alice")
    assert client.get("/auth/me", headers=auth).json()["username"] == "alice"
    assert client.post("/auth/register", json={"username": "ALICE",
                                               "password": "whatever123"}).status_code == 409
    assert client.post("/auth/register", json={"username": "a",
                                               "password": "whatever123"}).status_code == 400
    assert client.post("/auth/register", json={"username": "bob",
                                               "password": "short"}).status_code == 400
    login = client.post("/auth/login", json={"username": "alice", "password": "correct horse"})
    assert login.status_code == 200
    token2 = {"Authorization": "Bearer " + login.json()["token"]}
    assert client.post("/auth/logout", headers=token2).status_code == 204
    assert client.get("/auth/me", headers=token2).status_code == 401
    assert client.get("/auth/me", headers=auth).status_code == 200  # other session unaffected
    assert client.get("/auth/me").status_code == 401
    assert client.get("/captures").status_code == 401


def test_password_is_hashed_and_login_is_throttled(client, app):
    register(client, "carol", "s3cret-password")
    with app.state.db.connect() as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert "s3cret-password" not in stored and stored.startswith("scrypt$")
    for _ in range(5):
        assert client.post("/auth/login", json={"username": "carol",
                                                "password": "wrong-password"}).status_code == 401
    assert client.post("/auth/login", json={"username": "carol",
                                            "password": "s3cret-password"}).status_code == 429


def test_capture_photo_by_photo_then_view(client, rotation_folder):
    auth = register(client, "dave")
    # photos arrive out of order (as background uploads can); the index decides the order
    capture_id = capture_photos(client, auth, rotation_folder, order=[1, 0, 3, 2])
    assert client.get(f"/captures/{capture_id}", headers=auth).json()["photo_count"] == 4
    assert client.get(f"/captures/{capture_id}/manifest.json", headers=auth).status_code == 409
    started = client.post(f"/captures/{capture_id}/finish", headers=auth, json={})
    assert started.status_code == 202 and started.json()["state"] in ("queued", "running", "done")
    capture = wait_done(client, auth, capture_id)
    assert capture["state"] == "done", capture
    assert capture["owner"]["username"] == "dave" and capture["width"] > 480

    manifest = client.get(f"/captures/{capture_id}/manifest.json", headers=auth).json()
    tile = client.get(f"/captures/{capture_id}/tiles/{manifest['max_level']}/0_0.jpg",
                      headers=auth)
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg"
    thumb = client.get(f"/captures/{capture_id}/thumb.jpg", headers=auth)
    assert cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR) is not None
    crop = client.get(f"/captures/{capture_id}/crop", headers=auth,
                      params={"x": 5, "y": 5, "w": 64, "h": 48})
    assert cv2.imdecode(np.frombuffer(crop.content, np.uint8), 1).shape[:2] == (48, 64)
    assert client.get(f"/captures/{capture_id}/report.json", headers=auth).status_code == 200
    assert client.get(f"/captures/{capture_id}/panorama.tif", headers=auth).status_code == 200
    page = client.get(f"/captures/{capture_id}/view")  # static page, no data inside
    assert page.status_code == 200 and "ajaxHeaders" in page.text
    library = client.get("/captures", headers=auth).json()
    assert [c["id"] for c in library] == [capture_id]
    # no more uploads once finished
    with open(os.path.join(rotation_folder, sorted(os.listdir(rotation_folder))[0]), "rb") as f:
        late = client.post(f"/captures/{capture_id}/photos", headers=auth,
                           data={"index": "4"}, files={"file": ("x.png", f)})
    assert late.status_code == 409


def test_finish_requires_all_photos(client, rotation_folder):
    auth = register(client, "erin")
    capture_id = capture_photos(client, auth, rotation_folder, order=[0, 2])
    response = client.post(f"/captures/{capture_id}/finish", headers=auth, json={})
    assert response.status_code == 400 and "[1]" in response.json()["error"]
    one = capture_photos(client, auth, rotation_folder, count=1)
    assert client.post(f"/captures/{one}/finish", headers=auth, json={}).status_code == 400


def test_upload_validation(client, rotation_folder):
    auth = register(client, "frank")
    capture_id = client.post("/captures", json={}, headers=auth).json()["id"]
    url = f"/captures/{capture_id}/photos"
    bad_type = client.post(url, headers=auth, data={"index": "0"},
                           files={"file": ("a.exe", b"MZ")})
    assert bad_type.status_code == 400
    too_big = client.post(url, headers=auth, data={"index": "0"},
                          files={"file": ("a.jpg", b"0" * 6_000_000)})
    assert too_big.status_code == 413
    bad_index = client.post(url, headers=auth, data={"index": "99"},
                            files={"file": ("a.jpg", b"x")})
    assert bad_index.status_code == 400
    inputs = os.path.join(client.app.state.settings.data_dir, "jobs", capture_id, "inputs")
    assert os.listdir(inputs) == []  # nothing (not even .part files) left behind
    # retrying an index replaces the photo instead of adding one
    for _ in range(2):
        client.post(url, headers=auth, data={"index": "0"}, files={"file": ("a.jpg", b"x")})
    assert client.get(f"/captures/{capture_id}", headers=auth).json()["photo_count"] == 1


def test_private_until_shared_and_failure_reported(client, rotation_folder, tmp_path):
    alice, bob = register(client, "alice"), register(client, "bob")
    capture_id = capture_photos(client, alice, rotation_folder)
    client.post(f"/captures/{capture_id}/finish", headers=alice, json={})
    assert wait_done(client, alice, capture_id)["state"] == "done"
    # bob cannot see, view, crop, upload to or delete alice's private capture
    for path in ("", "/manifest.json", "/thumb.jpg", "/panorama.tif"):
        assert client.get(f"/captures/{capture_id}{path}", headers=bob).status_code == 404
    assert client.delete(f"/captures/{capture_id}", headers=bob).status_code == 404
    shared = client.patch(f"/captures/{capture_id}", headers=alice, json={"public": True})
    assert shared.json()["is_public"] is True
    assert client.get(f"/captures/{capture_id}/manifest.json", headers=bob).status_code == 200
    assert client.patch(f"/captures/{capture_id}", headers=bob,
                        json={"public": False}).status_code == 404
    assert client.get("/captures", headers=bob).json() == []  # library is per user

    rng = np.random.default_rng(3)
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    for i in range(2):
        noise = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
        cv2.imwrite(str(noise_dir / f"n{i}.png"), noise)
    bad = capture_photos(client, alice, str(noise_dir), count=2)
    client.post(f"/captures/{bad}/finish", headers=alice, json={})
    failed = wait_done(client, alice, bad)
    assert failed["state"] == "failed" and "do not overlap each other" in failed["error"]
    assert client.patch(f"/captures/{bad}", headers=alice,
                        json={"public": True}).status_code == 409


def test_community_posts_comments_likes_profiles(client, rotation_folder):
    alice, bob = register(client, "alice"), register(client, "bob")
    capture_id = capture_photos(client, alice, rotation_folder)
    client.post(f"/captures/{capture_id}/finish", headers=alice, json={})
    wait_done(client, alice, capture_id)

    # bob cannot attach alice's capture
    assert client.post("/posts", headers=bob, json={"body": "mine!",
                                                    "capture_id": capture_id}).status_code == 404
    assert client.post("/posts", headers=bob, json={"body": "   "}).status_code == 400
    post = client.post("/posts", headers=alice,
                       json={"body": "Lake at sunrise", "capture_id": capture_id}).json()
    assert post["capture"]["id"] == capture_id and post["author"]["username"] == "alice"
    # attaching shared it, so bob can now open the panorama
    assert client.get(f"/captures/{capture_id}/manifest.json", headers=bob).status_code == 200
    idea = client.post("/posts", headers=bob, json={"body": "Tip: overlap 30%"}).json()

    feed = client.get("/posts", headers=bob).json()
    assert [p["id"] for p in feed] == [idea["id"], post["id"]]
    assert client.get("/posts", headers=bob, params={"before": idea["id"]}).json()[0]["id"] \
        == post["id"]

    liked = client.put(f"/posts/{post['id']}/like", headers=bob).json()
    assert liked["like_count"] == 1 and liked["liked"] is True
    assert client.put(f"/posts/{post['id']}/like", headers=bob).json()["like_count"] == 1
    assert client.get(f"/posts/{post['id']}", headers=alice).json()["liked"] is False
    assert client.delete(f"/posts/{post['id']}/like", headers=bob).json()["like_count"] == 0

    comment = client.post(f"/posts/{post['id']}/comments", headers=bob,
                          json={"body": "Beautiful!"}).json()
    assert client.get(f"/posts/{post['id']}", headers=alice).json()["comment_count"] == 1
    assert client.delete(f"/comments/{comment['id']}", headers=alice).status_code == 403
    assert client.delete(f"/comments/{comment['id']}", headers=bob).status_code == 204

    profile = client.get("/users/alice", headers=bob).json()
    assert [c["id"] for c in profile["captures"]] == [capture_id]
    assert [p["id"] for p in profile["posts"]] == [post["id"]]
    assert client.get("/users/nobody", headers=bob).status_code == 404

    # unsharing hides the panorama from the post; deleting it detaches it
    client.patch(f"/captures/{capture_id}", headers=alice, json={"public": False})
    assert client.get(f"/posts/{post['id']}", headers=bob).json()["capture"] is None
    assert client.delete(f"/captures/{capture_id}", headers=alice).status_code == 204
    assert client.get(f"/posts/{post['id']}", headers=alice).json()["capture"] is None
    assert client.delete(f"/posts/{post['id']}", headers=bob).status_code == 403
    assert client.delete(f"/posts/{post['id']}", headers=alice).status_code == 204


def test_restart_marks_interrupted_captures_failed(tmp_path):
    settings = ServerSettings(data_dir=str(tmp_path / "d"))
    first = create_app(settings)
    first.state.store.write_status("a" * 32, state="running")
    first.state.runner.shutdown()
    second = create_app(settings)
    assert second.state.store.read_status("a" * 32)["state"] == "failed"
    second.state.runner.shutdown()


def test_bad_ids_and_tile_names(client, rotation_folder):
    auth = register(client, "gina")
    assert client.get("/captures/" + "z" * 32, headers=auth).status_code == 404
    assert client.get("/captures/..%2f..%2fapp.db", headers=auth).status_code == 404
    capture_id = capture_photos(client, auth, rotation_folder)
    client.post(f"/captures/{capture_id}/finish", headers=auth, json={})
    wait_done(client, auth, capture_id)
    assert client.get(f"/captures/{capture_id}/tiles/5/0_0.exe", headers=auth).status_code == 404
    assert client.get(f"/captures/{capture_id}/crop", headers=auth,
                      params={"x": 0, "y": 0, "w": 0, "h": 5}).status_code == 400
