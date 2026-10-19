"""Registration, login, logout and the signed-in user."""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stitcher.server.app import current_user
from stitcher.server.security import hash_password, new_token, token_hash, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
# A real hash to compare against for unknown usernames, so timing does not reveal them.
_DUMMY_HASH = hash_password("not-a-real-password")


class Credentials(BaseModel):
    username: str
    password: str


def _issue_session(request: Request, user_id: int, username: str) -> Dict[str, Any]:
    token = new_token()
    now = time.time()
    expires = now + request.app.state.settings.session_days * 86400
    with request.app.state.db.write() as conn:
        conn.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (?, ?, ?, ?)",
            (token_hash(token), user_id, now, expires),
        )
    return {"token": token, "expires": expires, "user": {"id": user_id, "username": username}}


@router.post("/register", status_code=201)
def register(body: Credentials, request: Request) -> Dict[str, Any]:
    if not request.app.state.settings.allow_registration:
        raise HTTPException(403, "Registration is closed on this server.")
    if not USERNAME.match(body.username):
        raise HTTPException(400, "Username must be 3-32 letters, digits, '_', '.' or '-'.")
    if not 8 <= len(body.password) <= 256:
        raise HTTPException(400, "Password must be at least 8 characters.")
    try:
        with request.app.state.db.write() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, created) VALUES (?, ?, ?)",
                (body.username, hash_password(body.password), time.time()),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "That username is taken.") from exc
    return _issue_session(request, user_id, body.username)


@router.post("/login")
def login(body: Credentials, request: Request) -> Dict[str, Any]:
    throttle = request.app.state.throttle
    if throttle.blocked(body.username):
        raise HTTPException(429, "Too many failed attempts; wait a minute and try again.")
    with request.app.state.db.connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?", (body.username,)
        ).fetchone()
    if row is None:
        verify_password(body.password, _DUMMY_HASH)
        throttle.failed(body.username)
        raise HTTPException(401, "Wrong username or password.")
    if not verify_password(body.password, row["password_hash"]):
        throttle.failed(body.username)
        raise HTTPException(401, "Wrong username or password.")
    throttle.succeeded(body.username)
    return _issue_session(request, row["id"], row["username"])


@router.post("/logout", status_code=204)
def logout(request: Request, user: Dict[str, Any] = Depends(current_user)) -> None:  # noqa: B008
    token = request.headers["authorization"][7:].strip()
    with request.app.state.db.write() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))


@router.get("/me")
def me(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    return user
