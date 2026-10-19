"""Community: posts (optionally with a panorama), comments, likes and user profiles."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from stitcher.server.app import current_user
from stitcher.server.routes_captures import capture_json, load_capture

router = APIRouter(tags=["community"])
User = Dict[str, Any]

_POST_SELECT = """
SELECT p.id, p.body, p.created, p.capture_id, p.author_id, u.username AS author,
       (SELECT COUNT(*) FROM likes l WHERE l.post_id = p.id) AS like_count,
       (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
       EXISTS(SELECT 1 FROM likes l WHERE l.post_id = p.id AND l.user_id = :me) AS liked
FROM posts p JOIN users u ON u.id = p.author_id
"""


class PostCreate(BaseModel):
    body: str
    capture_id: Optional[str] = None


class CommentCreate(BaseModel):
    body: str


def _post_json(request: Request, row: Any) -> Dict[str, Any]:
    capture = None
    if row["capture_id"]:
        store = request.app.state.store
        with request.app.state.db.connect() as conn:
            crow = conn.execute(
                "SELECT c.*, u.username FROM captures c JOIN users u ON u.id = c.owner_id "
                "WHERE c.id = ? AND c.is_public = 1",
                (row["capture_id"],),
            ).fetchone()
        if crow is not None and store.exists(crow["id"]):
            capture = capture_json(crow, store.read_status(crow["id"]))
    return {
        "id": row["id"],
        "body": row["body"],
        "created": row["created"],
        "author": {"id": row["author_id"], "username": row["author"]},
        "capture": capture,  # None if detached, unshared or deleted
        "like_count": row["like_count"],
        "comment_count": row["comment_count"],
        "liked": bool(row["liked"]),
    }


def _clean(text: str, limit: int, what: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(400, f"{what} cannot be empty.")
    if len(text) > limit:
        raise HTTPException(400, f"{what} is longer than {limit} characters.")
    return text


def _get_post(request: Request, post_id: int, user: User) -> Dict[str, Any]:
    with request.app.state.db.connect() as conn:
        row = conn.execute(_POST_SELECT + " WHERE p.id = :id",
                           {"me": user["id"], "id": post_id}).fetchone()
    if row is None:
        raise HTTPException(404, "Post not found.")
    return _post_json(request, row)


@router.get("/posts")
def feed(request: Request, before: Optional[int] = None, limit: int = 20,
         user: User = Depends(current_user)) -> List[Dict[str, Any]]:  # noqa: B008
    """Newest posts first. Page with ``before=<id of the last post you have>``."""
    limit = max(1, min(limit, 50))
    with request.app.state.db.connect() as conn:
        rows = conn.execute(
            _POST_SELECT + " WHERE (:before IS NULL OR p.id < :before) ORDER BY p.id DESC "
            "LIMIT :limit",
            {"me": user["id"], "before": before, "limit": limit},
        ).fetchall()
    return [_post_json(request, r) for r in rows]


@router.post("/posts", status_code=201)
def create_post(body: PostCreate, request: Request,
                user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    text = _clean(body.body, request.app.state.settings.max_post_chars, "Post")
    if body.capture_id:
        # Attaching a panorama is the explicit choice to share it, so it becomes public.
        load_capture(request, body.capture_id, user, owner=True, done=True)
    with request.app.state.db.write() as conn:
        if body.capture_id:
            conn.execute(
                "UPDATE captures SET is_public = 1, published_at = COALESCE(published_at, ?) "
                "WHERE id = ?",
                (time.time(), body.capture_id),
            )
        cursor = conn.execute(
            "INSERT INTO posts (author_id, body, capture_id, created) VALUES (?, ?, ?, ?)",
            (user["id"], text, body.capture_id, time.time()),
        )
        post_id = int(cursor.lastrowid)
    return _get_post(request, post_id, user)


@router.get("/posts/{post_id}")
def get_post(post_id: int, request: Request,
             user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    return _get_post(request, post_id, user)


@router.delete("/posts/{post_id}", status_code=204)
def delete_post(post_id: int, request: Request,
                user: User = Depends(current_user)) -> Response:  # noqa: B008
    post = _get_post(request, post_id, user)
    if post["author"]["id"] != user["id"]:
        raise HTTPException(403, "You can only delete your own posts.")
    with request.app.state.db.write() as conn:
        conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    return Response(status_code=204)


@router.put("/posts/{post_id}/like")
def like(post_id: int, request: Request,
         user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    _get_post(request, post_id, user)
    with request.app.state.db.write() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO likes (post_id, user_id, created) VALUES (?, ?, ?)",
            (post_id, user["id"], time.time()),
        )
    return _get_post(request, post_id, user)


@router.delete("/posts/{post_id}/like")
def unlike(post_id: int, request: Request,
           user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    _get_post(request, post_id, user)
    with request.app.state.db.write() as conn:
        conn.execute("DELETE FROM likes WHERE post_id = ? AND user_id = ?",
                     (post_id, user["id"]))
    return _get_post(request, post_id, user)


@router.get("/posts/{post_id}/comments")
def comments(post_id: int, request: Request,
             user: User = Depends(current_user)) -> List[Dict[str, Any]]:  # noqa: B008
    _get_post(request, post_id, user)
    with request.app.state.db.connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.body, c.created, c.author_id, u.username FROM comments c "
            "JOIN users u ON u.id = c.author_id WHERE c.post_id = ? ORDER BY c.id",
            (post_id,),
        ).fetchall()
    return [{"id": r["id"], "body": r["body"], "created": r["created"],
             "author": {"id": r["author_id"], "username": r["username"]}} for r in rows]


@router.post("/posts/{post_id}/comments", status_code=201)
def add_comment(post_id: int, body: CommentCreate, request: Request,
                user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    _get_post(request, post_id, user)
    text = _clean(body.body, request.app.state.settings.max_comment_chars, "Comment")
    now = time.time()
    with request.app.state.db.write() as conn:
        cursor = conn.execute(
            "INSERT INTO comments (post_id, author_id, body, created) VALUES (?, ?, ?, ?)",
            (post_id, user["id"], text, now),
        )
    return {"id": int(cursor.lastrowid), "body": text, "created": now,
            "author": {"id": user["id"], "username": user["username"]}}


@router.delete("/comments/{comment_id}", status_code=204)
def delete_comment(comment_id: int, request: Request,
                   user: User = Depends(current_user)) -> Response:  # noqa: B008
    with request.app.state.db.write() as conn:
        row = conn.execute("SELECT author_id FROM comments WHERE id = ?",
                           (comment_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Comment not found.")
        if row["author_id"] != user["id"]:
            raise HTTPException(403, "You can only delete your own comments.")
        conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    return Response(status_code=204)


@router.get("/users/{username}")
def profile(username: str, request: Request,
            user: User = Depends(current_user)) -> Dict[str, Any]:  # noqa: B008
    """A user's public panoramas and posts (their own private captures are not shown)."""
    store = request.app.state.store
    with request.app.state.db.connect() as conn:
        target = conn.execute("SELECT id, username, created FROM users WHERE username = ?",
                              (username,)).fetchone()
        if target is None:
            raise HTTPException(404, "User not found.")
        captures = conn.execute(
            "SELECT c.*, u.username FROM captures c JOIN users u ON u.id = c.owner_id "
            "WHERE c.owner_id = ? AND c.is_public = 1 ORDER BY c.published_at DESC",
            (target["id"],),
        ).fetchall()
        posts = conn.execute(
            _POST_SELECT + " WHERE p.author_id = :author ORDER BY p.id DESC LIMIT 50",
            {"me": user["id"], "author": target["id"]},
        ).fetchall()
    return {
        "user": {"id": target["id"], "username": target["username"],
                 "joined": target["created"]},
        "captures": [capture_json(c, store.read_status(c["id"]))
                     for c in captures if store.exists(c["id"])],
        "posts": [_post_json(request, p) for p in posts],
    }
