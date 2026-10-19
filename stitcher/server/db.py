"""SQLite storage for users, sessions, captures and the community (standard library only)."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,           -- sha256 of the bearer token; the token is never stored
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created    REAL NOT NULL,
    expires    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
    id           TEXT PRIMARY KEY,         -- also the job directory name
    owner_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT '',
    created      REAL NOT NULL,
    is_public    INTEGER NOT NULL DEFAULT 0,
    published_at REAL
);
CREATE INDEX IF NOT EXISTS captures_owner ON captures(owner_id, created);
CREATE TABLE IF NOT EXISTS posts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body       TEXT NOT NULL,
    capture_id TEXT REFERENCES captures(id) ON DELETE SET NULL,
    created    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS posts_author ON posts(author_id, id);
CREATE TABLE IF NOT EXISTS comments (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id   INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body      TEXT NOT NULL,
    created   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS comments_post ON comments(post_id, id);
CREATE TABLE IF NOT EXISTS likes (
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created REAL NOT NULL,
    PRIMARY KEY (post_id, user_id)
);
"""


class Database:
    """Thin wrapper: one connection per operation, writes serialised by a lock."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._write_lock = threading.Lock()
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as conn:
            yield conn
