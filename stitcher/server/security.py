"""Password hashing, session tokens and a simple login throttle."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from typing import Dict, List

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32
    )
    return hmac.compare_digest(digest.hex(), expected)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class LoginThrottle:
    """Lock a username out for ``lockout_s`` after ``max_failures`` failures in that window."""

    def __init__(self, max_failures: int = 5, lockout_s: float = 60.0) -> None:
        self.max_failures = max_failures
        self.lockout_s = lockout_s
        self._failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> List[float]:
        return [t for t in self._failures.get(key, []) if now - t < self.lockout_s]

    def blocked(self, username: str) -> bool:
        with self._lock:
            return len(self._recent(username.lower(), time.time())) >= self.max_failures

    def failed(self, username: str) -> None:
        key, now = username.lower(), time.time()
        with self._lock:
            self._failures[key] = self._recent(key, now) + [now]

    def succeeded(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username.lower(), None)
