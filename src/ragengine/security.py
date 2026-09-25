"""Password hashing (scrypt, stdlib) and JWT access/refresh tokens (PyJWT, HS256)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

_N, _R, _P = 2**14, 8, 1  # scrypt cost parameters (~16 MiB memory per hash)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    enc = base64.b64encode
    return f"scrypt${_N}${_R}${_P}${enc(salt).decode()}${enc(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b64, digest_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        digest = hashlib.scrypt(
            password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)  # constant time: no timing side channel


class TokenError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    role: str
    token_type: str


class TokenService:
    def __init__(self, secret: str, access_ttl: timedelta, refresh_ttl: timedelta) -> None:
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        self._secret = secret
        self.access_ttl, self.refresh_ttl = access_ttl, refresh_ttl

    def _encode(self, claims: TokenClaims, ttl: timedelta) -> str:
        now = datetime.now(UTC)
        payload: dict[str, Any] = {
            "sub": str(claims.user_id),
            "tid": str(claims.tenant_id),
            "role": claims.role,
            "typ": claims.token_type,
            "iat": now,
            "exp": now + ttl,
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def issue(self, user_id: uuid.UUID, tenant_id: uuid.UUID, role: str) -> dict[str, str]:
        return {
            "access_token": self._encode(
                TokenClaims(user_id, tenant_id, role, "access"), self.access_ttl
            ),
            "refresh_token": self._encode(
                TokenClaims(user_id, tenant_id, role, "refresh"), self.refresh_ttl
            ),
            "token_type": "bearer",
        }

    def decode(self, token: str, expected_type: str) -> TokenClaims:
        try:
            # algorithms is an allow-list: never accept "none" or a key-confusion algorithm
            data = jwt.decode(token, self._secret, algorithms=["HS256"])
        except jwt.PyJWTError as exc:
            raise TokenError(str(exc)) from exc
        if data.get("typ") != expected_type:
            raise TokenError(f"expected a {expected_type} token")
        try:
            return TokenClaims(
                uuid.UUID(data["sub"]), uuid.UUID(data["tid"]), data["role"], data["typ"]
            )
        except (KeyError, ValueError) as exc:
            raise TokenError("malformed token claims") from exc
