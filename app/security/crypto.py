"""Encryption for secrets persisted in the application database.

This protects a stolen database file or backup.  It does not claim to protect a
host where an attacker can read the process environment or application memory.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

PREFIX = "enc:"


def _fernet() -> Fernet:
    raw_key = get_settings().security.encryption_key
    if not raw_key:
        raise RuntimeError("SECURITY_ENCRYPTION_KEY is required for persisted secrets")
    # A human-managed deployment secret can be any sufficiently long string.
    key = base64.urlsafe_b64encode(hashlib.sha256(raw_key.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_secret(value: str) -> str:
    if not value or value.startswith(PREFIX):
        return value
    return PREFIX + _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    if not value.startswith(PREFIX):
        return value
    try:
        return _fernet().decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError("Persisted secret cannot be decrypted with SECURITY_ENCRYPTION_KEY") from exc
