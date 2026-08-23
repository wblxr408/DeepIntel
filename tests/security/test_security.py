from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.config import SecuritySettings
from app.security import capabilities
from app.security.auth import hash_password, token_hash, verify_password
from app.security.crypto import decrypt_secret, encrypt_secret


def _settings():
    return SimpleNamespace(
        security=SimpleNamespace(encryption_key="test-encryption-key", session_ttl_seconds=3600),
        resilience=SimpleNamespace(capability_retry_seconds=0),
    )


def test_password_hash_is_bcrypt_and_verifiable():
    encoded = hash_password("a sufficiently strong test password")
    assert encoded.startswith("$2")
    assert verify_password("a sufficiently strong test password", encoded)
    assert not verify_password("wrong password", encoded)


def test_secret_encryption_is_idempotent_and_not_plaintext(monkeypatch):
    monkeypatch.setattr("app.security.crypto.get_settings", _settings)
    encrypted = encrypt_secret("sk-secret-value")
    assert encrypted.startswith("enc:")
    assert "sk-secret-value" not in encrypted
    assert encrypt_secret(encrypted) == encrypted
    assert decrypt_secret(encrypted) == "sk-secret-value"


def test_token_hash_is_stable_and_does_not_reveal_token():
    digest = token_hash("di_top-secret")
    assert digest == token_hash("di_top-secret")
    assert len(digest) == 64
    assert "top-secret" not in digest


@pytest.mark.asyncio
async def test_capability_cools_down_and_recovers(monkeypatch):
    registry = capabilities.CapabilityRegistry()
    monkeypatch.setattr("app.security.capabilities.get_settings", _settings)

    async def broken():
        raise ConnectionError("offline")

    with pytest.raises(capabilities.CapabilityUnavailable) as first:
        await registry.run("embedding", broken)
    assert first.value.status.state == capabilities.CapabilityHealth.DEGRADED

    async def healthy():
        return "ok"

    await asyncio.sleep(0)
    assert await registry.run("embedding", healthy) == "ok"
    assert registry.status("embedding").state == capabilities.CapabilityHealth.HEALTHY


def test_production_security_rejects_missing_or_placeholder_encryption_key():
    security = SecuritySettings(environment="production", encryption_key="replace-with-a-secret")
    with pytest.raises(ValueError, match="ENCRYPTION"):
        security.validate_production_requirements(["https://deepintel.example"])


def test_production_security_rejects_insecure_cookie_and_wildcard_cors():
    security = SecuritySettings(environment="production", encryption_key="real-secret", secure_cookies=False)
    with pytest.raises(ValueError, match="COOKIES"):
        security.validate_production_requirements(["https://deepintel.example"])

    security = SecuritySettings(environment="production", encryption_key="real-secret")
    with pytest.raises(ValueError, match="CORS"):
        security.validate_production_requirements(["*"])
