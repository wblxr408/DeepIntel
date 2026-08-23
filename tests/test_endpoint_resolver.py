from __future__ import annotations

import pytest

from app.endpoint_resolver import resolve_endpoint


@pytest.mark.parametrize("provider", ["qwen", "deepseek", "openai", "ollama"])
def test_openai_compatible_provider_resolution(provider: str):
    endpoint = resolve_endpoint(provider, "key")
    assert endpoint.openai_compatible
    assert endpoint.headers["Authorization"] == "Bearer key"
    assert endpoint.base_url


def test_anthropic_uses_native_auth_headers():
    endpoint = resolve_endpoint("anthropic", "key")
    assert not endpoint.openai_compatible
    assert endpoint.headers["x-api-key"] == "key"
    assert endpoint.headers["anthropic-version"]


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unsupported_llm_provider"):
        resolve_endpoint("unknown", "key")
