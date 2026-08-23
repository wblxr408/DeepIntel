"""Normalize LLM provider endpoint and authentication differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EndpointConfig:
    provider: str
    base_url: str
    headers: dict[str, str]
    openai_compatible: bool


DEFAULT_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "anthropic": "https://api.anthropic.com",
}


def resolve_endpoint(provider: str, api_key: str, api_base: str | None = None) -> EndpointConfig:
    normalized = provider.strip().lower()
    if normalized not in DEFAULT_BASE_URLS:
        raise ValueError(f"unsupported_llm_provider:{provider}")
    base_url = (api_base or DEFAULT_BASE_URLS[normalized]).rstrip("/")
    if normalized == "anthropic":
        return EndpointConfig(
            provider=normalized,
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            openai_compatible=False,
        )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return EndpointConfig(provider=normalized, base_url=base_url, headers=headers, openai_compatible=True)


def provider_catalog() -> list[dict[str, Any]]:
    return [
        {"id": "qwen", "name": "Qwen", "api_base_default": DEFAULT_BASE_URLS["qwen"]},
        {"id": "deepseek", "name": "DeepSeek", "api_base_default": DEFAULT_BASE_URLS["deepseek"]},
        {"id": "openai", "name": "OpenAI", "api_base_default": DEFAULT_BASE_URLS["openai"]},
        {"id": "ollama", "name": "Ollama", "api_base_default": DEFAULT_BASE_URLS["ollama"]},
        {"id": "anthropic", "name": "Anthropic", "api_base_default": DEFAULT_BASE_URLS["anthropic"]},
    ]
