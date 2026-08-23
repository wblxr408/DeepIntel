"""Small compatibility adapters for providers that are not OpenAI wire-compatible."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx


class _AnthropicCompletions:
    def __init__(self, client: httpx.Client, base_url: str, headers: dict[str, str]) -> None:
        self._client = client
        self._base_url = base_url
        self._headers = headers

    def create(self, *, model: str, messages: list[dict[str, Any]], stream: bool = False, max_tokens: int = 1024, temperature: float | None = None, **_: Any) -> Any:
        system = "\n\n".join(str(item.get("content", "")) for item in messages if item.get("role") == "system")
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": item.get("role", "user"), "content": item.get("content", "")} for item in messages if item.get("role") != "system"],
            "stream": stream,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if stream:
            return self._stream(body)
        response = self._client.post(f"{self._base_url}/v1/messages", headers=self._headers, json=body)
        response.raise_for_status()
        payload = response.json()
        content = "".join(item.get("text", "") for item in payload.get("content", []) if item.get("type") == "text")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=payload.get("usage", {}).get("input_tokens"),
                completion_tokens=payload.get("usage", {}).get("output_tokens"),
            ),
        )

    def _stream(self, body: dict[str, Any]) -> Iterator[Any]:
        with self._client.stream("POST", f"{self._base_url}/v1/messages", headers=self._headers, json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                delta = payload.get("delta", {}).get("text", "") if payload.get("type") == "content_block_delta" else ""
                if delta:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])


class AnthropicCompatibleClient:
    """Expose Anthropic Messages as the limited chat.completions surface agents use."""
    def __init__(self, http_client: httpx.Client, base_url: str, headers: dict[str, str]) -> None:
        self.chat = SimpleNamespace(completions=_AnthropicCompletions(http_client, base_url, headers))
        self.base_url = base_url
