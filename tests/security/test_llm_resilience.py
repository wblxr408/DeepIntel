from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.report import ReportAgent
from app.llm_client import StreamInterruptedError, call_chat_with_fallback


def _client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_non_streaming_request_falls_back_before_a_response():
    primary = _client(lambda **kwargs: (_ for _ in ()).throw(ConnectionError("primary offline")))
    fallback = _client(lambda **kwargs: "fallback response")

    response, model, used_fallback = call_chat_with_fallback(
        primary, "primary", (fallback, "fallback"), messages=[]
    )

    assert response == "fallback response"
    assert model == "fallback"
    assert used_fallback is True


def test_report_stream_falls_back_only_before_first_text(monkeypatch):
    agent = ReportAgent()
    agent._client = _client(lambda **kwargs: (_ for _ in ()).throw(ConnectionError("primary offline")))
    fallback_event = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="fallback text"))])
    fallback = _client(lambda **kwargs: iter([fallback_event]))
    monkeypatch.setattr("app.agents.report.create_fallback_llm_client", lambda: (fallback, "fallback"))

    content, _ = agent.generate_stream("test topic", "analysis", [], None)

    assert "fallback text" in content


def test_report_stream_never_switches_after_emitting_text(monkeypatch):
    agent = ReportAgent()

    def interrupted_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="primary text"))])
        raise ConnectionError("stream lost")

    agent._client = _client(lambda **kwargs: interrupted_stream())
    monkeypatch.setattr(
        "app.agents.report.create_fallback_llm_client",
        lambda: (_ for _ in ()).throw(AssertionError("fallback must not be called after output")),
    )

    with pytest.raises(StreamInterruptedError):
        agent.generate_stream("test topic", "analysis", [], None)
