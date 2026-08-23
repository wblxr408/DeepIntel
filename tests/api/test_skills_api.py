from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.time import utc_now_naive
from app.main import app
from app.security.auth import Principal, require_principal
from app.skills.models import SkillContentRecord, SkillMetaRecord, SkillRecord
from app.skills.parser import derive_coarse_terms, parse_skill_markdown

SAMPLE_SKILL = """---
name: Equity Research
slug: equity-research
description: Equity and earnings focused research skill.
version: 1
enabled: true
priority: 150
trigger_patterns:
  - "财报"
allowed_tools:
  - search
  - rag
agent_hints:
  planner: Prioritize earnings call, filings, and guidance.
---

# Overview
Equity research domain instructions.

# Prompt
Focus on public company disclosures.

# Constraints
Avoid browser unless a filing or official page is necessary.
"""


@pytest.fixture(autouse=True)
def authenticated_api_client_context(monkeypatch):
    """Production routes require a verified principal; bypass only token lookup."""
    async def principal_override():
        return Principal(id="00000000-0000-0000-0000-000000000001", username="test-admin", auth_type="test")
    # These route tests replace database initialization; the startup-time runtime
    # config restore must be isolated too, otherwise TestClient can reuse a pool
    # created on an already-closed event loop.
    monkeypatch.setattr("app.main.load_runtime_llm_config_from_db", AsyncMock(return_value=None))
    app.dependency_overrides[require_principal] = principal_override
    yield
    app.dependency_overrides.pop(require_principal, None)


AUTH_HEADERS = {"Authorization": "Bearer test-token"}


def _make_skill(skill_id: str = "00000000-0000-0000-0000-000000000001") -> SkillRecord:
    metadata, body, sections = parse_skill_markdown(SAMPLE_SKILL)
    now = utc_now_naive()
    meta = SkillMetaRecord(
        id=skill_id,
        metadata=metadata,
        coarse_terms=derive_coarse_terms(metadata),
        created_at=now,
        updated_at=now,
        is_deleted=False,
    )
    content = SkillContentRecord(
        skill_id=skill_id,
        version=metadata.version,
        markdown_content=SAMPLE_SKILL,
        body=body,
        prompt=sections.get("prompt", ""),
        constraints=sections.get("constraints", ""),
        overview=sections.get("overview", ""),
        created_at=now,
    )
    return SkillRecord(
        meta=meta,
        content=content,
    )


class _FakeRegistry:
    def __init__(self, skills: list[SkillRecord] | None = None) -> None:
        self.skills = skills or [_make_skill()]
        self.upsert_called = 0
        self.remove_called = 0
        self.load_called = 0
        self.loaded = False

    async def load(self) -> None:
        self.load_called += 1
        self.loaded = True

    async def ensure_loaded(self) -> None:
        if not self.loaded:
            await self.load()

    async def upsert(self, skill: SkillRecord) -> None:
        self.upsert_called += 1

    async def remove(self, skill_id: str) -> None:
        self.remove_called += 1

    def list_runtime_skills(self) -> list[SkillMetaRecord]:
        return [item.meta for item in self.skills]

    async def get_skill_record(self, skill_id: str) -> SkillRecord:
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        raise AssertionError("skill not found in fake registry")


def test_list_skills_endpoint(monkeypatch):
    registry = _FakeRegistry()
    monkeypatch.setattr("app.main.init_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.close_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.get_skill_registry", lambda: registry)
    monkeypatch.setattr("app.api.skills.get_skill_registry", lambda owner_id: registry)

    with TestClient(app) as client:
        response = client.get("/api/v1/skills", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["slug"] == "equity-research"
    assert payload["items"][0]["markdown_content"] is None
    assert payload["items"][0]["prompt"] == ""
    assert registry.load_called == 1


def test_get_skill_endpoint_returns_markdown_content(monkeypatch):
    registry = _FakeRegistry()
    skill = registry.skills[0]
    monkeypatch.setattr("app.main.init_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.close_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.get_skill_registry", lambda: registry)
    monkeypatch.setattr("app.api.skills.get_skill_registry", lambda owner_id: registry)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/skills/{skill.id}", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == skill.id
    assert payload["markdown_content"] == SAMPLE_SKILL
    assert "Focus on public company disclosures." in payload["prompt"]
    assert "Avoid browser unless a filing" in payload["constraints"]


def test_create_update_delete_and_upload_skill_endpoints(monkeypatch):
    registry = _FakeRegistry()
    created_skill = _make_skill("00000000-0000-0000-0000-000000000002")
    updated_skill = _make_skill("00000000-0000-0000-0000-000000000002")

    async def fake_create(markdown_content: str, owner_id: str):
        assert owner_id == "00000000-0000-0000-0000-000000000001"
        assert "Equity Research" in markdown_content
        return created_skill

    async def fake_update(skill_id: str, markdown_content: str, owner_id: str):
        assert owner_id == "00000000-0000-0000-0000-000000000001"
        assert skill_id == created_skill.id
        assert "Focus on public company disclosures." in markdown_content
        return updated_skill

    async def fake_delete(skill_id: str, owner_id: str):
        assert owner_id == "00000000-0000-0000-0000-000000000001"
        assert skill_id == created_skill.id

    monkeypatch.setattr("app.main.init_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.close_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.get_skill_registry", lambda: registry)
    monkeypatch.setattr("app.api.skills.get_skill_registry", lambda owner_id: registry)
    monkeypatch.setattr("app.api.skills.create_skill", fake_create)
    monkeypatch.setattr("app.api.skills.update_skill", fake_update)
    monkeypatch.setattr("app.api.skills.delete_skill", fake_delete)

    with TestClient(app) as client:
        create_response = client.post("/api/v1/skills", json={"markdown_content": SAMPLE_SKILL}, headers=AUTH_HEADERS)
        update_response = client.put(f"/api/v1/skills/{created_skill.id}", json={"markdown_content": SAMPLE_SKILL}, headers=AUTH_HEADERS)
        delete_response = client.delete(f"/api/v1/skills/{created_skill.id}", headers=AUTH_HEADERS)
        upload_response = client.post(
            "/api/v1/skills/upload",
            files={"file": ("equity-research.md", BytesIO(SAMPLE_SKILL.encode("utf-8")), "text/markdown")},
            headers=AUTH_HEADERS,
        )

    assert create_response.status_code == 200
    assert update_response.status_code == 200
    assert delete_response.status_code == 200
    assert upload_response.status_code == 200
    assert registry.upsert_called == 3
    assert registry.remove_called == 1
    assert upload_response.json()["slug"] == "equity-research"


def test_upload_skill_rejects_non_markdown_files(monkeypatch):
    registry = _FakeRegistry()
    monkeypatch.setattr("app.main.init_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.close_db", AsyncMock(return_value=None))
    monkeypatch.setattr("app.main.get_skill_registry", lambda: registry)
    monkeypatch.setattr("app.api.skills.get_skill_registry", lambda owner_id: registry)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/skills/upload",
            files={"file": ("equity-research.txt", BytesIO(b"not markdown"), "text/plain")},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .md skill files are supported"
    assert registry.upsert_called == 0
