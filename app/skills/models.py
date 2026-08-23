from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_SKILL_TOOLS = {"search", "browser", "rag", "mcp"}
SKILL_SCOPE_VALUES = {"global", "tenant", "project"}


class AgentHints(BaseModel):
    planner: str | None = None
    analyst: str | None = None
    reflection: str | None = None
    report: str | None = None
    browser: str | None = None
    search: str | None = None
    rag: str | None = None


class SkillMetadata(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    slug: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=400)
    version: int = Field(default=1, ge=1)
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    trigger_patterns: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    agent_hints: AgentHints = Field(default_factory=AgentHints)
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    domain: str | None = Field(default=None, max_length=100)
    tenant_id: str | None = Field(default=None, max_length=100)
    project_id: str | None = Field(default=None, max_length=100)
    scope: Literal["global", "tenant", "project"] = "global"

    @field_validator("slug")
    @classmethod
    def normalize_slug(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "-")
        if not normalized:
            raise ValueError("slug must not be empty")
        return normalized

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            tool_name = item.strip().lower()
            if tool_name not in ALLOWED_SKILL_TOOLS:
                raise ValueError(f"unsupported tool '{item}'")
            if tool_name not in normalized:
                normalized.append(tool_name)
        return normalized

    @field_validator("trigger_patterns", "tags", "keywords")
    @classmethod
    def normalize_string_lists(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            cleaned = item.strip().lower()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    @field_validator("domain", "tenant_id", "project_id")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        return cleaned or None

    @model_validator(mode="after")
    def validate_scope_binding(self) -> SkillMetadata:
        if self.scope == "global" and (self.tenant_id or self.project_id):
            raise ValueError("global skill must not set tenant_id or project_id")
        if self.scope == "tenant":
            if not self.tenant_id:
                raise ValueError("tenant scoped skill must set tenant_id")
            if self.project_id:
                raise ValueError("tenant scoped skill must not set project_id")
        if self.scope == "project" and (not self.tenant_id or not self.project_id):
            raise ValueError("project scoped skill must set tenant_id and project_id")
        return self


class SkillContentRecord(BaseModel):
    skill_id: str
    version: int
    markdown_content: str
    body: str
    prompt: str = ""
    constraints: str = ""
    overview: str = ""
    content_hash: str | None = None
    created_at: datetime


class SkillMetaRecord(BaseModel):
    id: str
    metadata: SkillMetadata
    coarse_terms: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False

    @property
    def version(self) -> int:
        return self.metadata.version

    def to_runtime_view(self, *, include_content: bool = False, content: SkillContentRecord | None = None) -> SkillRuntimeView:
        return SkillRuntimeView(
            id=self.id,
            name=self.metadata.name,
            slug=self.metadata.slug,
            description=self.metadata.description,
            version=self.metadata.version,
            enabled=self.metadata.enabled and not self.is_deleted,
            priority=self.metadata.priority,
            trigger_patterns=self.metadata.trigger_patterns,
            allowed_tools=self.metadata.allowed_tools,
            agent_hints=self.metadata.agent_hints.model_dump(),
            tags=self.metadata.tags,
            keywords=self.metadata.keywords,
            domain=self.metadata.domain,
            tenant_id=self.metadata.tenant_id,
            project_id=self.metadata.project_id,
            scope=self.metadata.scope,
            coarse_terms=self.coarse_terms,
            markdown_content=content.markdown_content if include_content and content else None,
            prompt=content.prompt if include_content and content else "",
            constraints=content.constraints if include_content and content else "",
            overview=content.overview if include_content and content else "",
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
        )


class SkillRecord(BaseModel):
    meta: SkillMetaRecord
    content: SkillContentRecord

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def metadata(self) -> SkillMetadata:
        return self.meta.metadata

    @property
    def created_at(self) -> datetime:
        return self.meta.created_at

    @property
    def updated_at(self) -> datetime:
        return self.meta.updated_at

    @property
    def is_deleted(self) -> bool:
        return self.meta.is_deleted

    def to_runtime_view(self) -> SkillRuntimeView:
        return self.meta.to_runtime_view(include_content=True, content=self.content)


class SkillRuntimeView(BaseModel):
    id: str
    name: str
    slug: str
    description: str
    version: int
    enabled: bool
    priority: int
    trigger_patterns: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    agent_hints: dict[str, str | None] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    domain: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    scope: str = "global"
    coarse_terms: list[str] = Field(default_factory=list)
    markdown_content: str | None = None
    prompt: str = ""
    constraints: str = ""
    overview: str = ""
    created_at: str
    updated_at: str


class SkillCreateRequest(BaseModel):
    markdown_content: str = Field(..., min_length=1)


class SkillUpdateRequest(BaseModel):
    markdown_content: str = Field(..., min_length=1)


class SkillMatchResult(BaseModel):
    skill_id: str
    slug: str
    name: str
    reason: str
    priority: int
    allowed_tools: list[str] = Field(default_factory=list)
    score: float = 0.0


class ResolvedSkillRef(BaseModel):
    skill_id: str
    slug: str
    name: str
    version: int
    priority: int
    matched_reason: str
    allowed_tools: list[str] = Field(default_factory=list)
    prompt_sections: dict[str, str] = Field(default_factory=dict)
    agent_hints: dict[str, list[str]] = Field(default_factory=dict)


class SessionSkillSelection(BaseModel):
    auto_matched_skill_ids: list[str] = Field(default_factory=list)
    manually_enabled_skill_ids: list[str] = Field(default_factory=list)
    manually_disabled_skill_ids: list[str] = Field(default_factory=list)
    effective_skill_ids: list[str] = Field(default_factory=list)
    effective_tool_allowlist: list[str] = Field(default_factory=list)
    match_results: list[SkillMatchResult] = Field(default_factory=list)
    effective_prompt_sections: dict[str, str] = Field(default_factory=dict)
    effective_agent_hints: dict[str, list[str]] = Field(default_factory=dict)
    resolved_skills: list[ResolvedSkillRef] = Field(default_factory=list)
    candidate_skill_ids: list[str] = Field(default_factory=list)
    scope_filters: dict[str, str | None] = Field(default_factory=dict)
    truncated_skill_count: int = 0
    snapshot_created_at: str | None = None
    match_cache_key: str | None = None
    registry_revision: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()
