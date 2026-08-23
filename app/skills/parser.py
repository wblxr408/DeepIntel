from __future__ import annotations

import re
from typing import Any

import yaml

from app.skills.models import SkillMetadata

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
SECTION_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[a-z0-9_\-\u4e00-\u9fff]{2,}", re.IGNORECASE)


class SkillParseError(ValueError):
    pass


def parse_skill_markdown(markdown_content: str) -> tuple[SkillMetadata, str, dict[str, str]]:
    text = markdown_content.strip()
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        raise SkillParseError("Skill markdown must start with YAML frontmatter delimited by ---")

    frontmatter_text, body = match.groups()
    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(f"Invalid YAML frontmatter: {exc}") from exc

    if not isinstance(frontmatter, dict):
        raise SkillParseError("YAML frontmatter must be a mapping object")

    metadata = SkillMetadata.model_validate(frontmatter)
    sections = _parse_sections(body)
    return metadata, body.strip(), sections


def _parse_sections(body: str) -> dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(body))
    if not matches:
        return {"overview": body.strip()}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def build_skill_markdown(
    *,
    metadata: dict[str, Any],
    body: str,
) -> str:
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    normalized_body = body.strip()
    return f"---\n{frontmatter}\n---\n\n{normalized_body}\n"


def derive_coarse_terms(metadata: SkillMetadata) -> list[str]:
    terms: list[str] = []

    for item in metadata.tags + metadata.keywords:
        _append_term(terms, item)
    if metadata.domain:
        _append_term(terms, metadata.domain)

    for pattern in metadata.trigger_patterns:
        literal_candidate = re.sub(r"[\^\$\.\*\+\?\(\)\[\]\{\}\|\\]", " ", pattern)
        for token in TOKEN_PATTERN.findall(literal_candidate.lower()):
            _append_term(terms, token)

    return terms


def tokenize_query(query: str) -> list[str]:
    seen: list[str] = []
    for token in TOKEN_PATTERN.findall(query.lower()):
        _append_term(seen, token)
    return seen


def _append_term(container: list[str], value: str) -> None:
    cleaned = value.strip().lower()
    if len(cleaned) < 2:
        return
    if cleaned not in container:
        container.append(cleaned)
