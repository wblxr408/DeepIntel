from __future__ import annotations

from typing import Any

from app.guardrails.policy import (
    BASE_GUARDRAIL_PREFIX,
    EvidenceGateResult,
    GuardrailDecision,
    PromptProfile,
    ResearchLength,
    RiskLevel,
    TaskIntent,
    get_research_budget,
    normalize_research_length,
)

__all__ = [
    "BASE_GUARDRAIL_PREFIX",
    "EvidenceGateResult",
    "GuardrailDecision",
    "PromptProfile",
    "ResearchLength",
    "RiskLevel",
    "TaskIntent",
    "build_answer_gate_message",
    "build_evidence_gate",
    "build_guardrail_decision",
    "build_prompt_profile_message",
    "build_review_status",
    "compose_guardrail_prompt",
    "get_research_budget",
    "is_tool_allowed",
    "normalize_research_length",
    "record_guardrail_event",
    "should_require_action_approval",
    "validate_tool_invocation",
]


def __getattr__(name: str) -> Any:
    if name in {
        "build_answer_gate_message",
        "build_evidence_gate",
        "build_review_status",
        "build_guardrail_decision",
        "build_prompt_profile_message",
        "compose_guardrail_prompt",
        "get_research_budget",
        "is_tool_allowed",
        "normalize_research_length",
        "record_guardrail_event",
        "should_require_action_approval",
        "validate_tool_invocation",
    }:
        from app.guardrails import policy

        return getattr(policy, name)
    raise AttributeError(name)
