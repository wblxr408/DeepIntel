"""
LangGraph StateGraph workflow for autonomous research.

Keep package imports lazy to avoid circular import during app startup.
"""

from __future__ import annotations

from typing import Any

from app.graph.state import (
    AgentEvent,
    AgentType,
    BrowserResult,
    Citation,
    ClaimConflict,
    ClaimEvidence,
    DAGDefinition,
    ErrorRecord,
    Evidence,
    HallucinatedClaim,
    PageType,
    PlanEdge,
    PlanNode,
    PlanStep,
    RAGResult,
    ReflectionResult,
    ResearchState,
    SearchResult,
    SessionMetadata,
    StepStatus,
    TaskStatus,
    ToolCallRecord,
    ToolInvocationHistory,
    VerificationDimension,
    create_initial_state,
    deserialize_dag,
    deserialize_evidence,
    deserialize_steps,
    serialize_dag,
)

__all__ = [
    "AgentEvent",
    "AgentType",
    "BrowserResult",
    "Citation",
    "ClaimConflict",
    "ClaimEvidence",
    "DAGDefinition",
    "ErrorRecord",
    "Evidence",
    "HallucinatedClaim",
    "PageType",
    "PlanEdge",
    "PlanNode",
    "PlanStep",
    "RAGResult",
    "ReflectionResult",
    "ResearchState",
    "SearchResult",
    "SessionMetadata",
    "StepStatus",
    "TaskStatus",
    "ToolCallRecord",
    "ToolInvocationHistory",
    "VerificationDimension",
    "VerificationResult",
    "compile_research_graph",
    "create_initial_state",
    "deserialize_dag",
    "deserialize_evidence",
    "deserialize_steps",
    "serialize_dag",
]


def __getattr__(name: str) -> Any:
    if name == "compile_research_graph":
        from app.graph.compiler import compile_research_graph

        return compile_research_graph
    if name == "VerificationResult":
        return ReflectionResult
    raise AttributeError(name)
