from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.graph.state import AgentEvent, Evidence


class TaskIntent(str, Enum):
    SEARCH_COMPARE = "search_compare"
    FACT_LOOKUP = "fact_lookup"
    DOC_ANSWER = "doc_answer"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PromptProfile(str, Enum):
    RESEARCH_STRICT = "research_strict"
    RESEARCH_FACTS_ONLY = "research_facts_only"
    RESEARCH_ANALYSIS = "research_analysis"


class ResearchLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


RESEARCH_LENGTH_BUDGETS: dict[ResearchLength, dict[str, int | float]] = {
    ResearchLength.SHORT: {
        "report_max_tokens": 3000,
        "search_max_queries": 2,
        "search_max_results": 5,
        "rag_max_results": 4,
        "citation_max": 8,
        "browser_max_results": 1,
        "max_total_tokens": 20000,
        "max_cost_usd": 0.20,
        "max_tool_calls": 6,
        "max_wall_clock_seconds": 180,
        "max_retries_per_tool": 1,
    },
    ResearchLength.MEDIUM: {
        "report_max_tokens": 6000,
        "search_max_queries": 4,
        "search_max_results": 10,
        "rag_max_results": 8,
        "citation_max": 15,
        "browser_max_results": 2,
        "max_total_tokens": 60000,
        "max_cost_usd": 0.50,
        "max_tool_calls": 12,
        "max_wall_clock_seconds": 420,
        "max_retries_per_tool": 2,
    },
    ResearchLength.LONG: {
        "report_max_tokens": 12000,
        "search_max_queries": 6,
        "search_max_results": 15,
        "rag_max_results": 12,
        "citation_max": 30,
        "browser_max_results": 3,
        "max_total_tokens": 120000,
        "max_cost_usd": 1.00,
        "max_tool_calls": 24,
        "max_wall_clock_seconds": 900,
        "max_retries_per_tool": 3,
    },
}


def normalize_research_length(value: str | ResearchLength | None) -> ResearchLength:
    if isinstance(value, ResearchLength):
        return value
    if value:
        normalized = value.lower()
        if normalized in ResearchLength._value2member_map_:
            return ResearchLength(normalized)
    return ResearchLength.MEDIUM


def get_research_budget(value: str | ResearchLength | None) -> dict[str, int | float]:
    length = normalize_research_length(value)
    return dict(RESEARCH_LENGTH_BUDGETS[length])


BASE_GUARDRAIL_PREFIX = """你是一个研究型 Agent。
必须遵守：
1. 不知道就说不知道。
2. 只能基于工具结果和检索文档回答。
3. 文档没有提到的信息不要补充。
4. 结论必须给出来源或置信度。
5. 高风险动作必须先确认。
"""


class GuardrailDecision(BaseModel):
    intent: TaskIntent = TaskIntent.UNKNOWN
    risk_level: RiskLevel = RiskLevel.MEDIUM
    prompt_profile: PromptProfile = PromptProfile.RESEARCH_STRICT
    enabled_tools: list[str] = Field(default_factory=list)
    must_confirm: bool = False
    answer_gate_required: bool = True
    reject_if_no_evidence: bool = True


class EvidenceGateResult(BaseModel):
    allowed: bool
    reason: str = ""
    confidence: float = 0.0
    evidence_count: int = 0


def build_review_status(
    *,
    blocked: bool,
    requires_confirmation: bool = False,
    approved: bool | None = None,
    reason: str | None = None,
    risk_level: RiskLevel | str | None = None,
    intent: TaskIntent | str | None = None,
    prompt_profile: PromptProfile | str | None = None,
    evidence_gate: EvidenceGateResult | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "blocked": blocked,
        "requires_confirmation": requires_confirmation,
        "approved": approved if approved is not None else not blocked,
    }
    if reason:
        status["reason"] = reason
    if risk_level is not None:
        status["risk_level"] = risk_level.value if isinstance(risk_level, Enum) else str(risk_level)
    if intent is not None:
        status["intent"] = intent.value if isinstance(intent, Enum) else str(intent)
    if prompt_profile is not None:
        status["prompt_profile"] = (
            prompt_profile.value if isinstance(prompt_profile, Enum) else str(prompt_profile)
        )
    if evidence_gate is not None:
        status["evidence_gate"] = evidence_gate.model_dump()
    if verification is not None:
        status["verification"] = verification
    return status


class ToolInvocationSpec(BaseModel):
    tool_name: str
    args_schema: dict[str, Any] = Field(default_factory=dict)
    readonly: bool = True
    requires_confirmation: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    mcp_server_id: str | None = None


TOOL_SPECS: dict[str, ToolInvocationSpec] = {
    "duckduckgo_search": ToolInvocationSpec(
        tool_name="duckduckgo_search",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 2}},
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "browse_webpage": ToolInvocationSpec(
        tool_name="browse_webpage",
        args_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 10000},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        risk_level=RiskLevel.LOW,
    ),
    "knowledge_base_search": ToolInvocationSpec(
        tool_name="knowledge_base_search",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "group": {"type": "string", "minLength": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        risk_level=RiskLevel.LOW,
    ),
}


def should_require_action_approval(
    *,
    tool_name: str,
    decision: GuardrailDecision | None,
    readonly: bool = True,
    server_id: str | None = None,
) -> tuple[bool, str | None]:
    spec = TOOL_SPECS.get(tool_name)
    if spec and spec.requires_confirmation:
        return True, "tool_requires_confirmation"
    if decision and decision.risk_level == RiskLevel.HIGH:
        return True, "high_risk_task"
    if not readonly:
        return True, "write_operation_requires_approval"
    if server_id:
        return True, "unknown_mcp_server_requires_approval"
    return False, None


def build_guardrail_decision(query: str, *, user_confirmed: bool = False) -> GuardrailDecision:
    text = query.lower()
    intent = TaskIntent.UNKNOWN
    if any(keyword in text for keyword in ("对比", "比较", "分析", "研究", "market", "market analysis")):
        intent = TaskIntent.SEARCH_COMPARE
    elif any(keyword in text for keyword in ("是什么", "多少", "谁", "哪里", "什么时候")):
        intent = TaskIntent.FACT_LOOKUP
    elif any(keyword in text for keyword in ("政策", "规则", "文档", "公司", "内部")):
        intent = TaskIntent.DOC_ANSWER
    elif len(text) > 30:
        intent = TaskIntent.ANALYSIS

    risk_level = RiskLevel.LOW
    if any(keyword in text for keyword in ("删除", "退款", "发邮件", "转账", "修改", "写入", "提交")):
        risk_level = RiskLevel.HIGH
    elif intent in (TaskIntent.DOC_ANSWER, TaskIntent.ANALYSIS):
        risk_level = RiskLevel.MEDIUM

    profile = {
        TaskIntent.SEARCH_COMPARE: PromptProfile.RESEARCH_ANALYSIS,
        TaskIntent.FACT_LOOKUP: PromptProfile.RESEARCH_FACTS_ONLY,
        TaskIntent.DOC_ANSWER: PromptProfile.RESEARCH_STRICT,
        TaskIntent.ANALYSIS: PromptProfile.RESEARCH_ANALYSIS,
        TaskIntent.UNKNOWN: PromptProfile.RESEARCH_STRICT,
    }[intent]

    enabled_tools = ["search", "rag"]
    if intent in (TaskIntent.SEARCH_COMPARE, TaskIntent.ANALYSIS, TaskIntent.DOC_ANSWER):
        enabled_tools.append("browser")

    must_confirm = risk_level == RiskLevel.HIGH and not user_confirmed

    reject_if_no_evidence = intent != TaskIntent.FACT_LOOKUP

    return GuardrailDecision(
        intent=intent,
        risk_level=risk_level,
        prompt_profile=profile,
        enabled_tools=enabled_tools,
        must_confirm=must_confirm,
        answer_gate_required=True,
        reject_if_no_evidence=reject_if_no_evidence,
    )


def compose_guardrail_prompt(query: str, decision: GuardrailDecision) -> str:
    if decision.prompt_profile == PromptProfile.RESEARCH_FACTS_ONLY:
        body = "只回答可被检索证据直接支持的事实，不能扩写。"
    elif decision.prompt_profile == PromptProfile.RESEARCH_ANALYSIS:
        body = "允许做分析，但每个关键结论必须绑定证据。"
    else:
        body = "优先保守回答，证据不足时明确拒答。"
    return f"{BASE_GUARDRAIL_PREFIX}\n{body}\n\n研究问题：{query}"


def build_prompt_profile_message(decision: GuardrailDecision, query: str) -> str:
    return compose_guardrail_prompt(query, decision)


def validate_tool_invocation(tool_name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return False, f"tool_not_allowed:{tool_name}"

    schema = spec.args_schema
    try:
        _validate_schema(args, schema)
        return True, None
    except ValidationError as e:
        return False, f"schema_validation_failed:{e}"
    except ValueError as e:
        return False, str(e)


def is_tool_allowed(decision: GuardrailDecision | None, tool_name: str) -> bool:
    if decision is None:
        return True
    return tool_name in decision.enabled_tools


def build_evidence_gate(
    evidence_list: list[Evidence],
    *,
    minimum_evidence: int = 1,
    minimum_confidence: float = 0.5,
) -> EvidenceGateResult:
    if not evidence_list:
        return EvidenceGateResult(
            allowed=False,
            reason="no_evidence",
            confidence=0.0,
            evidence_count=0,
        )

    avg_conf = sum(ev.reliability for ev in evidence_list) / len(evidence_list)
    allowed = len(evidence_list) >= minimum_evidence and avg_conf >= minimum_confidence
    return EvidenceGateResult(
        allowed=allowed,
        reason="" if allowed else "insufficient_evidence",
        confidence=avg_conf,
        evidence_count=len(evidence_list),
    )


def build_answer_gate_message(evidence_gate: EvidenceGateResult) -> str:
    if evidence_gate.allowed:
        return "evidence_gate_passed"
    return f"拒答：{evidence_gate.reason}"


def _validate_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError("unsupported_schema_type")

    required = set(schema.get("required", []))
    props = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)

    missing = required - set(data)
    if missing:
        raise ValueError(f"missing_required_fields:{sorted(missing)}")

    if not additional:
        extra = set(data) - set(props)
        if extra:
            raise ValueError(f"extra_fields_not_allowed:{sorted(extra)}")

    for field_name, rules in props.items():
        if field_name not in data:
            continue
        value = data[field_name]
        expected_type = rules.get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"field_type_error:{field_name}")
        if expected_type == "integer" and not isinstance(value, int):
            raise ValueError(f"field_type_error:{field_name}")
        if "enum" in rules and value not in rules["enum"]:
            raise ValueError(f"enum_error:{field_name}")
        if expected_type == "string":
            if "minLength" in rules and len(value) < rules["minLength"]:
                raise ValueError(f"min_length_error:{field_name}")
            if rules.get("format") == "uri" and not re.match(r"^https?://", value):
                raise ValueError(f"uri_format_error:{field_name}")
        if expected_type == "integer":
            if "minimum" in rules and value < rules["minimum"]:
                raise ValueError(f"minimum_error:{field_name}")
            if "maximum" in rules and value > rules["maximum"]:
                raise ValueError(f"maximum_error:{field_name}")


def record_guardrail_event(
    state: dict[str, Any],
    *,
    agent: str,
    event_type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    event = AgentEvent(agent=agent, event_type=event_type, content=content)
    trace = state.setdefault("guardrail_trace", [])
    trace.append({
        **event.model_dump(),
        "metadata": metadata or {},
    })
