"""
Analyst Agent: reasoning, synthesis, and comparison of collected evidence.

Produces a structured analysis with evidence attribution.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.graph.state import Evidence
from app.guardrails import build_guardrail_decision, build_prompt_profile_message
from app.llm_client import (
    call_chat_with_fallback,
    collect_usage_metrics,
    create_fallback_llm_client,
)
from app.security.prompt_security import untrusted_context

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Synthesizes evidence from multiple agents into a coherent analysis.

    Uses structured output to ensure parseable, evidence-linked analysis.
    """

    SYSTEM_PROMPT = """You are a senior business research analyst. Your role is to synthesize diverse evidence into a structured, well-reasoned analysis.

## Analysis Framework

1. **Fact vs. Inference**: Clearly distinguish hard facts from analyst inferences
2. **Evidence Attribution**: Every key claim must cite its source using [citation:N] format
3. **Conflict Resolution**: Identify contradictions between sources and provide judgment
4. **Confidence Assessment**: Rate each major finding as HIGH/MEDIUM/LOW confidence
5. **Completeness Check**: Ensure all research dimensions are covered

## Response Format

Return a JSON object with these fields:
- "findings": array of {finding, evidence_ids, confidence, category}
- "data_points": array of {value, source_citation, context}
- "conflicts": array of {claim_a, claim_b, resolution}
- "gaps": array of {topic, severity} for areas needing more research
- "analysis_text": comprehensive narrative analysis (main output)
"""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model
        self._client: OpenAI | None = None
        self.last_usage: dict | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            from app.llm_client import create_llm_client, get_llm_model
            self._client = create_llm_client()
            self.model = get_llm_model()
        return self._client

    def analyze(
        self,
        user_query: str,
        evidence_list: list[Evidence],
        skill_prompt: str | None = None,
        analyst_hints: list[str] | None = None,
    ) -> str:
        """
        Generate analysis from collected evidence.

        Args:
            user_query: The original research question
            evidence_list: All evidence collected by sub-agents

        Returns:
            Analysis text (Markdown format)
        """
        logger.info(f"Analyst: analyzing {len(evidence_list)} evidence items")
        decision = build_guardrail_decision(user_query)
        system_prompt = f"{build_prompt_profile_message(decision, user_query)}\n\n{self.SYSTEM_PROMPT}"
        if analyst_hints:
            system_prompt = (
                f"{system_prompt}\n\n## Analyst Skill Instructions\n"
                + "\n".join(f"- {item}" for item in analyst_hints if item and item.strip())
            )
        if not evidence_list and not decision.reject_if_no_evidence:
            system_prompt += (
                "\n\n当前没有检索到外部证据。你可以基于模型已有知识回答简单事实问题，"
                "但必须明确标注“未检索验证”，不要编造来源、引用或实时数据。"
            )

        # Format evidence for the prompt
        formatted_evidence = self._format_evidence(evidence_list)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Research Question: {user_query}

{untrusted_context('skill_content', skill_prompt or '') if skill_prompt else ''}

Collected Evidence:
{formatted_evidence}

Please provide a comprehensive analysis addressing the research question. Use [citation:N] format to attribute claims to evidence items (e.g., [citation:1] for the first evidence item).

Also provide the structured JSON output."""
            },
        ]

        try:
            response, response_model, fallback_used = call_chat_with_fallback(
                self.client,
                self.model,
                create_fallback_llm_client(),
                messages=messages,
                temperature=0.4,
                max_tokens=4096,
            )
            if fallback_used:
                self.model = response_model
            content = response.choices[0].message.content
            self.last_usage = collect_usage_metrics(
                response=response,
                model=response_model,
                messages=messages,
                completion_text=content,
            )

            if not content:
                return self._fallback_analysis(user_query, evidence_list)

            # Try to parse structured output
            try:
                # Extract JSON if present
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    structured = json.loads(json_str)
                    logger.info(f"Analyst: parsed {len(structured.get('findings', []))} findings")
                else:
                    structured = {}
            except json.JSONDecodeError:
                structured = {}

            # Return the narrative text portion
            if structured.get("analysis_text"):
                return structured["analysis_text"]

            # Fallback: return the full response
            return content

        except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Analyst error: {e}")
            self.last_usage = None
            return self._fallback_analysis(user_query, evidence_list)

    def _format_evidence(self, evidence_list: list[Evidence]) -> str:
        """Format evidence list for the prompt."""
        lines = []
        for i, ev in enumerate(evidence_list[:20], 1):  # Limit to 20 for token budget
            source_info = f"[Source: {ev.source_title}]" if ev.source_title else ""
            url_info = f"({ev.source_url})" if ev.source_url else ""
            lines.append(
                f"[{i}] {source_info} {url_info}\n"
                f"Type: {ev.source_type} | Agent: {ev.agent_type.value}\n"
                f"Content: {ev.content[:500]}"
                + ("..." if len(ev.content) > 500 else "")
            )
        return "\n\n".join(lines)

    def _fallback_analysis(self, user_query: str, evidence_list: list[Evidence]) -> str:
        """Generate a basic analysis when the LLM fails."""
        lines = [f"# Research Analysis: {user_query}\n"]

        if evidence_list:
            lines.append(f"\nBased on {len(evidence_list)} evidence items:\n")
            for i, ev in enumerate(evidence_list[:5], 1):
                lines.append(f"- {ev.content[:200]}...")
        else:
            lines.append("\nNo sufficient evidence collected for analysis. I don't know.")

        return "\n".join(lines)
