"""
Reflection Agent: hallucination detection and evidence validation.

Performs multi-dimensional quality checks on the analysis.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.graph.state import (
    ClaimConflict,
    Evidence,
    HallucinatedClaim,
    ReflectionResult,
)
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


class ReflectionAgent:
    """
    Validates analysis quality through multiple dimensions:

    1. Fact Verification: Cross-reference claims with evidence
    2. Numerical Accuracy: Check if numbers match sources
    3. Temporal Validity: Ensure data is within reasonable timeframe
    4. Consistency: Detect self-contradicting claims
    5. Completeness: Check coverage of research dimensions
    """

    SYSTEM_PROMPT = """You are a critical fact-checker and quality auditor for AI research outputs. Your job is to rigorously validate analysis quality.

## Validation Dimensions

### 1. Fact Verification
Check each claim against provided evidence. Flag as hallucination if:
- The claim directly contradicts evidence
- The claim makes specific claims (numbers, dates, names) not present in any evidence
- The claim is an overgeneralization not supported by evidence

### 2. Numerical Accuracy
Extract all numbers/dates from the analysis and verify they match evidence sources.

### 3. Temporal Validity
Ensure statistical claims are from reasonable timeframes (generally within 2 years for market data).

### 4. Self-Consistency
Detect if the analysis contains contradictory statements.

### 5. Citation Coverage
Calculate what percentage of claims have supporting citations.

## Response Format

Return JSON:
{
  "total_claims": <int>,
  "verified_claims": <int>,
  "hallucinated_claims": [
    {
      "claim": "<exact text of hallucinated claim>",
      "severity": "high|medium|low",
      "reason": "<why it is flagged>",
      "suggested_action": "<how to fix>"
    }
  ],
  "conflicts": [
    {
      "claim_a": "<text>",
      "claim_b": "<text>",
      "conflict_description": "<explanation>"
    }
  ],
  "citation_coverage": <float 0-1>,
  "overall_confidence": <float 0-1>,
  "needs_revision": <bool>,
  "revision_focus": "<specific guidance for revision>" or null
}

Be strict but fair. Flag real hallucinations but don't over-flag.
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

    def reflect(
        self,
        user_query: str,
        analysis: str,
        evidence_list: list[Evidence],
        skill_prompt: str | None = None,
        reflection_hints: list[str] | None = None,
    ) -> ReflectionResult:
        """
        Perform reflection on the analysis.

        Args:
            user_query: Original research question
            analysis: The analysis text to validate
            evidence_list: Evidence supporting the analysis

        Returns:
            ReflectionResult with quality metrics
        """
        logger.info(f"Reflection: validating analysis ({len(analysis)} chars, {len(evidence_list)} evidence items)")
        decision = build_guardrail_decision(user_query)
        system_prompt = f"{build_prompt_profile_message(decision, user_query)}\n\n{self.SYSTEM_PROMPT}"
        if reflection_hints:
            system_prompt = (
                f"{system_prompt}\n\n## Reflection Skill Instructions\n"
                + "\n".join(f"- {item}" for item in reflection_hints if item and item.strip())
            )

        formatted_evidence = self._format_evidence(evidence_list)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Research Question: {user_query}

Analysis to Validate:
{untrusted_context('analyst_output', analysis)}

{untrusted_context('skill_content', skill_prompt or '') if skill_prompt else ''}

Evidence Sources:
{untrusted_context('external_evidence', formatted_evidence)}

Perform a rigorous quality check and return the structured JSON output."""
            },
        ]

        try:
            response, response_model, fallback_used = call_chat_with_fallback(
                self.client,
                self.model,
                create_fallback_llm_client(),
                messages=messages,
                temperature=0.1,  # Low temp for consistency
                max_tokens=2048,
                response_format={"type": "json_object"},
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
                return self._default_result()

            data = json.loads(content)

            # Parse hallucinated claims
            hallucinated = []
            for h in data.get("hallucinated_claims", []):
                try:
                    hallucinated.append(HallucinatedClaim(
                        claim=h.get("claim", ""),
                        severity=h.get("severity", "medium"),
                        reason=h.get("reason", ""),
                        suggested_fix=h.get("suggested_action") or h.get("suggested_fix", ""),
                    ))
                except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
                    continue

            # Parse conflicts
            conflicts = []
            for c in data.get("conflicts", []):
                try:
                    conflicts.append(ClaimConflict(
                        claim_a=c.get("claim_a", ""),
                        claim_b=c.get("claim_b", ""),
                        conflict_description=c.get("conflict_description", ""),
                    ))
                except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
                    continue

            result = ReflectionResult(
                total_claims=data.get("total_claims", 0),
                verified_claims=data.get("verified_claims", 0),
                hallucinated_claims=hallucinated,
                conflicts=conflicts,
                citation_coverage=data.get("citation_coverage", 0.0),
                overall_confidence=data.get("overall_confidence", 0.5),
                needs_revision=data.get("needs_revision", len(hallucinated) > 0),
                revision_focus=data.get("revision_focus"),
            )

            logger.info(
                f"Reflection: confidence={result.overall_confidence:.2f}, "
                f"hallucinations={len(hallucinated)}, "
                f"conflicts={len(conflicts)}, "
                f"needs_revision={result.needs_revision}"
            )
            return result

        except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Reflection error: {e}")
            self.last_usage = None
            return self._default_result()

    def _format_evidence(self, evidence_list: list[Evidence]) -> str:
        """Format evidence for the prompt."""
        lines = []
        for i, ev in enumerate(evidence_list[:15], 1):
            source = f"{ev.source_title or 'Unknown'} ({ev.source_url})" if ev.source_url else (ev.source_title or "Unknown")
            lines.append(f"[{i}] {source}\n{ev.content[:400]}")
        return "\n\n".join(lines)

    def _default_result(self) -> ReflectionResult:
        """Return a default result when reflection fails."""
        return ReflectionResult(
            total_claims=0,
            verified_claims=0,
            hallucinated_claims=[],
            conflicts=[],
            citation_coverage=0.0,
            overall_confidence=0.0,
            needs_revision=True,
            revision_focus="Reflection validation failed. Re-run verification and inspect evidence coverage.",
        )
