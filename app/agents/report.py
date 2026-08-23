"""
Report Generator Agent: produces the final Markdown report with citations.

Generates a structured, well-formatted research report with source attribution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.graph.state import Citation, Evidence
from app.guardrails import (
    build_guardrail_decision,
    build_prompt_profile_message,
    get_research_budget,
)
from app.llm_client import (
    StreamInterruptedError,
    collect_usage_metrics,
    create_fallback_llm_client,
    create_llm_client,
    get_llm_model,
)
from app.security.prompt_security import untrusted_context

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Converts the analysis and evidence into a polished Markdown research report.

    The report follows a standard business research format with:
    - Executive summary
    - Structured findings
    - Data analysis
    - Expert perspectives
    - Conclusions
    - References
    """

    SYSTEM_PROMPT = """You are an expert research report writer. Your job is to produce a polished, well-structured Markdown research report.

## Report Structure

```
# {Research Topic}

> Generated: {timestamp} | Research Depth: Comprehensive

## Executive Summary
{2-3 sentence core conclusion}

## 1. Background and Context
{Research background and scope}

## 2. Key Findings
### 2.1 [Finding Category 1]
{Detailed finding with evidence}

### 2.2 [Finding Category 2]
{Detailed finding with evidence}

## 3. Data and Statistics
{Key data points with sources}

## 4. Expert Perspectives
{Expert opinions and analyses}

## 5. Conclusions and Recommendations
{Actionable conclusions}

## References
{citation list}
```

## Formatting Rules

1. Use [citation:N] format for every factual claim
2. Use **bold** for key terms and important numbers
3. Use tables for data comparisons
4. Use blockquotes for expert quotes
5. Keep paragraphs focused (3-5 sentences max)
6. Use ## for main sections, ### for subsections

## Quality Standards

- Every factual claim must be cited
- Distinguish between facts (verified) and opinions (analyst interpretation)
- Include confidence levels for uncertain claims
- Provide actionable, specific conclusions
"""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model
        self._client: OpenAI | None = None
        self.last_usage: dict | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = create_llm_client()
            self.model = get_llm_model()
        return self._client

    def generate(
        self,
        user_query: str,
        analysis: str,
        evidence_list: list[Evidence],
        reflection: dict | None,
    ) -> tuple[str, list[Citation]]:
        """Generate the final report without streaming callbacks."""
        return self.generate_stream(
            user_query=user_query,
            analysis=analysis,
            evidence_list=evidence_list,
            reflection=reflection,
        )

    def generate_stream(
        self,
        user_query: str,
        analysis: str,
        evidence_list: list[Evidence],
        reflection: dict | None,
        output_length: str | None = None,
        skill_prompt: str | None = None,
        report_hints: list[str] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_citation: Callable[[Citation], None] | None = None,
    ) -> tuple[str, list[Citation]]:
        """
        Generate the final research report.

        Args:
            user_query: Original research question
            analysis: Analyst's synthesized analysis
            evidence_list: All collected evidence
            reflection: Reflection quality result (optional)

        Returns:
            Tuple of (report_markdown, citations_list)
        """
        logger.info(f"Report: generating report for query: {user_query[:80]}")
        decision = build_guardrail_decision(user_query)
        budget = get_research_budget(output_length)
        system_prompt = f"{build_prompt_profile_message(decision, user_query)}\n\n{self.SYSTEM_PROMPT}"
        if report_hints:
            system_prompt = (
                f"{system_prompt}\n\n## Report Skill Instructions\n"
                + "\n".join(f"- {item}" for item in report_hints if item and item.strip())
            )
        system_prompt += (
            f"\n\n输出长度要求：{output_length or 'medium'}。"
            f"请优先控制在约 {budget['report_max_tokens']} tokens 内。"
            f"引用数量上限约 {budget['citation_max']} 条。"
        )
        if not evidence_list and not decision.reject_if_no_evidence:
            system_prompt += (
                "\n\n当前没有外部证据。允许回答简单事实问题，但必须明确说明“未检索验证”。"
                "不要生成虚假的 [citation:N]，不要声称已检索来源。"
            )

        citations = self._build_citations(evidence_list)
        for citation in citations:
            if on_citation:
                on_citation(citation)

        # Build reference list
        ref_list = self._build_reference_list(citations)

        # Format evidence for prompt
        formatted_evidence = self._format_evidence(evidence_list, citations)

        # Confidence note if reflection is available
        confidence_note = ""
        if reflection:
            conf = reflection.get("overall_confidence", 0.5)
            if conf < 0.7:
                confidence_note = f"\n\n**Quality Note**: This report has moderate confidence ({conf:.0%}). Some claims may need verification."

        # Generate citation range note
        citation_note = self._citation_range_note(len(citations))

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Research Topic: {user_query}

{untrusted_context('skill_content', skill_prompt or '') if skill_prompt else ''}

Analyst's Analysis:
{untrusted_context('analyst_output', analysis, limit=12000)}

Evidence Sources:
{untrusted_context('external_evidence', formatted_evidence, limit=18000)}

{citation_note}

{confidence_note}

Generate the complete research report in Markdown format. Include all sections specified in the system prompt.
Make sure every factual claim has a [citation:N] reference."""
            },
        ]

        try:
            chunks: list[str] = []
            active_model = self.model

            def consume(stream: Any) -> None:
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta.content or ""
                    if not delta:
                        continue
                    chunks.append(delta)
                    if on_chunk:
                        on_chunk(delta)

            try:
                consume(self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=budget["report_max_tokens"],
                    stream=True,
                ))
            except Exception as primary_error:
                if chunks:
                    raise StreamInterruptedError(
                        "primary_llm_stream_interrupted_after_output"
                    ) from primary_error
                fallback = create_fallback_llm_client()
                if fallback is None:
                    raise
                fallback_client, active_model = fallback
                logger.warning("Primary report stream failed before output; using fallback: %s", type(primary_error).__name__)
                consume(fallback_client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=budget["report_max_tokens"],
                    stream=True,
                ))

            content = "".join(chunks).strip()
            if not content:
                content = self._fallback_report(user_query, evidence_list, citations)
                self._emit_fallback_chunks(content, on_chunk)

            # Append references section if not present
            if "## References" not in content and "## 参考" not in content:
                references_block = f"\n\n---\n\n## References\n\n{ref_list}"
                content += references_block
                if on_chunk:
                    on_chunk(references_block)

            self.last_usage = collect_usage_metrics(
                response=None,
                model=active_model,
                messages=messages,
                completion_text=content,
            )

            logger.info(f"Report: generated {len(content)} chars, {len(citations)} citations")
            return content, citations

        except StreamInterruptedError:
            # Do not replace a partially emitted report with a second model's
            # response; the workflow must mark this as recoverable instead.
            self.last_usage = None
            raise
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Report generation error: {e}")
            self.last_usage = None
            fallback = self._fallback_report(user_query, evidence_list, citations)
            self._emit_fallback_chunks(fallback, on_chunk)
            return fallback, citations

    def _build_citations(self, evidence_list: list[Evidence]) -> list[Citation]:
        """Build a stable citation list from collected evidence."""
        citations: list[Citation] = []
        for i, ev in enumerate(evidence_list[:30], 1):
            citation_text = getattr(ev, "citation", None)
            citations.append(Citation(
                citation_id=f"citation:{i}",
                source_url=ev.source_url or "",
                source_title=ev.source_title or f"Source {i}",
                source_type=ev.source_type,
                extracted_evidence=(citation_text[:300] if citation_text else ev.content[:300]),
                relevance_score=0.5,
            ))
        return citations

    def _emit_fallback_chunks(
        self,
        content: str,
        on_chunk: Callable[[str], None] | None,
        chunk_size: int = 800,
    ) -> None:
        """Emit chunk events for fallback/non-streamed content."""
        if not on_chunk:
            return
        for start in range(0, len(content), chunk_size):
            on_chunk(content[start:start + chunk_size])

    def _format_evidence(self, evidence_list: list[Evidence], citations: list[Citation]) -> str:
        """Format evidence with citation numbers."""
        lines = []
        for i, (ev, citation) in enumerate(zip(evidence_list[:20], citations[:20]), 1):
            source = f"{ev.source_title or 'Unknown'} - {ev.source_url}" if ev.source_url else (ev.source_title or "Unknown")
            lines.append(f"[{i}] {source}\nType: {ev.source_type}\n{ev.content[:300]}")
        return "\n\n".join(lines)

    def _build_reference_list(self, citations: list[Citation]) -> str:
        """Build the references section."""
        if not citations:
            return "No citations available."

        refs = []
        for i, c in enumerate(citations, 1):
            title = c.source_title or "Untitled"
            url = c.source_url or ""
            ref_str = f"[{i}] **{title}**"
            if url:
                ref_str += f" - {url}"
            refs.append(ref_str)

        return "\n".join(refs)

    def _citation_range_note(self, n: int) -> str:
        """Generate a note about available citations."""
        if n == 0:
            return "No evidence sources available."
        return f"Available citations: {n} sources (use [citation:1] through [citation:{n}] to reference them)"

    def _fallback_report(
        self,
        user_query: str,
        evidence_list: list[Evidence],
        citations: list[Citation],
    ) -> str:
        """Generate a minimal fallback report when LLM fails."""
        lines = [f"# {user_query}", "", "## 摘要", ""]
        for i, ev in enumerate(evidence_list[:10], 1):
            lines.append(f"### 来源 {i}")
            lines.append(ev.content[:500])
            lines.append("")
        if citations:
            lines.append("## 参考资料")
            for i, c in enumerate(citations[:10], 1):
                title = c.source_title or "Untitled"
                url = c.source_url or ""
                lines.append(f"[{i}] {title} - {url}")
        return "\n".join(lines)
