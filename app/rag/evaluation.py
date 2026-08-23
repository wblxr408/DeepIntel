"""
Offline layered RAG evaluation utilities.

This module runs benchmark datasets against the real RAG retrieval path and
optionally a generation judge model for faithfulness / relevancy scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.agents.analyst import AnalystAgent
from app.graph.state import AgentType, Evidence
from app.llm_client import LLMClientWrapper
from metrics.rag_evaluation.collector import RAGEvaluationCollector

logger = logging.getLogger(__name__)


@dataclass
class RAGEvalExample:
    """Single offline RAG evaluation example."""

    question_id: str
    query: str
    expected_chunk_ids: list[str]
    ground_truth_answer: str | None = None
    top_k: int = 5
    query_type: str = "unlabeled"


class RetrievalRunner(Protocol):
    """Protocol for retrieval execution."""

    def execute_retrieval(self, query: str, context: str = "", group: str | None = None) -> list[Any]:
        """Run retrieval and return ranked RAG results."""


class AnswerGenerator(Protocol):
    """Protocol for generation execution."""

    def generate_answer(self, query: str, retrieved_results: list[Any]) -> str:
        """Generate an answer from retrieved results."""


def load_eval_dataset(path: str | Path) -> list[RAGEvalExample]:
    """Load evaluation dataset from JSON or JSONL."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"evaluation dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("items", [])

    examples: list[RAGEvalExample] = []
    for idx, item in enumerate(records):
        examples.append(
            RAGEvalExample(
                question_id=str(item.get("question_id") or item.get("id") or f"q{idx + 1}"),
                query=item["query"],
                expected_chunk_ids=[str(chunk_id) for chunk_id in item.get("expected_chunk_ids", [])],
                ground_truth_answer=item.get("ground_truth_answer"),
                top_k=int(item.get("top_k", 5)),
                query_type=str(item.get("query_type", "unlabeled")),
            )
        )
    return examples


class RAGOfflineEvaluator:
    """Run layered offline RAG evaluation against the real retrieval path."""

    def __init__(
        self,
        *,
        retrieval_runner: RetrievalRunner,
        answer_generator: AnswerGenerator | None = None,
        collector: RAGEvaluationCollector | None = None,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        self.retrieval_runner = retrieval_runner
        self.answer_generator = answer_generator or AnalystAnswerGenerator()
        self.collector = collector or RAGEvaluationCollector()
        self.client_wrapper = LLMClientWrapper()
        self.judge_model = judge_model

    def evaluate_dataset(
        self,
        examples: list[RAGEvalExample],
        *,
        group: str | None = None,
        run_generation_eval: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a dataset and return aggregated metrics."""
        for example in examples:
            rag_results = self.retrieval_runner.execute_retrieval(example.query, example.query, group=group)
            retrieved_chunk_ids = [str(result.chunk_id) for result in rag_results]

            self.collector.record_retrieval_eval(
                question_id=example.question_id,
                query=example.query,
                expected_chunk_ids=example.expected_chunk_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                top_k=example.top_k,
                query_type=example.query_type,
            )

            if run_generation_eval and example.ground_truth_answer:
                generated_answer = self.answer_generator.generate_answer(
                    example.query,
                    rag_results[: example.top_k],
                )
                context_metrics = self._compute_context_metrics(
                    expected_chunk_ids=example.expected_chunk_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    top_k=example.top_k,
                )
                generation_scores = self._judge_generation(
                    query=example.query,
                    retrieved_contexts=[result.content for result in rag_results[: example.top_k]],
                    answer=generated_answer,
                )
                self.collector.record_generation_eval(
                    question_id=example.question_id,
                    query=example.query,
                    answer=generated_answer,
                    ground_truth_answer=example.ground_truth_answer,
                    faithfulness=generation_scores["faithfulness"],
                    answer_relevancy=generation_scores["answer_relevancy"],
                    context_recall=context_metrics["context_recall"],
                    context_precision=context_metrics["context_precision"],
                    judge_model=self.judge_model,
                )

        return self.collector.get_metrics()

    def _judge_generation(
        self,
        *,
        query: str,
        retrieved_contexts: list[str],
        answer: str,
    ) -> dict[str, float | None]:
        """
        Score generation-layer metrics with an LLM judge.

        The judge returns floats in [0, 1].
        """
        prompt = (
            "You are evaluating a RAG system answer.\n"
            "Score each metric from 0.0 to 1.0 and return strict JSON with keys:\n"
            "faithfulness, answer_relevancy, context_recall, context_precision.\n\n"
            f"User question:\n{query}\n\n"
            f"Retrieved contexts:\n{json.dumps(retrieved_contexts, ensure_ascii=False)}\n\n"
            f"Answer to evaluate:\n{answer}\n"
        )

        response = self.client_wrapper.client.chat.completions.create(
            model=self.judge_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Be strict, numeric, and only return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "answer_relevancy": float(data.get("answer_relevancy", 0.0)),
            "context_recall": _optional_float(data.get("context_recall")),
            "context_precision": _optional_float(data.get("context_precision")),
        }

    @staticmethod
    def _compute_context_metrics(
        *,
        expected_chunk_ids: list[str],
        retrieved_chunk_ids: list[str],
        top_k: int,
    ) -> dict[str, float]:
        """Context-layer metrics derived from the chunk labels (no LLM judge)."""
        return {
            "context_recall": _recall_at_k(expected_chunk_ids, retrieved_chunk_ids, top_k),
            "context_precision": _average_precision(expected_chunk_ids, retrieved_chunk_ids, top_k),
        }


def _optional_float(value: Any) -> float | None:
    """Convert judge score to float when available."""
    if value is None:
        return None
    return float(value)


class AnalystAnswerGenerator:
    """Generate offline evaluation answers using the real Analyst agent."""

    def __init__(self) -> None:
        self.analyst = AnalystAgent()

    def generate_answer(self, query: str, retrieved_results: list[Any]) -> str:
        evidence_list = [
            Evidence(
                content=result.content,
                source_url=(result.metadata or {}).get("url"),
                source_title=(result.metadata or {}).get("title", "Knowledge Base"),
                source_type="knowledge_base",
                collected_by=AgentType.RAG,
            )
            for result in retrieved_results
        ]
        return self.analyst.analyze(query, evidence_list)


    
def _precision_at_k(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str], top_k: int) -> float:
    expected_set = set(expected_chunk_ids)
    ranked = retrieved_chunk_ids[:top_k]
    if not ranked:
        return 0.0
    hits = sum(1 for chunk_id in ranked if chunk_id in expected_set)
    return hits / len(ranked)


def _recall_at_k(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str], top_k: int) -> float:
    expected_set = set(expected_chunk_ids)
    if not expected_set:
        return 0.0
    ranked = retrieved_chunk_ids[:top_k]
    hits = sum(1 for chunk_id in ranked if chunk_id in expected_set)
    return hits / len(expected_set)


def _average_precision(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str], top_k: int) -> float:
    expected_set = set(expected_chunk_ids)
    if not expected_set:
        return 0.0

    ranked = retrieved_chunk_ids[:top_k]
    hit_count = 0
    precision_sum = 0.0
    for idx, chunk_id in enumerate(ranked, start=1):
        if chunk_id in expected_set:
            hit_count += 1
            precision_sum += hit_count / idx
    if hit_count == 0:
        return 0.0
    return precision_sum / min(len(expected_set), top_k)

