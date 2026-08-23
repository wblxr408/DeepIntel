"""
BGE Reranker: cross-encoder reranking for improved relevance.

Uses BAAI/bge-reranker-v2-m3 for cross-encoder scoring.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranking using BGE Reranker v2 M3.

    Cross-encoder vs bi-encoder:
    - Bi-encoder: encodes query and document separately, then computes similarity
    - Cross-encoder: encodes query+document together, capturing fine-grained interaction

    Trade-off: Cross-encoder is more accurate but slower (requires forward pass per pair).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.rag.rerank_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        logger.info(f"Reranker: using model={self.model_name}, device={self.device}")

    @property
    def model(self) -> AutoModelForSequenceClassification:
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("Reranker: model loaded successfully")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10,
        batch_size: int = 8,
    ) -> list[float]:
        """
        Rerank documents using cross-encoder scores.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_n: Number of top results to return
            batch_size: Batch size for inference

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not documents:
            return []

        logger.info(f"Reranker: reranking {len(documents)} documents for query '{query[:50]}'")

        all_scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]

                # Tokenize query-document pairs
                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).float().cpu().tolist()

                # Convert to probabilities using sigmoid (for binary classification)
                probs = [1 / (1 + abs(s)) * max(0, s) + 0.5 for s in scores]
                probs = [max(0.0, min(1.0, p)) for p in probs]

                all_scores.extend(probs)

        # Sort by score descending and return top_n
        scored = list(zip(documents, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_scores = [s for _, s in scored[:top_n]]

        logger.info(f"Reranker: top score={top_scores[0] if top_scores else 0:.3f}")
        return top_scores
