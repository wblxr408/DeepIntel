"""
BGE Embedder: generates vector embeddings for queries and documents.

Supports BAAI/bge-zh-qwen2-int8 (Chinese-optimized, INT8 quantized).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates vector embeddings using BGE models.

    Design decisions:
    - Lazy loading to avoid startup latency
    - Automatic device detection (CUDA/CPU)
    - Batch processing for document embedding
    - INT8 quantized model for memory efficiency
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.rag.embed_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SentenceTransformer | None = None
        logger.info(f"Embedder: using model={self.model_name}, device={self.device}")

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "sentence_transformers is required to load the embedder model"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedder: model loaded successfully")
        return self._model

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text query.

        Args:
            text: Query text

        Returns:
            1024-dimensional embedding vector (list of floats)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
