# app/core/embedding.py
"""Singleton TextEmbedding model to avoid reinitializing."""

from fastembed import TextEmbedding
from app.core import settings

# Lazy-loaded singleton
_embedding_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Get or initialize the embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=settings.embedding_model)
    return _embedding_model
