"""
neurokit.knowledge.embedding

Embedding generation and management for vector representations.
"""

from neurokit.knowledge.embedding.entity import Embedding, EmbeddingBatch
from neurokit.knowledge.embedding.provider import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    CohereEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
)

__all__ = [
    # Entities
    "Embedding",
    "EmbeddingBatch",
    # Providers
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
]
