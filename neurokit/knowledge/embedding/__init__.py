"""
neurokit.knowledge.embedding

Embedding generation and management for vector representations.
"""

from neurokit.knowledge.embedding.entity import Embedding, EmbeddingBatch
from neurokit.knowledge.embedding.providers.base import EmbeddingProvider
from neurokit.knowledge.embedding.providers.openai import OpenAIEmbeddingProvider
from neurokit.knowledge.embedding.providers.cohere import CohereEmbeddingProvider
from neurokit.knowledge.embedding.providers.hugging_face import HuggingFaceEmbeddingProvider


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
