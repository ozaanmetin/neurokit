"""
neurokit.knowledge.embedding.providers.base

Embedding provider implementations for various services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.embedding.entity import Embedding, EmbeddingBatch


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Defines the interface for generating embeddings from text chunks
    using various embedding models and services.
    """

    def __init__(
        self,
        model: str,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        **kwargs
    ):
        """
        Initialize embedding provider.

        Args:
            model: Model identifier (e.g., "text-embedding-3-small")
            dimensions: Optional dimension override (if model supports it)
            batch_size: Maximum number of texts to embed in one API call
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.kwargs = kwargs

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Dense vector representation
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of dense vectors
        """
        pass

    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """
        Generate embedding for a chunk.

        Args:
            chunk: Chunk to embed

        Returns:
            Embedding object
        """
        vector = self.embed_text(chunk.content)

        return Embedding(
            chunk_id=chunk.id,
            vector=vector,
            model=self.model,
            dimensions=self.dimensions or len(vector),
            metadata={
                "document_id": chunk.document_id,
            }
        )

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingBatch:
        """
        Generate embeddings for multiple chunks in batch.

        Args:
            chunks: List of chunks to embed

        Returns:
            EmbeddingBatch object
        """
        texts = [chunk.content for chunk in chunks]
        vectors = self.embed_texts(texts)

        embeddings = []
        for chunk, vector in zip(chunks, vectors):
            embedding = Embedding(
                chunk_id=chunk.id,
                vector=vector,
                model=self.model,
                dimensions=self.dimensions or len(vector),
                metadata={
                    "document_id": chunk.document_id,
                }
            )
            embeddings.append(embedding)

        return EmbeddingBatch(
            embeddings=embeddings,
            model=self.model,
            metadata={"total_chunks": len(chunks)}
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
        }