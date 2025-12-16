"""
neurokit.knowledge.embedding.provider

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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.

    Supports models like:
    - text-embedding-3-small (1536 dims, can be reduced)
    - text-embedding-3-large (3072 dims, can be reduced)
    - text-embedding-ada-002 (1536 dims, legacy)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        **kwargs
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            dimensions: Optional dimension override for v3 models
            batch_size: Batch size for API calls (max 2048)
            **kwargs: Additional parameters for OpenAI API
        """
        super().__init__(model, dimensions, min(batch_size, 2048), **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            dimensions=self.dimensions,
            **self.kwargs
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batches using OpenAI API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions,
                **self.kwargs
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Cohere embedding provider.

    Supports models like:
    - embed-english-v3.0 (1024 dims)
    - embed-multilingual-v3.0 (1024 dims)
    - embed-english-light-v3.0 (384 dims)
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
        batch_size: int = 96,
        **kwargs
    ):
        """
        Initialize Cohere embedding provider.

        Args:
            model: Cohere model name
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            input_type: Type of input ("search_document", "search_query", etc.)
            batch_size: Batch size for API calls (max 96)
            **kwargs: Additional parameters for Cohere API
        """
        super().__init__(model, None, min(batch_size, 96), **kwargs)
        self.input_type = input_type

        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. "
                "Install with: pip install cohere"
            )

        self.client = cohere.Client(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding using Cohere API."""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type,
            **self.kwargs
        )
        return response.embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batches using Cohere API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                **self.kwargs
            )

            all_embeddings.extend(response.embeddings)

        return all_embeddings


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace embedding provider using sentence-transformers.

    Supports any sentence-transformers model, including:
    - all-MiniLM-L6-v2 (384 dims, fast)
    - all-mpnet-base-v2 (768 dims, quality)
    - multi-qa-mpnet-base-dot-v1 (768 dims, for Q&A)
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize HuggingFace embedding provider.

        Args:
            model: HuggingFace model name or path
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
            normalize_embeddings: Whether to L2 normalize embeddings
            batch_size: Batch size for local inference
            **kwargs: Additional parameters for SentenceTransformer
        """
        super().__init__(model, None, batch_size, **kwargs)
        self.normalize_embeddings = normalize_embeddings

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_instance = SentenceTransformer(
            model,
            device=device,
            **kwargs
        )

        # Get actual dimensions from model
        self.dimensions = self.model_instance.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding using local model."""
        embedding = self.model_instance.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batches using local model."""
        embeddings = self.model_instance.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        return embeddings.tolist()

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            "max_seq_length": self.model_instance.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
        })
        return info
