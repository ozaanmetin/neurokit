from typing import Any, Optional

from neurokit.knowledge.embedding.providers.base import EmbeddingProvider


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