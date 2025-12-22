from typing import Optional

from neurokit.knowledge.embedding.providers.base import EmbeddingProvider


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