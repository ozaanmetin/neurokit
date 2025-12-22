from typing import Optional

from neurokit.knowledge.embedding.providers.base import EmbeddingProvider


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