from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result from a knowledge base or document store.
    """

    id: str
    content: str
    score: Optional[float] = field(default=None)
    chunk_id: Optional[str] = field(default=None)
    document_id: Optional[str] = field(default=None)
    metadata: Optional[dict[str, Any]] = field(default=None)
    ranking: Optional[dict[str, Any]] = field(default=None)

    def to_dict(self) -> dict:
        """
        Convert the RetrievalResult to a dictionary.

        Returns:
            dict: A dictionary representation of the RetrievalResult.
        """
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "ranking": self.ranking,
        }
    

@dataclass
class RetrievalContext:
    """
    Represents the context for a retrieval operation.
    """
    query: str
    top_k: int = field(default=5)
    filters: Optional[Any] = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)
