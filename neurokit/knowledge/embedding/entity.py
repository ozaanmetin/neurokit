"""
neurokit.knowledge.embedding.entity

Embedding entity definitions for vector representations of text chunks.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

from neurokit.core.utils.id import IDHelper
from neurokit.core.utils.math import cosine_similarity, cosine_distance


@dataclass
class Embedding:
    """
    Represents a vector embedding of a text chunk.
    
    Links a chunk to its dense vector representation, enabling
    similarity search and retrieval operations.
    
    Attributes:
        chunk_id: ID of the source chunk this embedding represents
        vector: Dense vector representation (list of floats)
        model: Name of the embedding model used (e.g., "text-embedding-3-small")
        dimensions: Vector dimensions (auto-calculated from vector if not provided)
        metadata: Optional metadata (e.g., token count, processing time)
        id: Unique identifier (auto-generated if not provided)
    """
    
    chunk_id: str
    vector: list[float]
    model: str
    
    dimensions: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.id is None:
            self.id = IDHelper.generate_embedding_id(
                chunk_id=self.chunk_id, 
                model=self.model
            )
        
        if self.dimensions is None:
            self.dimensions = len(self.vector)
        
        # Validate vector dimensions match
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"Vector length ({len(self.vector)}) does not match "
                f"dimensions ({self.dimensions})"
            )
    
    def similarity(self, other: "Embedding") -> float:
        """
        Calculate cosine similarity with another embedding.
        
        Args:
            other: Another Embedding to compare with
            
        Returns:
            Cosine similarity score between -1 and 1
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Cannot compare embeddings with different dimensions: "
                f"{self.dimensions} vs {other.dimensions}"
            )
        
        return cosine_similarity(self.vector, other.vector)
    
    def distance(self, other: "Embedding") -> float:
        """
        Calculate cosine distance with another embedding.
        
        Args:
            other: Another Embedding to compare with
            
        Returns:
            Cosine distance (1 - similarity), range 0 to 2
        """
        return cosine_distance(self.vector, other.vector)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "vector": self.vector,
            "model": self.model,
            "dimensions": self.dimensions,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Embedding":
        """Create an Embedding instance from a dictionary."""
        return cls(
            id=data.get("id"),
            chunk_id=data["chunk_id"],
            vector=data["vector"],
            model=data["model"],
            dimensions=data.get("dimensions"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EmbeddingBatch:
    """
    A batch of embeddings, typically from processing multiple chunks.
    
    Useful for batch operations and bulk storage.
    
    Attributes:
        embeddings: List of Embedding objects
        model: Common model used for all embeddings
        metadata: Batch-level metadata
    """
    
    embeddings: list[Embedding]
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __iter__(self):
        return iter(self.embeddings)
    
    def __getitem__(self, index: int) -> Embedding:
        return self.embeddings[index]
    
    @property
    def vectors(self) -> list[list[float]]:
        """Get all vectors as a list."""
        return [e.vector for e in self.embeddings]
    
    @property
    def chunk_ids(self) -> list[str]:
        """Get all chunk IDs."""
        return [e.chunk_id for e in self.embeddings]
