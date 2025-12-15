"""
neurokit.knowledge.vector_store.types
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


Vector = Sequence[float]
Metadata = dict[str, Any]
Payload = dict[str, Any]


class DistanceMetric(str, Enum):
    """Distance/similarity metric used by a collection.

    Interface convention:
    - Query results MUST return a `score` where higher is better.
    """

    COSINE = "cosine"


@dataclass
class CollectionConfig:
    dimension: int
    distance: DistanceMetric = DistanceMetric.COSINE


@dataclass
class VectorRecord:
    """A single vector record.

    Notes:
    - `metadata` is meant for filtering.
    - `payload` is stored and returned as-is; its schema is intentionally flexible.
    """

    id: str
    vector: Vector
    metadata: Metadata = field(default_factory=dict)
    payload: Payload = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """A single search result.

    `score` semantics:
    - Always "higher is better".
    """

    id: str
    score: float
    metadata: Metadata = field(default_factory=dict)
    payload: Payload = field(default_factory=dict)
    vector: list[float] | None = None
