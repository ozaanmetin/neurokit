"""neurokit.knowledge.vector_store.in_memory

A zero-dependency, sync in-memory vector store.

This backend is intended for:
- tests
- local prototyping

Collections are created lazily (on first upsert/query) with cosine distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from neurokit.core.utils.math import cosine_similarity
from neurokit.knowledge.vector_store.base import DEFAULT_COLLECTION, VectorStore
from neurokit.knowledge.vector_store.exceptions import DimensionMismatch
from neurokit.knowledge.vector_store.filter import Filter
from neurokit.knowledge.vector_store.entity import (
    CollectionConfig,
    DistanceMetric,
    Vector,
    VectorRecord,
    VectorSearchResult,
)


@dataclass
class _CollectionState:
    config: CollectionConfig
    records: dict[str, VectorRecord]


class InMemoryVectorStore(VectorStore):
    backend_name = "in-memory"

    def __init__(self) -> None:
        self._collections: dict[str, _CollectionState] = {}

    def ensure_collection(self, name: str, *, config: CollectionConfig) -> None:
        existing = self._collections.get(name)
       
        if existing is None:
            self._collections[name] = _CollectionState(config=config, records={})
            return

        if existing.config.dimension != config.dimension:
            raise DimensionMismatch(
                expected=existing.config.dimension,
                actual=config.dimension,
                collection=name,
                backend=self.backend_name,
            )

        if existing.config.distance != config.distance:
            # Keep it strict: mixed metric semantics lead to confusing scores.
            raise ValueError(
                f"Collection '{name}' already exists with distance={existing.config.distance} "
                f"but requested distance={config.distance}."
            )

    def _get_or_create_collection(self, name: str, dimension: int) -> _CollectionState:
        existing = self._collections.get(name)
        if existing is None:
            config = CollectionConfig(dimension=dimension, distance=DistanceMetric.COSINE)
            self._collections[name] = _CollectionState(config=config, records={})
            return self._collections[name]

        if existing.config.dimension != dimension:
            raise DimensionMismatch(
                expected=existing.config.dimension,
                actual=dimension,
                collection=name,
                backend=self.backend_name,
            )

        return existing

    def upsert(self, records: Sequence[VectorRecord], *, collection: str = DEFAULT_COLLECTION) -> None:
        if not records:
            return

        dimension = len(records[0].vector)
        state = self._get_or_create_collection(collection, dimension=dimension)

        for record in records:
            if len(record.vector) != state.config.dimension:
                raise DimensionMismatch(
                    expected=state.config.dimension,
                    actual=len(record.vector),
                    collection=collection,
                    backend=self.backend_name,
                )

            # Store a shallow copy to avoid accidental caller mutation.
            state.records[record.id] = VectorRecord(
                id=record.id,
                vector=list(record.vector),
                content=record.content,
                metadata=dict(record.metadata or {}),
                payload=dict(record.payload or {}),
            )

    def delete(self, ids: Sequence[str], *, collection: str = DEFAULT_COLLECTION) -> int:
        state = self._collections.get(collection)
        if state is None:
            return 0

        deleted = 0
        for record_id in ids:
            if record_id in state.records:
                del state.records[record_id]
                deleted += 1
        return deleted

    def get(
        self,
        ids: Sequence[str],
        *,
        collection: str = DEFAULT_COLLECTION,
        include_vectors: bool = False,
        include_payloads: bool = True,
    ) -> list[VectorRecord]:
        
        state = self._collections.get(collection)
        if state is None:
            return []

        results: list[VectorRecord] = []
        for record_id in ids:
            record = state.records.get(record_id)
            if record is None:
                continue

            results.append(
                VectorRecord(
                    id=record.id,
                    vector=list(record.vector) if include_vectors else [],
                    content=record.content,
                    metadata=dict(record.metadata),
                    payload=dict(record.payload) if include_payloads else {},
                )
            )

        return results

    def query(
        self,
        vector: Vector,
        *,
        top_k: int = 10,
        filter: Filter | None = None,
        collection: str = DEFAULT_COLLECTION,
        include_vectors: bool = False,
        include_payloads: bool = True,
    ) -> list[VectorSearchResult]:
        query_vector = list(vector)
        state = self._get_or_create_collection(collection, dimension=len(query_vector))

        candidates: Iterable[VectorRecord] = state.records.values()
        if filter is not None:
            candidates = (r for r in candidates if filter.evaluate(r.metadata))

        scored: list[tuple[float, VectorRecord]] = []
        for record in candidates:
            score = cosine_similarity(query_vector, record.vector)
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[VectorSearchResult] = []
        for score, record in scored[: max(0, top_k)]:
            results.append(
                VectorSearchResult(
                    id=record.id,
                    score=score,
                    content=record.content,
                    metadata=dict(record.metadata),
                    payload=dict(record.payload) if include_payloads else {},
                    vector=list(record.vector) if include_vectors else None,
                )
            )

        return results

    def count(self, *, collection: str = DEFAULT_COLLECTION, filter: Filter | None = None) -> int:
        state = self._collections.get(collection)
        if state is None:
            return 0

        if filter is None:
            return len(state.records)

        return sum(1 for record in state.records.values() if filter.evaluate(record.metadata))
