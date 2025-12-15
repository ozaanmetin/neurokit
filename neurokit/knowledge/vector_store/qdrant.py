"""neurokit.knowledge.vector_store.qdrant

Qdrant vector store backend.

Notes:
- Metadata fields are stored at top-level payload keys to support filtering.
- Payload is stored under a reserved key: `_nk_payload`.
"""

from __future__ import annotations

from typing import Any, Sequence

from neurokit.knowledge.vector_store.base import DEFAULT_COLLECTION, VectorStore
from neurokit.knowledge.vector_store.entity import CollectionConfig, DistanceMetric, Vector, VectorRecord, VectorSearchResult
from neurokit.knowledge.vector_store.exceptions import BackendNotInstalled, CollectionNotFound, DimensionMismatch
from neurokit.knowledge.vector_store.filter import And, Eq, Filter, In, Not, Or, Range


_RESERVED_PAYLOAD_KEY = "_nk_payload"


def _compile_filter(expr: Filter, *, models: Any) -> Any:
    """Compile Filter expression to qdrant_client.models.Filter."""

    if isinstance(expr, Eq):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=expr.field,
                    match=models.MatchValue(value=expr.value),
                )
            ]
        )

    if isinstance(expr, In):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=expr.field,
                    match=models.MatchAny(any=list(expr.values)),
                )
            ]
        )

    if isinstance(expr, Range):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=expr.field,
                    range=models.Range(
                        gt=expr.gt,
                        gte=expr.gte,
                        lt=expr.lt,
                        lte=expr.lte,
                    ),
                )
            ]
        )

    if isinstance(expr, And):
        left = _compile_filter(expr.left, models=models)
        right = _compile_filter(expr.right, models=models)
        return models.Filter(
            must=(left.must or []) + (right.must or []),
            should=(left.should or []) + (right.should or []),
            must_not=(left.must_not or []) + (right.must_not or []),
        )

    if isinstance(expr, Or):
        left = _compile_filter(expr.left, models=models)
        right = _compile_filter(expr.right, models=models)
        return models.Filter(should=[left, right], must=[])

    if isinstance(expr, Not):
        inner = _compile_filter(expr.inner, models=models)
        return models.Filter(must_not=[inner])

    raise ValueError(f"Unsupported filter expression for Qdrant: {type(expr).__name__}")


class QdrantVectorStore(VectorStore):
    backend_name = "qdrant"

    def __init__(
        self,
        *,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        path: str | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client import models  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendNotInstalled("qdrant", "qdrant") from exc

        self._models = models
        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(url=url, host=host, port=port, api_key=api_key, path=path)

    def ensure_collection(self, name: str, *, config: CollectionConfig) -> None:
        if config.distance != DistanceMetric.COSINE:
            raise ValueError("QdrantVectorStore currently supports only cosine distance.")

        models = self._models
        try:
            info = self._client.get_collection(name)
            existing = info.config.params.vectors
            size = existing.size
            distance = existing.distance
            if int(size) != int(config.dimension):
                raise DimensionMismatch(
                    expected=int(size),
                    actual=int(config.dimension),
                    collection=name,
                    backend=self.backend_name,
                )
            if distance != models.Distance.COSINE:
                raise ValueError(f"Collection '{name}' exists with distance={distance}.")
            return
        except Exception:
            pass

        self._client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=config.dimension, distance=models.Distance.COSINE),
        )

    def upsert(self, records: Sequence[VectorRecord], *, collection: str = DEFAULT_COLLECTION) -> None:
        if not records:
            return

        dim = len(records[0].vector)
        self.ensure_collection(collection, config=CollectionConfig(dimension=dim, distance=DistanceMetric.COSINE))
        models = self._models

        points = []
        for record in records:
            if len(record.vector) != dim:
                raise DimensionMismatch(
                    expected=dim,
                    actual=len(record.vector),
                    collection=collection,
                    backend=self.backend_name,
                )

            payload = dict(record.metadata or {})
            payload[_RESERVED_PAYLOAD_KEY] = dict(record.payload or {})

            points.append(models.PointStruct(id=record.id, vector=list(record.vector), payload=payload))

        self._client.upsert(collection_name=collection, points=points)

    def delete(self, ids: Sequence[str], *, collection: str = DEFAULT_COLLECTION) -> int:
        models = self._models
        res = self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(ids)),
        )
        # Qdrant doesn't always return deleted count; return input length.
        return len(ids)

    def get(
        self,
        ids: Sequence[str],
        *,
        collection: str = DEFAULT_COLLECTION,
        include_vectors: bool = False,
        include_payloads: bool = True,
    ) -> list[VectorRecord]:
        models = self._models
        try:
            points = self._client.retrieve(
                collection_name=collection,
                ids=list(ids),
                with_vectors=include_vectors,
                with_payload=include_payloads,
            )
        except Exception as exc:
            raise CollectionNotFound(collection, backend=self.backend_name) from exc

        out: list[VectorRecord] = []
        for p in points:
            payload = dict(p.payload or {})
            user_payload = payload.pop(_RESERVED_PAYLOAD_KEY, {}) if include_payloads else {}
            out.append(
                VectorRecord(
                    id=str(p.id),
                    vector=list(p.vector) if include_vectors and p.vector is not None else [],
                    metadata=payload,
                    payload=dict(user_payload) if include_payloads else {},
                )
            )
        return out

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
        q = list(vector)
        self.ensure_collection(collection, config=CollectionConfig(dimension=len(q), distance=DistanceMetric.COSINE))

        q_filter = _compile_filter(filter, models=self._models) if filter is not None else None

        result = self._client.query_points(
            collection_name=collection,
            query=q,
            limit=max(1, top_k),
            query_filter=q_filter,
            with_payload=True,
            with_vectors=include_vectors,
        )
        
        hits = result.points

        out: list[VectorSearchResult] = []
        for h in hits:
            payload = dict(h.payload or {})
            user_payload = payload.pop(_RESERVED_PAYLOAD_KEY, {}) if include_payloads else {}
            out.append(
                VectorSearchResult(
                    id=str(h.id),
                    score=float(h.score),
                    metadata=payload,
                    payload=dict(user_payload) if include_payloads else {},
                    vector=list(h.vector) if include_vectors and h.vector is not None else None,
                )
            )
        return out

    def count(self, *, collection: str = DEFAULT_COLLECTION, filter: Filter | None = None) -> int:
        q_filter = _compile_filter(filter, models=self._models) if filter is not None else None
        res = self._client.count(collection_name=collection, count_filter=q_filter, exact=True)
        return int(res.count)

    def close(self) -> None:
        # qdrant-client doesn't require explicit close in most cases.
        return
