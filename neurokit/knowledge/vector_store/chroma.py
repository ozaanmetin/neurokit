"""neurokit.knowledge.vector_store.chroma

Chroma vector store backend.

Notes:
- Chroma's metadata has type restrictions; callers should keep metadata values simple.
- Payload is stored as JSON in a reserved metadata key: `_nk_payload`.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from neurokit.knowledge.vector_store.base import DEFAULT_COLLECTION, VectorStore
from neurokit.knowledge.vector_store.entity import CollectionConfig, DistanceMetric, Vector, VectorRecord, VectorSearchResult
from neurokit.knowledge.vector_store.exceptions import BackendNotInstalled, DimensionMismatch
from neurokit.knowledge.vector_store.filter import And, Eq, Filter, In, Or, Range

try:
    import chromadb  # type: ignore
except Exception as exc:  # pragma: no cover
    raise BackendNotInstalled("chroma", "chroma") from exc


_RESERVED_PAYLOAD_KEY = "_nk_payload"


def _split_meta_and_payload(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload_json = metadata.get(_RESERVED_PAYLOAD_KEY)
    clean_meta = dict(metadata)
    clean_meta.pop(_RESERVED_PAYLOAD_KEY, None)

    if payload_json is None:
        return clean_meta, {}

    if isinstance(payload_json, str):
        try:
            payload = json.loads(payload_json)
            if isinstance(payload, dict):
                return clean_meta, payload
        except json.JSONDecodeError:
            pass

    return clean_meta, {}


def _compile_filter(expr: Filter) -> dict[str, Any]:
    if isinstance(expr, Eq):
        return {expr.field: expr.value}

    if isinstance(expr, In):
        return {expr.field: {"$in": list(expr.values)}}

    if isinstance(expr, Range):
        ops: dict[str, Any] = {}
        if expr.gt is not None:
            ops["$gt"] = expr.gt
        if expr.gte is not None:
            ops["$gte"] = expr.gte
        if expr.lt is not None:
            ops["$lt"] = expr.lt
        if expr.lte is not None:
            ops["$lte"] = expr.lte
        return {expr.field: ops}

    if isinstance(expr, And):
        return {"$and": [_compile_filter(expr.left), _compile_filter(expr.right)]}

    if isinstance(expr, Or):
        return {"$or": [_compile_filter(expr.left), _compile_filter(expr.right)]}

    raise ValueError(f"Unsupported filter expression for Chroma: {type(expr).__name__}")


class ChromaVectorStore(VectorStore):
    backend_name = "chroma"

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        client: Any | None = None,
    ) -> None:
        

        if client is not None:
            self._client = client
        else:
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()

        self._collection_configs: dict[str, CollectionConfig] = {}

    def ensure_collection(self, name: str, *, config: CollectionConfig) -> None:
        if config.distance != DistanceMetric.COSINE:
            raise ValueError("ChromaVectorStore currently supports only cosine distance.")

        existing = self._collection_configs.get(name)
        if existing is None:
            self._collection_configs[name] = config
        else:
            if existing.dimension != config.dimension:
                raise DimensionMismatch(
                    expected=existing.dimension,
                    actual=config.dimension,
                    collection=name,
                    backend=self.backend_name,
                )

        # Chroma has no strict dimension schema; ensure the collection exists.
        # Using get_or_create_collection to avoid recreating existing collections.
        try:
            self._client.get_collection(name=name)
        except Exception:
            self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    def _get_collection(self, name: str, *, dimension: int | None = None):
        if dimension is not None:
            self.ensure_collection(name, config=CollectionConfig(dimension=dimension, distance=DistanceMetric.COSINE))
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: Sequence[VectorRecord], *, collection: str = DEFAULT_COLLECTION) -> None:
        if not records:
            return

        dim = len(records[0].vector)
        col = self._get_collection(collection, dimension=dim)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for record in records:
            if len(record.vector) != dim:
                raise DimensionMismatch(
                    expected=dim,
                    actual=len(record.vector),
                    collection=collection,
                    backend=self.backend_name,
                )

            ids.append(record.id)
            embeddings.append(list(record.vector))

            metadata = dict(record.metadata or {})
            try:
                metadata[_RESERVED_PAYLOAD_KEY] = json.dumps(dict(record.payload or {}))
            except TypeError as exc:
                raise ValueError("Payload must be JSON-serializable for Chroma backend.") from exc

            metadatas.append(metadata)

        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def delete(self, ids: Sequence[str], *, collection: str = DEFAULT_COLLECTION) -> int:
        col = self._get_collection(collection)
        # Chroma doesn't return deleted count; approximate by checking existing ids.
        existing = col.get(ids=list(ids), include=[])
        count = len(existing.get("ids", [])) if isinstance(existing, dict) else 0
        col.delete(ids=list(ids))
        return count

    def get(
        self,
        ids: Sequence[str],
        *,
        collection: str = DEFAULT_COLLECTION,
        include_vectors: bool = False,
        include_payloads: bool = True,
    ) -> list[VectorRecord]:
        col = self._get_collection(collection)
        include: list[str] = ["metadatas"]
        if include_vectors:
            include.append("embeddings")

        resp = col.get(ids=list(ids), include=include)

        out: list[VectorRecord] = []
        resp_ids = resp.get("ids", [])
        resp_embeddings = resp.get("embeddings") if resp.get("embeddings") is not None else []
        resp_metadatas = resp.get("metadatas") if resp.get("metadatas") is not None else []

        for idx, record_id in enumerate(resp_ids):
            metadata = resp_metadatas[idx] or {}
            clean_meta, payload = _split_meta_and_payload(metadata)
            out.append(
                VectorRecord(
                    id=record_id,
                    vector=list(resp_embeddings[idx]) if include_vectors else [],
                    metadata=clean_meta,
                    payload=payload if include_payloads else {},
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
        col = self._get_collection(collection, dimension=len(q))

        where = _compile_filter(filter) if filter is not None else None

        include: list[str] = ["distances", "metadatas"]
        if include_vectors:
            include.append("embeddings")

        resp = col.query(
            query_embeddings=[q],
            n_results=max(0, top_k),
            where=where,
            include=include,
        )

        ids = (resp.get("ids") or [[]])[0]
        distances = (resp.get("distances") or [[]])[0]
        metadatas = (resp.get("metadatas") or [[]])[0]
        embeddings = (resp.get("embeddings") or [[]])
        embedding_row = embeddings[0] if embeddings else []

        out: list[VectorSearchResult] = []
        for i, record_id in enumerate(ids):
            distance = distances[i]
            score = 1.0 - float(distance)
            metadata = metadatas[i] or {}
            clean_meta, payload = _split_meta_and_payload(metadata)

            out.append(
                VectorSearchResult(
                    id=record_id,
                    score=score,
                    metadata=clean_meta,
                    payload=payload if include_payloads else {},
                    vector=list(embedding_row[i]) if include_vectors and i < len(embedding_row) else None,
                )
            )

        return out

    def count(self, *, collection: str = DEFAULT_COLLECTION, filter: Filter | None = None) -> int:
        col = self._get_collection(collection)
        where = _compile_filter(filter) if filter is not None else None
        try:
            return int(col.count(where=where))
        except TypeError:
            # Older versions may not support where in count
            resp = col.get(where=where, include=[])
            return len(resp.get("ids", []))
