from abc import ABC, abstractmethod
from typing import Sequence

from neurokit.knowledge.vector_store.filter import Filter
from neurokit.knowledge.vector_store.entity import (
	CollectionConfig,
	Vector,
	VectorRecord,
	VectorSearchResult,
)


DEFAULT_COLLECTION = "default"


class VectorStore(ABC):
	"""Sync vector store interface."""

	@abstractmethod
	def ensure_collection(
		self,
		name: str,
		*,
		config: CollectionConfig,
	) -> None:
		"""Create or validate a collection.

		Implementations should raise if an existing collection is incompatible
		(e.g. dimension mismatch).
		"""

	@abstractmethod
	def upsert(
		self,
		records: Sequence[VectorRecord],
		*,
		collection: str = DEFAULT_COLLECTION,
	) -> None:
		"""Insert or update records by id."""

	@abstractmethod
	def delete(
		self,
		ids: Sequence[str],
		*,
		collection: str = DEFAULT_COLLECTION,
	) -> int:
		"""Delete records by id.

		Returns:
			Number of deleted records.
		"""

	@abstractmethod
	def get(
		self,
		ids: Sequence[str],
		*,
		collection: str = DEFAULT_COLLECTION,
		include_vectors: bool = False,
		include_payloads: bool = True,
	) -> list[VectorRecord]:
		"""Fetch records by id.

		Implementations may return fewer records than requested if ids are missing.
		"""

	@abstractmethod
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
		"""Query by vector similarity.

		Args:
			vector: Query vector
			top_k: Number of results
			filter: Optional metadata filter expression
			collection: Collection name
			include_vectors: Include vectors in returned results
			include_payloads: Include payloads in returned results
		"""

	@abstractmethod
	def count(
		self,
		*,
		collection: str = DEFAULT_COLLECTION,
		filter: Filter | None = None,
	) -> int:
		"""Count records, optionally filtered."""

	def persist(self) -> None:
		"""Persist any pending changes.

		Default is a no-op.
		"""

	def close(self) -> None:
		"""Release any held resources.

		Default is a no-op.
		"""
