"""
neurokit.knowledge.vector_store.exceptions
Vector store specific exceptions.
"""

from __future__ import annotations

from typing import Any

from neurokit.core.exceptions.base import NeuroKitError


class VectorStoreError(NeuroKitError):
    """Base exception for all vector store errors."""


class BackendNotInstalled(VectorStoreError):
    """Raised when an optional backend dependency is missing."""

    def __init__(self, backend: str, extra: str) -> None:
        super().__init__(
            message=f"Vector store backend '{backend}' is not installed.",
            details={"backend": backend, "install": f"pip install neurokit[{extra}]"},
        )


class CollectionNotFound(VectorStoreError):
    """Raised when a collection is missing."""

    def __init__(self, collection: str, *, backend: str | None = None) -> None:
        details: dict[str, Any] = {"collection": collection}
        if backend:
            details["backend"] = backend
        super().__init__(message="Collection not found.", details=details)


class DimensionMismatch(VectorStoreError):
    """Raised when vector dimensions don't match the collection schema."""

    def __init__(
        self,
        *,
        expected: int,
        actual: int,
        collection: str,
        backend: str | None = None,
    ) -> None:
        details: dict[str, Any] = {
            "expected": expected,
            "actual": actual,
            "collection": collection,
        }
        if backend:
            details["backend"] = backend
        super().__init__(message="Vector dimension mismatch.", details=details)
