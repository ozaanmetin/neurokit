"""
neurokit.knowledge.vector_store.filter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class Filter:
    """Base class for filter expressions."""

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:  # pragma: no cover
        raise NotImplementedError

    def __and__(self, other: "Filter") -> "Filter":
        return And(self, other)

    def __or__(self, other: "Filter") -> "Filter":
        return Or(self, other)

    def __invert__(self) -> "Filter":
        return Not(self)


@dataclass(frozen=True)
class Field:
    name: str

    def eq(self, value: Any) -> Filter:
        return Eq(self.name, value)

    def in_(self, values: Sequence[Any]) -> Filter:
        return In(self.name, tuple(values))

    def range(
        self,
        *,
        gt: Any | None = None,
        gte: Any | None = None,
        lt: Any | None = None,
        lte: Any | None = None,
    ) -> Filter:
        return Range(self.name, gt=gt, gte=gte, lt=lt, lte=lte)


def F(name: str) -> Field:
    return Field(name)


@dataclass(frozen=True)
class And(Filter):
    left: Filter
    right: Filter

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        return self.left.evaluate(metadata) and self.right.evaluate(metadata)


@dataclass(frozen=True)
class Or(Filter):
    left: Filter
    right: Filter

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        return self.left.evaluate(metadata) or self.right.evaluate(metadata)


@dataclass(frozen=True)
class Not(Filter):
    inner: Filter

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        return not self.inner.evaluate(metadata)


@dataclass(frozen=True)
class Eq(Filter):
    field: str
    value: Any

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        return metadata.get(self.field) == self.value


@dataclass(frozen=True)
class In(Filter):
    field: str
    values: tuple[Any, ...]

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        return metadata.get(self.field) in self.values


@dataclass(frozen=True)
class Range(Filter):
    field: str
    gt: Any | None = None
    gte: Any | None = None
    lt: Any | None = None
    lte: Any | None = None

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        value = metadata.get(self.field)
        if value is None:
            return False

        try:
            if self.gt is not None and not (value > self.gt):
                return False
            if self.gte is not None and not (value >= self.gte):
                return False
            if self.lt is not None and not (value < self.lt):
                return False
            if self.lte is not None and not (value <= self.lte):
                return False
        except TypeError:
            return False

        return True
