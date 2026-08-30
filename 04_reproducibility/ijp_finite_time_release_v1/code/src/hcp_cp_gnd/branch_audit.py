"""Neutral branch metadata shared by continuous constitutive utilities.

This module has no endpoint-integration, UEL, or Schur-complement dependency.
It keeps the continuous spectral export independent of the backward-Euler
implementation while retaining a machine-comparable active-set contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _frozen(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


class SpectralNondifferentiableError(RuntimeError):
    """A continuous derivative probe cannot remain on one smooth branch."""


class SpectralAdmissibilityError(ValueError):
    """A derivative probe left the open physical domain of the base point."""

    def __init__(self, message: str, *, boundary_label: str) -> None:
        super().__init__(message)
        if not boundary_label.startswith("admissibility."):
            raise ValueError("admissibility boundary labels must use the registered prefix")
        self.boundary_label = boundary_label


@dataclass(frozen=True)
class SpectralBranchAudit:
    labels: tuple[str, ...]
    categories: tuple[int, ...]
    signed_distances: Array
    distance_scales: Array

    def __post_init__(self) -> None:
        count = len(self.labels)
        if len(self.categories) != count or count == 0:
            raise ValueError("branch labels and categories must be non-empty and aligned")
        object.__setattr__(
            self,
            "signed_distances",
            _frozen(self.signed_distances, (count,), "branch signed distances"),
        )
        object.__setattr__(
            self,
            "distance_scales",
            _frozen(self.distance_scales, (count,), "branch distance scales"),
        )
        if np.any(self.distance_scales <= 0.0):
            raise ValueError("branch distance scales must be positive")

    @property
    def normalized_distances(self) -> Array:
        result = np.abs(self.signed_distances) / self.distance_scales
        result.setflags(write=False)
        return result

    @property
    def category_tokens(self) -> tuple[str, ...]:
        return tuple(
            f"{label}={category}" for label, category in zip(self.labels, self.categories)
        )

    def compatible_with(
        self,
        plus: "SpectralBranchAudit",
        minus: "SpectralBranchAudit",
        *,
        normalized_distance_floor: float = 1.0e-10,
    ) -> bool:
        if self.labels != plus.labels or self.labels != minus.labels:
            return False
        for index, label in enumerate(self.labels):
            if (
                label.startswith("switch.")
                or label.startswith("admissibility.")
                or label.endswith(".dissipation")
            ):
                continue
            if (
                self.categories[index] != plus.categories[index]
                or self.categories[index] != minus.categories[index]
            ):
                return False
            scale = max(
                float(self.distance_scales[index]),
                float(plus.distance_scales[index]),
                float(minus.distance_scales[index]),
            )
            distance = min(
                abs(float(self.signed_distances[index])),
                abs(float(plus.signed_distances[index])),
                abs(float(minus.signed_distances[index])),
            ) / scale
            if distance <= normalized_distance_floor:
                return False
        return True


__all__ = [
    "SpectralAdmissibilityError",
    "SpectralBranchAudit",
    "SpectralNondifferentiableError",
]
