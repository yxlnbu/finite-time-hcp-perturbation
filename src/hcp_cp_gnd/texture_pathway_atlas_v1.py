"""Deterministic orientation design and pathway summaries for CP-Ti."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable

import numpy as np


ORIENTATION_GRID_DEG = tuple(
    (float(phi1), float(Phi), float(phi2))
    for phi1 in (0, 30, 60, 90)
    for Phi in (0, 30, 60, 90)
    for phi2 in (0, 30)
)
LOG_GAIN_THRESHOLDS = (1.0, 5.0, log(1.0e4))


def projective_hemisphere_directions() -> tuple[np.ndarray, ...]:
    """Return a deterministic projective-hemisphere selector grid."""

    raw = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0),
        (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    ]
    output: list[np.ndarray] = []
    for values in raw:
        direction = np.asarray(values, dtype=float)
        direction /= np.linalg.norm(direction)
        # Canonical representative of n~-n: first nonzero component positive.
        first = int(np.flatnonzero(np.abs(direction) > 1.0e-14)[0])
        if direction[first] < 0.0:
            direction *= -1.0
        if not any(np.allclose(direction, item, atol=1.0e-14) for item in output):
            output.append(direction)
    return tuple(output)


def select_storage_representatives(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select minimum, median-nearest, and maximum storage rows a priori."""

    values = list(rows)
    if len(values) < 3:
        raise ValueError("at least three orientation rows are required")
    if any("storage_fraction_at_shear_0p2" not in row for row in values):
        raise ValueError("orientation rows lack the registered storage metric")
    ordered = sorted(
        values,
        key=lambda row: (
            float(row["storage_fraction_at_shear_0p2"]),
            tuple(row["orientation_bunge_deg"]),
        ),
    )
    median = float(np.median([row["storage_fraction_at_shear_0p2"] for row in ordered]))
    middle = min(
        ordered[1:-1],
        key=lambda row: (
            abs(float(row["storage_fraction_at_shear_0p2"]) - median),
            tuple(row["orientation_bunge_deg"]),
        ),
    )
    return {"minimum_storage": ordered[0], "median_storage": middle, "maximum_storage": ordered[-1]}


def crossing_time_log_linear(
    times_s: Any, gains: Any, log_threshold: float
) -> float | None:
    """First log-linear gain crossing with an explicit threshold."""

    times = np.asarray(times_s, dtype=float)
    values = np.asarray(gains, dtype=float)
    threshold = float(log_threshold)
    if (
        times.ndim != 1
        or times.shape != values.shape
        or times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(times) <= 0.0)
        or np.any(values <= 0.0)
        or not np.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError("invalid gain-crossing data")
    logs = np.log(values)
    crossed = np.flatnonzero(logs >= threshold)
    if not crossed.size:
        return None
    right = int(crossed[0])
    if right == 0:
        return float(times[0])
    left = right - 1
    denominator = logs[right] - logs[left]
    if abs(denominator) <= np.finfo(float).tiny:
        return float(times[right])
    fraction = (threshold - logs[left]) / denominator
    return float(times[left] + fraction * (times[right] - times[left]))


def pathway_group_scores(attribution: dict[str, float]) -> dict[str, Any]:
    """Allocate positive edge Shapley values equally to their endpoints."""

    groups = ("mechanical_inertia", "thermal", "dislocation", "plastic_kinematics")
    scores = {group: 0.0 for group in groups}
    signed = {group: 0.0 for group in groups}
    for edge, raw_value in attribution.items():
        left, right = edge.split("__")
        if left not in scores or right not in scores:
            raise ValueError(f"unknown mechanism edge {edge}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError("non-finite Shapley attribution")
        signed[left] += 0.5 * value
        signed[right] += 0.5 * value
        positive = 0.5 * max(value, 0.0)
        scores[left] += positive
        scores[right] += positive
    thermal = scores["thermal"]
    defect = scores["dislocation"]
    reference = max(thermal, defect, 1.0e-300)
    if abs(thermal - defect) <= 0.10 * reference:
        label = "mixed_thermal_dislocation"
    elif thermal > defect:
        label = "thermally_assisted"
    else:
        label = "defect_storage_coupled"
    return {
        "positive_endpoint_scores": scores,
        "signed_endpoint_allocations": signed,
        "thermal_to_dislocation_positive_score_ratio": float(thermal / max(defect, 1.0e-300)),
        "pathway_label": label,
        "definition": "positive cross-group Shapley edge values split equally between endpoints; 10% mixed band",
    }


@dataclass(frozen=True)
class SelectorGridV1:
    wavenumbers_m_inv: tuple[float, ...] = tuple(np.geomspace(3.0e2, 3.0e5, 9))
    directions_n: tuple[tuple[float, float, float], ...] = tuple(
        tuple(float(value) for value in direction)
        for direction in projective_hemisphere_directions()
    )

    def validate(self) -> None:
        k = np.asarray(self.wavenumbers_m_inv, dtype=float)
        n = np.asarray(self.directions_n, dtype=float)
        if (
            k.ndim != 1
            or k.size < 5
            or np.any(~np.isfinite(k))
            or np.any(k <= 0.0)
            or np.any(np.diff(k) <= 0.0)
            or n.ndim != 2
            or n.shape[1] != 3
            or np.any(~np.isfinite(n))
            or np.any(np.abs(np.linalg.norm(n, axis=1) - 1.0) > 1.0e-12)
        ):
            raise ValueError("invalid selector grid")


__all__ = [
    "LOG_GAIN_THRESHOLDS",
    "ORIENTATION_GRID_DEG",
    "SelectorGridV1",
    "crossing_time_log_linear",
    "pathway_group_scores",
    "projective_hemisphere_directions",
    "select_storage_representatives",
]
