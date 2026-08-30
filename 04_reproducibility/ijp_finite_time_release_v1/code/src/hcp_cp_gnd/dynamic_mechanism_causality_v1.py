"""Causal coupling interventions for the finite-time HCP generator.

The intervention keeps the registered nonlinear base trajectory fixed and
modifies only cross-group blocks of the 69-state perturbation generator.  It
therefore answers a perturbation-level question: which admitted couplings have
a marginal effect on the finite-time observable gain?  The returned values are
not dissipation fractions or experimental mechanism probabilities.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import factorial
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, expm_frechet

from .dynamic_crystal_perturbation_v1 import (
    N_GENERATOR,
    DynamicCrystalOperatorV1,
    finite_time_amplification_history,
)


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

MECHANISM_GROUPS: dict[str, NDArray[np.int64]] = {
    "mechanical_inertia": np.arange(0, 6, dtype=np.int64),
    "thermal": np.arange(6, 7, dtype=np.int64),
    "dislocation": np.arange(15, 51, dtype=np.int64),
    "plastic_kinematics": np.r_[
        np.arange(7, 15, dtype=np.int64),
        np.arange(51, 69, dtype=np.int64),
    ],
}


def _edge_name(first: str, second: str) -> str:
    order = list(MECHANISM_GROUPS)
    a, b = sorted((first, second), key=order.index)
    if a == b:
        raise ValueError("a coupling edge requires two distinct mechanism groups")
    return f"{a}__{b}"


COUPLING_EDGES: tuple[str, ...] = tuple(
    _edge_name(first, second)
    for first, second in combinations(MECHANISM_GROUPS, 2)
)


def _matrix(value: Any, name: str) -> ComplexArray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (N_GENERATOR, N_GENERATOR):
        raise ValueError(f"{name} must have shape {(N_GENERATOR, N_GENERATOR)}")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_partition() -> None:
    concatenated = np.concatenate(list(MECHANISM_GROUPS.values()))
    if not np.array_equal(np.sort(concatenated), np.arange(N_GENERATOR)):
        raise RuntimeError("mechanism groups must partition all generator coordinates")
    if len(np.unique(concatenated)) != N_GENERATOR:
        raise RuntimeError("mechanism groups may not overlap")


_validate_partition()


def generator_coupling_components(generator_A: Any) -> dict[str, ComplexArray]:
    """Split a generator into within-group and bidirectional edge blocks."""

    generator = _matrix(generator_A, "generator_A")
    diagonal = np.zeros_like(generator)
    for indices in MECHANISM_GROUPS.values():
        diagonal[np.ix_(indices, indices)] = generator[np.ix_(indices, indices)]
    result = {"within_group": diagonal}
    names = list(MECHANISM_GROUPS)
    for first, second in combinations(names, 2):
        first_indices = MECHANISM_GROUPS[first]
        second_indices = MECHANISM_GROUPS[second]
        component = np.zeros_like(generator)
        component[np.ix_(first_indices, second_indices)] = generator[
            np.ix_(first_indices, second_indices)
        ]
        component[np.ix_(second_indices, first_indices)] = generator[
            np.ix_(second_indices, first_indices)
        ]
        result[_edge_name(first, second)] = component
    reconstructed = sum(result.values(), np.zeros_like(generator))
    if not np.array_equal(reconstructed, generator):
        error = float(np.max(np.abs(reconstructed - generator)))
        raise RuntimeError(f"generator partition failed exact reconstruction: {error}")
    return result


def intervened_generator(
    generator_A: Any,
    *,
    edge_scales: Mapping[str, float] | None = None,
) -> ComplexArray:
    """Return a generator with registered cross-group coupling scales.

    Missing edges have scale zero.  Supplying all edges with unit scale returns
    the original generator exactly; an empty mapping returns the within-group
    block diagonal generator.
    """

    components = generator_coupling_components(generator_A)
    scales = {} if edge_scales is None else dict(edge_scales)
    unknown = set(scales) - set(COUPLING_EDGES)
    if unknown:
        raise ValueError(f"unknown coupling edges: {sorted(unknown)}")
    result = components["within_group"].copy()
    for edge in COUPLING_EDGES:
        scale = float(scales.get(edge, 0.0))
        if not np.isfinite(scale):
            raise ValueError("coupling scales must be finite")
        result += scale * components[edge]
    return result


def intervene_operator(
    operator: DynamicCrystalOperatorV1,
    *,
    edge_scales: Mapping[str, float] | None = None,
) -> DynamicCrystalOperatorV1:
    if not isinstance(operator, DynamicCrystalOperatorV1):
        raise TypeError("operator must be DynamicCrystalOperatorV1")
    return replace(
        operator,
        generator_A=intervened_generator(
            operator.generator_A,
            edge_scales=edge_scales,
        ),
    )


def _coalition_key(coalition: Iterable[str]) -> tuple[str, ...]:
    members = set(coalition)
    unknown = members - set(COUPLING_EDGES)
    if unknown:
        raise ValueError(f"unknown coalition edges: {sorted(unknown)}")
    return tuple(edge for edge in COUPLING_EDGES if edge in members)


def all_couplings_coalition_values(
    operators: Iterable[DynamicCrystalOperatorV1],
    times_s: Iterable[float],
    *,
    coordinate_scales: Any,
    input_indices: Any,
    output_indices: Any,
    gain_threshold: float = float(np.e),
    integration_substeps_per_interval: int = 1,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """Evaluate all 2**6 cross-group coupling coalitions."""

    operator_list = list(operators)
    if len(operator_list) < 2:
        raise ValueError("coalition analysis needs at least two operators")
    values: dict[tuple[str, ...], dict[str, Any]] = {}
    for mask in range(1 << len(COUPLING_EDGES)):
        coalition = tuple(
            edge for bit, edge in enumerate(COUPLING_EDGES) if mask & (1 << bit)
        )
        edge_scales = {edge: 1.0 for edge in coalition}
        intervened = [
            intervene_operator(operator, edge_scales=edge_scales)
            for operator in operator_list
        ]
        raw = finite_time_amplification_history(
            intervened,
            times_s,
            coordinate_scales=coordinate_scales,
            gain_threshold=gain_threshold,
            input_indices=input_indices,
            output_indices=output_indices,
            integration_substeps_per_interval=integration_substeps_per_interval,
        )
        values[coalition] = {
            "gain": float(raw["final_gain"]),
            "log_gain": float(raw["final_log_gain"]),
            "critical_time_s": raw["critical_time_s"],
            "output_mechanism_participation": raw[
                "output_mechanism_participation"
            ],
            "full_state_output_mechanism_participation": raw[
                "full_state_output_mechanism_participation"
            ],
        }
    return values


def exact_shapley_log_gain(
    coalition_values: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    """Return exact Shapley values for the finite-time log gain."""

    normalized = {
        _coalition_key(coalition): float(value["log_gain"])
        for coalition, value in coalition_values.items()
    }
    expected = 1 << len(COUPLING_EDGES)
    if len(normalized) != expected:
        raise ValueError(f"all {expected} coalitions are required")
    if () not in normalized or tuple(COUPLING_EDGES) not in normalized:
        raise ValueError("empty and full coalitions are required")
    m = len(COUPLING_EDGES)
    attribution: dict[str, float] = {}
    for edge in COUPLING_EDGES:
        others = [candidate for candidate in COUPLING_EDGES if candidate != edge]
        contribution = 0.0
        for mask in range(1 << len(others)):
            coalition = tuple(
                candidate
                for bit, candidate in enumerate(others)
                if mask & (1 << bit)
            )
            coalition = _coalition_key(coalition)
            augmented = _coalition_key((*coalition, edge))
            size = len(coalition)
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            contribution += weight * (normalized[augmented] - normalized[coalition])
        attribution[edge] = float(contribution)
    total_difference = float(
        normalized[tuple(COUPLING_EDGES)] - normalized[()]
    )
    efficiency_error = float(sum(attribution.values()) - total_difference)
    absolute_sum = float(sum(abs(value) for value in attribution.values()))
    return {
        "quantity": "finite_time_log_gain",
        "players": list(COUPLING_EDGES),
        "disconnected_log_gain": normalized[()],
        "full_log_gain": normalized[tuple(COUPLING_EDGES)],
        "full_minus_disconnected_log_gain": total_difference,
        "attribution": attribution,
        "share_of_net_difference": {
            edge: (
                None
                if abs(total_difference) <= np.finfo(float).tiny
                else float(value / total_difference)
            )
            for edge, value in attribution.items()
        },
        "share_of_absolute_attribution": {
            edge: (
                None
                if absolute_sum <= np.finfo(float).tiny
                else float(abs(value) / absolute_sum)
            )
            for edge, value in attribution.items()
        },
        "efficiency_error": efficiency_error,
    }


def finite_time_log_gain_edge_sensitivities(
    operators: Iterable[DynamicCrystalOperatorV1],
    times_s: Iterable[float],
    *,
    coordinate_scales: Any,
    input_indices: Any,
    output_indices: Any,
    integration_substeps_per_interval: int = 1,
) -> dict[str, Any]:
    """Differentiate the full-coupling log gain using exponential Fréchet derivatives."""

    operator_list = list(operators)
    times = np.asarray(list(times_s), dtype=np.float64)
    if len(operator_list) < 2 or times.shape != (len(operator_list),):
        raise ValueError("sensitivity analysis needs matching operator/time histories")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    substeps = int(integration_substeps_per_interval)
    if substeps < 1:
        raise ValueError("integration substeps must be positive")
    scales = np.asarray(coordinate_scales, dtype=float)
    inputs = np.asarray(input_indices, dtype=np.int64)
    outputs = np.asarray(output_indices, dtype=np.int64)
    if scales.shape != (N_GENERATOR,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("coordinate scales must be a positive finite 69-vector")
    if inputs.ndim != 1 or outputs.ndim != 1 or inputs.size == 0 or outputs.size == 0:
        raise ValueError("input and output selectors must be nonempty vectors")

    # Work in the locked dimensionless coordinates q=Dz.  This similarity
    # transform is mathematically identical to scaling the final propagator,
    # but it prevents mixed-unit HCP generator entries from overflowing the
    # Fréchet scaling-and-squaring algorithm at intermediate stages.
    scaled_generators = [
        (operator.generator_A * scales[None, :]) / scales[:, None]
        for operator in operator_list
    ]
    phi = np.eye(N_GENERATOR, dtype=np.complex128)
    derivatives = {
        edge: np.zeros_like(phi) for edge in COUPLING_EDGES
    }
    components = [
        {
            name: (component * scales[None, :]) / scales[:, None]
            for name, component in generator_coupling_components(
                operator.generator_A
            ).items()
        }
        for operator in operator_list
    ]
    for interval, duration in enumerate(np.diff(times)):
        step = float(duration) / substeps
        for substep in range(substeps):
            fraction = (substep + 0.5) / substeps
            generator = (
                (1.0 - fraction) * scaled_generators[interval]
                + fraction * scaled_generators[interval + 1]
            )
            scaled_generator = generator * step
            exponential = expm(scaled_generator)
            old_phi = phi
            old_derivatives = derivatives
            derivatives = {}
            for edge in COUPLING_EDGES:
                edge_midpoint = (
                    (1.0 - fraction) * components[interval][edge]
                    + fraction * components[interval + 1][edge]
                )
                frechet = expm_frechet(
                    scaled_generator,
                    edge_midpoint * step,
                    compute_expm=False,
                )
                derivatives[edge] = (
                    frechet @ old_phi + exponential @ old_derivatives[edge]
                )
            phi = exponential @ old_phi

    observed = phi[np.ix_(outputs, inputs)]
    left, singular, right_h = np.linalg.svd(observed, full_matrices=False)
    if singular[0] <= np.finfo(float).tiny:
        raise FloatingPointError("leading observable gain is zero")
    singular_gap = (
        float((singular[0] - singular[1]) / singular[0])
        if singular.size > 1 else 1.0
    )
    if singular_gap <= 1.0e-10:
        raise FloatingPointError("leading singular value is not simple enough to differentiate")
    left_vector = left[:, 0]
    right_vector = right_h[0].conj()
    sensitivities: dict[str, float] = {}
    for edge, derivative in derivatives.items():
        observed_derivative = derivative[np.ix_(outputs, inputs)]
        derivative_gain = float(
            np.real(np.vdot(left_vector, observed_derivative @ right_vector))
        )
        sensitivities[edge] = float(derivative_gain / singular[0])
    return {
        "gain": float(singular[0]),
        "log_gain": float(np.log(singular[0])),
        "relative_leading_singular_gap": singular_gap,
        "d_log_gain_d_edge_scale_at_full_coupling": sensitivities,
    }


__all__ = [
    "COUPLING_EDGES",
    "MECHANISM_GROUPS",
    "all_couplings_coalition_values",
    "exact_shapley_log_gain",
    "finite_time_log_gain_edge_sensitivities",
    "generator_coupling_components",
    "intervene_operator",
    "intervened_generator",
]
