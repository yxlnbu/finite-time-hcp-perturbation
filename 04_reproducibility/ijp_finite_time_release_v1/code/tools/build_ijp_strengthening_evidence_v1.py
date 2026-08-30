"""Build the reviewer-strengthening evidence for the finite-time HCP paper.

The audit adds four evidence layers without changing the registered material
model or observable norm:

1. stationary-normal and commuting-normal positive controls for which the
   frozen spectral integral and the propagator norm must agree;
2. same-branch finite-time/frozen discrimination for three crystal
   orientations selected independently by the existing texture atlas;
3. a nested projective-sphere/log-wavenumber anchor-assisted search audit with
   multistart local refinement, basin recurrence, boundary checks, and a
   129-state verification of the retained winners; and
4. release arrays containing every 69 x 69 generator checkpoint and the
   complete complex singular vectors for every branch reported in the paper.

This is an anchor-assisted basin-retention and convergence audit on the
registered compact domain.  The blind Sobol miss is retained as a failed
discovery test, so the result is not called a global-optimum certificate.
Likewise, it does not claim a strict HCP-to-Bai asymptotic reduction.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import cos, pi, sin
from pathlib import Path
import sys
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)
from tools.audit_cp_ti_continuous_spectrum_robustness_v1 import (  # noqa: E402
    BOUNDS,
    K_BOUNDS,
    OBSERVED,
    SpectrumAudit,
    angle_degrees,
    coordinates_from_direction,
    direction_from_coordinates,
)
from tools.run_cp_ti_texture_pathway_v1 import (  # noqa: E402
    BASE_FACTOR,
    BASE_SHEARS,
    build_spectral_context,
    refined_shears,
)


ROBUSTNESS = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2.json"
TEXTURE = ROOT / "05_results/cp_ti_texture_pathway_v1.json"
RESULT = ROOT / "05_results/ijp_strengthening_evidence_v1.json"
ARRAYS = ROOT / "05_results/ijp_strengthening_evidence_v1_arrays.npz"
TABLE_DIR = ROOT / "05_results/ijp_strengthening_evidence_v1"
FIGURE_DIR = ROOT / "06_manuscript/ijp_spectral_hcp/figures"
FIGURE_STEM = FIGURE_DIR / "fig08_positive_orientation_anchor_audit"

QMC_LEVELS = (64, 256, 1024)
QMC_CONTEXT = "factor2"
QMC_SUBSTEPS = 1
LOCAL_SUBSTEPS = 2
REFERENCE_CONTEXT = "factor16"
REFERENCE_SUBSTEPS = 4
GAIN_THRESHOLD = float(np.e)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative_change(first: float, second: float) -> float:
    return float(abs(first / second - 1.0))


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    output[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(times))
    return output


def full_state_prefix_log_gains(
    generators: np.ndarray,
    times: np.ndarray,
    scales: np.ndarray,
    *,
    substeps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the complete scaled state for a like-for-like norm audit."""

    dimension = int(generators.shape[1])
    phi = np.eye(dimension, dtype=np.complex128)
    propagators = [phi.copy()]
    log_gains = [0.0]
    scaled = [
        (matrix * scales[None, :]) / scales[:, None] for matrix in generators
    ]
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index]) / substeps
        for substep in range(substeps):
            fraction = (substep + 0.5) / substeps
            midpoint = (
                (1.0 - fraction) * scaled[index]
                + fraction * scaled[index + 1]
            )
            phi = expm(midpoint * dt) @ phi
        propagators.append(phi.copy())
        log_gains.append(
            float(np.log(max(np.linalg.svd(phi, compute_uv=False)[0], np.finfo(float).tiny)))
        )
    return np.asarray(propagators), np.asarray(log_gains)


def reassess_orientation_full_state(
    orientation: dict[str, Any],
) -> dict[str, Any]:
    """Replace historical projected differences by complete-state comparisons."""

    require(ARRAYS.is_file(), "missing strengthening arrays for orientation reassessment")
    with np.load(ARRAYS) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    table_rows = []
    for label, record in orientation["orientations"].items():
        prefix = f"orientation_{label}"
        times = np.asarray(payload[f"{prefix}_times_s"], dtype=float)
        scales = np.asarray(payload[f"{prefix}_scales"], dtype=float)
        generators = np.asarray(payload[f"{prefix}_generators"])
        propagators, full_log_gain = full_state_prefix_log_gains(
            generators, times, scales, substeps=2
        )
        released = np.asarray(payload[f"{prefix}_propagator"])
        endpoint_error = float(
            np.linalg.norm(propagators[-1] - released, ord="fro")
            / max(np.linalg.norm(released, ord="fro"), np.finfo(float).tiny)
        )
        require(endpoint_error <= 2.0e-10, f"{label} full-state endpoint mismatch")
        frozen = np.asarray(payload[f"{prefix}_frozen_integral_log_gain"], dtype=float)
        discrepancy = full_log_gain - frozen
        target_index = int(
            np.argmin(np.abs(times - float(record["selection_horizon_s"])))
        )
        record["projected_selection_horizon_log_gain_historical"] = record[
            "selection_horizon_finite_time_log_gain"
        ]
        record["projected_terminal_log_gain_historical"] = record[
            "terminal_finite_time_log_gain"
        ]
        record["comparison_contract"] = {
            "state_space": "complete reduced 69-state generator",
            "norm": "fixed D-scaled Euclidean norm",
            "input_selector": "identity",
            "output_selector": "identity",
        }
        record["released_propagator_relative_error"] = endpoint_error
        record["selection_horizon_finite_time_log_gain"] = float(
            full_log_gain[target_index]
        )
        record["selection_horizon_log_discrepancy"] = float(discrepancy[target_index])
        record["terminal_finite_time_log_gain"] = float(full_log_gain[-1])
        record["terminal_log_discrepancy"] = float(discrepancy[-1])
        record["maximum_abs_log_discrepancy"] = float(np.max(np.abs(discrepancy)))
        record["non_equivalence_gate"] = bool(
            max(
                abs(record["selection_horizon_log_discrepancy"]),
                abs(record["terminal_log_discrepancy"]),
            )
            >= 0.10
        )
        payload[f"{prefix}_full_state_prefix_propagators"] = propagators
        payload[f"{prefix}_full_state_log_gain"] = full_log_gain
        table_rows.append(record)
    np.savez_compressed(ARRAYS, **payload)
    write_csv(TABLE_DIR / "orientation_transfer.csv", table_rows)
    orientation["selection_rule"] = (
        "Each orientation retains its independently refined direction-wavenumber "
        "selector. The propagated gain and frozen spectral integral are reassessed "
        "on the same complete 69-state space, D-scaled Euclidean norm and identity "
        "input/output selectors; historical projected gains remain labeled diagnostics."
    )
    orientation["status"] = (
        "THREE_ORIENTATION_TRANSFER_PASS"
        if all(item["non_equivalence_gate"] for item in orientation["orientations"].values())
        else "ORIENTATION_TRANSFER_INCOMPLETE"
    )
    return orientation


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def fixed_orthogonal_basis() -> np.ndarray:
    raw = np.asarray(
        [[1.0, 2.0, -1.0], [2.0, -1.0, 1.0], [1.0, 1.0, 2.0]], dtype=float
    )
    basis, _ = np.linalg.qr(raw)
    return basis


def positive_control_case(
    identifier: str,
    eigenvalues: Callable[[float], np.ndarray],
    *,
    expected_equivalence: bool,
    rationale: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Propagate a common-basis normal family using the same midpoint rule."""

    basis = fixed_orthogonal_basis().astype(np.complex128)
    times = np.linspace(0.0, 2.0, 401)
    propagator = np.eye(3, dtype=np.complex128)
    frozen = [0.0]
    propagated = [0.0]
    alpha_midpoints = []
    normality = []
    commutators = []
    previous = None
    modal_integrals = np.zeros(3)
    modal_integral_history = [modal_integrals.copy()]
    for left, right in zip(times[:-1], times[1:], strict=True):
        midpoint = 0.5 * (left + right)
        dt = float(right - left)
        lambdas = np.asarray(eigenvalues(midpoint), dtype=np.complex128)
        operator = basis @ np.diag(lambdas) @ basis.conj().T
        normality.append(
            float(
                np.linalg.norm(operator.conj().T @ operator - operator @ operator.conj().T)
                / max(np.linalg.norm(operator, ord="fro") ** 2, np.finfo(float).tiny)
            )
        )
        if previous is not None:
            commutators.append(
                float(
                    np.linalg.norm(operator @ previous - previous @ operator, ord="fro")
                    / max(
                        np.linalg.norm(operator, ord="fro")
                        * np.linalg.norm(previous, ord="fro"),
                        np.finfo(float).tiny,
                    )
                )
            )
        previous = operator
        propagator = expm(operator * dt) @ propagator
        alpha = float(np.max(lambdas.real))
        alpha_midpoints.append(alpha)
        frozen.append(frozen[-1] + alpha * dt)
        propagated.append(float(np.log(np.linalg.svd(propagator, compute_uv=False)[0])))
        modal_integrals += lambdas.real * dt
        modal_integral_history.append(modal_integrals.copy())
    frozen_array = np.asarray(frozen)
    propagated_array = np.asarray(propagated)
    discrepancy = propagated_array - frozen_array
    final_error = float(discrepancy[-1])
    tolerance = 2.0e-11
    equivalence_observed = abs(final_error) <= tolerance
    require(
        equivalence_observed == expected_equivalence,
        f"positive-control expectation failed for {identifier}: {final_error}",
    )
    record = {
        "case_id": identifier,
        "operator_family": "pairwise_commuting_common-basis_normal",
        "rationale": rationale,
        "expected_equivalence": expected_equivalence,
        "observed_equivalence": equivalence_observed,
        "final_propagator_log_gain": float(propagated_array[-1]),
        "final_frozen_integral_log_gain": float(frozen_array[-1]),
        "final_log_discrepancy": final_error,
        "maximum_abs_log_discrepancy": float(np.max(np.abs(discrepancy))),
        "maximum_normality_residual": float(max(normality)),
        "maximum_adjacent_commutator_residual": float(max(commutators, default=0.0)),
        "same_pointwise_leading_mode": bool(
            np.unique(
                [int(np.argmax(np.asarray(eigenvalues(t)).real)) for t in times]
            ).size
            == 1
        ),
        "quadrature": "common exponential-midpoint / midpoint frozen integral",
        "tolerance": tolerance,
    }
    arrays = {
        "times": times,
        "propagator_log_gain": propagated_array,
        "frozen_integral_log_gain": frozen_array,
        "modal_integral_history": np.asarray(modal_integral_history),
        "alpha_midpoints": np.asarray(alpha_midpoints),
    }
    return record, arrays


def positive_controls() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stationary, stationary_arrays = positive_control_case(
        "stationary_normal",
        lambda _t: np.asarray([2.0 + 0.35j, 0.4 - 0.2j, -0.8 + 0.1j]),
        expected_equivalence=True,
        rationale="A constant normal generator has one fixed orthonormal eigenbasis and one fixed leading mode.",
    )
    commuting, commuting_arrays = positive_control_case(
        "commuting_normal_fixed_leader",
        lambda t: np.asarray(
            [1.6 + 0.20 * sin(pi * t) + 0.25j, 0.35 + 0.10 * cos(pi * t) - 0.1j, -0.5]
        ),
        expected_equivalence=True,
        rationale=(
            "The generator varies in time but remains normal, pairwise commuting, and has the same "
            "pointwise leading common eigenvector throughout the window."
        ),
    )
    switching, switching_arrays = positive_control_case(
        "commuting_normal_switching_leader",
        lambda t: np.asarray(
            [1.0 + 0.9 * sin(pi * t), 1.0 - 0.9 * sin(pi * t), -0.4]
        ),
        expected_equivalence=False,
        rationale=(
            "Pairwise commutation alone is insufficient: when the pointwise leading common mode "
            "switches, integral(max alpha_j) exceeds max_j integral(alpha_j)."
        ),
    )
    cases = [stationary, commuting, switching]
    gate = all(item["observed_equivalence"] for item in cases[:2]) and not cases[2][
        "observed_equivalence"
    ]
    arrays = {}
    for prefix, source in (
        ("positive_stationary", stationary_arrays),
        ("positive_commuting", commuting_arrays),
        ("positive_switching", switching_arrays),
    ):
        for key, value in source.items():
            arrays[f"{prefix}_{key}"] = value
    return {
        "status": "POSITIVE_CONTROLS_PASS" if gate else "POSITIVE_CONTROL_FAILURE",
        "theorem_scope": (
            "For a pairwise-commuting normal family with a common orthonormal eigenbasis, "
            "log||Phi||_2=max_j integral Re(lambda_j)dt.  Equality with integral alpha(t)dt "
            "additionally requires one common mode to attain the pointwise maximum almost everywhere."
        ),
        "cases": cases,
        "gates": {
            "stationary_normal_equality": stationary["observed_equivalence"],
            "commuting_normal_fixed_leader_equality": commuting["observed_equivalence"],
            "switching_leader_non_equivalence_detected": not switching["observed_equivalence"],
        },
    }, arrays


def operator_spectral_abscissae(operators: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    abscissae = []
    residuals = []
    for operator in operators:
        roots, _, errors = operator.admitted_eigenpairs()
        dominant = int(np.argmax(roots.real))
        abscissae.append(float(roots[dominant].real))
        residuals.append(float(errors[dominant]))
    return np.asarray(abscissae), np.asarray(residuals)


def orientation_transfer(
    texture: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fine_shears = refined_shears(BASE_SHEARS)
    analyses = texture["analyses"]
    records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    table_rows = []
    for label in ("minimum_storage", "median_storage", "maximum_storage"):
        source = analyses[label]
        orientation = tuple(float(value) for value in source["orientation_bunge_deg"])
        selector = source["refined_selector"]
        context = build_spectral_context(
            orientation,
            BASE_FACTOR,
            fine_shears,
            evolving_base=True,
            evolving_operator=True,
            label=f"strengthening_{label}",
        )
        normal = np.asarray(selector["direction_n"], dtype=float)
        wavenumber = float(selector["wavenumber_m_inv"])
        operators = context.operators(wavenumber, normal)
        propagated = context.propagation(wavenumber, normal, substeps=2)
        alpha, residuals = operator_spectral_abscissae(operators)
        frozen = cumulative_trapezoid(alpha, context.times_s)
        finite_time = np.asarray([row["log_gain"] for row in propagated["prefix"]])
        discrepancy = finite_time - frozen
        coarse_target_time = float(
            source["coarse_selector"]["envelope"][int(selector["target_prefix_index"])][
                "time_s"
            ]
        )
        target_index = int(
            np.argmin(np.abs(np.asarray(context.times_s) - coarse_target_time))
        )
        commutators = []
        scaled = [
            (item.generator_A * context.scales[None, :]) / context.scales[:, None]
            for item in operators
        ]
        for first, second in zip(scaled[:-1], scaled[1:], strict=True):
            commutators.append(
                float(
                    np.linalg.norm(second @ first - first @ second, ord="fro")
                    / max(
                        np.linalg.norm(first, ord="fro")
                        * np.linalg.norm(second, ord="fro"),
                        np.finfo(float).tiny,
                    )
                )
            )
        selection_difference = float(discrepancy[target_index])
        terminal_difference = float(discrepancy[-1])
        record = {
            "orientation_id": label,
            "orientation_bunge_deg": list(orientation),
            "loading_path": "homogeneous simple shear F12 at 1e4 s^-1",
            "direction_n": normal.tolist(),
            "k_m_inv": wavenumber,
            "selection_horizon_s": float(context.times_s[target_index]),
            "selection_horizon_finite_time_log_gain": float(finite_time[target_index]),
            "selection_horizon_frozen_integral_log_gain": float(frozen[target_index]),
            "selection_horizon_log_discrepancy": selection_difference,
            "terminal_horizon_s": float(context.times_s[-1]),
            "terminal_finite_time_log_gain": float(finite_time[-1]),
            "terminal_frozen_integral_log_gain": float(frozen[-1]),
            "terminal_log_discrepancy": terminal_difference,
            "maximum_abs_log_discrepancy": float(np.max(np.abs(discrepancy))),
            "maximum_modal_backward_error": float(np.max(residuals)),
            "maximum_normalized_adjacent_commutator": float(max(commutators)),
            "non_equivalence_gate": bool(
                max(abs(selection_difference), abs(terminal_difference)) >= 0.10
            ),
        }
        records[label] = record
        table_rows.append(record)
        prefix = f"orientation_{label}"
        arrays[f"{prefix}_times_s"] = np.asarray(context.times_s)
        arrays[f"{prefix}_scales"] = np.asarray(context.scales)
        arrays[f"{prefix}_direction_n"] = normal
        arrays[f"{prefix}_k_m_inv"] = np.asarray([wavenumber])
        arrays[f"{prefix}_generators"] = np.stack(
            [item.generator_A for item in operators]
        )
        arrays[f"{prefix}_propagator"] = propagated["propagator_dimensionless"]
        arrays[f"{prefix}_input_vector"] = propagated["input_vector_dimensionless"]
        arrays[f"{prefix}_output_vector"] = propagated["output_vector_dimensionless"]
        arrays[f"{prefix}_full_output_response"] = propagated[
            "full_state_output_response_dimensionless"
        ]
        arrays[f"{prefix}_finite_time_log_gain"] = finite_time
        arrays[f"{prefix}_frozen_integral_log_gain"] = frozen
    write_csv(TABLE_DIR / "orientation_transfer.csv", table_rows)
    all_pass = all(item["non_equivalence_gate"] for item in records.values())
    return {
        "status": (
            "THREE_ORIENTATION_TRANSFER_PASS"
            if all_pass
            else "ORIENTATION_TRANSFER_INCOMPLETE"
        ),
        "orientation_count": len(records),
        "selection_rule": (
            "Each orientation uses its independently refined direction-wavenumber selector "
            "from the pre-existing texture pathway audit; finite-time and frozen quantities "
            "then share that branch, base history, norm, and observation subspace."
        ),
        "non_equivalence_tolerance_log_gain": 0.10,
        "orientations": records,
        "gates": {
            "at_least_three_orientations": len(records) >= 3,
            "non_equivalence_on_every_orientation": all_pass,
            "all_modal_backward_errors_below_1e_minus_8": all(
                item["maximum_modal_backward_error"] <= 1.0e-8
                for item in records.values()
            ),
        },
    }, arrays


def qmc_coordinates(count: int = QMC_LEVELS[-1]) -> np.ndarray:
    require(count > 0 and count & (count - 1) == 0, "Sobol count must be a power of two")
    engine = qmc.Sobol(d=3, scramble=True, seed=20260829)
    unit = engine.random_base2(int(np.log2(count)))
    z = unit[:, 0]
    phi = -pi + 2.0 * pi * unit[:, 1]
    theta = np.arccos(z)
    log_k = BOUNDS[2][0] + unit[:, 2] * (BOUNDS[2][1] - BOUNDS[2][0])
    return np.c_[theta, phi, log_k]


def anchor_coordinates(robustness: dict[str, Any]) -> list[np.ndarray]:
    records = robustness["continuous_selection"]
    values = [
        records["onset_branches"]["sample_x_branch"]["upper"],
        records["onset_branches"]["sample_y_branch"]["upper"],
    ]
    values.extend(records["final_branches"].values())
    starts = [
        coordinates_from_direction(item["direction_n"], float(item["k_m_inv"]))
        for item in values
    ]
    for normal in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        for k_value in np.geomspace(K_BOUNDS[0], K_BOUNDS[1], 9):
            starts.append(coordinates_from_direction(normal, float(k_value)))
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for value in starts:
        unique[tuple(np.round(value, 9))] = value
    return list(unique.values())


def coverage_estimate(coordinates: np.ndarray) -> dict[str, float]:
    directions = np.stack([direction_from_coordinates(value) for value in coordinates])
    tree = cKDTree(np.vstack((directions, -directions)))
    probe_count = 20000
    golden = pi * (3.0 - np.sqrt(5.0))
    index = np.arange(probe_count)
    z = (index + 0.5) / probe_count
    radius = np.sqrt(1.0 - z * z)
    probes = np.c_[radius * np.cos(golden * index), radius * np.sin(golden * index), z]
    distance, _ = tree.query(probes, k=1)
    angular = 2.0 * np.arcsin(np.clip(distance / 2.0, 0.0, 1.0))
    logs = np.sort(np.r_[BOUNDS[2][0], coordinates[:, 2], BOUNDS[2][1]])
    return {
        "empirical_projective_sphere_fill_angle_deg": float(np.degrees(np.max(angular))),
        "maximum_log_k_gap": float(np.max(np.diff(logs))),
        "probe_direction_count": probe_count,
    }


def choose_separated_starts(
    coordinates: np.ndarray,
    scores: np.ndarray,
    anchors: list[np.ndarray],
    maximum: int = 8,
) -> list[np.ndarray]:
    candidates = [coordinates[index] for index in np.argsort(scores)[::-1]]
    candidates.extend(anchors)
    selected: list[np.ndarray] = []
    for candidate in candidates:
        normal = direction_from_coordinates(candidate)
        if all(
            angle_degrees(normal, direction_from_coordinates(existing)) >= 2.0
            or abs(float(candidate[2] - existing[2])) >= 0.08
            for existing in selected
        ):
            selected.append(np.asarray(candidate, dtype=float))
        if len(selected) >= maximum:
            break
    return selected


def local_refinement(
    audit: SpectrumAudit,
    prefix_index: int,
    start: np.ndarray,
    run_id: str,
) -> dict[str, Any]:
    trace = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(x: Any) -> dict[str, Any]:
        raw = np.asarray(x, dtype=float)
        key = tuple(np.round(raw, 11))
        if key not in cache:
            value = audit.history(
                QMC_CONTEXT,
                prefix_index,
                raw,
                integration_substeps_per_interval=LOCAL_SUBSTEPS,
            )
            cache[key] = value
            trace.append(
                {
                    "evaluation": len(trace),
                    "theta_rad": float(raw[0]),
                    "phi_rad": float(raw[1]),
                    "log_k": float(raw[2]),
                    "n1": float(value["direction_n"][0]),
                    "n2": float(value["direction_n"][1]),
                    "n3": float(value["direction_n"][2]),
                    "k_m_inv": float(value["k_m_inv"]),
                    "log_gain": float(value["log_gain"]),
                }
            )
        return cache[key]

    start_value = evaluate(start)
    result = minimize(
        lambda x: -evaluate(x)["log_gain"],
        start,
        method="Powell",
        bounds=BOUNDS,
        options={"xtol": 1.0e-5, "ftol": 1.0e-8, "maxiter": 90},
    )
    optimized = evaluate(result.x)
    retained = start_value if start_value["log_gain"] > optimized["log_gain"] else optimized
    return {
        "run_id": run_id,
        "start_coordinates": np.asarray(start).tolist(),
        "winner_coordinates": coordinates_from_direction(
            retained["direction_n"], retained["k_m_inv"]
        ).tolist(),
        "direction_n": retained["direction_n"],
        "k_m_inv": retained["k_m_inv"],
        "gain": retained["gain"],
        "log_gain": retained["log_gain"],
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "retained_start_instead_of_optimizer": retained is start_value,
        "trace": trace,
    }


def high_resolution_verification(
    audit: SpectrumAudit,
    prefix_index: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    x = coordinates_from_direction(candidate["direction_n"], candidate["k_m_inv"])
    value = audit.history(
        REFERENCE_CONTEXT,
        prefix_index,
        x,
        integration_substeps_per_interval=REFERENCE_SUBSTEPS,
    )
    return {
        "direction_n": value["direction_n"],
        "k_m_inv": value["k_m_inv"],
        "gain": value["gain"],
        "log_gain": value["log_gain"],
        "integration_substeps_per_interval": REFERENCE_SUBSTEPS,
        "base_state_count": int(prefix_index + 1),
    }


def reassess_global_search(
    search: dict[str, Any], robustness: dict[str, Any]
) -> dict[str, Any]:
    """Apply like-for-like high-resolution acceptance gates.

    The low-order global scan is a basin-discovery calculation.  Its objective
    value is not compared directly with the 129-state objective because the
    early gain is known to be time-grid sensitive.  The retained coordinates
    are instead re-evaluated on the 129-state/four-substep reference and
    compared with the independently registered reference candidates on that
    same discretization.
    """

    selection = robustness["continuous_selection"]
    reference_onset = max(
        float(value["upper"]["gain"])
        for value in selection["onset_branches"].values()
    )
    reference_terminal = float(selection["final_winner"]["gain"])
    comparisons = {
        "onset": {
            "candidate_gain": float(search["high_resolution_verification"]["onset"]["gain"]),
            "registered_reference_gain": reference_onset,
        },
        "terminal": {
            "candidate_gain": float(search["high_resolution_verification"]["terminal"]["gain"]),
            "registered_reference_gain": reference_terminal,
        },
    }
    for value in comparisons.values():
        value["relative_gain_deficit"] = float(
            max(0.0, 1.0 - value["candidate_gain"] / value["registered_reference_gain"])
        )
    comparisons["coarse_to_reference_objective_shift"] = {
        key: float(
            search["high_resolution_verification"][key]["log_gain"]
            - search["winners"][key]["log_gain"]
        )
        for key in ("onset", "terminal")
    }
    search["high_resolution_comparison"] = comparisons
    search.pop("certification_boundary", None)
    search["audit_boundary"] = (
        "This is an anchor-assisted basin-retention and convergence audit over the "
        "registered compact domain. Because the blind Sobol sample misses the narrow "
        "terminal basin, it is not a global-optimum certificate."
    )
    search["gates"].pop("high_resolution_log_gain_within_5_percent", None)
    search["gates"]["high_resolution_candidates_within_1_percent_of_registered_reference"] = all(
        comparisons[key]["relative_gain_deficit"] <= 0.01
        for key in ("onset", "terminal")
    )
    search["status"] = (
        "ANCHOR_ASSISTED_SEARCH_AUDIT_PASS"
        if all(search["gates"].values())
        else "ANCHOR_ASSISTED_SEARCH_AUDIT_HAS_FAILED_GATES"
    )
    return search


def global_search(
    robustness: dict[str, Any], audit: SpectrumAudit
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    context = audit.contexts[QMC_CONTEXT]
    times = np.asarray(context["times"], dtype=float)
    onset_time = float(robustness["continuous_selection"]["onset_candidate_evaluation_time_s"])
    onset_index = int(np.argmin(np.abs(times - onset_time)))
    require(abs(times[onset_index] - onset_time) <= 1.0e-15, "QMC context lacks onset time")
    final_index = len(times) - 1
    coordinates = qmc_coordinates()
    onset_scores = np.empty(len(coordinates))
    final_scores = np.empty(len(coordinates))
    started = time.perf_counter()
    for index, x in enumerate(coordinates):
        value = audit.history(
            QMC_CONTEXT,
            final_index,
            x,
            integration_substeps_per_interval=QMC_SUBSTEPS,
        )
        onset_scores[index] = float(value["prefix"][onset_index]["log_gain"])
        final_scores[index] = float(value["log_gain"])
        if (index + 1) % 128 == 0:
            print(
                json.dumps(
                    {
                        "stage": "nested_global_scan",
                        "completed": index + 1,
                        "total": len(coordinates),
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    anchors = anchor_coordinates(robustness)
    anchor_onset = []
    anchor_final = []
    for x in anchors:
        value = audit.history(
            QMC_CONTEXT,
            final_index,
            x,
            integration_substeps_per_interval=QMC_SUBSTEPS,
        )
        anchor_onset.append(float(value["prefix"][onset_index]["log_gain"]))
        anchor_final.append(float(value["log_gain"]))
    anchor_onset_array = np.asarray(anchor_onset)
    anchor_final_array = np.asarray(anchor_final)
    level_rows = []
    previous = None
    for count in QMC_LEVELS:
        active_coordinates = np.vstack((coordinates[:count], np.asarray(anchors)))
        qmc_onset_max = float(np.max(onset_scores[:count]))
        qmc_final_max = float(np.max(final_scores[:count]))
        combined_onset = max(qmc_onset_max, float(np.max(anchor_onset_array)))
        combined_final = max(qmc_final_max, float(np.max(anchor_final_array)))
        coverage = coverage_estimate(active_coordinates)
        row = {
            "qmc_count": count,
            "anchor_count": len(anchors),
            "qmc_only_onset_max_log_gain": qmc_onset_max,
            "qmc_only_final_max_log_gain": qmc_final_max,
            "combined_onset_max_log_gain": combined_onset,
            "combined_final_max_log_gain": combined_final,
            **coverage,
            "combined_onset_relative_change_from_previous": (
                None
                if previous is None
                else relative_change(combined_onset, previous[0])
            ),
            "combined_final_relative_change_from_previous": (
                None
                if previous is None
                else relative_change(combined_final, previous[1])
            ),
        }
        level_rows.append(row)
        previous = (combined_onset, combined_final)
    write_csv(TABLE_DIR / "global_search_levels.csv", level_rows)

    refinements: dict[str, list[dict[str, Any]]] = {}
    for objective, prefix_index, scores in (
        ("onset", onset_index, onset_scores),
        ("terminal", final_index, final_scores),
    ):
        starts = choose_separated_starts(coordinates, scores, anchors)
        runs = []
        for index, start in enumerate(starts):
            print(
                json.dumps(
                    {"stage": "local_refinement", "objective": objective, "run": index + 1}
                ),
                flush=True,
            )
            runs.append(local_refinement(audit, prefix_index, start, f"{objective}_{index:02d}"))
        refinements[objective] = runs
    winners = {
        objective: max(runs, key=lambda item: item["log_gain"])
        for objective, runs in refinements.items()
    }

    reference_times = np.asarray(audit.contexts[REFERENCE_CONTEXT]["times"])
    reference_onset_index = int(np.argmin(np.abs(reference_times - onset_time)))
    require(
        abs(reference_times[reference_onset_index] - onset_time) <= 1.0e-15,
        "reference context lacks onset time",
    )
    reference_final_index = len(reference_times) - 1
    high_resolution = {
        "onset": high_resolution_verification(
            audit, reference_onset_index, winners["onset"]
        ),
        "terminal": high_resolution_verification(
            audit, reference_final_index, winners["terminal"]
        ),
    }
    recurrence = {}
    for objective, runs in refinements.items():
        winner = winners[objective]
        in_basin = [
            item
            for item in runs
            if angle_degrees(item["direction_n"], winner["direction_n"]) <= 2.5
            and relative_change(item["k_m_inv"], winner["k_m_inv"]) <= 0.10
            and abs(item["log_gain"] - winner["log_gain"]) <= 0.02
        ]
        recurrence[objective] = {
            "independent_start_count": len(runs),
            "winner_basin_recurrence_count": len(in_basin),
            "winner_basin_fraction": len(in_basin) / len(runs),
        }
    trace_rows = []
    for objective, runs in refinements.items():
        for run in runs:
            for row in run["trace"]:
                trace_rows.append({"objective": objective, "run_id": run["run_id"], **row})
    write_csv(TABLE_DIR / "optimizer_trajectories.csv", trace_rows)

    last = level_rows[-1]
    previous_level = level_rows[-2]
    gates = {
        "nested_levels_completed": [row["qmc_count"] for row in level_rows]
        == list(QMC_LEVELS),
        "sphere_coverage_improves": last["empirical_projective_sphere_fill_angle_deg"]
        < level_rows[0]["empirical_projective_sphere_fill_angle_deg"],
        "log_k_coverage_improves": last["maximum_log_k_gap"]
        < level_rows[0]["maximum_log_k_gap"],
        "last_two_combined_maxima_within_1_percent": (
            relative_change(
                last["combined_onset_max_log_gain"],
                previous_level["combined_onset_max_log_gain"],
            )
            <= 0.01
            and relative_change(
                last["combined_final_max_log_gain"],
                previous_level["combined_final_max_log_gain"],
            )
            <= 0.01
        ),
        "independent_basin_recurrence": all(
            value["winner_basin_recurrence_count"] >= 2
            for value in recurrence.values()
        ),
        "winners_strictly_inside_k_domain": all(
            K_BOUNDS[0] * 1.01 < item["k_m_inv"] < K_BOUNDS[1] / 1.01
            for item in winners.values()
        ),
    }
    result = {
        "status": (
            "ANCHOR_ASSISTED_SEARCH_AUDIT_PASS"
            if all(gates.values())
            else "ANCHOR_ASSISTED_SEARCH_AUDIT_HAS_FAILED_GATES"
        ),
        "domain": {
            "direction": "real projective plane represented by one closed hemisphere",
            "wavenumber_bounds_m_inv": list(K_BOUNDS),
            "wavenumber_coordinate": "uniform in log(k)",
        },
        "scan": {
            "sampler": "nested scrambled Sobol sequence plus symmetry-axis and prior-branch anchors",
            "seed": 20260829,
            "levels": level_rows,
            "qmc_context_state_count": len(times),
            "qmc_integration_substeps_per_interval": QMC_SUBSTEPS,
            "onset_evaluation_time_s": onset_time,
            "wall_time_s": time.perf_counter() - started,
        },
        "local_refinements": refinements,
        "winners": winners,
        "basin_recurrence": recurrence,
        "high_resolution_verification": high_resolution,
        "gates": gates,
        "audit_boundary": (
            "This is an anchor-assisted basin-retention and convergence audit over the "
            "registered compact domain. Because the blind Sobol sample misses the narrow "
            "terminal basin, it is not a global-optimum certificate."
        ),
    }
    result = reassess_global_search(result, robustness)
    return result, {
        "global_qmc_coordinates": coordinates,
        "global_qmc_onset_log_gain": onset_scores,
        "global_qmc_terminal_log_gain": final_scores,
        "global_anchor_coordinates": np.asarray(anchors),
        "global_anchor_onset_log_gain": anchor_onset_array,
        "global_anchor_terminal_log_gain": anchor_final_array,
    }


def branch_records(robustness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selection = robustness["continuous_selection"]
    final_label = selection["final_winner_label"]
    return {
        "onset_x": selection["onset_branches"]["sample_x_branch"]["upper"],
        "onset_y": selection["onset_branches"]["sample_y_branch"]["upper"],
        "terminal_y": selection["final_branches"][final_label],
    }


def release_branch_arrays(
    audit: SpectrumAudit, robustness: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    context = audit.contexts[REFERENCE_CONTEXT]
    times = np.asarray(context["times"], dtype=float)
    arrays: dict[str, np.ndarray] = {
        "baseline_reference_times_s": times,
        "baseline_coordinate_scales": np.asarray(audit.scales),
        "baseline_observed_indices": np.asarray(OBSERVED),
    }
    receipt = {}
    for identifier, record in branch_records(robustness).items():
        normal = np.asarray(record["direction_n"], dtype=float)
        normal /= np.linalg.norm(normal)
        wavenumber = float(record["k_m_inv"])
        operators = [
            assemble_dynamic_crystal_operator_v1(
                point,
                wavenumber_m_inv=wavenumber,
                direction_n=normal,
                admission=audit.admission,
            )
            for point in context["points"]
        ]
        raw = finite_time_amplification_history(
            operators,
            times,
            coordinate_scales=audit.scales,
            gain_threshold=GAIN_THRESHOLD,
            input_indices=OBSERVED,
            output_indices=OBSERVED,
            integration_substeps_per_interval=REFERENCE_SUBSTEPS,
        )
        observed = raw["propagator_dimensionless"][np.ix_(OBSERVED, OBSERVED)]
        singular_values = np.linalg.svd(observed, compute_uv=False)
        prefix = f"baseline_{identifier}"
        arrays[f"{prefix}_generators"] = np.stack(
            [item.generator_A for item in operators]
        )
        arrays[f"{prefix}_direction_n"] = normal
        arrays[f"{prefix}_k_m_inv"] = np.asarray([wavenumber])
        arrays[f"{prefix}_propagator"] = raw["propagator_dimensionless"]
        arrays[f"{prefix}_input_vector"] = raw["input_vector_dimensionless"]
        arrays[f"{prefix}_output_vector"] = raw["output_vector_dimensionless"]
        arrays[f"{prefix}_full_output_response"] = raw[
            "full_state_output_response_dimensionless"
        ]
        arrays[f"{prefix}_all_observed_singular_values"] = singular_values
        arrays[f"{prefix}_prefix_log_gain"] = np.asarray(
            [item["log_gain"] for item in raw["prefix"]]
        )
        receipt[identifier] = {
            "generator_shape": list(arrays[f"{prefix}_generators"].shape),
            "checkpoint_count": len(operators),
            "direction_n": normal.tolist(),
            "k_m_inv": wavenumber,
            "final_gain": raw["final_gain"],
            "leading_singular_triplet_released": True,
            "all_observed_singular_values_released": True,
        }
    return {
        "status": "FULL_REPORTED_BRANCH_ARRAYS_ASSEMBLED",
        "array_archive": ARRAYS.relative_to(ROOT).as_posix(),
        "branches": receipt,
        "complex_storage": "NumPy complex128 arrays in compressed NPZ",
    }, arrays


def make_figure(
    positive: dict[str, Any],
    positive_arrays: dict[str, np.ndarray],
    orientation: dict[str, Any],
    search: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11.4,
            "axes.labelsize": 11.6,
            "axes.titlesize": 11.8,
            "legend.fontsize": 9.6,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    ax = axes[0, 0]
    styles = (
        ("positive_stationary", "stationary normal", "#2166ac"),
        ("positive_commuting", "commuting, fixed leader", "#1b7837"),
        ("positive_switching", "commuting, switching leader", "#b2182b"),
    )
    for prefix, label, color in styles:
        time_values = positive_arrays[f"{prefix}_times"]
        ax.plot(
            time_values,
            positive_arrays[f"{prefix}_propagator_log_gain"],
            color=color,
            lw=1.8,
            label=label,
        )
        ax.plot(
            time_values,
            positive_arrays[f"{prefix}_frozen_integral_log_gain"],
            color=color,
            lw=1.1,
            ls="--",
        )
    ax.set_xlabel("normalized time")
    ax.set_ylabel("logarithmic gain")
    ax.set_title("(a) Normal-operator controls")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)

    ax = axes[0, 1]
    # Keep the manuscript and figure order identical even though the JSON
    # receipt is serialized with sorted keys.
    labels = ["minimum_storage", "median_storage", "maximum_storage"]
    x = np.arange(len(labels))
    selection = [
        orientation["orientations"][label]["selection_horizon_log_discrepancy"]
        for label in labels
    ]
    terminal = [
        orientation["orientations"][label]["terminal_log_discrepancy"]
        for label in labels
    ]
    ax.bar(x - 0.18, selection, width=0.36, color="#67a9cf", label="selection horizon")
    ax.bar(x + 0.18, terminal, width=0.36, color="#b2182b", label="terminal horizon")
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xticks(x, [label.replace("_storage", "") for label in labels], rotation=12)
    ax.set_ylabel(r"$\log G-\int\alpha\,dt$")
    ax.set_title("(b) Three independently selected orientations")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.22)

    ax = axes[1, 0]
    levels = search["scan"]["levels"]
    counts = np.asarray([row["qmc_count"] for row in levels])
    ax.plot(
        counts,
        [row["qmc_only_onset_max_log_gain"] for row in levels],
        "o-",
        color="#2166ac",
        label="onset, QMC only",
    )
    ax.plot(
        counts,
        [row["qmc_only_final_max_log_gain"] for row in levels],
        "s-",
        color="#b2182b",
        label="terminal, QMC only",
    )
    ax.plot(
        counts,
        [row["combined_onset_max_log_gain"] for row in levels],
        "o--",
        color="#67a9cf",
        label="onset + anchors",
    )
    ax.plot(
        counts,
        [row["combined_final_max_log_gain"] for row in levels],
        "s--",
        color="#ef8a62",
        label="terminal + anchors",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nested Sobol samples")
    ax.set_ylabel("maximum finite-time log gain")
    ax.set_title("(c) Anchor-assisted search convergence")
    # The four long labels overrun the left margin in a two-column legend.
    # Keep them inside the otherwise empty middle of the panel.
    ax.legend(frameon=False, ncol=1, loc="center")
    ax.grid(True, alpha=0.22)

    ax = axes[1, 1]
    colors = {"onset": "#2166ac", "terminal": "#b2182b"}
    for objective, runs in search["local_refinements"].items():
        angles = [
            angle_degrees(run["direction_n"], search["winners"][objective]["direction_n"])
            for run in runs
        ]
        k_ratio = [run["k_m_inv"] / search["winners"][objective]["k_m_inv"] for run in runs]
        ax.scatter(
            angles,
            k_ratio,
            s=30,
            alpha=0.82,
            color=colors[objective],
            label=f"{objective} multistarts",
        )
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.axvline(2.5, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("angle from retained winner [deg]")
    ax.set_ylabel(r"$k/k_{\rm winner}$")
    ax.set_title("(d) Independent-basin recurrence")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reassess-existing",
        action="store_true",
        help="reapply like-for-like acceptance gates without rerunning expensive operators",
    )
    parser.add_argument(
        "--refresh-figure",
        action="store_true",
        help="redraw the strengthening figure from the released JSON/NPZ only",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    robustness = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    texture = json.loads(TEXTURE.read_text(encoding="utf-8"))

    if args.refresh_figure:
        require(RESULT.is_file() and ARRAYS.is_file(), "released evidence is incomplete")
        existing = json.loads(RESULT.read_text(encoding="utf-8"))
        with np.load(ARRAYS) as source:
            payload = {key: np.asarray(source[key]) for key in source.files}
        make_figure(
            existing["positive_controls"],
            payload,
            existing["orientation_transfer"],
            existing["global_search"],
        )
        print(json.dumps({"figure": str(FIGURE_STEM.with_suffix('.pdf'))}, indent=2))
        return 0

    if args.reassess_existing:
        require(RESULT.is_file(), "no existing strengthening result to reassess")
        existing = json.loads(RESULT.read_text(encoding="utf-8"))
        existing["orientation_transfer"] = reassess_orientation_full_state(
            existing["orientation_transfer"]
        )
        existing["global_search"] = reassess_global_search(
            existing["global_search"], robustness
        )
        existing["gates"]["anchor_assisted_search_audit_pass"] = (
            existing["global_search"]["status"]
            == "ANCHOR_ASSISTED_SEARCH_AUDIT_PASS"
        )
        existing["gates"]["three_orientation_transfer_pass"] = (
            existing["orientation_transfer"]["status"]
            == "THREE_ORIENTATION_TRANSFER_PASS"
        )
        existing["gates"].pop("numerical_global_search_certificate_pass", None)
        required = {
            key: value
            for key, value in existing["gates"].items()
            if not key.startswith("strict_")
        }
        existing["status"] = (
            "ALL_REQUIRED_STRENGTHENING_GATES_PASS"
            if all(required.values())
            else "STRENGTHENING_HAS_FAILED_REQUIRED_GATES"
        )
        existing["gate_reassessment"] = {
            "reason": "like-for-like 129-state candidate/reference comparison",
            "script": "tools/build_ijp_strengthening_evidence_v1.py --reassess-existing",
        }
        existing["outputs"]["figure_pdf"] = FIGURE_STEM.with_suffix(".pdf").relative_to(ROOT).as_posix()
        existing["outputs"]["figure_png"] = FIGURE_STEM.with_suffix(".png").relative_to(ROOT).as_posix()
        RESULT.write_text(
            json.dumps(clean_json(existing), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "result": str(RESULT),
                    "status": existing["status"],
                    "global_gates": existing["global_search"]["gates"],
                },
                indent=2,
            )
        )
        return 0

    print(json.dumps({"stage": "positive_controls"}), flush=True)
    positive, positive_arrays = positive_controls()
    positive_rows = positive["cases"]
    write_csv(TABLE_DIR / "positive_controls.csv", positive_rows)

    print(json.dumps({"stage": "orientation_transfer"}), flush=True)
    orientation, orientation_arrays = orientation_transfer(texture)

    print(json.dumps({"stage": "build_129_state_reference"}), flush=True)
    audit = SpectrumAudit(maximum_refinement=16, context_factors=(1, 2, 16))
    search, search_arrays = global_search(robustness, audit)
    print(json.dumps({"stage": "release_branch_arrays"}), flush=True)
    release, release_arrays = release_branch_arrays(audit, robustness)

    all_arrays = {}
    all_arrays.update(positive_arrays)
    all_arrays.update(orientation_arrays)
    all_arrays.update(search_arrays)
    all_arrays.update(release_arrays)
    ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS, **all_arrays)

    make_figure(positive, positive_arrays, orientation, search)
    gates = {
        "positive_controls_pass": positive["status"] == "POSITIVE_CONTROLS_PASS",
        "three_orientation_transfer_pass": orientation["status"]
        == "THREE_ORIENTATION_TRANSFER_PASS",
        "anchor_assisted_search_audit_pass": search["status"]
        == "ANCHOR_ASSISTED_SEARCH_AUDIT_PASS",
        "full_reported_generators_and_singular_vectors_released": release["status"]
        == "FULL_REPORTED_BRANCH_ARRAYS_ASSEMBLED",
        "strict_hcp_to_bai_analytic_reduction_closed": False,
    }
    required_gates = {key: value for key, value in gates.items() if not key.startswith("strict_")}
    result = {
        "schema": "IJP_FINITE_TIME_STRENGTHENING_EVIDENCE_V1",
        "status": (
            "ALL_REQUIRED_STRENGTHENING_GATES_PASS"
            if all(required_gates.values())
            else "STRENGTHENING_HAS_FAILED_REQUIRED_GATES"
        ),
        "positive_controls": positive,
        "orientation_transfer": orientation,
        "global_search": search,
        "reproducibility_arrays": release,
        "gates": gates,
        "claim_boundary": {
            "supports": (
                "The frozen integral implementation reproduces the propagator under its stated "
                "normal/commuting/fixed-leading-mode conditions; the HCP non-equivalence persists "
                "for three independently selected crystal orientations; and the reported selector "
                "is stable under the registered anchor-assisted basin-retention audit."
            ),
            "does_not_support": (
                "An analytic global optimum proof, specimen-calibrated onset or width, nonlinear "
                "propagation resistance, or a strict analytical HCP-to-Bai reduction."
            ),
            "strict_hcp_to_bai_statement": (
                "The scalar Bai implementation remains an independently verified classical anchor. "
                "A strict HCP-to-Bai analytical degeneration is not closed and is not required for "
                "the finite-time same-generator discrimination claim."
            ),
        },
        "sources": [
            ROBUSTNESS.relative_to(ROOT).as_posix(),
            TEXTURE.relative_to(ROOT).as_posix(),
            "src/hcp_cp_gnd/dynamic_crystal_perturbation_v1.py",
            "tools/audit_cp_ti_continuous_spectrum_robustness_v1.py",
            "tools/run_cp_ti_texture_pathway_v1.py",
        ],
        "outputs": {
            "arrays": ARRAYS.relative_to(ROOT).as_posix(),
            "figure_pdf": FIGURE_STEM.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            "figure_png": FIGURE_STEM.with_suffix(".png").relative_to(ROOT).as_posix(),
            "tables": TABLE_DIR.relative_to(ROOT).as_posix(),
        },
        "wall_time_s": time.perf_counter() - started,
    }
    RESULT.write_text(
        json.dumps(clean_json(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "arrays": str(ARRAYS),
                "status": result["status"],
                "gates": gates,
                "wall_time_s": result["wall_time_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
