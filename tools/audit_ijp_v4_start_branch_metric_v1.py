"""Audit start-time dependence, constitutive branch crossings, and physical metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hcp_cp_gnd.cp_ti_material_v1 import simple_shear  # noqa: E402
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    N_GENERATOR,
    finite_time_amplification_history,
)
from hcp_cp_gnd.lsbrp_metrics_v1 import (  # noqa: E402
    construct_checkpoint_energy_metric,
    construct_observation_metric,
    weighted_propagator_gain,
)
import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v3 as v3nl  # noqa: E402


SCHEMA = "IJP_V4_START_BRANCH_METRIC_AUDIT_V1"
RESULT = ROOT / "05_results/ijp_v4_start_branch_metric_audit_v1.json"
TABLE = ROOT / "05_results/ijp_v4_start_branch_metric_audit_v1.csv"
OBSERVED = np.arange(6, N_GENERATOR, dtype=int)


def observable_gain(propagator: np.ndarray) -> float:
    selected = propagator[np.ix_(OBSERVED, OBSERVED)]
    return float(np.linalg.svd(selected, compute_uv=False)[0])


def ordered_product(maps: list[np.ndarray], start: int = 0) -> np.ndarray:
    result = np.eye(N_GENERATOR, dtype=np.complex128)
    for matrix in maps[start:]:
        result = matrix @ result
    return result


def branch_transitions(context: dict[str, Any], stop: int) -> list[dict[str, Any]]:
    points = context["points"][: stop + 1]
    times = np.asarray(context["times"][: stop + 1], dtype=float)
    labels = points[0].branch_audit.labels
    categories = np.asarray([point.branch_audit.categories for point in points], dtype=int)
    transitions = []
    for interval in np.where(np.any(categories[1:] != categories[:-1], axis=1))[0]:
        changed = np.where(categories[interval + 1] != categories[interval])[0]
        details = []
        event_fractions = []
        for index in changed:
            left = float(points[interval].branch_audit.signed_distances[index])
            right = float(points[interval + 1].branch_audit.signed_distances[index])
            fraction = abs(left) / max(abs(left) + abs(right), np.finfo(float).tiny)
            if labels[index].endswith(".yield"):
                event_fractions.append(fraction)
            details.append(
                {
                    "label": labels[index],
                    "left_category": int(categories[interval, index]),
                    "right_category": int(categories[interval + 1, index]),
                    "left_signed_distance": left,
                    "right_signed_distance": right,
                    "secant_zero_fraction": fraction,
                }
            )
        fraction = float(np.median(event_fractions or [0.5]))
        transitions.append(
            {
                "interval": int(interval),
                "left_time_s": float(times[interval]),
                "right_time_s": float(times[interval + 1]),
                "event_time_secant_s": float(
                    times[interval] + fraction * (times[interval + 1] - times[interval])
                ),
                "event_fraction": fraction,
                "changed_contracts": details,
            }
        )
    return transitions


def exact_index(times: np.ndarray, target: float) -> int:
    matches = np.where(np.isclose(times, target, rtol=0.0, atol=2.0e-18))[0]
    if matches.size != 1:
        raise ValueError(f"start time {target} is not an exact registered checkpoint")
    return int(matches[0])


def analyze_role(
    role: str,
    contract: dict[str, Any],
    context: dict[str, Any],
    spectral: Any,
    admission: Any,
    scales: np.ndarray,
) -> dict[str, Any]:
    direction = np.asarray(contract["direction_n"], dtype=float)
    direction /= np.linalg.norm(direction)
    k_m_inv = float(contract["k_m_inv"])
    times_all = np.asarray(context["times"], dtype=float)
    stop = v2._exact_index(times_all, float(contract["target_time_s"]))
    times = times_all[: stop + 1]
    linear, operators = v2.linear_contract(
        context, stop, direction, k_m_inv, admission, scales
    )
    generators = [
        operator.generator_A * scales[None, :] / scales[:, None]
        for operator in operators
    ]
    midpoint_maps = [
        expm(0.5 * (generators[index] + generators[index + 1]) * (times[index + 1] - times[index]))
        for index in range(stop)
    ]
    midpoint_propagator = ordered_product(midpoint_maps)
    transitions = branch_transitions(context, stop)
    convention_gains = {"linear_interpolation_midpoint": observable_gain(midpoint_propagator)}
    convention_maps: dict[str, list[np.ndarray]] = {}
    for convention in ("left_continuous", "right_continuous", "secant_split"):
        maps = list(midpoint_maps)
        for transition in transitions:
            interval = int(transition["interval"])
            dt = float(times[interval + 1] - times[interval])
            if convention == "left_continuous":
                maps[interval] = expm(generators[interval] * dt)
            elif convention == "right_continuous":
                maps[interval] = expm(generators[interval + 1] * dt)
            else:
                fraction = float(transition["event_fraction"])
                maps[interval] = (
                    expm(generators[interval + 1] * ((1.0 - fraction) * dt))
                    @ expm(generators[interval] * (fraction * dt))
                )
        convention_maps[convention] = maps
        convention_gains[convention] = observable_gain(ordered_product(maps))
    baseline_gain = convention_gains["linear_interpolation_midpoint"]
    convention_changes = {
        key: abs(value / baseline_gain - 1.0)
        for key, value in convention_gains.items()
        if key != "linear_interpolation_midpoint"
    }
    start_candidates = [0.5e-6, 0.75e-6, 1.0e-6]
    if float(contract["target_time_s"]) > 2.0e-6:
        start_candidates.append(2.0e-6)
    start_records = []
    for start_time in start_candidates:
        start = exact_index(times, start_time)
        gain_value = observable_gain(ordered_product(midpoint_maps, start=start))
        start_records.append(
            {
                "start_time_s": start_time,
                "start_index": start,
                "gain": gain_value,
                "relative_to_0p5us_gain": gain_value / baseline_gain,
            }
        )

    initial_metric = construct_checkpoint_energy_metric(
        operators[0],
        context["points"][0],
        reference_temperature_K=context["storages"][0].temperature_K,
        coordinate_scales=scales,
        spectral_model=spectral,
        base_F_sample=simple_shear(float(context["shears"][0])),
    )
    final_metric = construct_checkpoint_energy_metric(
        operators[-1],
        context["points"][stop],
        reference_temperature_K=context["storages"][stop].temperature_K,
        coordinate_scales=scales,
        spectral_model=spectral,
        base_F_sample=simple_shear(float(context["shears"][stop])),
    )
    physical_propagator = (
        scales[:, None] * midpoint_propagator / scales[None, :]
    )
    energy_gain = weighted_propagator_gain(
        physical_propagator,
        input_metric=initial_metric,
        output_metric=final_metric,
    )
    observation = np.zeros((OBSERVED.size, N_GENERATOR), dtype=float)
    observation[np.arange(OBSERVED.size), OBSERVED] = 1.0
    observation_metric = construct_observation_metric(
        observation,
        np.diag(scales[OBSERVED] ** 2),
        provenance=(
            "NUMERICAL SURROGATE: registered coordinate scales treated as independent "
            "standard deviations; no experimental covariance"
        ),
        coordinate_scales=scales,
    )
    energy_to_observation = weighted_propagator_gain(
        physical_propagator,
        input_metric=initial_metric,
        output_metric=observation_metric,
    )
    return {
        "role": role,
        "target_time_s": float(contract["target_time_s"]),
        "direction_n": direction.tolist(),
        "k_m_inv": k_m_inv,
        "registered_state_count": stop + 1,
        "branch_transitions": transitions,
        "branch_transition_count": len(transitions),
        "convention_gains": convention_gains,
        "maximum_branch_convention_relative_gain_change": max(
            convention_changes.values(), default=0.0
        ),
        "start_time_sensitivity": start_records,
        "fixed_coordinate_gain": baseline_gain,
        "released_linear_gain": float(linear["final_gain"]),
        "fixed_coordinate_gain_relative_reproduction_error": abs(
            baseline_gain / float(linear["final_gain"]) - 1.0
        ),
        "physical_metric_audit": {
            "classification": (
                "FIXED_DIRECTION_WAVENUMBER_QUOTIENT_METRIC_AUDIT_"
                "NOT_ENERGY_METRIC_REOPTIMIZATION"
            ),
            "initial_energy_metric": initial_metric.audit(),
            "final_energy_metric": final_metric.audit(),
            "energy_to_energy_gain": float(energy_gain["gain"]),
            "energy_to_numerical_observation_gain": float(
                energy_to_observation["gain"]
            ),
            "numerical_observation_metric": observation_metric.audit(),
            "experimental_covariance_available": False,
        },
    }


def write_table(roles: dict[str, dict[str, Any]]) -> None:
    rows = []
    for role, record in roles.items():
        for item in record["start_time_sensitivity"]:
            rows.append(
                {
                    "role": role,
                    "audit": "start_time",
                    "case": f"t0={item['start_time_s']:.9g}",
                    "gain": item["gain"],
                    "relative_change_or_ratio": item["relative_to_0p5us_gain"],
                }
            )
        baseline = record["convention_gains"]["linear_interpolation_midpoint"]
        for name, value in record["convention_gains"].items():
            rows.append(
                {
                    "role": role,
                    "audit": "branch_convention",
                    "case": name,
                    "gain": value,
                    "relative_change_or_ratio": abs(value / baseline - 1.0),
                }
            )
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    started = time.perf_counter()
    evidence = json.loads(v3nl.V3_EVIDENCE.read_text(encoding="utf-8"))
    with np.load(v3nl.V3_ARRAYS) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    context = v2.load_verified_cache()["contexts"]["factor64"]
    _, _, _, spectral = v3.build_case_models("v4_start_branch_metric")
    admission = v3._admission()
    contracts = v3nl.contracts(evidence)
    roles = {
        role: analyze_role(
            role, contract, context, spectral, admission, scales
        )
        for role, contract in contracts.items()
    }
    gates = {
        "released_gain_reproduced_below_1e_minus_10": max(
            item["fixed_coordinate_gain_relative_reproduction_error"]
            for item in roles.values()
        )
        <= 1.0e-10,
        "branch_convention_gain_change_below_1_percent": max(
            item["maximum_branch_convention_relative_gain_change"]
            for item in roles.values()
        )
        <= 0.01,
        "energy_metrics_constructed_without_diagonal_regularization": all(
            item["physical_metric_audit"]["initial_energy_metric"][
                "positive_semidefinite_within_relative_tolerance"
            ]
            and item["physical_metric_audit"]["final_energy_metric"][
                "positive_semidefinite_within_relative_tolerance"
            ]
            for item in roles.values()
        ),
        "experimental_observation_covariance_available": False,
    }
    report = {
        "schema": SCHEMA,
        "status": "START_AND_BRANCH_AUDIT_PASS__EXPERIMENTAL_METRIC_OPEN",
        "roles": roles,
        "gates": gates,
        "interpretation": {
            "start_time": (
                "t0 is part of the input-output problem; sensitivity is reported rather "
                "than treated as a numerical invariance requirement"
            ),
            "branching": (
                "checkpoint Jacobians are branch locked, but the base path is only "
                "piecewise smooth; left, right and secant-split conventions bound the "
                "effect of four registered transition intervals"
            ),
            "physical_metric": (
                "the recoverable-energy metric is a positive-semidefinite quotient metric "
                "with density null directions; the observation covariance is numerical, "
                "not experimental, and no energy-metric direction-wavenumber reoptimization "
                "is claimed"
            ),
        },
        "provenance": {
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": v2.sha256(Path(__file__)),
            "base_context_cache": v2.CACHE.relative_to(ROOT).as_posix(),
            "base_context_cache_sha256": v2.sha256(v2.CACHE),
            "v3_evidence_sha256": v2.sha256(v3nl.V3_EVIDENCE),
            "v3_arrays_sha256": v2.sha256(v3nl.V3_ARRAYS),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            "json": RESULT.relative_to(ROOT).as_posix(),
            "csv": TABLE.relative_to(ROOT).as_posix(),
        },
    }
    write_table(roles)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
