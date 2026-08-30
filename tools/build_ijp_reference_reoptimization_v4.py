"""Directly re-optimize the IJP selector audit on the reference propagator.

The V3 audit discovered direction--wavenumber basins on a reduced temporal
map and evaluated the retained coordinates on the 129-state/four-substep
map.  That separation is useful for reconnaissance, but a reference
reevaluation is not a reference-objective optimization.  This V4 audit closes
that gap by

1. evaluating one shared Sobol/anchor/candidate pool on the reference map;
2. refining two separated starts for every horizon--norm--selector contract
   directly on the 129-state/four-substep objective;
3. checking a deterministic six-point neighbourhood of every winner; and
4. releasing the final propagators, singular vectors, traces and gates.

The search remains an anchor-assisted finite audit, not a mathematical proof
of a global optimum.  A working checkpoint is written after every completed
contract so an interrupted production run can be resumed without promoting a
partial result to submission evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from tools.build_ijp_operator_consistency_v2 import (  # noqa: E402
    log_gain,
    norm_variants,
    rescale_propagator,
    selectors,
)
from tools.build_ijp_revision_evidence_v3 import (  # noqa: E402
    BOUNDS,
    K_BOUNDS,
    LOCALIZATION_INTERIOR_SELECTORS,
    REOPTIMIZED_NORMS,
    RESULT as V3_RESULT,
    build_context,
    clean_json,
    coordinates_from_direction,
    direction_angle_degrees,
    direction_from_coordinates,
    evaluate_pool,
    local_refinement,
    objective_score,
    qmc_coordinates,
    registered_anchors,
    require,
    singular_diagnostics,
)


RESULT = ROOT / "05_results/ijp_reference_reoptimization_v4.json"
WORKING = ROOT / "05_results/ijp_reference_reoptimization_v4_working.json"
ARRAYS = ROOT / "05_results/ijp_reference_reoptimization_v4_arrays.npz"
TABLE_DIR = ROOT / "05_results/ijp_reference_reoptimization_v4"
WINNERS_CSV = TABLE_DIR / "reference_reoptimized_winners.csv"
TRACE_CSV = TABLE_DIR / "reference_optimizer_trace.csv"

REFERENCE_FACTOR = 16
REFERENCE_SUBSTEPS = 4
REFERENCE_QMC_COUNT = 64
REFERENCE_START_COUNT = 2
DEFAULT_MAXIMUM_ITERATIONS = 24
NEIGHBOUR_DELTA = 2.0e-3
LOCAL_STATIONARITY_LOG_TOL = 2.0e-4
BOUNDARY_EXTENSION_LOG_TOL = 2.0e-4
FINITE_LOCALIZATION_INTERIOR_SELECTORS = (
    set(LOCALIZATION_INTERIOR_SELECTORS) - {"constitutive_to_temperature"}
)
REVERSIBLE_WAVE_SELECTORS = {
    "full_to_full",
    "mechanical_to_mechanical",
}


def boundary_classification(
    *,
    selector_id: str,
    wavenumber_interior: bool,
    boundary_side: str | None,
    reference_log_gain: float,
    boundary_extension: list[dict[str, Any]],
) -> str:
    """Classify the winning branch from the extension evidence, not its label."""
    if boundary_side == "lower_long_wave":
        return (
            "compact_domain_lower_boundary_long_wave_branch_"
            "not_a_finite_localization_wavelength"
        )
    extension_exceeds = any(
        float(item["log_gain"]) > reference_log_gain + BOUNDARY_EXTENSION_LOG_TOL
        for item in boundary_extension
    )
    if not wavenumber_interior or (
        selector_id in REVERSIBLE_WAVE_SELECTORS and extension_exceeds
    ):
        return (
            "high_frequency_reversible_wave_branch_"
            "not_a_finite_localization_wavelength"
        )
    return "interior_finite_band_candidate"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique_coordinates(values: list[np.ndarray]) -> np.ndarray:
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for value in values:
        raw = np.asarray(value, dtype=float)
        raw[0] = np.clip(raw[0], BOUNDS[0][0], BOUNDS[0][1])
        raw[1] = np.clip(raw[1], BOUNDS[1][0], BOUNDS[1][1])
        raw[2] = np.clip(raw[2], BOUNDS[2][0], BOUNDS[2][1])
        unique[tuple(np.round(raw, 10))] = raw
    return np.asarray(list(unique.values()), dtype=float)


def _reference_pool(v3_records: dict[str, Any], horizon_id: str) -> np.ndarray:
    values = [
        *list(qmc_coordinates(REFERENCE_QMC_COUNT, 20260830)),
        *registered_anchors(),
    ]
    for record in v3_records.values():
        if record["horizon_id"] != horizon_id:
            continue
        values.append(
            coordinates_from_direction(record["direction_n"], record["k_m_inv"])
        )
        for run in record.get("search_local_runs", []):
            values.append(np.asarray(run["coordinates"], dtype=float))
    return _unique_coordinates(values)


def _separated_starts(
    coordinates: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
) -> list[np.ndarray]:
    order = np.argsort(scores)[::-1]
    starts: list[np.ndarray] = []
    for index in order:
        candidate = np.asarray(coordinates[int(index)], dtype=float)
        if not starts or all(
            direction_angle_degrees(
                direction_from_coordinates(candidate), direction_from_coordinates(other)
            )
            >= 3.0
            or abs(float(candidate[2] - other[2])) >= 0.10
            for other in starts
        ):
            starts.append(candidate)
        if len(starts) == count:
            break
    require(bool(starts), "reference pool did not produce a start")
    return starts


def _neighbourhood_coordinates(winner: np.ndarray) -> list[np.ndarray]:
    probes: list[np.ndarray] = []
    for coordinate in range(3):
        for sign in (-1.0, 1.0):
            value = np.asarray(winner, dtype=float).copy()
            value[coordinate] += sign * NEIGHBOUR_DELTA
            value[coordinate] = np.clip(
                value[coordinate], BOUNDS[coordinate][0], BOUNDS[coordinate][1]
            )
            if not np.allclose(value, winner, rtol=0.0, atol=1.0e-14):
                probes.append(value)
    return probes


def _neighbourhood_audit(
    context: Any,
    indices: np.ndarray,
    winner_coordinates: np.ndarray,
    winner_log_gain: float,
    active_scales: np.ndarray,
    output_indices: np.ndarray,
    input_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], float, np.ndarray | None]:
    rows: list[dict[str, Any]] = []
    for probe in _neighbourhood_coordinates(winner_coordinates):
        propagated = context.propagator(
            probe, indices, substeps=REFERENCE_SUBSTEPS
        )
        score = objective_score(
            propagated["propagator"],
            context.scales,
            active_scales,
            output_indices,
            input_indices,
        )
        rows.append(
            {
                "coordinates": probe.tolist(),
                "direction_n": propagated["direction_n"].tolist(),
                "k_m_inv": float(propagated["k_m_inv"]),
                "log_gain": float(score),
            }
        )
    if not rows:
        return rows, -np.inf, None
    best = max(rows, key=lambda item: item["log_gain"])
    return (
        rows,
        float(best["log_gain"] - winner_log_gain),
        np.asarray(best["coordinates"], dtype=float),
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed contracts from the working checkpoint",
    )
    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=DEFAULT_MAXIMUM_ITERATIONS,
        help="Powell iteration cap for each reference start",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="development-only contract limit; a limited run cannot write final evidence",
    )
    args = parser.parse_args()
    require(args.maximum_iterations >= 1, "maximum iterations must be positive")
    require(V3_RESULT.is_file(), "missing V3 re-optimization evidence")

    started = time.perf_counter()
    v3 = json.loads(V3_RESULT.read_text(encoding="utf-8"))
    v3_records = v3["reoptimization"]["records"]
    require(len(v3_records) == 70, "the V3 input must contain all 70 contracts")

    completed: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    if args.resume and WORKING.is_file():
        checkpoint = json.loads(WORKING.read_text(encoding="utf-8"))
        completed = {
            key: value
            for key, value in checkpoint.get("records", {}).items()
            if value.get("local_stationarity_pass", False)
            and (
                value.get("wavenumber_interior", False)
                or "boundary_side" in value
            )
            and (
                value.get("selector_id") not in REVERSIBLE_WAVE_SELECTORS
                or len(value.get("fixed_direction_extended_k_diagnostic", [])) == 2
            )
        }
        traces = checkpoint.get("traces", [])

    print(
        json.dumps(
            {
                "stage": "build_reference_context",
                "factor": REFERENCE_FACTOR,
                "substeps": REFERENCE_SUBSTEPS,
            }
        ),
        flush=True,
    )
    context = build_context("reference_reoptimization_v4", factor=REFERENCE_FACTOR)
    selector_map = selectors()
    active_norms = norm_variants(context.scales)
    reference_indices = np.arange(len(context.shears), dtype=int)
    onset_index = int(np.argmin(np.abs(context.times_s - 1.46e-6)))
    horizon_indices = {
        "onset": reference_indices[: onset_index + 1],
        "terminal": reference_indices,
    }

    pool_coordinates: dict[str, np.ndarray] = {}
    pool_propagators: dict[str, list[np.ndarray]] = {}
    for horizon_id in ("onset", "terminal"):
        coordinates = _reference_pool(v3_records, horizon_id)
        pool_coordinates[horizon_id] = coordinates
        print(
            json.dumps(
                {
                    "stage": "reference_pool",
                    "horizon": horizon_id,
                    "count": int(len(coordinates)),
                }
            ),
            flush=True,
        )
        pool_propagators[horizon_id] = evaluate_pool(
            context,
            horizon_indices[horizon_id],
            coordinates,
            substeps=REFERENCE_SUBSTEPS,
        )

    ordered_keys = sorted(v3_records)
    remaining = [key for key in ordered_keys if key not in completed]
    if args.max_records is not None:
        require(args.max_records >= 1, "max-records must be positive")
        remaining = remaining[: args.max_records]

    for ordinal, key in enumerate(remaining, start=1):
        old = v3_records[key]
        horizon_id = old["horizon_id"]
        norm_id = old["norm_id"]
        selector_id = old["selector_id"]
        output_indices, input_indices = selector_map[selector_id]
        active_scales = active_norms[norm_id]
        coordinates = pool_coordinates[horizon_id]
        scores = np.asarray(
            [
                objective_score(
                    phi,
                    context.scales,
                    active_scales,
                    output_indices,
                    input_indices,
                )
                for phi in pool_propagators[horizon_id]
            ]
        )
        starts = _separated_starts(
            coordinates, scores, count=REFERENCE_START_COUNT
        )
        runs: list[dict[str, Any]] = []
        for start_index, start in enumerate(starts, start=1):
            run, trace = local_refinement(
                context,
                horizon_indices[horizon_id],
                start,
                active_scales,
                output_indices,
                input_indices,
                run_id=f"{key}__reference_start{start_index}",
                substeps=REFERENCE_SUBSTEPS,
                maximum_iterations=args.maximum_iterations,
            )
            runs.append(run)
            traces.extend(trace)

        winner = max(runs, key=lambda item: item["log_gain"])
        winner_coordinates = np.asarray(winner["coordinates"], dtype=float)
        neighbour_scores, local_stationarity_excess, best_probe = _neighbourhood_audit(
            context,
            horizon_indices[horizon_id],
            winner_coordinates,
            float(winner["log_gain"]),
            active_scales,
            output_indices,
            input_indices,
        )
        stationarity_march_steps = 0
        for march_index in range(1, 65):
            if local_stationarity_excess <= LOCAL_STATIONARITY_LOG_TOL:
                break
            require(best_probe is not None, "failed stationarity audit has no probe")
            best_row = max(neighbour_scores, key=lambda item: item["log_gain"])
            winner_coordinates = np.asarray(best_row["coordinates"], dtype=float)
            winner = {
                "run_id": f"{key}__stationarity_march{march_index}",
                "log_gain": float(best_row["log_gain"]),
                "gain": float(np.exp(best_row["log_gain"])),
                "direction_n": best_row["direction_n"],
                "k_m_inv": float(best_row["k_m_inv"]),
                "coordinates": best_row["coordinates"],
                "optimizer_success": False,
                "optimizer_message": "deterministic local stationarity march",
                "optimizer_nfev": 6,
                "retained_start_instead_of_optimizer": True,
            }
            stationarity_march_steps = march_index
            neighbour_scores, local_stationarity_excess, best_probe = (
                _neighbourhood_audit(
                    context,
                    horizon_indices[horizon_id],
                    winner_coordinates,
                    float(winner["log_gain"]),
                    active_scales,
                    output_indices,
                    input_indices,
                )
            )
        if stationarity_march_steps:
            runs.append(winner)

        final = context.propagator(
            winner_coordinates,
            horizon_indices[horizon_id],
            substeps=REFERENCE_SUBSTEPS,
            return_generators=(norm_id == "baseline"),
        )
        active_phi = rescale_propagator(
            final["propagator"], context.scales, active_scales
        )
        final_log_gain = log_gain(active_phi, output_indices, input_indices)
        singular, _ = singular_diagnostics(active_phi, output_indices, input_indices)
        wavenumber_interior = bool(
            final["k_m_inv"] > 1.01 * K_BOUNDS[0]
            and final["k_m_inv"] < 0.99 * K_BOUNDS[1]
        )
        boundary_extension: list[dict[str, Any]] = []
        boundary_side: str | None = None
        if not wavenumber_interior or selector_id in REVERSIBLE_WAVE_SELECTORS:
            if final["k_m_inv"] <= 1.01 * K_BOUNDS[0]:
                boundary_side = "lower_long_wave"
                extension_wavenumbers = (
                    float(final["k_m_inv"]) / 10.0,
                    float(final["k_m_inv"]) / 100.0,
                )
            else:
                boundary_side = (
                    "upper_high_frequency"
                    if not wavenumber_interior
                    else "interior_point_high_frequency_extension"
                )
                extension_wavenumbers = (
                    min(float(final["k_m_inv"]) * 10.0, 3.0e7),
                    min(float(final["k_m_inv"]) * 100.0, 3.0e7),
                )
            for extension_wavenumber in extension_wavenumbers:
                extension_coordinates = coordinates_from_direction(
                    final["direction_n"],
                    extension_wavenumber,
                )
                extension = context.propagator(
                    extension_coordinates,
                    horizon_indices[horizon_id],
                    substeps=REFERENCE_SUBSTEPS,
                )
                extension_score = objective_score(
                    extension["propagator"],
                    context.scales,
                    active_scales,
                    output_indices,
                    input_indices,
                )
                boundary_extension.append(
                    {
                        "k_m_inv": float(extension["k_m_inv"]),
                        "log_gain": float(extension_score),
                    }
                )

        old_log_gain = float(old["reference_log_gain"])
        completed[key] = {
            "horizon_id": horizon_id,
            "time_s": float(final["times_s"][-1]),
            "norm_id": norm_id,
            "selector_id": selector_id,
            "input_dimension": int(len(input_indices)),
            "output_dimension": int(len(output_indices)),
            "reference_factor": REFERENCE_FACTOR,
            "reference_substeps": REFERENCE_SUBSTEPS,
            "reference_pool_count": int(len(coordinates)),
            "reference_pool_best_log_gain": float(np.max(scores)),
            "reference_local_runs": runs,
            "previous_v3_direction_n": old["direction_n"],
            "previous_v3_k_m_inv": float(old["k_m_inv"]),
            "previous_v3_reference_log_gain": old_log_gain,
            "previous_v3_reference_gain": float(old["reference_gain"]),
            "direction_n": final["direction_n"].tolist(),
            "k_m_inv": float(final["k_m_inv"]),
            "reference_log_gain": float(final_log_gain),
            "reference_gain": float(np.exp(final_log_gain)),
            "direct_reference_log_gain_improvement": float(final_log_gain - old_log_gain),
            "direct_reference_gain_ratio": float(np.exp(final_log_gain - old_log_gain)),
            "local_neighbourhood_delta": NEIGHBOUR_DELTA,
            "local_stationarity_march_steps": stationarity_march_steps,
            "local_neighbourhood": neighbour_scores,
            "local_stationarity_excess_log_gain": local_stationarity_excess,
            "local_stationarity_pass": bool(
                local_stationarity_excess <= LOCAL_STATIONARITY_LOG_TOL
            ),
            "wavenumber_interior": wavenumber_interior,
            "boundary_side": boundary_side,
            "boundary_classification": boundary_classification(
                selector_id=selector_id,
                wavenumber_interior=wavenumber_interior,
                boundary_side=boundary_side,
                reference_log_gain=float(final_log_gain),
                boundary_extension=boundary_extension,
            ),
            "fixed_direction_extended_k_diagnostic": boundary_extension,
            "singular_diagnostics": singular,
        }
        checkpoint = {
            "schema": "IJP_REFERENCE_REOPTIMIZATION_V4_WORKING",
            "status": "PARTIAL_NOT_SUBMISSION_EVIDENCE",
            "completed_count": len(completed),
            "required_count": len(v3_records),
            "records": completed,
            "traces": traces,
        }
        _write_json(WORKING, checkpoint)
        print(
            json.dumps(
                {
                    "stage": "reference_contract_complete",
                    "ordinal_this_run": ordinal,
                    "key": key,
                    "completed_total": len(completed),
                    "log_gain": final_log_gain,
                    "k_m_inv": final["k_m_inv"],
                    "improvement": final_log_gain - old_log_gain,
                    "stationarity_pass": completed[key]["local_stationarity_pass"],
                }
            ),
            flush=True,
        )

    complete = len(completed) == len(v3_records)
    if not complete:
        print(
            json.dumps(
                {
                    "status": "PARTIAL_NOT_SUBMISSION_EVIDENCE",
                    "completed_count": len(completed),
                    "required_count": len(v3_records),
                    "working_file": WORKING.relative_to(ROOT).as_posix(),
                },
                indent=2,
            )
        )
        return 3

    # A resumed run may contain records written by an older classification
    # rule.  Recompute every label from the released extension values before
    # forming gates, tables or the final receipt.
    for record in completed.values():
        record["boundary_classification"] = boundary_classification(
            selector_id=record["selector_id"],
            wavenumber_interior=bool(record["wavenumber_interior"]),
            boundary_side=record.get("boundary_side"),
            reference_log_gain=float(record["reference_log_gain"]),
            boundary_extension=record["fixed_direction_extended_k_diagnostic"],
        )

    array_payload: dict[str, np.ndarray] = {
        "reference_coordinate_scales": context.scales,
    }
    winner_rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        record = completed[key]
        output_indices, input_indices = selector_map[record["selector_id"]]
        active_scales = active_norms[record["norm_id"]]
        coordinates = coordinates_from_direction(
            record["direction_n"], record["k_m_inv"]
        )
        final = context.propagator(
            coordinates,
            horizon_indices[record["horizon_id"]],
            substeps=REFERENCE_SUBSTEPS,
            return_generators=(record["norm_id"] == "baseline"),
        )
        active_phi = rescale_propagator(
            final["propagator"], context.scales, active_scales
        )
        _, vectors = singular_diagnostics(active_phi, output_indices, input_indices)
        prefix = key.replace(".", "p")
        array_payload[f"{prefix}__propagator"] = active_phi
        array_payload[f"{prefix}__singular_values"] = vectors["singular_values"]
        array_payload[f"{prefix}__input_singular_vector"] = vectors[
            "input_singular_vector"
        ]
        array_payload[f"{prefix}__output_singular_vector"] = vectors[
            "output_singular_vector"
        ]
        if record["norm_id"] == "baseline":
            array_payload[f"{prefix}__generators"] = final["generators"]
        winner_rows.append(
            {
                "horizon_id": record["horizon_id"],
                "norm_id": record["norm_id"],
                "selector_id": record["selector_id"],
                "n1": record["direction_n"][0],
                "n2": record["direction_n"][1],
                "n3": record["direction_n"][2],
                "k_m_inv": record["k_m_inv"],
                "reference_log_gain": record["reference_log_gain"],
                "reference_gain": record["reference_gain"],
                "previous_v3_reference_log_gain": record[
                    "previous_v3_reference_log_gain"
                ],
                "direct_reference_log_gain_improvement": record[
                    "direct_reference_log_gain_improvement"
                ],
                "local_stationarity_pass": record["local_stationarity_pass"],
                "wavenumber_interior": record["wavenumber_interior"],
            }
        )

    gates = {
        "all_70_contracts_directly_reference_reoptimized": len(completed) == 70,
        "all_contracts_locally_stationary": all(
            record["local_stationarity_pass"] for record in completed.values()
        ),
        "no_reference_refinement_degrades_previous_candidate": all(
            record["direct_reference_log_gain_improvement"] >= -1.0e-9
            for record in completed.values()
        ),
        "all_constitutive_state_localization_selectors_retain_interior_wavenumber": all(
            record["wavenumber_interior"]
            for record in completed.values()
            if record["selector_id"] in FINITE_LOCALIZATION_INTERIOR_SELECTORS
        ),
        "all_boundary_cases_explicitly_extended": all(
            (
                record["wavenumber_interior"]
                and record["selector_id"] not in REVERSIBLE_WAVE_SELECTORS
            )
            or len(record["fixed_direction_extended_k_diagnostic"]) == 2
            for record in completed.values()
        ),
    }
    report = {
        "schema": "IJP_REFERENCE_REOPTIMIZATION_V4",
        "status": (
            "REFERENCE_REOPTIMIZATION_PASS"
            if all(gates.values())
            else "REFERENCE_REOPTIMIZATION_GATES_OPEN"
        ),
        "audit_boundary": (
            "All retained coordinates are optimized directly on the 129-state/"
            "four-substep objective. The shared reference Sobol/anchor/candidate "
            "pool and two separated starts audit basin retention but do not prove "
            "a mathematical global optimum."
        ),
        "reference_contract": {
            "history_factor": REFERENCE_FACTOR,
            "state_count": int(len(context.shears)),
            "midpoint_substeps_per_dense_interval": REFERENCE_SUBSTEPS,
            "sobol_count_per_horizon": REFERENCE_QMC_COUNT,
            "separated_reference_starts_per_contract": REFERENCE_START_COUNT,
            "maximum_powell_iterations": int(args.maximum_iterations),
            "local_stationarity_probe_delta": NEIGHBOUR_DELTA,
            "local_stationarity_log_tolerance": LOCAL_STATIONARITY_LOG_TOL,
            "boundary_extension_log_tolerance": BOUNDARY_EXTENSION_LOG_TOL,
        },
        "source_v3": V3_RESULT.relative_to(ROOT).as_posix(),
        "records": completed,
        "gates": gates,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "artifacts": {
            "arrays": ARRAYS.relative_to(ROOT).as_posix(),
            "winners_csv": WINNERS_CSV.relative_to(ROOT).as_posix(),
            "trace_csv": TRACE_CSV.relative_to(ROOT).as_posix(),
        },
    }
    np.savez_compressed(ARRAYS, **array_payload)
    _write_rows(WINNERS_CSV, winner_rows)
    _write_rows(TRACE_CSV, traces)
    _write_json(RESULT, report)
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
