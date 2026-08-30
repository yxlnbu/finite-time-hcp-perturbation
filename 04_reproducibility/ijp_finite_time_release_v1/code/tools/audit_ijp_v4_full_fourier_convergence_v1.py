"""Audit full-grid small-perturbation transport across spectral and time ladders.

The legacy 16-cell all-mode audit mixed a continuum Nyquist generator with an
even-grid real collocation residual.  This script preserves that case as a
negative control, then separates three questions: exclusion of the unmatched
Nyquist coefficient, standard two-thirds de-aliasing, and convergence with
grid, time density, and an independent Lawson--Euler integrating-factor
format.  Every trial uses the latest complete direct-reference singular input
and metric (V4 when available, otherwise the retained V3 release).
"""

from __future__ import annotations

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
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hcp_cp_gnd.periodic69_history_v1 import Periodic69IntegrationFailure  # noqa: E402
import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v3 as v3nl  # noqa: E402


SCHEMA = "IJP_V4_FULL_FOURIER_CONVERGENCE_AUDIT_V1"
RESULT = ROOT / "05_results/ijp_v4_full_fourier_convergence_audit_v1.json"
WORKING = ROOT / "05_results/ijp_v4_full_fourier_convergence_audit_v1_working.json"
TABLE = ROOT / "05_results/ijp_v4_full_fourier_convergence_audit_v1.csv"


def retained_modes(cells: int, spectral_rule: str) -> tuple[int, ...]:
    if spectral_rule == "raw_all_including_nyquist":
        return tuple(range(cells // 2 + 1))
    if spectral_rule == "all_resolvable_excluding_nyquist":
        return tuple(range(cells // 2))
    if spectral_rule == "two_thirds_dealiased":
        return tuple(range(cells // 3 + 1))
    raise ValueError(f"unknown spectral rule {spectral_rule}")


def definitions() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "case_id": "raw_nyquist_n16_factor64_s1_midpoint_onset",
            "role": "near_onset",
            "context": "factor64",
            "cells": 16,
            "spectral_rule": "raw_all_including_nyquist",
            "substeps": 1,
            "scheme": "exponential_midpoint",
            "control_role": "negative_control_for_discrete_continuum_mismatch",
        }
    ]
    for role in ("near_onset", "terminal"):
        cases.append(
            {
                "case_id": f"no_nyquist_n16_factor64_s1_midpoint_{role}",
                "role": role,
                "context": "factor64",
                "cells": 16,
                "spectral_rule": "all_resolvable_excluding_nyquist",
                "substeps": 1,
                "scheme": "exponential_midpoint",
                "control_role": "matched_discrete_state_space_control",
            }
        )
    for cells in (16, 32, 64):
        for role in ("near_onset", "terminal"):
            cases.append(
                {
                    "case_id": f"dealias_n{cells}_factor64_s1_midpoint_{role}",
                    "role": role,
                    "context": "factor64",
                    "cells": cells,
                    "spectral_rule": "two_thirds_dealiased",
                    "substeps": 1,
                    "scheme": "exponential_midpoint",
                    "control_role": "grid_convergence",
                }
            )
    for substeps in (1, 2, 4):
        for role in ("near_onset", "terminal"):
            cases.append(
                {
                    "case_id": f"dealias_n16_factor32_s{substeps}_midpoint_{role}",
                    "role": role,
                    "context": "factor32",
                    "cells": 16,
                    "spectral_rule": "two_thirds_dealiased",
                    "substeps": substeps,
                    "scheme": "exponential_midpoint",
                    "control_role": "time_density_convergence",
                }
            )
    for role in ("near_onset", "terminal"):
        cases.append(
            {
                "case_id": f"dealias_n16_factor64_s1_lawson_{role}",
                "role": role,
                "context": "factor64",
                "cells": 16,
                "spectral_rule": "two_thirds_dealiased",
                "substeps": 1,
                "scheme": "lawson_euler",
                "control_role": "independent_integration_format",
            }
        )
    return cases


def relative_change(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), np.finfo(float).tiny
    )


def gain(record: dict[str, Any]) -> float:
    return float(record["records"][0]["measured_nonlinear_gain"])


def run_case(
    definition: dict[str, Any],
    *,
    contexts: dict[str, Any],
    spectral: Any,
    admission: Any,
    scales: np.ndarray,
    contracts: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    cells = int(definition["cells"])
    modes = retained_modes(cells, str(definition["spectral_rule"]))
    v2.CELLS = cells
    v2.RETAINED_MODES = modes
    v2.LINEAR_SUBSTEPS = 1
    v2.NONLINEAR_SUBSTEPS = int(definition["substeps"])
    v2.NONLINEAR_SCHEME = str(definition["scheme"])
    output = {
        **definition,
        "retained_nonnegative_modes": list(modes),
        "cutoff_mode": int(modes[-1]),
    }
    print(json.dumps({"stage": "full_fourier_ladder", **definition}), flush=True)
    try:
        summary, records, _ = v2.execute_role(
            str(definition["role"]),
            contracts[str(definition["role"])],
            contexts[str(definition["context"])],
            spectral,
            admission,
            scales,
            amplitude_multipliers=(1.0,),
        )
        output.update({"completed": True, "summary": summary, "records": records})
    except Periodic69IntegrationFailure as error:
        output.update(
            {
                "completed": False,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "first_failure": error.diagnostics,
            }
        )
    except (FloatingPointError, OverflowError, ValueError) as error:
        output.update(
            {
                "completed": False,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "first_failure": {"structured_location_available": False},
            }
        )
    output["runtime_s"] = float(time.perf_counter() - started)
    return output


def summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grid: dict[str, Any] = {}
    time_ladder: dict[str, Any] = {}
    format_ladder: dict[str, Any] = {}
    for role in ("near_onset", "terminal"):
        grid_values = {
            cells: gain(records[f"dealias_n{cells}_factor64_s1_midpoint_{role}"])
            for cells in (16, 32, 64)
        }
        grid[role] = {
            "measured_gains": {str(key): value for key, value in grid_values.items()},
            "n16_to_n32_relative_change": relative_change(grid_values[16], grid_values[32]),
            "n32_to_n64_relative_change": relative_change(grid_values[32], grid_values[64]),
        }
        time_values = {
            substeps: gain(records[f"dealias_n16_factor32_s{substeps}_midpoint_{role}"])
            for substeps in (1, 2, 4)
        }
        reference = gain(records[f"dealias_n16_factor64_s1_midpoint_{role}"])
        time_ladder[role] = {
            "factor32_measured_gains": {
                str(key): value for key, value in time_values.items()
            },
            "factor64_one_substep_reference_gain": reference,
            "factor32_s1_to_s2_relative_change": relative_change(
                time_values[1], time_values[2]
            ),
            "factor32_s2_to_s4_relative_change": relative_change(
                time_values[2], time_values[4]
            ),
            "factor32_s2_to_factor64_s1_relative_change": relative_change(
                time_values[2], reference
            ),
            "factor32_s4_to_factor64_s1_relative_change": relative_change(
                time_values[4], reference
            ),
        }
        lawson = gain(records[f"dealias_n16_factor64_s1_lawson_{role}"])
        format_ladder[role] = {
            "exponential_midpoint_gain": reference,
            "lawson_euler_gain": lawson,
            "relative_change": relative_change(reference, lawson),
        }
    primary_dealiased = [
        records[f"dealias_n{cells}_factor64_s1_midpoint_{role}"]
        for cells in (16, 32, 64)
        for role in ("near_onset", "terminal")
    ]
    primary_records = [item["records"][0] for item in primary_dealiased]
    raw = records["raw_nyquist_n16_factor64_s1_midpoint_onset"]
    raw_failure = raw.get("first_failure", {})
    gates = {
        "raw_negative_control_fails_at_nyquist": (
            not raw["completed"]
            and raw_failure.get("conjugacy_defect_nonnegative_mode") == 8
        ),
        "matched_no_nyquist_control_completes_both_horizons": all(
            records[f"no_nyquist_n16_factor64_s1_midpoint_{role}"]["completed"]
            for role in ("near_onset", "terminal")
        ),
        "two_thirds_grid_ladder_completes": all(
            records[f"dealias_n{cells}_factor64_s1_midpoint_{role}"]["completed"]
            for cells in (16, 32, 64)
            for role in ("near_onset", "terminal")
        ),
        "n32_to_n64_gain_change_below_1_percent": max(
            item["n32_to_n64_relative_change"] for item in grid.values()
        )
        <= 0.01,
        "time_density_crosscheck_below_1_percent": max(
            max(
                item["factor32_s2_to_factor64_s1_relative_change"],
                item["factor32_s4_to_factor64_s1_relative_change"],
            )
            for item in time_ladder.values()
        )
        <= 0.01,
        "integration_format_change_below_2_percent": max(
            item["relative_change"] for item in format_ladder.values()
        )
        <= 0.02,
        "primary_dealiased_linear_gain_error_below_2_percent": max(
            float(item["gain_relative_error"]) for item in primary_records
        )
        <= 0.02,
        "primary_dealiased_output_vector_error_below_3_percent": max(
            float(item["complex_output_vector_relative_error"])
            for item in primary_records
        )
        <= 0.03,
        "primary_dealiased_discarded_rhs_energy_below_1_percent": max(
            float(item["integration"]["maximum_discarded_rhs_fourier_energy_fraction"])
            for item in primary_records
        )
        <= 0.01,
    }
    numerical_gate_names = (
        "raw_negative_control_fails_at_nyquist",
        "matched_no_nyquist_control_completes_both_horizons",
        "two_thirds_grid_ladder_completes",
        "n32_to_n64_gain_change_below_1_percent",
        "time_density_crosscheck_below_1_percent",
        "integration_format_change_below_2_percent",
        "primary_dealiased_linear_gain_error_below_2_percent",
        "primary_dealiased_output_vector_error_below_3_percent",
    )
    return {
        "first_failure_localization": raw_failure,
        "grid_convergence": grid,
        "time_density_convergence": time_ladder,
        "integration_format_sensitivity": format_ladder,
        "maximum_primary_dealiased_gain_relative_error": max(
            float(item["gain_relative_error"]) for item in primary_records
        ),
        "maximum_primary_dealiased_output_vector_relative_error": max(
            float(item["complex_output_vector_relative_error"])
            for item in primary_records
        ),
        "maximum_primary_dealiased_discarded_rhs_fourier_energy_fraction": max(
            float(item["integration"]["maximum_discarded_rhs_fourier_energy_fraction"])
            for item in primary_records
        ),
        "gates": gates,
        "numerical_convergence_passed": all(
            gates[name] for name in numerical_gate_names
        ),
        "full_nonlinear_rhs_closure_passed": gates[
            "primary_dealiased_discarded_rhs_energy_below_1_percent"
        ],
        "passed": all(gates.values()),
    }


def write_csv(records: dict[str, dict[str, Any]]) -> None:
    rows = []
    for item in records.values():
        base = {
            key: item.get(key)
            for key in (
                "case_id",
                "role",
                "context",
                "cells",
                "spectral_rule",
                "cutoff_mode",
                "substeps",
                "scheme",
                "control_role",
                "completed",
                "runtime_s",
                "failure_type",
                "failure_message",
            )
        }
        if item["completed"]:
            record = item["records"][0]
            base.update(
                {
                    "measured_nonlinear_gain": record["measured_nonlinear_gain"],
                    "gain_relative_error": record["gain_relative_error"],
                    "output_vector_relative_error": record[
                        "complex_output_vector_relative_error"
                    ],
                    "discarded_rhs_energy_fraction": record["integration"][
                        "maximum_discarded_rhs_fourier_energy_fraction"
                    ],
                }
            )
        else:
            base.update(
                {
                    "first_failure_time_s": item.get("first_failure", {}).get("time_s"),
                    "first_failure_stage": item.get("first_failure", {}).get("stage"),
                    "first_failure_mode": item.get("first_failure", {}).get(
                        "conjugacy_defect_nonnegative_mode"
                    ),
                    "first_failure_wavenumber_m_inv": item.get("first_failure", {}).get(
                        "conjugacy_defect_wavenumber_m_inv"
                    ),
                    "first_failure_state_group": item.get("first_failure", {}).get(
                        "conjugacy_defect_state_group"
                    ),
                }
            )
        rows.append(base)
    fields = sorted({key for row in rows for key in row})
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    started = time.perf_counter()
    use_v4 = v3nl.V4_EVIDENCE.is_file() and v3nl.V4_ARRAYS.is_file()
    evidence_path = v3nl.V4_EVIDENCE if use_v4 else v3nl.V3_EVIDENCE
    arrays_path = v3nl.V4_ARRAYS if use_v4 else v3nl.V3_ARRAYS
    version_label = "V4" if use_v4 else "V3"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if use_v4:
        v2.require(
            evidence.get("status") == "REFERENCE_REOPTIMIZATION_PASS",
            "V4 reference re-optimization gates are open",
        )
    with np.load(arrays_path) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    evidence_sha256 = v2.sha256(evidence_path)
    arrays_sha256 = v2.sha256(arrays_path)
    payload = v2.load_verified_cache()
    contexts = payload["contexts"]
    _, _, _, spectral = v3.build_case_models("v4_full_fourier_convergence")
    admission = v3._admission()
    contracts = v3nl.contracts(evidence)
    records: dict[str, dict[str, Any]] = {}
    if WORKING.is_file():
        prior = json.loads(WORKING.read_text(encoding="utf-8"))
        if (
            prior.get("schema") == SCHEMA
            and prior.get("optimization_evidence_sha256") == evidence_sha256
            and prior.get("optimization_arrays_sha256") == arrays_sha256
        ):
            records = dict(prior.get("records", {}))
    for definition in definitions():
        case_id = str(definition["case_id"])
        if case_id in records:
            print(json.dumps({"stage": "resume_skip", "case_id": case_id}), flush=True)
            continue
        records[case_id] = run_case(
            definition,
            contexts=contexts,
            spectral=spectral,
            admission=admission,
            scales=scales,
            contracts=contracts,
        )
        WORKING.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "optimization_version": version_label,
                    "optimization_evidence_sha256": evidence_sha256,
                    "optimization_arrays_sha256": arrays_sha256,
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    audit = summarize(records)
    report = {
        "schema": SCHEMA,
        "status": (
            "DEALIASED_FULL_GRID_SMALL_PERTURBATION_GATE_PASS"
            if audit["passed"]
            else (
                "NUMERICAL_CONVERGENCE_PASS_NONLINEAR_RHS_CLOSURE_OPEN"
                if audit["numerical_convergence_passed"]
                else "DEALIASED_FULL_GRID_SMALL_PERTURBATION_GATE_OPEN"
            )
        ),
        "classification": (
            "THEORY_METHOD_VERIFICATION_OF_SMALL_PERTURBATION_TRANSPORT_"
            "NOT_NONLINEAR_SHEAR_BAND_WIDTH_VALIDATION"
        ),
        "contract": {
            "released_selector": "constitutive_to_constitutive",
            "released_inputs": f"{version_label} direct-reference near-onset and terminal right singular vectors",
            "coordinate_metric": f"exact released {version_label} reference coordinate scales",
            "amplitude": "larger preregistered small-perturbation amplitude only",
            "dealiasing": "retain nonnegative modes 0..floor(N/3)",
            "grids": [16, 32, 64],
            "base_histories": {key: len(value["times"]) for key, value in contexts.items()},
            "integration_schemes": ["exponential_midpoint", "lawson_euler"],
        },
        "audit": audit,
        "records": records,
        "claim_boundary": (
            "Passing establishes grid-, time-density-, and integrator-insensitive transport "
            "of the released infinitesimal inputs on a dealiased periodic grid. It does not "
            "establish finite-amplitude saturation, a seed-independent band width, fracture "
            "energy, or propagation resistance."
        ),
        "provenance": {
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": v2.sha256(Path(__file__)),
            "optimization_evidence": evidence_path.relative_to(ROOT).as_posix(),
            "optimization_evidence_sha256": evidence_sha256,
            "optimization_arrays": arrays_path.relative_to(ROOT).as_posix(),
            "optimization_arrays_sha256": arrays_sha256,
            "base_context_cache": v2.CACHE.relative_to(ROOT).as_posix(),
            "base_context_cache_sha256": v2.sha256(v2.CACHE),
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
    write_csv(records)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": audit["gates"]}, indent=2), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
