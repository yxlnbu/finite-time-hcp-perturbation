"""Nonlinearly validate the re-optimized V3 constitutive selector vectors.

The V2 solver and acceptance gates are reused unchanged.  Only the two branch
contracts are replaced by the onset and terminal baseline
constitutive-to-constitutive winners from the V3 re-optimization receipt.
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

import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402
import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402


SCHEMA = "IJP_SINGULAR_VECTOR_NONLINEAR_VALIDATION_V3"
V3_EVIDENCE = ROOT / "05_results/ijp_revision_evidence_v3.json"
V3_ARRAYS = ROOT / "05_results/ijp_revision_evidence_v3_arrays.npz"
RESULT = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v3.json"
TABLE = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v3.csv"
FIELDS = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v3_fields.npz"
FIGURE_STEM = (
    ROOT
    / "06_manuscript/ijp_spectral_hcp/figures/fig12_v3_singular_vector_nonlinear_validation"
)


def write_table(records: list[dict[str, Any]]) -> None:
    rows = [
        {
            "case_id": item["case_id"],
            "role": item["role"],
            "epsilon": item["epsilon"],
            "predicted_linear_gain": item["predicted_linear_gain"],
            "measured_nonlinear_gain": item["measured_nonlinear_gain"],
            "gain_relative_error": item["gain_relative_error"],
            "complex_output_vector_relative_error": item[
                "complex_output_vector_relative_error"
            ],
            "nonretained_fourier_energy_fraction": item[
                "nonretained_fourier_energy_fraction"
            ],
            "ledger_partition_maximum_relative_residual": item[
                "ledger_partition_maximum_relative_residual"
            ],
        }
        for item in records
    ]
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contracts(report: dict[str, Any]) -> dict[str, Any]:
    records = report["reoptimization"]["records"]
    output = {}
    for horizon, role in (("onset", "near_onset"), ("terminal", "terminal")):
        source = records[f"{horizon}__baseline__constitutive_to_constitutive"]
        output[role] = {
            "target_time_s": float(source["time_s"]),
            "direction_n": source["direction_n"],
            "k_m_inv": float(source["k_m_inv"]),
            "source": (
                "V3 direction--wavenumber re-optimization for the baseline "
                "constitutive-to-constitutive selector"
            ),
        }
    return output


def main() -> int:
    started = time.perf_counter()
    for source in (V3_EVIDENCE, V3_ARRAYS):
        v2.require(source.is_file(), f"missing required source {source}")
    evidence = json.loads(V3_EVIDENCE.read_text(encoding="utf-8"))
    dense_payload = v2.load_verified_cache()
    context = dense_payload["contexts"]["factor64"]
    # Preserve the exact V3 reference metric while refining the base history.
    with np.load(V3_ARRAYS) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    _, _, _, spectral = v3.build_case_models(
        "v3_nonlinear_validation"
    )
    admission = v3._admission()
    # The 513-state history subdivides each V3 registered interval by four;
    # one midpoint step per refined interval therefore matches the temporal
    # density of the released 129-state/four-substep calculation while also
    # reducing interpolation mismatch in the nonlinear residual.
    v2.LINEAR_SUBSTEPS = 1
    v2.NONLINEAR_SUBSTEPS = 1
    # This is explicitly a four-nonnegative-mode Galerkin transport check.
    # A separate full-Fourier audit records that the grid-resolved nonlinear
    # continuation becomes non-finite when all resolvable harmonics are kept.
    v2.RETAINED_MODES = tuple(range(4))
    role_contracts = contracts(evidence)

    role_summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    field_payload: dict[str, np.ndarray] = {
        "coordinate_scales": scales,
        "base_times_s": np.asarray(context["times"]),
    }
    for role, contract in role_contracts.items():
        print(json.dumps({"stage": "v3_nonlinear_validation", "role": role}), flush=True)
        summary, records, fields = v2.execute_role(
            role, contract, context, spectral, admission, scales
        )
        role_summaries[role] = summary
        all_records.extend(records)
        for record, field in zip(records, fields, strict=True):
            field_payload[f"{record['case_id']}_final_active_delta"] = field

    reference_gain_errors = {}
    for horizon, role in (("onset", "near_onset"), ("terminal", "terminal")):
        expected = float(
            evidence["reoptimization"]["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]["reference_gain"]
        )
        observed = float(role_summaries[role]["predicted_linear_gain"])
        reference_gain_errors[role] = abs(observed / expected - 1.0)
    dense_convergence: dict[str, Any] = {}
    saved_linear_substeps = v2.LINEAR_SUBSTEPS
    v2.LINEAR_SUBSTEPS = 4
    try:
        for role, contract in role_contracts.items():
            values = {}
            for factor_name in ("factor32", "factor64"):
                dense_context = dense_payload["contexts"][factor_name]
                end_index = v2._exact_index(
                    np.asarray(dense_context["times"]),
                    float(contract["target_time_s"]),
                )
                linear, _ = v2.linear_contract(
                    dense_context,
                    end_index,
                    np.asarray(contract["direction_n"], dtype=float),
                    float(contract["k_m_inv"]),
                    dense_payload["admission"],
                    scales,
                )
                values[f"{factor_name}_four_substep_gain"] = float(
                    linear["final_gain"]
                )
            log32 = float(np.log(values["factor32_four_substep_gain"]))
            log64 = float(np.log(values["factor64_four_substep_gain"]))
            values["factor32_to64_relative_log_gain_change"] = abs(
                log32 / log64 - 1.0
            )
            values["factor64_one_to_four_substep_relative_log_gain_change"] = abs(
                role_summaries[role]["predicted_linear_gain"]
                / values["factor64_four_substep_gain"]
                - 1.0
            )
            dense_convergence[role] = values
    finally:
        v2.LINEAR_SUBSTEPS = saved_linear_substeps
    gates = {
        "both_reoptimized_horizons_completed": set(role_summaries)
        == {"near_onset", "terminal"},
        "257_to_513_state_relative_log_gain_change_below_2_percent": max(
            value["factor32_to64_relative_log_gain_change"]
            for value in dense_convergence.values()
        )
        <= 0.02,
        "513_state_one_to_four_substep_gain_change_below_0p5_percent": max(
            value["factor64_one_to_four_substep_relative_log_gain_change"]
            for value in dense_convergence.values()
        )
        <= 0.005,
        "gain_relative_error_below_2_percent": all(
            item["gain_relative_error"] <= v2.GAIN_ERROR_GATE for item in all_records
        ),
        "complex_output_vector_error_below_3_percent": all(
            item["complex_output_vector_relative_error"]
            <= v2.OUTPUT_VECTOR_ERROR_GATE
            for item in all_records
        ),
        "factor_two_amplitude_collapse_below_2_percent": all(
            item["factor_two_gain_collapse"] <= v2.AMPLITUDE_COLLAPSE_GATE
            for item in role_summaries.values()
        ),
        "nonretained_fourier_energy_below_1_percent": all(
            item["nonretained_fourier_energy_fraction"] <= v2.FOURIER_LEAKAGE_GATE
            for item in all_records
        ),
        "discarded_rhs_fourier_energy_below_1_percent": all(
            item["integration"]["maximum_discarded_rhs_fourier_energy_fraction"]
            <= v2.FOURIER_LEAKAGE_GATE
            for item in all_records
        ),
        "ledger_partition_residual_below_1e_minus_10": all(
            item["ledger_partition_maximum_relative_residual"] <= v2.LEDGER_GATE
            for item in all_records
        ),
        "minimum_16_cells_per_input_wavelength": all(
            item["cells_per_wavelength"] >= 16.0 for item in role_summaries.values()
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": (
            "BOTH_REOPTIMIZED_SINGULAR_VECTOR_GATES_PASS"
            if all(gates.values())
            else "OPEN_REOPTIMIZED_SINGULAR_VECTOR_GATES"
        ),
        "classification": (
            "FOUR_MODE_GALERKIN_SMALL_PERTURBATION_TRANSPORT_NOT_FULL_FOURIER_VALIDATION"
        ),
        "contract": {
            "source": V3_EVIDENCE.relative_to(ROOT).as_posix(),
            "selector": "constitutive_to_constitutive",
            "base_context": (
                "paper-consistent fixed-beta V3 factor64 / 513 registered checkpoints"
            ),
            "metric": "exact released V3 reference_coordinate_scales",
            "linear_and_nonlinear_substeps_per_interval": 1,
            "cells_per_wavelength": v2.CELLS,
            "retained_nonnegative_fourier_modes": list(v2.RETAINED_MODES),
            "target_final_dimensionless_norm": v2.TARGET_FINAL_DIMENSIONLESS_NORM,
            "released_129_to_513_gain_relative_changes": reference_gain_errors,
            "released_129_state_reference_status": (
                "superseded for central gain magnitudes by the converged 513-state map"
            ),
            "dense_history_convergence": dense_convergence,
        },
        "roles": role_summaries,
        "cases": all_records,
        "gates": gates,
        "claim_boundary": (
            "The matching-grid gain and output-vector checks validate only the declared "
            "four-mode Galerkin transport of the actual V3 re-optimized vectors. The "
            "discarded-RHS gate must pass before this can be called a complete Fourier "
            "nonlinear validation; mature width, resistance and calibration remain open."
        ),
        "provenance": {
            "v3_evidence_sha256": v2.sha256(V3_EVIDENCE),
            "v3_arrays": V3_ARRAYS.relative_to(ROOT).as_posix(),
            "v3_arrays_sha256": v2.sha256(V3_ARRAYS),
            "base_context_cache": v2.CACHE.relative_to(ROOT).as_posix(),
            "base_context_cache_sha256": v2.sha256(v2.CACHE),
            "paper_model_contract_builder": Path(v3.__file__).relative_to(ROOT).as_posix(),
            "paper_model_contract_builder_sha256": v2.sha256(Path(v3.__file__)),
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": v2.sha256(Path(__file__)),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            "table": TABLE.relative_to(ROOT).as_posix(),
            "fields": FIELDS.relative_to(ROOT).as_posix(),
            "figure_pdf": FIGURE_STEM.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            "figure_png": FIGURE_STEM.with_suffix(".png").relative_to(ROOT).as_posix(),
        },
    }
    write_table(all_records)
    np.savez_compressed(FIELDS, **field_payload)
    v2.FIGURE_STEM = FIGURE_STEM
    v2.make_figure(all_records, role_summaries)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2), flush=True)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
