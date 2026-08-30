"""Validate near-onset and terminal optimal inputs in the full nonlinear PDE.

For each registered direction--wavenumber branch, the leading right singular
vector is recomputed on the 513-checkpoint base history using the same
constitutive input/output selector as the localization analysis.  Two
factor-two amplitudes are then advanced by the full nonlinear periodic
69-state residual with the registered tangent used only as an integrating
factor.  The measured gain, complex Fourier output and amplitude collapse are
compared with the linear propagator prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hcp_cp_gnd.cp_ti_material_v1 import build_material_objects, load_card  # noqa: E402
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    GENERATOR_RHO_DIPOLE_SLICE,
    GENERATOR_RHO_MOBILE_SLICE,
    GENERATOR_T_SLICE,
    GENERATOR_THETA_SLICE,
    N_GENERATOR,
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)
from hcp_cp_gnd.periodic69_history_v1 import (  # noqa: E402
    PASSIVE_CP_WORK,
    PASSIVE_GENERATED_HEAT,
    PASSIVE_STORED_ENERGY,
    Periodic69HistoryV1,
)
from tools.run_cp_ti_lsbrp_periodic69_v1 import _spectral_model  # noqa: E402
from tools.run_cp_ti_lsbrp_q1_dense_history_v1 import (  # noqa: E402
    CACHE,
    _exact_index,
    load_verified_cache,
)


SCHEMA = "IJP_SINGULAR_VECTOR_NONLINEAR_VALIDATION_V2"
STRENGTHENING = ROOT / "05_results/ijp_strengthening_evidence_v1.json"
RELEASE = ROOT / "05_results/ijp_strengthening_evidence_v1_arrays.npz"
RESULT = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v2.json"
TABLE = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v2.csv"
FIELDS = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v2_fields.npz"
FIGURE_STEM = (
    ROOT
    / "06_manuscript/ijp_spectral_hcp/figures/fig10_singular_vector_nonlinear_validation"
)

OBSERVED = np.arange(6, N_GENERATOR, dtype=int)
CELLS = 16
MODE = 1
RETAINED_MODES = tuple(range(4))
LINEAR_SUBSTEPS = 1
NONLINEAR_SUBSTEPS = 1
NONLINEAR_SCHEME = "exponential_midpoint"
TARGET_FINAL_DIMENSIONLESS_NORM = 1.0e-3
GAIN_ERROR_GATE = 0.02
OUTPUT_VECTOR_ERROR_GATE = 0.03
AMPLITUDE_COLLAPSE_GATE = 0.02
FOURIER_LEAKAGE_GATE = 0.01
LEDGER_GATE = 1.0e-10
TINY = np.finfo(float).tiny


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_generator_state(storage: Any) -> np.ndarray:
    state = np.zeros(N_GENERATOR, dtype=float)
    state[GENERATOR_T_SLICE] = float(storage.temperature_K)
    state[GENERATOR_THETA_SLICE] = 0.0
    state[GENERATOR_RHO_MOBILE_SLICE] = np.asarray(storage.rho_mobile_m2)
    state[GENERATOR_RHO_DIPOLE_SLICE] = np.asarray(storage.rho_dipole_m2)
    state[51:69] = np.asarray(storage.gamma_signed)
    return state


def maximum_admissible_amplitude(
    base: np.ndarray, physical_mode: np.ndarray
) -> float:
    bounds = [1.0e-4]
    for part in (GENERATOR_RHO_MOBILE_SLICE, GENERATOR_RHO_DIPOLE_SLICE):
        amplitude = np.abs(physical_mode[part])
        active = amplitude > TINY
        if np.any(active):
            bounds.append(float(np.min(0.1 * base[part][active] / amplitude[active])))
    temperature_amplitude = float(abs(physical_mode[GENERATOR_T_SLICE][0]))
    if temperature_amplitude > TINY:
        bounds.append(0.1 * float(base[GENERATOR_T_SLICE][0]) / temperature_amplitude)
    chart_amplitude = float(np.max(np.abs(physical_mode[GENERATOR_THETA_SLICE])))
    if chart_amplitude > TINY:
        bounds.append(0.01 / chart_amplitude)
    value = min(bounds)
    require(np.isfinite(value) and value > 0.0, "no admissible perturbation amplitude")
    return value


def observed_rms(active_delta: np.ndarray, scales: np.ndarray) -> float:
    dimensionless = active_delta[:, OBSERVED] / scales[OBSERVED][None, :]
    return float(np.sqrt(np.mean(np.sum(dimensionless**2, axis=1))))


def fourier_leakage(active_delta: np.ndarray, scales: np.ndarray) -> float:
    dimensionless = active_delta[:, OBSERVED] / scales[OBSERVED][None, :]
    energy = np.sum(np.abs(np.fft.fft(dimensionless, axis=0)) ** 2, axis=1)
    retained = {0}
    for mode in RETAINED_MODES:
        retained.add(mode)
        if mode > 0:
            retained.add(CELLS - mode)
    total = float(np.sum(energy))
    kept = float(sum(energy[index] for index in retained))
    return 0.0 if total <= TINY else float(max(total - kept, 0.0) / total)


def branch_contracts(strengthening: dict[str, Any], release: Any) -> dict[str, Any]:
    onset = strengthening["global_search"]["high_resolution_verification"]["onset"]
    return {
        "near_onset": {
            "target_time_s": float(strengthening["global_search"]["scan"]["onset_evaluation_time_s"]),
            "direction_n": onset["direction_n"],
            "k_m_inv": float(onset["k_m_inv"]),
            "source": "anchor-assisted search candidate reevaluated on the 129-state history",
        },
        "terminal": {
            "target_time_s": 140.0e-6,
            "direction_n": np.asarray(release["baseline_terminal_y_direction_n"]).tolist(),
            "k_m_inv": float(release["baseline_terminal_y_k_m_inv"][0]),
            "source": "registered terminal branch reported in the manuscript",
        },
    }


def linear_contract(
    context: dict[str, Any],
    end_index: int,
    direction: np.ndarray,
    k_m_inv: float,
    admission: Any,
    scales: np.ndarray,
) -> tuple[dict[str, Any], list[Any]]:
    operators = [
        assemble_dynamic_crystal_operator_v1(
            point,
            wavenumber_m_inv=k_m_inv,
            direction_n=direction,
            admission=admission,
        )
        for point in context["points"][: end_index + 1]
    ]
    history = finite_time_amplification_history(
        operators,
        context["times"][: end_index + 1],
        coordinate_scales=scales,
        input_indices=OBSERVED,
        output_indices=OBSERVED,
        integration_substeps_per_interval=LINEAR_SUBSTEPS,
    )
    return history, operators


def execute_role(
    role: str,
    contract: dict[str, Any],
    context: dict[str, Any],
    spectral: Any,
    admission: Any,
    scales: np.ndarray,
    amplitude_multipliers: tuple[float, ...] = (0.5, 1.0),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    role_started = time.perf_counter()
    direction = np.asarray(contract["direction_n"], dtype=float)
    direction /= np.linalg.norm(direction)
    k_m_inv = float(contract["k_m_inv"])
    end_index = _exact_index(np.asarray(context["times"]), float(contract["target_time_s"]))
    linear, operators = linear_contract(
        context, end_index, direction, k_m_inv, admission, scales
    )
    vector = np.asarray(linear["input_vector_dimensionless"], dtype=np.complex128)
    predicted_response = np.asarray(
        linear["full_state_output_response_dimensionless"], dtype=np.complex128
    )
    predicted_gain = float(linear["final_gain"])
    base = base_generator_state(context["storages"][0])
    physical_mode = scales * vector
    admissible = maximum_admissible_amplitude(base, physical_mode)
    high = min(0.25 * admissible, TARGET_FINAL_DIMENSIONLESS_NORM / predicted_gain)
    multipliers = tuple(float(value) for value in amplitude_multipliers)
    require(
        bool(multipliers)
        and all(np.isfinite(value) and 0.0 < value <= 1.0 for value in multipliers),
        "amplitude multipliers must lie in (0,1]",
    )
    amplitudes = tuple(value * high for value in multipliers)
    require(high > 0.0, f"{role} has no positive small amplitude")

    domain_length = float(2.0 * np.pi * MODE / k_m_inv)
    history = Periodic69HistoryV1(
        spectral,
        np.asarray(context["shears"][: end_index + 1]),
        np.asarray(context["times"][: end_index + 1]),
        context["storages"][: end_index + 1],
        direction,
        domain_length,
        CELLS,
    )
    endpoint_generators: dict[tuple[int, int], np.ndarray] = {}

    def scaled_generator(interval: int, fraction: float, mode: int) -> np.ndarray:
        def endpoint(index: int) -> np.ndarray:
            key = (int(index), int(mode))
            cached = endpoint_generators.get(key)
            if cached is not None:
                return cached
            mode_k = 2.0 * np.pi * int(mode) / domain_length
            if mode == 0:
                mode_k = max(2.0 * np.pi / domain_length * 1.0e-12, 1.0e-9)
            operator = assemble_dynamic_crystal_operator_v1(
                context["points"][index],
                wavenumber_m_inv=mode_k,
                direction_n=direction,
                admission=admission,
            )
            value = operator.generator_A * scales[None, :] / scales[:, None]
            endpoint_generators[key] = value
            return value

        return (1.0 - fraction) * endpoint(interval) + fraction * endpoint(interval + 1)

    x = np.arange(CELLS, dtype=float)
    carrier = np.exp(2j * np.pi * MODE * x / CELLS)
    records: list[dict[str, Any]] = []
    final_fields: list[np.ndarray] = []
    for amplitude_index, epsilon in enumerate(amplitudes, start=1):
        case_started = time.perf_counter()
        initial = epsilon * np.real(carrier[:, None] * vector[None, :])
        initial_physical = initial * scales[None, :]
        initial_norm = observed_rms(initial_physical, scales)
        integrated = history.integrate_exponential_midpoint(
            initial_physical,
            coordinate_scales=scales,
            scaled_generator=scaled_generator,
            retained_nonnegative_modes=RETAINED_MODES,
            integration_substeps_per_interval=NONLINEAR_SUBSTEPS,
            integration_scheme=NONLINEAR_SCHEME,
            end_index=end_index,
        )
        final = np.asarray(integrated["active_delta"], dtype=float)
        passive = np.asarray(integrated["passive_delta"], dtype=float)
        final_norm = observed_rms(final, scales)
        measured_gain = final_norm / max(initial_norm, TINY)
        coefficient = 2.0 * np.fft.fft(final / scales[None, :], axis=0)[MODE] / CELLS
        measured_response = coefficient / epsilon
        output_error = float(
            np.linalg.norm(measured_response - predicted_response)
            / max(np.linalg.norm(predicted_response), TINY)
        )
        final_frame = history.base_frame(len(history.base_times_s) - 2, 1.0)
        cp_work = final_frame.observer.cp_work_density_J_m3 + passive[:, PASSIVE_CP_WORK]
        heat = final_frame.observer.generated_heat_density_J_m3 + passive[:, PASSIVE_GENERATED_HEAT]
        nonthermal = final_frame.observer.passive_storage_density_J_m3 + passive[:, PASSIVE_STORED_ENERGY]
        ledger_error = float(
            np.max(np.abs(cp_work - heat - nonthermal) / np.maximum(np.abs(cp_work), 1.0))
        )
        record = {
            "case_id": f"{role}_a{amplitude_index}",
            "role": role,
            "amplitude_index": amplitude_index,
            "epsilon": float(epsilon),
            "predicted_linear_gain": predicted_gain,
            "measured_nonlinear_gain": measured_gain,
            "gain_relative_error": abs(measured_gain - predicted_gain) / predicted_gain,
            "complex_output_vector_relative_error": output_error,
            "nonretained_fourier_energy_fraction": fourier_leakage(final, scales),
            "ledger_partition_maximum_relative_residual": ledger_error,
            "final_dimensionless_observed_rms": final_norm,
            "runtime_s": time.perf_counter() - case_started,
            "integration": {
                key: value
                for key, value in integrated.items()
                if key not in ("active_delta", "passive_delta", "monitor_records")
            },
        }
        records.append(record)
        final_fields.append(final)

    collapse = (
        abs(records[-1]["measured_nonlinear_gain"] - records[0]["measured_nonlinear_gain"])
        / max(abs(records[0]["measured_nonlinear_gain"]), TINY)
        if len(records) >= 2
        else None
    )
    summary = {
        **contract,
        "direction_n": direction.tolist(),
        "end_index": end_index,
        "base_state_count": end_index + 1,
        "cells": CELLS,
        "mode": MODE,
        "cells_per_wavelength": CELLS / MODE,
        "domain_length_m": domain_length,
        "linear_substeps_per_interval": LINEAR_SUBSTEPS,
        "nonlinear_substeps_per_interval": NONLINEAR_SUBSTEPS,
        "nonlinear_integration_scheme": NONLINEAR_SCHEME,
        "predicted_linear_gain": predicted_gain,
        "admissible_amplitude_ceiling": admissible,
        "tested_amplitudes": list(amplitudes),
        "factor_two_gain_collapse": collapse,
        "maximum_gain_relative_error": max(item["gain_relative_error"] for item in records),
        "maximum_complex_output_vector_relative_error": max(
            item["complex_output_vector_relative_error"] for item in records
        ),
        "maximum_nonretained_fourier_energy_fraction": max(
            item["nonretained_fourier_energy_fraction"] for item in records
        ),
        "maximum_ledger_partition_relative_residual": max(
            item["ledger_partition_maximum_relative_residual"] for item in records
        ),
        "runtime_s": time.perf_counter() - role_started,
    }
    return summary, records, final_fields


def write_table(records: list[dict[str, Any]]) -> None:
    rows = [
        {
            "case_id": item["case_id"],
            "role": item["role"],
            "epsilon": item["epsilon"],
            "predicted_linear_gain": item["predicted_linear_gain"],
            "measured_nonlinear_gain": item["measured_nonlinear_gain"],
            "gain_relative_error": item["gain_relative_error"],
            "complex_output_vector_relative_error": item["complex_output_vector_relative_error"],
            "nonretained_fourier_energy_fraction": item["nonretained_fourier_energy_fraction"],
            "ledger_partition_maximum_relative_residual": item["ledger_partition_maximum_relative_residual"],
        }
        for item in records
    ]
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(records: list[dict[str, Any]], roles: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.4,
            "legend.fontsize": 9.0,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = [
        ("O" if item["role"] == "near_onset" else "T")
        + str(item["amplitude_index"])
        for item in records
    ]
    x = np.arange(len(records))
    predicted = [item["predicted_linear_gain"] for item in records]
    measured = [item["measured_nonlinear_gain"] for item in records]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.45))
    ax = axes[0]
    width = 0.38
    ax.bar(x - width / 2, predicted, width, label="linear")
    ax.bar(x + width / 2, measured, width, label="nonlinear")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("gain")
    ax.set_title("(a) Linear/nonlinear gain")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x, [100.0 * item["gain_relative_error"] for item in records], color="#2166ac")
    ax.axhline(100.0 * GAIN_ERROR_GATE, color="#b2182b", ls="--", lw=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("gain error [%]")
    ax.set_title("(b) Small-amplitude gain gate")

    ax = axes[2]
    role_names = list(roles)
    role_labels = [
        "onset" if role == "near_onset" else "terminal" for role in role_names
    ]
    collapse_values = [
        100.0 * roles[role]["factor_two_gain_collapse"] for role in role_names
    ]
    bars = ax.bar(
        np.arange(len(role_names)),
        collapse_values,
        color="#4d9221",
    )
    ax.axhline(100.0 * AMPLITUDE_COLLAPSE_GATE, color="#b2182b", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-8, 3.0)
    ax.set_xticks(np.arange(len(role_names)), role_labels, rotation=20, ha="right")
    ax.set_ylabel("factor-two gain change [%]")
    ax.set_title("(c) Amplitude collapse")
    for bar, value in zip(bars, collapse_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value * 1.5, 2.0e-8),
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=8.6,
        )
    fig.tight_layout(pad=0.7, w_pad=1.0)
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    for source in (STRENGTHENING, RELEASE, CACHE):
        require(source.is_file(), f"missing required source {source}")
    strengthening = json.loads(STRENGTHENING.read_text(encoding="utf-8"))
    release = np.load(RELEASE)
    payload = load_verified_cache()
    context = payload["contexts"]["factor64"]
    scales = np.asarray(payload["coordinate_scales"], dtype=float)
    card = load_card()
    _, parameters, model = build_material_objects(card)
    spectral = _spectral_model(model, parameters, card)
    contracts = branch_contracts(strengthening, release)

    role_summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    field_payload: dict[str, np.ndarray] = {
        "coordinate_scales": scales,
        "base_times_s": np.asarray(context["times"]),
    }
    for role, contract in contracts.items():
        print(json.dumps({"stage": "nonlinear_validation", "role": role}), flush=True)
        summary, records, fields = execute_role(
            role, contract, context, spectral, payload["admission"], scales
        )
        role_summaries[role] = summary
        all_records.extend(records)
        for record, field in zip(records, fields, strict=True):
            field_payload[f"{record['case_id']}_final_active_delta"] = field

    gates = {
        "both_horizons_completed": set(role_summaries) == {"near_onset", "terminal"},
        "gain_relative_error_below_2_percent": all(
            item["gain_relative_error"] <= GAIN_ERROR_GATE for item in all_records
        ),
        "complex_output_vector_error_below_3_percent": all(
            item["complex_output_vector_relative_error"] <= OUTPUT_VECTOR_ERROR_GATE
            for item in all_records
        ),
        "factor_two_amplitude_collapse_below_2_percent": all(
            item["factor_two_gain_collapse"] <= AMPLITUDE_COLLAPSE_GATE
            for item in role_summaries.values()
        ),
        "nonretained_fourier_energy_below_1_percent": all(
            item["nonretained_fourier_energy_fraction"] <= FOURIER_LEAKAGE_GATE
            for item in all_records
        ),
        "ledger_partition_residual_below_1e_minus_10": all(
            item["ledger_partition_maximum_relative_residual"] <= LEDGER_GATE
            for item in all_records
        ),
        "minimum_16_cells_per_input_wavelength": all(
            item["cells_per_wavelength"] >= 16.0 for item in role_summaries.values()
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": "BOTH_SINGULAR_VECTOR_NONLINEAR_GATES_PASS" if all(gates.values()) else "OPEN_SINGULAR_VECTOR_NONLINEAR_GATES",
        "classification": "NONLINEAR_SMALL_PERTURBATION_TRANSPORT_VALIDATION_NOT_MATURE_BAND_WIDTH",
        "contract": {
            "base_context": "factor64 / 513 registered checkpoints",
            "active_state_count": N_GENERATOR,
            "input_output_indices": OBSERVED.tolist(),
            "full_nonlinear_residual": "periodic momentum--heat--micromorphic--crystal-state residual minus the homogeneous base residual",
            "integration": "registered-tangent exponential midpoint with nonlinear remainder",
            "target_final_dimensionless_norm": TARGET_FINAL_DIMENSIONLESS_NORM,
        },
        "roles": role_summaries,
        "cases": all_records,
        "gates": gates,
        "claim_boundary": (
            "Passing this gate validates small-amplitude transport of the prescribed "
            "near-onset and terminal singular vectors. It does not establish an emergent "
            "mature shear-band width, propagation resistance or specimen calibration."
        ),
        "provenance": {
            "strengthening_result": STRENGTHENING.relative_to(ROOT).as_posix(),
            "strengthening_sha256": sha256(STRENGTHENING),
            "release_archive": RELEASE.relative_to(ROOT).as_posix(),
            "release_sha256": sha256(RELEASE),
            "base_cache": CACHE.relative_to(ROOT).as_posix(),
            "base_cache_sha256": sha256(CACHE),
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": sha256(Path(__file__)),
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
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
    make_figure(all_records, role_summaries)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "result": str(RESULT)}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
