"""Finite-time HCP crystal perturbation scan along the CP-Ti base history.

This runner is intentionally independent of U903 mesh generation.  It builds
the complete continuous 84-coordinate pencil at exact homogeneous base-state
checkpoints, eliminates only the algebraic micromorphic slips, and propagates
the resulting 69-state generator over time for a fixed Fourier pair (k,n).
"""

from __future__ import annotations

import argparse
import csv
import json
from math import cos, pi, sin
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    load_card,
    local_state92_from_material_state,
    run_representative_checkpoint_states,
    simple_shear,
)
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)
from hcp_cp_gnd.micromorphic import MicromorphicParameters  # noqa: E402
from hcp_cp_gnd.qs_descriptor import QSDescriptorAdmission  # noqa: E402
from hcp_cp_gnd.spectral_export import ContinuousSpectralPointModel  # noqa: E402


SCHEMA = "CP_TI_FINITE_TIME_DYNAMIC_PERTURBATION_V2"
RESULT = ROOT / "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json"
SCAN_CSV = ROOT / "05_results/cp_ti_finite_time_scan_v2.csv"
SUMMARY = ROOT / "05_results/cp_ti_finite_time_dynamic_perturbation_v2.md"


def _admission() -> QSDescriptorAdmission:
    return QSDescriptorAdmission(
        microforce_abs_tolerance_Pa=1.0e-5,
        power_identity_abs_tolerance_W_m3=1.0e-2,
        power_partition_abs_tolerance_W_m3=1.0e-2,
        minimum_normalized_branch_distance=1.0e-10,
        direction_norm_abs_tolerance=2.0e-14,
        determinant_abs_tolerance=2.0e-12,
        symmetry_abs_tolerance=2.0e-12,
        psd_eigenvalue_abs_tolerance=2.0e-12,
    )


def _hemisphere_directions(count: int) -> list[dict[str, Any]]:
    """Return deterministic antipodal representatives including sample axes."""

    if count < 3:
        raise ValueError("direction count must be at least three")
    candidates: list[tuple[str, np.ndarray]] = [
        ("sample_x", np.array([1.0, 0.0, 0.0])),
        ("sample_y", np.array([0.0, 1.0, 0.0])),
        ("sample_z", np.array([0.0, 0.0, 1.0])),
    ]
    golden = pi * (3.0 - np.sqrt(5.0))
    for index in range(max(0, count - 3)):
        z = (index + 0.5) / max(1, count - 3)
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        azimuth = golden * index
        candidates.append(
            (
                f"fibonacci_{index:03d}",
                np.array([radius * cos(azimuth), radius * sin(azimuth), z]),
            )
        )
    output = []
    for identifier, value in candidates:
        unit = value / np.linalg.norm(value)
        output.append({"direction_id": identifier, "direction_n": unit.tolist(), "scan_level": "coarse"})
    return output


def _refined_directions(center: np.ndarray) -> list[dict[str, Any]]:
    """Construct a deterministic tangent-ring refinement around one normal."""

    normal = np.asarray(center, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    output = [
        {"direction_id": "refined_center", "direction_n": normal.tolist(), "scan_level": "refined"}
    ]
    for angle_deg in (2.5, 5.0, 10.0):
        angle = np.deg2rad(angle_deg)
        for azimuth_index in range(8):
            azimuth = 2.0 * pi * azimuth_index / 8.0
            value = (
                np.cos(angle) * normal
                + np.sin(angle)
                * (np.cos(azimuth) * tangent_1 + np.sin(azimuth) * tangent_2)
            )
            # Canonical antipodal representative: upper hemisphere, with a
            # deterministic tie-break on its equator.
            if value[2] < -1.0e-14 or (
                abs(value[2]) <= 1.0e-14
                and (value[1] < -1.0e-14 or (abs(value[1]) <= 1.0e-14 and value[0] < 0.0))
            ):
                value = -value
            output.append(
                {
                    "direction_id": f"refined_{angle_deg:g}deg_{azimuth_index:02d}",
                    "direction_n": value.tolist(),
                    "scan_level": "refined",
                }
            )
    return output


def _coordinate_scales(storages: list[Any]) -> np.ndarray:
    rho_mobile = np.median(np.stack([state.rho_mobile_m2 for state in storages]), axis=0)
    rho_dipole = np.median(np.stack([state.rho_dipole_m2 for state in storages]), axis=0)
    return np.r_[
        np.full(3, 1.0e-6),       # displacement amplitude
        np.full(3, 1.0e-1),       # velocity amplitude
        1.0,                      # temperature amplitude
        np.full(8, 2.0e-4),       # local SL(3) chart
        rho_mobile,
        rho_dipole,
        np.full(18, 2.0e-4),      # signed slip
    ]


def _json_history(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    result.pop("input_vector_dimensionless", None)
    result.pop("output_vector_dimensionless", None)
    result.pop("full_state_output_response_dimensionless", None)
    result.pop("propagator_dimensionless", None)
    return result


def _crossing_from_envelope(
    times: np.ndarray,
    gains: np.ndarray,
    threshold: float,
) -> float | None:
    target = float(np.log(threshold))
    logs = np.log(gains)
    for index in range(times.size - 1):
        if logs[index] < target <= logs[index + 1]:
            fraction = (target - logs[index]) / (logs[index + 1] - logs[index])
            return float(times[index] + fraction * (times[index + 1] - times[index]))
    return None


def run(*, direction_count: int, wavenumber_count: int, gain_threshold: float) -> dict[str, Any]:
    card = load_card()
    checkpoints = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1.40]
    card["representative_loading"]["spectral_checkpoint_shears"] = checkpoints
    records, states, model, parameters = run_representative_checkpoint_states(
        card, shear_increment=2.0e-3
    )
    storages = [
        local_state92_from_material_state(state, model, simple_shear(shear))
        for state, shear in zip(states, checkpoints, strict=True)
    ]
    micro = MicromorphicParameters(
        reference_shear_modulus_Pa=parameters.reference_shear_modulus,
        nye_length_scale_m=1.0e-6,
        penalty_modulus_Pa=parameters.reference_shear_modulus,
        slip_gradient_length_m=0.25e-6,
        burgers_m=parameters.burgers,
    )
    spectral_model = ContinuousSpectralPointModel(
        model,
        micro,
        conductivity_W_mK=card["thermal"]["conductivity_W_mK_at_300K"],
        parameter_provenance=card["status"],
    )
    reference_direction = np.array([1.0, 2.0, 3.0])
    reference_direction /= np.linalg.norm(reference_direction)
    points = []
    for storage, shear in zip(storages, checkpoints, strict=True):
        points.append(
            spectral_model.export(
                simple_shear(shear),
                storage.temperature_K,
                storage.gamma_signed,
                np.zeros((18, 3)),
                storage,
                direction_n=reference_direction,
                compute_jacobian=True,
            )
        )

    times = np.asarray([row["time_s"] for row in records], dtype=float)
    scales = _coordinate_scales(storages)
    directions = _hemisphere_directions(direction_count)
    wavenumbers = np.geomspace(3.0e2, 3.0e5, wavenumber_count)
    scans: list[dict[str, Any]] = []
    envelope = np.ones(times.size)
    admission = _admission()
    def evaluate_scan(direction_set: list[dict[str, Any]], k_set: np.ndarray) -> None:
        nonlocal envelope
        for direction in direction_set:
            normal = np.asarray(direction["direction_n"], dtype=float)
            for wavenumber in k_set:
                operators = [
                    assemble_dynamic_crystal_operator_v1(
                        point,
                        wavenumber_m_inv=float(wavenumber),
                        direction_n=normal,
                        admission=admission,
                    )
                    for point in points
                ]
                raw = finite_time_amplification_history(
                    operators,
                    times,
                    coordinate_scales=scales,
                    gain_threshold=gain_threshold,
                    input_indices=np.arange(6, 69),
                    output_indices=np.arange(6, 69),
                )
                history = _json_history(raw)
                prefix_gain = np.asarray(
                    [row["maximum_gain"] for row in history["prefix"]], dtype=float
                )
                envelope = np.maximum(envelope, prefix_gain)
                scans.append({**direction, **history})

    evaluate_scan(directions, wavenumbers)
    coarse_winner = max(scans, key=lambda row: row["final_log_gain"])
    coarse_index = int(
        np.argmin(np.abs(np.log(wavenumbers / coarse_winner["wavenumber_m_inv"])))
    )
    lower_index = max(0, coarse_index - 1)
    upper_index = min(wavenumbers.size - 1, coarse_index + 1)
    refined_wavenumbers = np.geomspace(
        wavenumbers[lower_index], wavenumbers[upper_index], 13
    )
    refined_directions = _refined_directions(
        np.asarray(coarse_winner["direction_n"], dtype=float)
    )
    evaluate_scan(refined_directions, refined_wavenumbers)

    final_winner = max(scans, key=lambda row: row["final_log_gain"])
    crossed = [row for row in scans if row["critical_time_s"] is not None]
    onset_winner = (
        min(crossed, key=lambda row: (row["critical_time_s"], -row["final_log_gain"]))
        if crossed
        else final_winner
    )
    tc = _crossing_from_envelope(times, envelope, gain_threshold)

    selection_trajectory = []
    for time_index, time in enumerate(times):
        winner = max(
            scans,
            key=lambda row: row["prefix"][time_index]["maximum_gain"],
        )
        selection_trajectory.append(
            {
                "time_s": float(time),
                "maximum_gain": float(winner["prefix"][time_index]["maximum_gain"]),
                "direction_id": winner["direction_id"],
                "scan_level": winner["scan_level"],
                "direction_n": winner["direction_n"],
                "k_star_m_inv": float(winner["wavenumber_m_inv"]),
                "wavelength_2pi_over_k_m": float(2.0 * pi / winner["wavenumber_m_inv"]),
            }
        )
    threshold_indices = np.flatnonzero(envelope >= gain_threshold)
    onset_checkpoint_index = int(threshold_indices[0]) if threshold_indices.size else times.size - 1
    onset_checkpoint_pair = selection_trajectory[onset_checkpoint_index]
    onset_checkpoint_normal = np.asarray(onset_checkpoint_pair["direction_n"], dtype=float)
    onset_checkpoint_k = float(onset_checkpoint_pair["k_star_m_inv"])
    onset_checkpoint_operators = [
        assemble_dynamic_crystal_operator_v1(
            point,
            wavenumber_m_inv=onset_checkpoint_k,
            direction_n=onset_checkpoint_normal,
            admission=admission,
        )
        for point in points[: onset_checkpoint_index + 1]
    ]
    onset_checkpoint_analysis = _json_history(
        finite_time_amplification_history(
            onset_checkpoint_operators,
            times[: onset_checkpoint_index + 1],
            coordinate_scales=scales,
            gain_threshold=gain_threshold,
            input_indices=np.arange(6, 69),
            output_indices=np.arange(6, 69),
        )
    )

    # Frozen eigenvalue information is retained as a separate diagnostic; it
    # is not substituted for the non-autonomous singular-value onset.
    selected_normal = np.asarray(onset_winner["direction_n"], dtype=float)
    selected_k = float(onset_winner["wavenumber_m_inv"])
    frozen = []
    for point, time in zip(points, times, strict=True):
        operator = assemble_dynamic_crystal_operator_v1(
            point,
            wavenumber_m_inv=selected_k,
            direction_n=selected_normal,
            admission=admission,
        )
        roots, _, residuals = operator.admitted_eigenpairs()
        dominant = int(np.argmax(roots.real))
        frozen.append(
            {
                "time_s": float(time),
                "spectral_abscissa_s_inv": float(roots[dominant].real),
                "frequency_s_inv": float(abs(roots[dominant].imag)),
                "relative_backward_error": float(residuals[dominant]),
                "algebraic_condition_number": float(operator.algebraic_condition_number),
            }
        )

    rows = []
    for item in scans:
        rows.append(
            {
                "direction_id": item["direction_id"],
                "scan_level": item["scan_level"],
                "n1": item["direction_n"][0],
                "n2": item["direction_n"][1],
                "n3": item["direction_n"][2],
                "k_m_inv": item["wavenumber_m_inv"],
                "wavelength_m": 2.0 * pi / item["wavenumber_m_inv"],
                "critical_time_s": item["critical_time_s"],
                "final_gain": item["final_gain"],
                "final_log_gain": item["final_log_gain"],
                "dominant_output_mechanism": item[
                    "full_state_output_mechanism_participation"
                ]["dominant_coarse"],
            }
        )
    SCAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SCAN_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_DISCRETE_DIRECTION_WAVENUMBER_SCAN__RESEARCH_BASELINE",
        "claim_boundary": (
            "Single-crystal, literature-constrained CP-Ti baseline. The reported n* and k* "
            "are discrete-scan candidates, not batch-specific experimental predictions."
        ),
        "base_history": records,
        "operator_contract": {
            "full_quadratic_state_dimension": 84,
            "first_order_descriptor_state_dimension": 87,
            "algebraic_micromorphic_slip_dimension": 18,
            "differential_generator_dimension": 69,
            "pencil": "K(k,n,t)+sE(t)+s^2M",
            "finite_time_integrator": "piecewise_linear_generator_exponential_midpoint",
            "fictitious_internal_inertia": False,
        },
        "scan_contract": {
            "antipodal_domain": "one hemisphere; real-coefficient Fourier conjugacy identifies n and -n",
            "direction_count": len(directions),
            "refined_direction_count": len(refined_directions),
            "wavenumber_count": int(wavenumbers.size),
            "refined_wavenumber_count": int(refined_wavenumbers.size),
            "wavenumber_bounds_m_inv": [float(wavenumbers[0]), float(wavenumbers[-1])],
            "refined_wavenumber_bounds_m_inv": [float(refined_wavenumbers[0]), float(refined_wavenumbers[-1])],
            "checkpoint_times_s": times.tolist(),
            "gain_norm": "fixed dimensionless Euclidean norm q=Dz",
            "gain_observable": "thermal plus 62 constitutive coordinates; displacement and velocity are propagated internally but excluded from input/output norms",
            "coordinate_scales": scales.tolist(),
            "gain_threshold": float(gain_threshold),
            "critical_time_definition": "first log-linear crossing of max_(k,n) sigma_max(Phi) >= threshold",
            "nye_length_m": 1.0e-6,
            "slip_gradient_length_m": 0.25e-6,
            "extra_local_chart_diffusivity_m2_s": 0.0,
        },
        "finite_time_selection": {
            "critical_time_s": tc,
            "onset_pair_full_horizon_record": onset_winner,
            "onset_checkpoint_pair": onset_checkpoint_pair,
            "onset_checkpoint_analysis": onset_checkpoint_analysis,
            "final_horizon_pair": final_winner,
            "selection_trajectory": selection_trajectory,
            "gain_envelope": [
                {"time_s": float(time), "maximum_gain": float(gain)}
                for time, gain in zip(times, envelope, strict=True)
            ],
            "linear_wavelength_2pi_over_k_star_m": 2.0 * pi / selected_k,
        },
        "selected_pair_frozen_spectrum": frozen,
        "scan": scans,
    }
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    mechanism = onset_checkpoint_analysis["full_state_output_mechanism_participation"]
    tc_text = "not reached" if tc is None else f"{tc * 1e6:.6g} us"
    SUMMARY.write_text(
        "# CP-Ti finite-time dynamic perturbation V2\n\n"
        f"- Critical time definition: first {gain_threshold:.6g} gain crossing of the global finite-time envelope.\n"
        f"- Candidate tc: {tc_text}.\n"
        f"- Onset candidate n*: {onset_checkpoint_pair['direction_n']}.\n"
        f"- Onset candidate k*: {onset_checkpoint_k:.9g} 1/m; 2pi/k*={2*pi/onset_checkpoint_k*1e6:.6g} um.\n"
        f"- Onset-checkpoint dominant full-state response mechanism: {mechanism['dominant_coarse']}; weights={json.dumps(mechanism['coarse'], sort_keys=True)}.\n"
        f"- Final-horizon candidate: n={final_winner['direction_n']}, k={final_winner['wavenumber_m_inv']:.9g} 1/m, 2pi/k={2*pi/final_winner['wavenumber_m_inv']*1e6:.6g} um.\n"
        "- Claim boundary: discrete single-crystal research baseline; direction and wavenumber need refinement and batch-specific validation.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", type=int, default=13)
    parser.add_argument("--wavenumbers", type=int, default=17)
    parser.add_argument("--gain-threshold", type=float, default=float(np.e))
    args = parser.parse_args()
    result = run(
        direction_count=args.directions,
        wavenumber_count=args.wavenumbers,
        gain_threshold=args.gain_threshold,
    )
    selected = result["finite_time_selection"]
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "scan_csv": str(SCAN_CSV),
                "summary": str(SUMMARY),
                "critical_time_s": selected["critical_time_s"],
                "n_star": selected["onset_checkpoint_pair"]["direction_n"],
                "k_star_m_inv": selected["onset_checkpoint_pair"]["k_star_m_inv"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
