"""Continuous-spectrum and robustness audit for the CP-Ti dynamic crystal operator.

This audit does not identify experimental parameters.  It tests the mathematical
research baseline against continuous (n,k) optimization, time-history refinement,
dimensionless-coordinate scaling, long/short-wave extension, principal-symbol
admissibility, algebraic-block conditioning, and modal backward errors.
"""

from __future__ import annotations

import csv
import json
from math import pi
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.run_cp_ti_finite_time_propagator_v2 import (  # noqa: E402
    _admission,
    _coordinate_scales,
    _crossing_from_envelope,
)
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
from hcp_cp_gnd.qs_descriptor import direction_maps  # noqa: E402
from hcp_cp_gnd.spectral_export import ContinuousSpectralPointModel  # noqa: E402


BASELINE = ROOT / "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json"
RESULT = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v1.json"
SUMMARY = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v1.md"
FIGURE = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v1.png"
TABLE_DIR = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v1"

BASE_SHEARS = np.asarray([0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1.40])
K_BOUNDS = (3.0e2, 3.0e5)
EXTENDED_K = np.geomspace(3.0, 3.0e7, 22)
GAIN_THRESHOLD = float(np.e)
OBSERVED = np.arange(6, 69)
BOUNDS = ((0.0, 0.5 * pi), (-pi, pi), (float(np.log(K_BOUNDS[0])), float(np.log(K_BOUNDS[1]))))

# Numerical acceptance gates.  These regulate approximation error; they are not material data.
ANGLE_GATE_DEG = 2.5
K_RELATIVE_GATE = 0.10
GAIN_RELATIVE_GATE = 0.02
TIME_REFINEMENT_TC_GATE = 0.02
TIME_REFINEMENT_K_GATE = 0.05
SCALE_ANGLE_GATE_DEG = 5.0
SCALE_K_RELATIVE_GATE = 0.20
SCALE_TC_RELATIVE_GATE = 0.10
GENERATOR_RESIDUAL_GATE = 2.0e-10
ALGEBRAIC_CONDITION_GATE = 1.0e14


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def refined_shears(factor: int) -> np.ndarray:
    require(factor >= 1, "refinement factor must be positive")
    values = [float(BASE_SHEARS[0])]
    for left, right in zip(BASE_SHEARS[:-1], BASE_SHEARS[1:], strict=True):
        values.extend(float(left + (right - left) * j / factor) for j in range(1, factor + 1))
    return np.asarray(values)


def direction_from_coordinates(x: Any) -> np.ndarray:
    theta, phi = float(x[0]), float(x[1])
    value = np.asarray([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return value / np.linalg.norm(value)


def coordinates_from_direction(direction: Any, k_m_inv: float) -> np.ndarray:
    value = np.asarray(direction, dtype=float)
    value /= np.linalg.norm(value)
    if value[2] < -1.0e-14:
        value = -value
    theta = float(np.arccos(np.clip(value[2], -1.0, 1.0)))
    phi = float(np.arctan2(value[1], value[0]))
    return np.asarray([theta, phi, np.log(float(k_m_inv))])


def angle_degrees(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=float); a /= np.linalg.norm(a)
    b = np.asarray(second, dtype=float); b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def relative_change(first: float, second: float) -> float:
    return float(abs(first / second - 1.0))


class SpectrumAudit:
    def __init__(self, maximum_refinement: int = 4, context_factors: tuple[int, ...] = (1, 2, 4)) -> None:
        require(maximum_refinement >= 1, "maximum refinement must be positive")
        require(all(maximum_refinement % value == 0 for value in context_factors),
                "context factors must divide the maximum refinement")
        card = load_card()
        fine_shears = refined_shears(maximum_refinement)
        card["representative_loading"]["spectral_checkpoint_shears"] = fine_shears.tolist()
        records, states, model, parameters = run_representative_checkpoint_states(
            card, shear_increment=2.0e-3,
        )
        storages = [
            local_state92_from_material_state(state, model, simple_shear(shear))
            for state, shear in zip(states, fine_shears, strict=True)
        ]
        micro = MicromorphicParameters(
            reference_shear_modulus_Pa=parameters.reference_shear_modulus,
            nye_length_scale_m=1.0e-6,
            penalty_modulus_Pa=parameters.reference_shear_modulus,
            slip_gradient_length_m=0.25e-6,
            burgers_m=parameters.burgers,
        )
        spectral = ContinuousSpectralPointModel(
            model,
            micro,
            conductivity_W_mK=card["thermal"]["conductivity_W_mK_at_300K"],
            parameter_provenance=card["status"],
        )
        reference = np.asarray([1.0, 2.0, 3.0]); reference /= np.linalg.norm(reference)
        points = [
            spectral.export(
                simple_shear(shear), storage.temperature_K, storage.gamma_signed,
                np.zeros((18, 3)), storage, direction_n=reference, compute_jacobian=True,
            )
            for storage, shear in zip(storages, fine_shears, strict=True)
        ]
        times = np.asarray([row["time_s"] for row in records], dtype=float)
        self.contexts = {}
        for factor in context_factors:
            target_shears = refined_shears(factor)
            indices = [
                int(np.flatnonzero(np.isclose(fine_shears, value, rtol=0.0, atol=1.0e-14))[0])
                for value in target_shears
            ]
            self.contexts[f"factor{factor}"] = {
                "shears": fine_shears[indices],
                "times": times[indices],
                "points": [points[index] for index in indices],
                "storages": [storages[index] for index in indices],
            }
        self.scales = _coordinate_scales(self.contexts["factor1"]["storages"])
        self.admission = _admission()
        self.cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def history(self, context_name: str, prefix_index: int, x: Any,
                scales: np.ndarray | None = None,
                integration_substeps_per_interval: int = 1) -> dict[str, Any]:
        raw_x = np.asarray(x, dtype=float)
        require(raw_x.shape == (3,), "optimizer coordinate must be length three")
        context = self.contexts[context_name]
        require(1 <= prefix_index < len(context["times"]), "invalid propagation prefix")
        active_scales = self.scales if scales is None else np.asarray(scales, dtype=float)
        scale_key = tuple(np.round(np.log(active_scales / self.scales), 8))
        key = (context_name, prefix_index, int(integration_substeps_per_interval),
               *np.round(raw_x, 10), scale_key)
        if key in self.cache:
            return self.cache[key]
        direction = direction_from_coordinates(raw_x)
        k_m_inv = float(np.exp(raw_x[2]))
        operators = [
            assemble_dynamic_crystal_operator_v1(
                point, wavenumber_m_inv=k_m_inv, direction_n=direction,
                admission=self.admission,
            )
            for point in context["points"][:prefix_index + 1]
        ]
        raw = finite_time_amplification_history(
            operators, context["times"][:prefix_index + 1],
            coordinate_scales=active_scales, gain_threshold=GAIN_THRESHOLD,
            input_indices=OBSERVED, output_indices=OBSERVED,
            integration_substeps_per_interval=integration_substeps_per_interval,
        )
        value = {
            "gain": float(raw["final_gain"]),
            "log_gain": float(raw["final_log_gain"]),
            "critical_time_s": raw["critical_time_s"],
            "direction_n": direction.tolist(),
            "k_m_inv": k_m_inv,
            "mechanism": raw["full_state_output_mechanism_participation"],
            "observed_subspace_mechanism": raw["output_mechanism_participation"],
            "prefix": raw["prefix"],
            "integration_substeps_per_interval": int(integration_substeps_per_interval),
        }
        self.cache[key] = value
        return value

    def objective(self, context_name: str, prefix_index: int, scales: np.ndarray | None = None,
                  integration_substeps_per_interval: int = 1):
        return lambda x: -self.history(
            context_name, prefix_index, x, scales, integration_substeps_per_interval,
        )["log_gain"]

    def global_optimum(self, context_name: str, prefix_index: int, seed: int,
                       *, maxiter: int = 11, popsize: int = 8,
                       integration_substeps_per_interval: int = 1) -> dict[str, Any]:
        result = differential_evolution(
            self.objective(
                context_name, prefix_index, None, integration_substeps_per_interval,
            ), BOUNDS,
            seed=seed, maxiter=maxiter, popsize=popsize, tol=2.0e-5,
            polish=True, updating="immediate", workers=1,
        )
        history = self.history(
            context_name, prefix_index, result.x, None, integration_substeps_per_interval,
        )
        return {
            **history,
            "optimizer_coordinates": result.x.tolist(),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_nfev": int(result.nfev),
            "objective": "maximum finite-time log gain",
        }

    def local_optimum(self, context_name: str, prefix_index: int, starts: list[np.ndarray],
                      scales: np.ndarray | None = None,
                      integration_substeps_per_interval: int = 1) -> dict[str, Any]:
        candidates = [
            {
                **self.history(
                    context_name, prefix_index, start, scales,
                    integration_substeps_per_interval,
                ),
                "optimizer_coordinates": np.asarray(start, dtype=float).tolist(),
                "optimizer_success": True,
                "optimizer_message": "retained input start",
                "optimizer_nfev": 1,
                "objective": "retained local-search start",
            }
            for start in starts
        ]
        for start in starts:
            result = minimize(
                self.objective(
                    context_name, prefix_index, scales, integration_substeps_per_interval,
                ), start,
                method="Powell", bounds=BOUNDS,
                options={"xtol": 2.0e-5, "ftol": 2.0e-7, "maxiter": 80},
            )
            history = self.history(
                context_name, prefix_index, result.x, scales,
                integration_substeps_per_interval,
            )
            candidates.append({
                **history,
                "optimizer_coordinates": result.x.tolist(),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_nfev": int(result.nfev),
                "objective": "local maximum finite-time log gain",
            })
        return max(candidates, key=lambda item: item["log_gain"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    audit = SpectrumAudit()
    coarse = audit.contexts["factor1"]
    times = np.asarray(coarse["times"])
    lower_index = int(np.flatnonzero(times < baseline["finite_time_selection"]["critical_time_s"])[-1])
    upper_index = lower_index + 1
    final_index = len(times) - 1

    opt_lower = audit.global_optimum("factor1", lower_index, 20260821)
    opt_upper = audit.global_optimum("factor1", upper_index, 20260822)
    opt_final = audit.global_optimum("factor1", final_index, 20260823)
    repeat_upper = audit.global_optimum("factor1", upper_index, 20260922, maxiter=8, popsize=7)
    repeat_final = audit.global_optimum("factor1", final_index, 20260923, maxiter=8, popsize=7)

    continuous_envelope = np.asarray([1.0, opt_lower["gain"], opt_upper["gain"]])
    continuous_tc = _crossing_from_envelope(
        times[[0, lower_index, upper_index]], continuous_envelope, GAIN_THRESHOLD,
    )
    require(continuous_tc is not None, "continuous envelope did not cross the gain threshold")
    discrete_onset = baseline["finite_time_selection"]["onset_checkpoint_pair"]
    discrete_final = baseline["finite_time_selection"]["final_horizon_pair"]
    discrete_onset_gain = baseline["finite_time_selection"]["onset_checkpoint_analysis"]["final_gain"]
    continuous_vs_discrete = {
        "onset_checkpoint": {
            "angle_change_deg": angle_degrees(opt_upper["direction_n"], discrete_onset["direction_n"]),
            "relative_k_change": relative_change(opt_upper["k_m_inv"], discrete_onset["k_star_m_inv"]),
            "relative_gain_change": relative_change(opt_upper["gain"], discrete_onset_gain),
        },
        "final_horizon": {
            "angle_change_deg": angle_degrees(opt_final["direction_n"], discrete_final["direction_n"]),
            "relative_k_change": relative_change(opt_final["k_m_inv"], discrete_final["wavenumber_m_inv"]),
            "relative_gain_change": relative_change(opt_final["gain"], discrete_final["final_gain"]),
        },
        "critical_time_relative_change": relative_change(continuous_tc, baseline["finite_time_selection"]["critical_time_s"]),
    }
    repeatability = {
        "onset_checkpoint": {
            "angle_deg": angle_degrees(opt_upper["direction_n"], repeat_upper["direction_n"]),
            "relative_k": relative_change(opt_upper["k_m_inv"], repeat_upper["k_m_inv"]),
            "relative_gain": relative_change(opt_upper["gain"], repeat_upper["gain"]),
        },
        "final_horizon": {
            "angle_deg": angle_degrees(opt_final["direction_n"], repeat_final["direction_n"]),
            "relative_k": relative_change(opt_final["k_m_inv"], repeat_final["k_m_inv"]),
            "relative_gain": relative_change(opt_final["gain"], repeat_final["gain"]),
        },
    }

    # Profiles establish that the maximum is isolated and not selected by a k boundary.
    profile_rows = []
    for label, prefix_index, optimum in (
        ("onset_checkpoint", upper_index, opt_upper), ("final_horizon", final_index, opt_final),
    ):
        center = np.asarray(optimum["optimizer_coordinates"])
        for offset in np.linspace(-1.5, 1.5, 31):
            x = center.copy(); x[2] = np.clip(center[2] + offset, BOUNDS[2][0], BOUNDS[2][1])
            value = audit.history("factor1", prefix_index, x)
            profile_rows.append({
                "selector": label, "log_k_offset": float(offset), "k_m_inv": value["k_m_inv"],
                "gain": value["gain"], "log_gain": value["log_gain"],
            })
    direction_rows = []
    center_n = np.asarray(opt_final["direction_n"])
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(center_n @ reference)) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0])
    tangent1 = np.cross(center_n, reference); tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(center_n, tangent1)
    for angle_deg in (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0):
        count = 1 if angle_deg == 0.0 else 12
        for azimuth_index in range(count):
            azimuth = 2.0 * pi * azimuth_index / count
            angle = np.deg2rad(angle_deg)
            normal = np.cos(angle) * center_n + np.sin(angle) * (
                np.cos(azimuth) * tangent1 + np.sin(azimuth) * tangent2
            )
            x = coordinates_from_direction(normal, opt_final["k_m_inv"])
            value = audit.history("factor1", final_index, x)
            direction_rows.append({
                "angle_deg": angle_deg, "azimuth_index": azimuth_index,
                "n1": value["direction_n"][0], "n2": value["direction_n"][1],
                "n3": value["direction_n"][2], "gain": value["gain"], "log_gain": value["log_gain"],
            })

    # Time-checkpoint convergence: optimize every early envelope checkpoint and final horizon.
    time_results = []
    start_lower = np.asarray(opt_lower["optimizer_coordinates"])
    start_upper = np.asarray(opt_upper["optimizer_coordinates"])
    start_final = np.asarray(opt_final["optimizer_coordinates"])
    for context_name in ("factor1", "factor2", "factor4"):
        context = audit.contexts[context_name]
        ctimes = np.asarray(context["times"])
        early_indices = [index for index, time in enumerate(ctimes) if index >= 1 and time <= times[upper_index] + 1.0e-18]
        envelope = np.ones(len(early_indices) + 1)
        trajectory = []
        for output_index, prefix_index in enumerate(early_indices, start=1):
            optimum = audit.local_optimum(
                context_name, prefix_index, [start_lower, start_upper],
            )
            envelope[output_index] = optimum["gain"]
            trajectory.append({
                "time_s": float(ctimes[prefix_index]), "gain": optimum["gain"],
                "direction_n": optimum["direction_n"], "k_m_inv": optimum["k_m_inv"],
                "optimizer_coordinates": optimum["optimizer_coordinates"],
            })
        envelope_times = np.r_[ctimes[0], [row["time_s"] for row in trajectory]]
        tc = _crossing_from_envelope(envelope_times, envelope, GAIN_THRESHOLD)
        require(tc is not None, f"{context_name} envelope did not cross")
        onset_upper = next(row for row in trajectory if row["time_s"] >= tc)
        final = audit.local_optimum(context_name, len(ctimes) - 1, [start_final])
        time_results.append({
            "context": context_name,
            "checkpoint_count": len(ctimes),
            "critical_time_s": tc,
            "onset_upper_checkpoint": onset_upper,
            "final_horizon": final,
            "early_trajectory": trajectory,
        })
    time_pair_changes = []
    for first, second in zip(time_results[:-1], time_results[1:], strict=True):
        time_pair_changes.append({
            "pair": f"{first['context']}->{second['context']}",
            "critical_time_relative_change": relative_change(first["critical_time_s"], second["critical_time_s"]),
            "onset_k_relative_change": relative_change(first["onset_upper_checkpoint"]["k_m_inv"], second["onset_upper_checkpoint"]["k_m_inv"]),
            "onset_angle_change_deg": angle_degrees(first["onset_upper_checkpoint"]["direction_n"], second["onset_upper_checkpoint"]["direction_n"]),
            "final_k_relative_change": relative_change(first["final_horizon"]["k_m_inv"], second["final_horizon"]["k_m_inv"]),
            "final_angle_change_deg": angle_degrees(first["final_horizon"]["direction_n"], second["final_horizon"]["direction_n"]),
            "final_log_gain_relative_change": relative_change(first["final_horizon"]["log_gain"], second["final_horizon"]["log_gain"]),
        })

    # Dimensionless-coordinate norm sensitivity over explicit factor-two group changes.
    groups = {
        "temperature": np.arange(6, 7),
        "plastic_chart": np.arange(7, 15),
        "dislocation": np.arange(15, 51),
        "signed_slip": np.arange(51, 69),
    }
    scale_scenarios = [("baseline", None, 1.0)]
    for group in groups:
        scale_scenarios.extend([(f"{group}_x0p5", group, 0.5), (f"{group}_x2", group, 2.0)])
    scale_results = []
    for identifier, group, factor in scale_scenarios:
        scales = audit.scales.copy()
        if group is not None:
            scales[groups[group]] *= factor
        lower = audit.local_optimum("factor1", lower_index, [start_lower, start_upper], scales)
        upper = audit.local_optimum("factor1", upper_index, [start_lower, start_upper], scales)
        final = audit.local_optimum("factor1", final_index, [start_final, start_upper], scales)
        envelope = np.asarray([1.0, lower["gain"], upper["gain"]])
        tc = _crossing_from_envelope(times[[0, lower_index, upper_index]], envelope, GAIN_THRESHOLD)
        scale_results.append({
            "scenario": identifier, "group": group, "factor": factor,
            "critical_time_s": tc, "onset_checkpoint": upper, "final_horizon": final,
        })
    baseline_scale = scale_results[0]
    scale_changes = []
    for value in scale_results[1:]:
        scale_changes.append({
            "scenario": value["scenario"],
            "critical_time_relative_change": relative_change(value["critical_time_s"], baseline_scale["critical_time_s"]),
            "onset_angle_change_deg": angle_degrees(value["onset_checkpoint"]["direction_n"], baseline_scale["onset_checkpoint"]["direction_n"]),
            "onset_k_relative_change": relative_change(value["onset_checkpoint"]["k_m_inv"], baseline_scale["onset_checkpoint"]["k_m_inv"]),
            "final_angle_change_deg": angle_degrees(value["final_horizon"]["direction_n"], baseline_scale["final_horizon"]["direction_n"]),
            "final_k_relative_change": relative_change(value["final_horizon"]["k_m_inv"], baseline_scale["final_horizon"]["k_m_inv"]),
            "onset_dominant_mechanism": value["onset_checkpoint"]["mechanism"]["dominant_coarse"],
            "final_dominant_mechanism": value["final_horizon"]["mechanism"]["dominant_coarse"],
        })

    # Extended-k behavior, block conditioning, and generator backward errors.
    asymptotic_rows = []
    max_generator_residual = 0.0
    max_algebraic_condition = 0.0
    context = audit.contexts["factor1"]
    for k_m_inv in EXTENDED_K:
        x = coordinates_from_direction(opt_final["direction_n"], float(k_m_inv))
        failure = None
        try:
            value = audit.history("factor1", final_index, x)
            operator = assemble_dynamic_crystal_operator_v1(
                context["points"][-1], wavenumber_m_inv=float(k_m_inv),
                direction_n=opt_final["direction_n"], admission=audit.admission,
            )
            roots, _, residuals = operator.admitted_eigenpairs(GENERATOR_RESIDUAL_GATE)
            spectral_abscissa = float(np.max(roots.real))
            residual = float(np.max(residuals))
            condition = float(operator.algebraic_condition_number)
            max_generator_residual = max(max_generator_residual, residual)
            max_algebraic_condition = max(max_algebraic_condition, condition)
            asymptotic_rows.append({
                "k_m_inv": float(k_m_inv), "wavelength_m": float(2.0*pi/k_m_inv),
                "finite_time_gain": value["gain"], "finite_time_log_gain": value["log_gain"],
                "final_spectral_abscissa_s_inv": spectral_abscissa,
                "maximum_admitted_generator_residual": residual,
                "algebraic_condition_number": condition, "failure": failure,
            })
        except Exception as exc:  # preserved as an audit result
            asymptotic_rows.append({
                "k_m_inv": float(k_m_inv), "wavelength_m": float(2.0*pi/k_m_inv),
                "finite_time_gain": None, "finite_time_log_gain": None,
                "final_spectral_abscissa_s_inv": None,
                "maximum_admitted_generator_residual": None,
                "algebraic_condition_number": None, "failure": f"{type(exc).__name__}: {exc}",
            })

    # Principal-symbol audit on a deterministic hemisphere grid and all fine base states.
    principal_min = {"acoustic_symmetric_Pa": float("inf"), "conductivity_W_mK": float("inf"),
                     "slip_gradient_Pa_m2": float("inf")}
    principal_location: dict[str, Any] = {}
    golden = pi * (3.0 - np.sqrt(5.0))
    principal_directions = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])]
    for index in range(61):
        z = (index + 0.5) / 61.0
        radius = np.sqrt(1.0 - z*z)
        principal_directions.append(np.asarray([radius*np.cos(golden*index), radius*np.sin(golden*index), z]))
    for point_index, point in enumerate(audit.contexts["factor4"]["points"]):
        derivative = point.derivatives
        assert derivative is not None
        for normal in principal_directions:
            B, N = direction_maps(normal, norm_tolerance=audit.admission.direction_norm_abs_tolerance)
            acoustic = N @ derivative.dP_dF @ B
            acoustic_min = float(np.min(np.linalg.eigvalsh(0.5*(acoustic + acoustic.T))))
            conductivity_min = float(normal @ point.conductivity_W_mK @ normal)
            H = np.einsum("aibj,i,j->ab", point.gradient_hessian_Pa_m2, normal, normal, optimize=True)
            gradient_min = float(np.min(np.linalg.eigvalsh(0.5*(H + H.T))))
            values = {
                "acoustic_symmetric_Pa": acoustic_min,
                "conductivity_W_mK": conductivity_min,
                "slip_gradient_Pa_m2": gradient_min,
            }
            for name, observed in values.items():
                if observed < principal_min[name]:
                    principal_min[name] = observed
                    principal_location[name] = {
                        "point_index": point_index,
                        "time_s": float(audit.contexts["factor4"]["times"][point_index]),
                        "direction_n": normal.tolist(),
                    }

    successful_asymptotic = [row for row in asymptotic_rows if row["failure"] is None]
    high_tail = successful_asymptotic[-4:]
    high_k_log_gain_slope = float(np.polyfit(
        np.log([row["k_m_inv"] for row in high_tail]),
        [row["finite_time_log_gain"] for row in high_tail], 1,
    )[0])
    high_k_abscissa_over_k = [row["final_spectral_abscissa_s_inv"] / row["k_m_inv"] for row in high_tail]
    high_k_bounded_or_damped = bool(
        high_k_log_gain_slope <= 0.0
        and max(row["final_spectral_abscissa_s_inv"] for row in high_tail) <= 0.0
    )

    gates = {
        "continuous_optimizer_repeatable": all(
            value <= limit for value, limit in (
                (repeatability["onset_checkpoint"]["angle_deg"], ANGLE_GATE_DEG),
                (repeatability["onset_checkpoint"]["relative_k"], K_RELATIVE_GATE),
                (repeatability["onset_checkpoint"]["relative_gain"], GAIN_RELATIVE_GATE),
                (repeatability["final_horizon"]["angle_deg"], ANGLE_GATE_DEG),
                (repeatability["final_horizon"]["relative_k"], K_RELATIVE_GATE),
                (repeatability["final_horizon"]["relative_gain"], GAIN_RELATIVE_GATE),
            )
        ),
        "discrete_scan_agrees_with_continuous_optimum": all(
            value <= limit for value, limit in (
                (continuous_vs_discrete["onset_checkpoint"]["angle_change_deg"], ANGLE_GATE_DEG),
                (continuous_vs_discrete["onset_checkpoint"]["relative_k_change"], K_RELATIVE_GATE),
                (continuous_vs_discrete["onset_checkpoint"]["relative_gain_change"], GAIN_RELATIVE_GATE),
                (continuous_vs_discrete["final_horizon"]["angle_change_deg"], ANGLE_GATE_DEG),
                (continuous_vs_discrete["final_horizon"]["relative_k_change"], K_RELATIVE_GATE),
                (continuous_vs_discrete["final_horizon"]["relative_gain_change"], GAIN_RELATIVE_GATE),
            )
        ),
        "time_checkpoint_factor2_to_factor4_converged": all(
            value <= limit for value, limit in (
                (time_pair_changes[-1]["critical_time_relative_change"], TIME_REFINEMENT_TC_GATE),
                (time_pair_changes[-1]["onset_k_relative_change"], TIME_REFINEMENT_K_GATE),
                (time_pair_changes[-1]["onset_angle_change_deg"], ANGLE_GATE_DEG),
                (time_pair_changes[-1]["final_k_relative_change"], TIME_REFINEMENT_K_GATE),
                (time_pair_changes[-1]["final_angle_change_deg"], ANGLE_GATE_DEG),
                (time_pair_changes[-1]["final_log_gain_relative_change"], GAIN_RELATIVE_GATE),
            )
        ),
        "factor_two_coordinate_scale_robust": all(
            value[name] <= limit
            for value in scale_changes
            for name, limit in (
                ("critical_time_relative_change", SCALE_TC_RELATIVE_GATE),
                ("onset_angle_change_deg", SCALE_ANGLE_GATE_DEG),
                ("onset_k_relative_change", SCALE_K_RELATIVE_GATE),
                ("final_angle_change_deg", SCALE_ANGLE_GATE_DEG),
                ("final_k_relative_change", SCALE_K_RELATIVE_GATE),
            )
        ),
        "all_scale_variants_keep_plastic_dominance": all(
            value["onset_dominant_mechanism"] == "plastic_kinematics"
            and value["final_dominant_mechanism"] == "plastic_kinematics"
            for value in scale_changes
        ),
        "algebraic_block_conditioned_on_extended_k": (
            len(successful_asymptotic) == len(asymptotic_rows)
            and max_algebraic_condition < ALGEBRAIC_CONDITION_GATE
        ),
        "generator_backward_error_on_extended_k": (
            len(successful_asymptotic) == len(asymptotic_rows)
            and max_generator_residual <= GENERATOR_RESIDUAL_GATE
        ),
        "conductivity_principal_symbol_positive": principal_min["conductivity_W_mK"] > 0.0,
        "slip_gradient_principal_symbol_nonnegative": principal_min["slip_gradient_Pa_m2"] >= -audit.admission.psd_eigenvalue_abs_tolerance,
        "acoustic_principal_symbol_positive": principal_min["acoustic_symmetric_Pa"] > 0.0,
        "high_k_finite_time_and_spectral_tail_damped": high_k_bounded_or_damped,
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    report = {
        "schema": "CP_TI_CONTINUOUS_SPECTRUM_ROBUSTNESS_V1",
        "status": "ALL_DECLARED_GATES_PASS" if not failed_gates else "ROBUSTNESS_GAPS_IDENTIFIED",
        "baseline_source": BASELINE.relative_to(ROOT).as_posix(),
        "problem_contract": {
            "state_dimensions": {"quadratic_pencil": 84, "algebraic_slip": 18, "generator": 69},
            "continuous_domain": {"direction": "real projective plane represented by z>=0 hemisphere",
                                  "k_m_inv": list(K_BOUNDS)},
            "gain_threshold": GAIN_THRESHOLD,
            "observable": "thermal plus 62 constitutive coordinates; mechanics propagated internally",
            "coordinate_scaling": "q=Dz with fixed physical scales",
            "experimental_parameter_identification": False,
            "independent_u903_validation": False,
        },
        "continuous_optimization": {
            "lower_bracket": opt_lower,
            "upper_onset_checkpoint": opt_upper,
            "final_horizon": opt_final,
            "repeat_upper_onset_checkpoint": repeat_upper,
            "repeat_final_horizon": repeat_final,
            "critical_time_s": continuous_tc,
            "continuous_vs_discrete": continuous_vs_discrete,
            "repeatability": repeatability,
        },
        "time_checkpoint_refinement": {"results": time_results, "pair_changes": time_pair_changes},
        "coordinate_scale_sensitivity": {"scenarios": scale_results, "changes_from_baseline": scale_changes},
        "extended_wavenumber_audit": {
            "range_m_inv": [float(EXTENDED_K[0]), float(EXTENDED_K[-1])],
            "rows": asymptotic_rows,
            "maximum_algebraic_condition_number": max_algebraic_condition,
            "maximum_admitted_generator_residual": max_generator_residual,
            "high_k_log_gain_log_k_slope": high_k_log_gain_slope,
            "high_k_spectral_abscissa_over_k_s_inv_per_m_inv": high_k_abscissa_over_k,
            "high_k_bounded_or_damped": high_k_bounded_or_damped,
        },
        "principal_symbol_audit": {
            "direction_count": len(principal_directions),
            "base_state_count": len(audit.contexts["factor4"]["points"]),
            "minimum_eigenvalues": principal_min,
            "minimum_locations": principal_location,
        },
        "profiles": {"wavenumber": profile_rows, "direction": direction_rows},
        "acceptance_thresholds": {
            "angle_deg": ANGLE_GATE_DEG, "relative_k": K_RELATIVE_GATE,
            "relative_gain": GAIN_RELATIVE_GATE, "time_refinement_relative_tc": TIME_REFINEMENT_TC_GATE,
            "time_refinement_relative_k": TIME_REFINEMENT_K_GATE,
            "scale_angle_deg": SCALE_ANGLE_GATE_DEG, "scale_relative_k": SCALE_K_RELATIVE_GATE,
            "scale_relative_tc": SCALE_TC_RELATIVE_GATE,
            "generator_residual": GENERATOR_RESIDUAL_GATE,
            "algebraic_condition": ALGEBRAIC_CONDITION_GATE,
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "claim_boundary": {
            "continuous_global_proof": False,
            "numerical_global_search_plus_repeat_multistart": True,
            "well_posedness_claim_requires_all_principal_and_high_k_gates": True,
            "experimental_validation_claimed": False,
            "batch_specific_prediction": False,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(TABLE_DIR / "wavenumber_profiles.csv", profile_rows)
    write_csv(TABLE_DIR / "direction_profiles.csv", direction_rows)
    write_csv(TABLE_DIR / "extended_wavenumber.csv", asymptotic_rows)
    write_csv(TABLE_DIR / "scale_sensitivity.csv", scale_changes)
    write_csv(TABLE_DIR / "time_refinement.csv", time_pair_changes)

    lines = [
        "# CP-Ti continuous-spectrum robustness audit V1", "",
        f"Status: **{report['status']}**.", "",
        "This is a mathematical/numerical robustness audit of the single-crystal research baseline; it is not experimental parameter identification or U903 validation.", "",
        "## Continuous selectors", "",
        f"- Continuous tc: {continuous_tc*1e6:.9g} us.",
        f"- Onset checkpoint: n={opt_upper['direction_n']}, k={opt_upper['k_m_inv']:.9g} 1/m, gain={opt_upper['gain']:.9g}.",
        f"- Final horizon: n={opt_final['direction_n']}, k={opt_final['k_m_inv']:.9g} 1/m, gain={opt_final['gain']:.9g}.",
        "", "## Gate table", "", "| gate | passed |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {passed} |" for name, passed in gates.items())
    lines.extend([
        "", "## Principal and asymptotic findings", "",
        f"- Minimum symmetric acoustic eigenvalue: {principal_min['acoustic_symmetric_Pa']:.9g} Pa.",
        f"- Minimum directional conductivity: {principal_min['conductivity_W_mK']:.9g} W/(m K).",
        f"- Minimum directional slip-gradient eigenvalue: {principal_min['slip_gradient_Pa_m2']:.9g} Pa m2.",
        f"- High-k log-gain/log-k slope: {high_k_log_gain_slope:.9g}.",
        f"- Failed gates: {failed_gates}.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for label, axis in (("onset_checkpoint", axes[0, 0]), ("final_horizon", axes[0, 1])):
        subset = [row for row in profile_rows if row["selector"] == label]
        axis.semilogx([row["k_m_inv"] for row in subset], [row["gain"] for row in subset], "o-")
        axis.set_title(label.replace("_", " ")); axis.set_xlabel("k (m$^{-1}$)"); axis.set_ylabel("finite-time gain")
    grouped_angles = sorted({row["angle_deg"] for row in direction_rows})
    axes[0, 2].plot(grouped_angles, [max(row["gain"] for row in direction_rows if row["angle_deg"] == value) for value in grouped_angles], "o-")
    axes[0, 2].set_xlabel("angular distance from n* (deg)"); axes[0, 2].set_ylabel("maximum ring gain")
    axes[0, 2].set_title("direction isolation")
    for value in time_results:
        rows = value["early_trajectory"]
        axes[1, 0].semilogy([0.5e6*value["critical_time_s"]*0+0.5] + [1e6*row["time_s"] for row in rows], [1.0]+[row["gain"] for row in rows], "o-", label=value["context"])
    axes[1, 0].axhline(GAIN_THRESHOLD, color="k", ls="--", lw=1); axes[1, 0].set_xlabel("time (µs)"); axes[1, 0].set_ylabel("envelope gain"); axes[1, 0].legend(frameon=False)
    axes[1, 1].semilogx([row["k_m_inv"] for row in successful_asymptotic], [row["finite_time_log_gain"] for row in successful_asymptotic], "o-", label="log gain")
    axes[1, 1].set_xlabel("k (m$^{-1}$)"); axes[1, 1].set_ylabel("final log gain")
    axes[1, 2].semilogx([row["k_m_inv"] for row in successful_asymptotic], [row["final_spectral_abscissa_s_inv"] for row in successful_asymptotic], "o-")
    axes[1, 2].axhline(0.0, color="k", ls="--", lw=1); axes[1, 2].set_xlabel("k (m$^{-1}$)"); axes[1, 2].set_ylabel("final spectral abscissa (s$^{-1}$)")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=220); plt.close(fig)
    print(json.dumps({
        "result": str(RESULT), "summary": str(SUMMARY), "figure": str(FIGURE),
        "status": report["status"], "continuous_tc_s": continuous_tc,
        "onset_n": opt_upper["direction_n"], "onset_k_m_inv": opt_upper["k_m_inv"],
        "final_n": opt_final["direction_n"], "final_k_m_inv": opt_final["k_m_inv"],
        "failed_gates": failed_gates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
