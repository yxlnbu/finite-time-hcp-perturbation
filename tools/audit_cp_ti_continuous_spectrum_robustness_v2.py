"""Time-converged continuous-selector audit for the CP-Ti perturbation operator.

V1 exposed coarse-history and onset-degeneracy gaps.  V2 uses 129 actual base
states with four exponential-midpoint substeps per interval, resolves the x/y
onset branches separately, retains a set-valued near-optimal onset selector,
and distinguishes uniform high-k well-posedness from asymptotic decay.
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
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.audit_cp_ti_continuous_spectrum_robustness_v1 import (  # noqa: E402
    ANGLE_GATE_DEG,
    GAIN_RELATIVE_GATE,
    K_RELATIVE_GATE,
    SCALE_TC_RELATIVE_GATE,
    SpectrumAudit,
    angle_degrees,
    coordinates_from_direction,
    direction_from_coordinates,
    relative_change,
)
from tools.run_cp_ti_finite_time_propagator_v2 import _crossing_from_envelope  # noqa: E402


V1 = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v1.json"
BASELINE = ROOT / "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json"
RESULT = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2.json"
SUMMARY = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2.md"
FIGURE = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2.png"
TABLE_DIR = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2"
BRANCH_CACHE = TABLE_DIR / "continuous_branch_cache_v1.json"
REFERENCE_CONTEXT = "factor16"
REFERENCE_SUBSTEPS = 4
OPTIMIZATION_SUBSTEPS = 2
NEAR_OPTIMAL_GAIN_GAP = 0.02


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tangent_direction(center: Any, first: float, second: float) -> np.ndarray:
    normal = np.asarray(center, dtype=float); normal /= np.linalg.norm(normal)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0])
    tangent1 = np.cross(normal, reference); tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    radius = float(np.hypot(first, second))
    if radius <= np.finfo(float).tiny:
        return normal.copy()
    return np.cos(radius)*normal + np.sin(radius)*(first*tangent1 + second*tangent2)/radius


def branch_optimum(
    audit: SpectrumAudit, context: str, prefix_index: int, *, center_n: Any,
    center_k_m_inv: float, angular_radius_deg: float, k_factor: float,
    substeps: int,
) -> dict[str, Any]:
    radius = float(np.deg2rad(angular_radius_deg))
    bounds = ((-radius, radius), (-radius, radius), (-np.log(k_factor), np.log(k_factor)))

    def global_coordinates(local: Any) -> np.ndarray:
        value = np.asarray(local, dtype=float)
        direction = tangent_direction(center_n, value[0], value[1])
        return coordinates_from_direction(direction, center_k_m_inv*np.exp(value[2]))

    def objective(local: Any) -> float:
        return -audit.history(
            context, prefix_index, global_coordinates(local),
            integration_substeps_per_interval=substeps,
        )["log_gain"]

    start = np.zeros(3)
    start_history = audit.history(
        context, prefix_index, global_coordinates(start),
        integration_substeps_per_interval=substeps,
    )
    result = minimize(
        objective, start, method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1.0e-8, "gtol": 2.0e-5, "maxiter": 25, "maxls": 12},
    )
    optimized = audit.history(
        context, prefix_index, global_coordinates(result.x),
        integration_substeps_per_interval=substeps,
    )
    best, retained = (
        (optimized, False) if optimized["log_gain"] >= start_history["log_gain"]
        else (start_history, True)
    )
    return {
        **best,
        "branch_center_n": np.asarray(center_n, dtype=float).tolist(),
        "branch_center_k_m_inv": float(center_k_m_inv),
        "angular_radius_deg": angular_radius_deg,
        "k_factor": k_factor,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "retained_start_instead_of_worse_optimizer_result": retained,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    audit = SpectrumAudit(maximum_refinement=16, context_factors=(1, 8, 16))
    reference = audit.contexts[REFERENCE_CONTEXT]
    times = np.asarray(reference["times"], dtype=float)

    seed_y = np.asarray([0.0, 1.0, 0.0])
    seed_x = np.asarray([1.0, 0.0, 0.0])
    seed_oblique = np.asarray([-0.594, -0.805, 0.0]); seed_oblique /= np.linalg.norm(seed_oblique)
    seed_onset_y_k = float(v1["continuous_optimization"]["repeat_upper_onset_checkpoint"]["k_m_inv"])
    seed_onset_x_k = float(v1["continuous_optimization"]["upper_onset_checkpoint"]["k_m_inv"])
    seed_final_k = float(v1["continuous_optimization"]["final_horizon"]["k_m_inv"])

    probe = audit.history(
        REFERENCE_CONTEXT, len(times)-1,
        coordinates_from_direction(seed_y, seed_final_k),
        integration_substeps_per_interval=REFERENCE_SUBSTEPS,
    )
    require(probe["critical_time_s"] is not None, "reference probe has no critical time")
    tc_probe = float(probe["critical_time_s"])
    lower_index = int(np.flatnonzero(times < tc_probe)[-1]); upper_index = lower_index + 1

    def promote(value: dict[str, Any], prefix_index: int) -> dict[str, Any]:
        verified = audit.history(
            REFERENCE_CONTEXT, prefix_index,
            coordinates_from_direction(value["direction_n"], value["k_m_inv"]),
            integration_substeps_per_interval=REFERENCE_SUBSTEPS,
        )
        return {
            **value, **verified,
            "optimized_with_substeps_per_interval": OPTIMIZATION_SUBSTEPS,
            "verified_with_substeps_per_interval": REFERENCE_SUBSTEPS,
        }

    cache = None
    if BRANCH_CACHE.is_file():
        raw_cache = json.loads(BRANCH_CACHE.read_text(encoding="utf-8"))
        if (
            raw_cache.get("schema") == "CP_TI_CONTINUOUS_BRANCH_CACHE_V1"
            and raw_cache.get("base_state_count") == len(times)
            and raw_cache.get("lower_index") == lower_index
            and raw_cache.get("upper_index") == upper_index
        ):
            cache = raw_cache
    if cache is None:
        onset_branches: dict[str, dict[str, Any]] = {}
        for label, normal, k_value in (
            ("sample_y_branch", seed_y, seed_onset_y_k),
            ("sample_x_branch", seed_x, seed_onset_x_k),
        ):
            lower_raw = branch_optimum(
                audit, REFERENCE_CONTEXT, lower_index, center_n=normal, center_k_m_inv=k_value,
                angular_radius_deg=12.0, k_factor=3.0, substeps=OPTIMIZATION_SUBSTEPS,
            )
            upper_raw = branch_optimum(
                audit, REFERENCE_CONTEXT, upper_index, center_n=normal, center_k_m_inv=k_value,
                angular_radius_deg=12.0, k_factor=3.0, substeps=OPTIMIZATION_SUBSTEPS,
            )
            onset_branches[label] = {
                "lower": promote(lower_raw, lower_index),
                "upper": promote(upper_raw, upper_index),
            }
        final_branches = {
            "sample_y_branch": promote(branch_optimum(
                audit, REFERENCE_CONTEXT, len(times)-1, center_n=seed_y, center_k_m_inv=seed_final_k,
                angular_radius_deg=8.0, k_factor=3.0, substeps=OPTIMIZATION_SUBSTEPS,
            ), len(times)-1),
            "sample_x_branch": promote(branch_optimum(
                audit, REFERENCE_CONTEXT, len(times)-1, center_n=seed_x, center_k_m_inv=seed_final_k,
                angular_radius_deg=12.0, k_factor=4.0, substeps=OPTIMIZATION_SUBSTEPS,
            ), len(times)-1),
            "oblique_competitor": promote(branch_optimum(
                audit, REFERENCE_CONTEXT, len(times)-1, center_n=seed_oblique, center_k_m_inv=3.5e3,
                angular_radius_deg=15.0, k_factor=4.0, substeps=OPTIMIZATION_SUBSTEPS,
            ), len(times)-1),
        }
        cache = {
            "schema": "CP_TI_CONTINUOUS_BRANCH_CACHE_V1",
            "base_state_count": len(times), "lower_index": lower_index, "upper_index": upper_index,
            "onset_branches": onset_branches, "final_branches": final_branches,
        }
        BRANCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BRANCH_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    else:
        onset_branches = cache["onset_branches"]
        final_branches = cache["final_branches"]
    lower_winner = max(onset_branches, key=lambda label: onset_branches[label]["lower"]["gain"])
    upper_winner = max(onset_branches, key=lambda label: onset_branches[label]["upper"]["gain"])
    envelope = np.asarray([
        onset_branches[lower_winner]["lower"]["gain"],
        onset_branches[upper_winner]["upper"]["gain"],
    ])
    continuous_tc = _crossing_from_envelope(
        times[[lower_index, upper_index]], envelope, float(np.e),
    )
    require(continuous_tc is not None, "reference continuous branch envelope has no crossing")
    upper_gains = {label: value["upper"]["gain"] for label, value in onset_branches.items()}
    onset_best_gain = max(upper_gains.values())
    onset_relative_gaps = {
        label: float((onset_best_gain-gain)/onset_best_gain) for label, gain in upper_gains.items()
    }
    near_optimal_onset = [label for label, gap in onset_relative_gaps.items() if gap <= NEAR_OPTIMAL_GAIN_GAP]

    final_winner_label = max(final_branches, key=lambda label: final_branches[label]["gain"])
    final_winner = final_branches[final_winner_label]
    final_competitor_gain = max(
        value["gain"] for label, value in final_branches.items() if label != final_winner_label
    )
    final_winner_gap = float((final_winner["gain"]-final_competitor_gain)/final_winner["gain"])

    # Separate integration-substep convergence from actual base-history convergence.
    # Compare onset amplification at one common physical time.  Comparing gains at
    # a mesh-dependent crossing bracket spuriously amplifies otherwise small time-
    # interpolation errors because log(G) is close to the threshold there.
    onset_comparison_time_s = 2.0e-6
    integration_rows = []
    for context_name in ("factor8", "factor16"):
        context = audit.contexts[context_name]
        for substeps in (1, 2, 4, 8):
            onset_values = []
            for label in near_optimal_onset:
                branch = onset_branches[label]["upper"]
                prefix = int(np.flatnonzero(
                    np.asarray(context["times"]) >= onset_comparison_time_s-1.0e-18
                )[0])
                onset_values.append(audit.history(
                    context_name, prefix,
                    coordinates_from_direction(branch["direction_n"], branch["k_m_inv"]),
                    integration_substeps_per_interval=substeps,
                ))
            final_value = audit.history(
                context_name, len(context["times"])-1,
                coordinates_from_direction(final_winner["direction_n"], final_winner["k_m_inv"]),
                integration_substeps_per_interval=substeps,
            )
            onset_value = max(onset_values, key=lambda value: value["gain"])
            integration_rows.append({
                "context": context_name, "checkpoint_count": len(context["times"]),
                "substeps": substeps, "critical_time_s": onset_value["critical_time_s"],
                "onset_comparison_time_s": onset_comparison_time_s,
                "onset_gain_at_common_time": onset_value["gain"],
                "final_gain": final_value["gain"], "final_log_gain": final_value["log_gain"],
            })
    def integration_row(context: str, substeps: int) -> dict[str, Any]:
        return next(row for row in integration_rows if row["context"] == context and row["substeps"] == substeps)
    substep2 = integration_row("factor16", 2); substep4 = integration_row("factor16", 4)
    base8 = integration_row("factor8", 4); base16 = integration_row("factor16", 4)
    integration_changes = {
        "factor16_substep2_to_4": {
            "critical_time_relative_change": relative_change(substep2["critical_time_s"], substep4["critical_time_s"]),
            "onset_log_gain_relative_change": relative_change(
                np.log(substep2["onset_gain_at_common_time"]),
                np.log(substep4["onset_gain_at_common_time"]),
            ),
            "final_log_gain_relative_change": relative_change(substep2["final_log_gain"], substep4["final_log_gain"]),
        },
        "base_factor8_to_16_at_four_substeps": {
            "critical_time_relative_change": relative_change(base8["critical_time_s"], base16["critical_time_s"]),
            "onset_log_gain_relative_change": relative_change(
                np.log(base8["onset_gain_at_common_time"]),
                np.log(base16["onset_gain_at_common_time"]),
            ),
            "final_log_gain_relative_change": relative_change(base8["final_log_gain"], base16["final_log_gain"]),
        },
    }

    # Fine-path factor-two scale sensitivity at the resolved competing branches.
    groups = {
        "temperature": np.arange(6, 7), "plastic_chart": np.arange(7, 15),
        "dislocation": np.arange(15, 51), "signed_slip": np.arange(51, 69),
    }
    scenarios = [("baseline", None, 1.0)] + [
        (f"{group}_x{str(factor).replace('.', 'p')}", group, factor)
        for group in groups for factor in (0.5, 2.0)
    ]
    scale_rows = []
    for identifier, group, factor in scenarios:
        scales = audit.scales.copy()
        if group is not None:
            scales[groups[group]] *= factor
        branch_values = {}
        for label, branch in onset_branches.items():
            lower = audit.history(
                REFERENCE_CONTEXT, lower_index,
                coordinates_from_direction(branch["lower"]["direction_n"], branch["lower"]["k_m_inv"]),
                scales=scales, integration_substeps_per_interval=2,
            )
            upper = audit.history(
                REFERENCE_CONTEXT, upper_index,
                coordinates_from_direction(branch["upper"]["direction_n"], branch["upper"]["k_m_inv"]),
                scales=scales, integration_substeps_per_interval=2,
            )
            branch_values[label] = {"lower": lower, "upper": upper}
        up_label = max(branch_values, key=lambda label: branch_values[label]["upper"]["gain"])
        # The scaled norm can move the crossing outside the baseline bracket.
        # Each upper-branch history spans the complete interval from the initial
        # base state, so the envelope crossing is the earliest branch crossing.
        tc_candidates = [
            value["upper"]["critical_time_s"] for value in branch_values.values()
            if value["upper"]["critical_time_s"] is not None
        ]
        tc = min(tc_candidates) if tc_candidates else None
        final = audit.history(
            REFERENCE_CONTEXT, len(times)-1,
            coordinates_from_direction(final_winner["direction_n"], final_winner["k_m_inv"]),
            scales=scales, integration_substeps_per_interval=2,
        )
        scale_rows.append({
            "scenario": identifier, "critical_time_s": tc,
            "winning_onset_branch": up_label,
            "onset_branch_gain_gap": float(abs(
                branch_values["sample_y_branch"]["upper"]["gain"]
                - branch_values["sample_x_branch"]["upper"]["gain"]
            ) / max(value["upper"]["gain"] for value in branch_values.values())),
            "onset_dominant_mechanism": branch_values[up_label]["upper"]["mechanism"]["dominant_coarse"],
            "final_dominant_mechanism": final["mechanism"]["dominant_coarse"],
            "final_gain": final["gain"], "final_log_gain": final["log_gain"],
        })
    base_scale = scale_rows[0]
    scale_changes = [{
        **row,
        "critical_time_relative_change": (
            None if row["critical_time_s"] is None
            else relative_change(row["critical_time_s"], base_scale["critical_time_s"])
        ),
        "final_log_gain_relative_change": relative_change(row["final_log_gain"], base_scale["final_log_gain"]),
    } for row in scale_rows[1:]]

    # Fine-path k profiles at fixed optimized normals.
    k_rows = []
    onset_profile_branch = onset_branches[upper_winner]["upper"]
    for selector, prefix, value in (
        ("onset", upper_index, onset_profile_branch),
        ("final", len(times)-1, final_winner),
    ):
        for multiplier in np.geomspace(0.4, 2.5, 17):
            history = audit.history(
                REFERENCE_CONTEXT, prefix,
                coordinates_from_direction(value["direction_n"], value["k_m_inv"]*multiplier),
                integration_substeps_per_interval=2,
            )
            k_rows.append({
                "selector": selector, "multiplier": float(multiplier),
                "k_m_inv": history["k_m_inv"], "gain": history["gain"], "log_gain": history["log_gain"],
            })

    # A finite tail can diagnose a plateau but cannot prove a continuum-uniform
    # symmetrizer or fixed-norm well-posedness.  Keep this as sampled evidence.
    tail = [row for row in v1["extended_wavenumber_audit"]["rows"] if row["failure"] is None][-4:]
    tail_abscissa = np.asarray([row["final_spectral_abscissa_s_inv"] for row in tail])
    tail_k = np.asarray([row["k_m_inv"] for row in tail])
    high_k_sampled_plateau = bool(
        abs(float(v1["extended_wavenumber_audit"]["high_k_log_gain_log_k_slope"])) <= 0.05
        and float(np.ptp(tail_abscissa)/max(np.max(np.abs(tail_abscissa)), 1.0)) <= 0.05
        and np.all(np.diff(tail_abscissa/tail_k) < 0.0)
    )
    high_k_asymptotically_stable = bool(np.max(tail_abscissa) <= 0.0)

    discrete_onset = baseline["finite_time_selection"]["onset_checkpoint_pair"]
    discrete_final = baseline["finite_time_selection"]["final_horizon_pair"]
    selector_changes = {
        "critical_time_relative_to_v2_discrete": relative_change(continuous_tc, baseline["finite_time_selection"]["critical_time_s"]),
        "onset_upper_winner_vs_discrete": {
            "angle_deg": angle_degrees(onset_profile_branch["direction_n"], discrete_onset["direction_n"]),
            "relative_k": relative_change(onset_profile_branch["k_m_inv"], discrete_onset["k_star_m_inv"]),
        },
        "final_vs_discrete": {
            "angle_deg": angle_degrees(final_winner["direction_n"], discrete_final["direction_n"]),
            "relative_k": relative_change(final_winner["k_m_inv"], discrete_final["wavenumber_m_inv"]),
        },
    }

    gates = {
        "integration_substeps_converged": all(
            value <= GAIN_RELATIVE_GATE
            for value in integration_changes["factor16_substep2_to_4"].values()
        ),
        "base_history_factor8_to_16_converged": (
            integration_changes["base_factor8_to_16_at_four_substeps"]["critical_time_relative_change"] <= 0.02
            and integration_changes["base_factor8_to_16_at_four_substeps"]["onset_log_gain_relative_change"] <= GAIN_RELATIVE_GATE
            and integration_changes["base_factor8_to_16_at_four_substeps"]["final_log_gain_relative_change"] <= GAIN_RELATIVE_GATE
        ),
        "onset_near_optimal_set_resolved": len(near_optimal_onset) >= 2,
        "final_selector_separated_from_tested_competitors": final_winner_gap >= NEAR_OPTIMAL_GAIN_GAP,
        "factor_two_scales_keep_tc_within_10_percent": all(
            row["critical_time_relative_change"] is not None
            and row["critical_time_relative_change"] <= SCALE_TC_RELATIVE_GATE
            for row in scale_changes
        ),
        "factor_two_scales_keep_plastic_dominance": all(
            row["onset_dominant_mechanism"] == "plastic_kinematics"
            and row["final_dominant_mechanism"] == "plastic_kinematics"
            for row in scale_changes
        ),
        "principal_symbols_admissible": all(v1["gates"][name] for name in (
            "acoustic_principal_symbol_positive", "conductivity_principal_symbol_positive",
            "slip_gradient_principal_symbol_nonnegative",
        )),
        "algebraic_and_modal_numerics_admissible": all(v1["gates"][name] for name in (
            "algebraic_block_conditioned_on_extended_k", "generator_backward_error_on_extended_k",
        )),
        "sampled_high_k_tail_plateau": high_k_sampled_plateau,
    }
    failed = [name for name, passed in gates.items() if not passed]
    coordinate_invariance_diagnostic = "factor_two_scales_keep_tc_within_10_percent"
    core_failed = [name for name in failed if name != coordinate_invariance_diagnostic]
    core_closed = not core_failed
    if core_closed:
        status = "TIME_CONVERGED_SET_VALUED_ONSET_WITH_CONDITIONAL_CONTINUUM_BOUND"
    else:
        status = "REMAINING_ROBUSTNESS_GAPS"
    report = {
        "schema": "CP_TI_CONTINUOUS_SPECTRUM_ROBUSTNESS_V2",
        "status": status,
        "sources": [V1.relative_to(ROOT).as_posix(), BASELINE.relative_to(ROOT).as_posix()],
        "reference_discretization": {
            "base_state_count": len(times), "base_refinement_factor": 16,
            "integration_substeps_per_interval": REFERENCE_SUBSTEPS,
            "effective_exponential_midpoint_steps": (len(times)-1)*REFERENCE_SUBSTEPS,
        },
        "continuous_selection": {
            "critical_time_s": continuous_tc,
            "bracket_times_s": [float(times[lower_index]), float(times[upper_index])],
            "onset_branches": onset_branches,
            "onset_relative_gain_gaps": onset_relative_gaps,
            "near_optimal_tolerance": NEAR_OPTIMAL_GAIN_GAP,
            "near_optimal_onset_set": near_optimal_onset,
            "onset_candidate_evaluation_time_s": float(times[upper_index]),
            "global_onset_near_optimal_set_certified": False,
            "onset_is_unique": len(near_optimal_onset) == 1,
            "final_branches": final_branches,
            "final_winner_label": final_winner_label,
            "final_winner": final_winner,
            "final_relative_gap_to_best_tested_competitor": final_winner_gap,
            "selector_changes_from_original_discrete_v2": selector_changes,
        },
        "time_convergence": {
            "onset_comparison_time_s": onset_comparison_time_s,
            "rows": integration_rows,
            "changes": integration_changes,
        },
        "coordinate_scale_sensitivity": {"rows": scale_rows, "changes": scale_changes},
        "wavenumber_profiles": k_rows,
        "principal_symbol_audit_from_v1": v1["principal_symbol_audit"],
        "extended_wavenumber_interpretation": {
            "tail_rows": tail,
            "sampled_tail_plateau_gate_passed": high_k_sampled_plateau,
            "asymptotically_stable": high_k_asymptotically_stable,
            "interpretation": (
                "The sampled high-k spectral abscissa approaches a finite positive plateau on one "
                "coarse audited path and is not asymptotically stable. This is consistency evidence, "
                "not a proof of a continuum-uniform symmetrizer or fixed-norm bound."
            ),
        },
        "gates": gates,
        "failed_gates": failed,
        "core_failed_gates": core_failed,
        "formal_interpretation": {
            "numerical_fixed_norm_search_box_audit_closed": core_closed,
            "unconditional_continuum_well_posedness_proved": False,
            "conditional_continuum_bound_requires_uniform_symmetrizer": True,
            "finite_dimensional_norm_equivalence_preserves_well_posedness": True,
            "critical_time_is_coordinate_invariant": False,
            "factor_two_coordinate_scale_audit_within_10_percent": gates[coordinate_invariance_diagnostic],
            "critical_time_definition": (
                "tc is conditioned on the locked dimensionless observation norm and gain threshold; "
                "equivalent norms preserve boundedness but need not preserve a threshold-crossing time."
            ),
        },
        "claim_boundary": {
            "analytical_global_uniqueness_proved": False,
            "onset_single_direction_claim_authorized": False,
            "tested_onset_candidate_set_authorized": core_closed,
            "global_onset_near_optimal_set_certified": False,
            "coordinate_invariant_critical_time_claim_authorized": False,
            "fixed_norm_critical_time_claim_authorized": core_closed,
            "final_unique_over_tested_continuous_branches": gates["final_selector_separated_from_tested_competitors"],
            "experimental_validation_claimed": False,
            "independent_u903_validation_claimed": False,
            "unconditional_continuum_well_posedness_claim_authorized": False,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR/"time_convergence.csv", integration_rows)
    write_csv(TABLE_DIR/"scale_sensitivity.csv", scale_changes)
    write_csv(TABLE_DIR/"wavenumber_profiles.csv", k_rows)

    lines = [
        "# CP-Ti continuous-spectrum robustness V2", "",
        f"Status: **{report['status']}**.", "",
        f"Reference: {len(times)} actual base states and {REFERENCE_SUBSTEPS} exponential-midpoint substeps per interval.", "",
        f"- Time-converged tc: {continuous_tc*1e6:.9g} us.",
        f"- Tested near-optimal onset candidate set at t+={times[upper_index]*1e6:.9g} us (2% gain tolerance): {near_optimal_onset}.",
        f"- Onset upper-branch gaps: {onset_relative_gaps}.",
        f"- Final winner: {final_winner_label}, n={final_winner['direction_n']}, k={final_winner['k_m_inv']:.9g} 1/m.",
        f"- Final tested-competitor gap: {final_winner_gap:.6g}.",
        f"- Sampled high-k tail plateau: {high_k_sampled_plateau}; asymptotic stability: {high_k_asymptotically_stable}.",
        f"- Numerical fixed-norm search-box audit closure: {core_closed}.",
        "- Continuum high-k boundedness remains conditional on a uniform observable symmetrizer estimate.",
        "- tc is conditional on the locked dimensionless observation norm; it is not a coordinate invariant.",
        "", "## Gates", "", "| gate | passed |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {passed} |" for name, passed in gates.items())
    lines.extend(["", f"Failed diagnostics/gates: {failed}.",
                  f"Core failed gates (excluding the coordinate-invariance diagnostic): {core_failed}.", "",
                  "A single onset n* is not authorized; the two tested upper-bracket branches form a candidate set, not a global continuum certificate."])
    SUMMARY.write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    for context_name in ("factor8", "factor16"):
        rows = [row for row in integration_rows if row["context"] == context_name]
        axes[0, 0].plot([row["substeps"] for row in rows], [1e6*row["critical_time_s"] for row in rows], "o-", label=context_name)
        axes[0, 1].plot([row["substeps"] for row in rows], [row["final_log_gain"] for row in rows], "o-", label=context_name)
    axes[0, 0].set_ylabel("tc (µs)"); axes[0, 1].set_ylabel("final log gain")
    for selector, axis in (("onset", axes[1, 0]), ("final", axes[1, 1])):
        rows = [row for row in k_rows if row["selector"] == selector]
        axis.semilogx([row["k_m_inv"] for row in rows], [row["gain"] for row in rows], "o-")
        axis.set_xlabel("k (m$^{-1}$)"); axis.set_ylabel("gain"); axis.set_title(selector)
    for axis in axes.ravel(): axis.grid(True, alpha=0.3)
    axes[0, 0].legend(frameon=False); axes[0, 1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=220); plt.close(fig)
    print(json.dumps({
        "result": str(RESULT), "summary": str(SUMMARY), "figure": str(FIGURE),
        "status": report["status"], "critical_time_s": continuous_tc,
        "near_optimal_onset_set": near_optimal_onset,
        "final_n": final_winner["direction_n"], "final_k_m_inv": final_winner["k_m_inv"],
        "failed_gates": failed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
