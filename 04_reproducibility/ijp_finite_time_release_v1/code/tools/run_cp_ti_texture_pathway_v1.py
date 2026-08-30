"""Build the CP-Ti orientation/energy atlas and finite-time pathway paper data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from math import log, pi
from pathlib import Path
import sys
import time
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

from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    build_material_objects,
    load_card,
    local_state92_from_material_state,
    simple_shear,
)
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)
from hcp_cp_gnd.dynamic_mechanism_causality_v1 import (  # noqa: E402
    COUPLING_EDGES,
    all_couplings_coalition_values,
    exact_shapley_log_gain,
)
from hcp_cp_gnd.evolving_work_partition_v1 import (  # noqa: E402
    DislocationStoredEnergyPartitionV1,
    EvolvingPartitionHCPMaterialPointV1,
)
from hcp_cp_gnd.micromorphic import MicromorphicParameters  # noqa: E402
from hcp_cp_gnd.spectral_export import ContinuousSpectralPointModel  # noqa: E402
from hcp_cp_gnd.texture_pathway_atlas_v1 import (  # noqa: E402
    LOG_GAIN_THRESHOLDS,
    ORIENTATION_GRID_DEG,
    SelectorGridV1,
    crossing_time_log_linear,
    pathway_group_scores,
    select_storage_representatives,
)
from tools.run_cp_ti_finite_time_propagator_v2 import (  # noqa: E402
    _admission,
    _coordinate_scales,
)


ATLAS = ROOT / "05_results/cp_ti_texture_energy_atlas_v1.json"
RESULT = ROOT / "05_results/cp_ti_texture_pathway_v1.json"
SUMMARY = ROOT / "05_results/cp_ti_texture_pathway_v1.md"
TABLE_DIR = ROOT / "05_results/cp_ti_texture_pathway_v1"
FIGURE = ROOT / "05_results/cp_ti_texture_pathway_v1.png"
PAPER_FIG_DIR = ROOT / "06_manuscript/ijp_texture_pathway/figures"

BASE_SHEARS = np.asarray([0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1.40])
OBSERVED = np.arange(6, 69)
BASE_FACTOR = 0.5
SENSITIVITY_FACTORS = (0.25, 0.5, 1.0)
HIGH_THRESHOLD = float(log(1.0e4))
K_BOUNDS = (3.0e2, 3.0e5)
SPHERICAL_BOUNDS = ((0.0, 0.5 * pi), (-pi, pi), (log(K_BOUNDS[0]), log(K_BOUNDS[1])))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def refined_shears(values: np.ndarray) -> np.ndarray:
    output = [float(values[0])]
    for left, right in zip(values[:-1], values[1:], strict=True):
        output.extend((float(0.5 * (left + right)), float(right)))
    return np.asarray(output)


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
    return np.asarray([
        np.arccos(np.clip(value[2], -1.0, 1.0)),
        np.arctan2(value[1], value[0]),
        np.log(float(k_m_inv)),
    ])


def angle_degrees(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=float); a /= np.linalg.norm(a)
    b = np.asarray(second, dtype=float); b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def build_oriented_model(orientation_deg: tuple[float, float, float], factor: float,
                         *, evolving: bool) -> tuple[Any, Any, Any]:
    card = load_card()
    systems, parameters, baseline = build_material_objects(card)
    from hcp_cp.model import HCPMaterialPoint, orientation_from_bunge
    orientation = orientation_from_bunge(*orientation_deg)
    law = DislocationStoredEnergyPartitionV1(float(factor))
    if evolving:
        model = EvolvingPartitionHCPMaterialPointV1(
            systems, parameters, orientation, baseline.switches, partition_law=law,
        )
    else:
        model = HCPMaterialPoint(systems, parameters, orientation, baseline.switches)
    return model, parameters, law


def run_history(orientation_deg: tuple[float, float, float], factor: float,
                checkpoints: np.ndarray, *, evolving: bool = True,
                shear_increment: float = 2.0e-3) -> dict[str, Any]:
    model, parameters, law = build_oriented_model(orientation_deg, factor, evolving=evolving)
    rate = float(load_card()["representative_loading"]["macroscopic_shear_rate_s"])
    state = model.initial_state()
    gamma = 0.0
    records: list[dict[str, Any]] = []
    states: list[Any] = []
    maximum_energy_error = 0.0
    maximum_partition_error = 0.0
    for target in np.asarray(checkpoints, dtype=float):
        result = None
        while gamma < target - 4.0 * np.finfo(float).eps:
            next_gamma = min(gamma + shear_increment, float(target))
            result = model.advance(
                simple_shear(gamma), simple_shear(next_gamma), state,
                (next_gamma - gamma) / rate,
            )
            state = result.state
            gamma = next_gamma
            maximum_energy_error = max(maximum_energy_error, result.energy_balance_relative_error)
            maximum_partition_error = max(maximum_partition_error, result.work_partition_relative_error)
        response = model.evaluate(simple_shear(float(target)), state)
        work = float(state.plastic_work_density)
        heat = float(state.heat_density)
        storage = float(state.stored_energy_density)
        records.append({
            "shear": float(target),
            "time_s": float(target / rate),
            "temperature_K": float(state.temperature),
            "cauchy_shear_Pa": float(response.cauchy[0, 1]),
            "cp_work_J_m3": work,
            "generated_heat_J_m3": heat,
            "defect_storage_J_m3": storage,
            "cumulative_heat_fraction": float(heat / max(work, 1.0)),
            "cumulative_storage_fraction": float(storage / max(work, 1.0)),
            "dominant_slip_index": int(np.argmax(np.abs(response.slip_rate))),
            "dominant_slip_family": model.systems.slip_families[int(np.argmax(np.abs(response.slip_rate)))],
            "maximum_slip_rate_s_inv": float(np.max(np.abs(response.slip_rate))),
            "ledger_residual_J_m3": float(work - heat - storage),
            "substeps_last_increment": 0 if result is None else int(result.substeps),
        })
        states.append(state)
    return {
        "orientation_bunge_deg": list(orientation_deg),
        "line_energy_factor": float(factor),
        "partition_kind": "evolving_dislocation_line_energy" if evolving else "constant_beta_0p9",
        "records": records,
        "states": states,
        "model": model,
        "parameters": parameters,
        "partition_law": law if evolving else None,
        "maximum_energy_balance_relative_error": float(maximum_energy_error),
        "maximum_work_partition_relative_error": float(maximum_partition_error),
    }


def atlas_row(history: dict[str, Any]) -> dict[str, Any]:
    final = history["records"][-1]
    return {
        "orientation_bunge_deg": history["orientation_bunge_deg"],
        "line_energy_factor": history["line_energy_factor"],
        "storage_fraction_at_shear_0p2": final["cumulative_storage_fraction"],
        "heat_fraction_at_shear_0p2": final["cumulative_heat_fraction"],
        "temperature_at_shear_0p2_K": final["temperature_K"],
        "shear_stress_at_shear_0p2_Pa": final["cauchy_shear_Pa"],
        "dominant_slip_family_at_shear_0p2": final["dominant_slip_family"],
        "cp_work_at_shear_0p2_J_m3": final["cp_work_J_m3"],
        "defect_storage_at_shear_0p2_J_m3": final["defect_storage_J_m3"],
        "ledger_residual_at_shear_0p2_J_m3": final["ledger_residual_J_m3"],
        "maximum_energy_balance_relative_error": history["maximum_energy_balance_relative_error"],
        "maximum_work_partition_relative_error": history["maximum_work_partition_relative_error"],
    }


def run_atlas() -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for index, orientation in enumerate(ORIENTATION_GRID_DEG, start=1):
        rows.append(atlas_row(run_history(orientation, BASE_FACTOR, np.asarray([0.2]))))
        print(json.dumps({"atlas_orientation": index, "total": len(ORIENTATION_GRID_DEG),
                          "orientation_deg": orientation, "storage_fraction": rows[-1]["storage_fraction_at_shear_0p2"]}))
    selected = select_storage_representatives(rows)
    ledger_scale = max(max(abs(row["cp_work_at_shear_0p2_J_m3"]) for row in rows), 1.0)
    maximum_ledger_residual = max(abs(row["ledger_residual_at_shear_0p2_J_m3"]) for row in rows)
    report = {
        "schema": "CP_TI_TEXTURE_ENERGY_ATLAS_V1",
        "status": "ATLAS_ACCEPTED" if maximum_ledger_residual <= 1.0e-10 * ledger_scale else "ATLAS_LEDGER_FAILURE",
        "orientation_design": {
            "kind": "deterministic_Bunge_grid_not_experimental_ODF",
            "count": len(rows),
            "phi1_deg": [0, 30, 60, 90],
            "Phi_deg": [0, 30, 60, 90],
            "phi2_deg": [0, 30],
        },
        "line_energy_factor": BASE_FACTOR,
        "selection_metric": "cumulative defect-storage fraction at shear 0.2",
        "rows": rows,
        "selected_representatives": selected,
        "ledger_audit": {
            "maximum_absolute_residual_J_m3": float(maximum_ledger_residual),
            "relative_to_maximum_work": float(maximum_ledger_residual / ledger_scale),
        },
        "wall_time_s": float(time.perf_counter() - started),
        "claim_boundary": {
            "experimental_ODF": False,
            "batch_specific_prediction": False,
            "twinning_included": False,
            "damage_included": False,
        },
    }
    ATLAS.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


@dataclass
class SpectralContext:
    label: str
    history: dict[str, Any]
    points: list[Any]
    storages: list[Any]
    times_s: np.ndarray
    scales: np.ndarray
    cache: dict[tuple[Any, ...], list[Any]]

    def operators(self, k_m_inv: float, direction_n: Any, prefix_index: int | None = None) -> list[Any]:
        direction = np.asarray(direction_n, dtype=float); direction /= np.linalg.norm(direction)
        stop = len(self.points) if prefix_index is None else int(prefix_index) + 1
        key = (round(float(k_m_inv), 8), *(np.round(direction, 10)), stop)
        if key not in self.cache:
            self.cache[key] = [
                assemble_dynamic_crystal_operator_v1(
                    point,
                    wavenumber_m_inv=float(k_m_inv),
                    direction_n=direction,
                    admission=_admission(),
                )
                for point in self.points[:stop]
            ]
        return self.cache[key]

    def propagation(self, k_m_inv: float, direction_n: Any, prefix_index: int | None = None,
                    substeps: int = 1) -> dict[str, Any]:
        stop = len(self.points) if prefix_index is None else int(prefix_index) + 1
        return finite_time_amplification_history(
            self.operators(k_m_inv, direction_n, prefix_index),
            self.times_s[:stop],
            coordinate_scales=self.scales,
            gain_threshold=float(np.e),
            input_indices=OBSERVED,
            output_indices=OBSERVED,
            integration_substeps_per_interval=substeps,
        )


def build_spectral_context(orientation: tuple[float, float, float], factor: float,
                           checkpoints: np.ndarray, *, evolving_base: bool,
                           evolving_operator: bool, label: str) -> SpectralContext:
    history = run_history(orientation, factor, checkpoints, evolving=evolving_base)
    model = history["model"]
    parameters = history["parameters"]
    storages = [
        local_state92_from_material_state(state, model, simple_shear(float(shear)))
        for state, shear in zip(history["states"], checkpoints, strict=True)
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
        conductivity_W_mK=load_card()["thermal"]["conductivity_W_mK_at_300K"],
        parameter_provenance=load_card()["status"],
        power_partition_law=history["partition_law"] if evolving_operator else None,
    )
    reference = np.asarray([1.0, 2.0, 3.0]); reference /= np.linalg.norm(reference)
    points = [
        spectral.export(
            simple_shear(float(shear)),
            storage.temperature_K,
            storage.gamma_signed,
            np.zeros((18, 3)),
            storage,
            direction_n=reference,
            compute_jacobian=True,
        )
        for storage, shear in zip(storages, checkpoints, strict=True)
    ]
    return SpectralContext(
        label=label,
        history=history,
        points=points,
        storages=storages,
        times_s=np.asarray([row["time_s"] for row in history["records"]]),
        scales=_coordinate_scales(storages),
        cache={},
    )


def compact_propagation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "wavenumber_m_inv": value["wavenumber_m_inv"],
        "wavelength_m": float(2.0 * pi / value["wavenumber_m_inv"]),
        "direction_n": value["direction_n"],
        "integration_substeps_per_interval": value["integration_substeps_per_interval"],
        "initial_time_s": value["initial_time_s"],
        "final_time_s": value["final_time_s"],
        "final_gain": value["final_gain"],
        "final_log_gain": value["final_log_gain"],
        "prefix": value["prefix"],
        "input_mechanism_participation": value["input_mechanism_participation"],
        "output_mechanism_participation": value["output_mechanism_participation"],
        "full_state_output_mechanism_participation": value[
            "full_state_output_mechanism_participation"
        ],
    }


def coarse_selector(context: SpectralContext) -> dict[str, Any]:
    grid = SelectorGridV1(); grid.validate()
    branches = []
    for direction in grid.directions_n:
        for k_m_inv in grid.wavenumbers_m_inv:
            raw = context.propagation(k_m_inv, direction)
            branches.append(compact_propagation(raw))
    gain_matrix = np.asarray([[row["maximum_gain"] for row in branch["prefix"]] for branch in branches])
    envelope = np.max(gain_matrix, axis=0)
    threshold_candidates = np.flatnonzero(np.log(envelope) >= HIGH_THRESHOLD)
    target_index = int(threshold_candidates[0]) if threshold_candidates.size else len(context.times_s) - 1
    winner_index = int(np.argmax(gain_matrix[:, target_index]))
    thresholds = {
        f"log_gain_{threshold:.12g}": crossing_time_log_linear(context.times_s, envelope, threshold)
        for threshold in LOG_GAIN_THRESHOLDS
    }
    return {
        "grid": {
            "directions_n": [list(value) for value in grid.directions_n],
            "wavenumbers_m_inv": list(grid.wavenumbers_m_inv),
            "branch_count": len(branches),
        },
        "branches": branches,
        "envelope": [
            {"time_s": float(time_s), "maximum_gain": float(gain), "log_gain": float(np.log(gain))}
            for time_s, gain in zip(context.times_s, envelope, strict=True)
        ],
        "threshold_crossing_times_s": thresholds,
        "high_threshold_reached": bool(threshold_candidates.size),
        "target_prefix_index": target_index,
        "coarse_winner": branches[winner_index],
    }


def refine_selector(context: SpectralContext, coarse: dict[str, Any]) -> dict[str, Any]:
    target = int(coarse["target_prefix_index"])
    winner = coarse["coarse_winner"]
    start = coordinates_from_direction(winner["direction_n"], winner["wavenumber_m_inv"])
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(x: Any) -> dict[str, Any]:
        raw = np.asarray(x, dtype=float)
        key = tuple(np.round(raw, 9))
        if key not in cache:
            cache[key] = context.propagation(
                float(np.exp(raw[2])), direction_from_coordinates(raw), prefix_index=target,
            )
        return cache[key]

    result = minimize(
        lambda x: -evaluate(x)["final_log_gain"],
        start,
        method="Powell",
        bounds=SPHERICAL_BOUNDS,
        options={"xtol": 2.0e-5, "ftol": 2.0e-7, "maxiter": 55},
    )
    raw = evaluate(result.x)
    compact = compact_propagation(raw)
    compact.update({
        "optimizer_coordinates": np.asarray(result.x).tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "coarse_to_refined_log_gain_change": float(raw["final_log_gain"] - winner["prefix"][target]["log_gain"]),
        "k_strictly_inside_bounds": bool(K_BOUNDS[0] * 1.001 < raw["wavenumber_m_inv"] < K_BOUNDS[1] / 1.001),
        "target_prefix_index": target,
    })
    return compact


def shapley_case(context: SpectralContext, selector: dict[str, Any], prefix_index: int) -> dict[str, Any]:
    operators = context.operators(selector["wavenumber_m_inv"], selector["direction_n"], prefix_index)
    values = all_couplings_coalition_values(
        operators,
        context.times_s[:prefix_index + 1],
        coordinate_scales=context.scales,
        input_indices=OBSERVED,
        output_indices=OBSERVED,
        integration_substeps_per_interval=1,
    )
    shapley = exact_shapley_log_gain(values)
    return {
        "shapley": shapley,
        "pathway": pathway_group_scores(shapley["attribution"]),
        "coalition_count": len(values),
        "prefix_index": int(prefix_index),
        "final_time_s": float(context.times_s[prefix_index]),
    }


def point_partition_audit(context: SpectralContext) -> dict[str, Any]:
    rows = []
    for record, point in zip(context.history["records"], context.points, strict=True):
        beta = float(point.heat_source_W_m3 / point.cp_power_W_m3) if point.cp_power_W_m3 > 1.0e-30 else 1.0
        rows.append({
            "shear": record["shear"],
            "time_s": record["time_s"],
            "cp_power_W_m3": float(point.cp_power_W_m3),
            "heat_source_W_m3": float(point.heat_source_W_m3),
            "storage_rate_W_m3": float(point.storage_rate_W_m3),
            "beta_eff": beta,
            "power_partition_residual_W_m3": float(point.power_partition_residual_W_m3),
            "cumulative_storage_fraction": record["cumulative_storage_fraction"],
            "temperature_K": record["temperature_K"],
        })
    maximum_relative = max(
        abs(row["power_partition_residual_W_m3"]) / max(abs(row["cp_power_W_m3"]), 1.0)
        for row in rows
    )
    return {
        "rows": rows,
        "maximum_relative_power_partition_residual": float(maximum_relative),
        "minimum_heat_source_W_m3": float(min(row["heat_source_W_m3"] for row in rows)),
        "minimum_beta_eff": float(min(row["beta_eff"] for row in rows)),
        "maximum_beta_eff": float(max(row["beta_eff"] for row in rows)),
    }


def evaluate_fixed_selector(context: SpectralContext, selector: dict[str, Any],
                            log_threshold: float = HIGH_THRESHOLD) -> tuple[dict[str, Any], int]:
    full = context.propagation(selector["wavenumber_m_inv"], selector["direction_n"])
    prefix_logs = np.asarray([row["log_gain"] for row in full["prefix"]])
    crossed = np.flatnonzero(prefix_logs >= log_threshold)
    prefix = int(crossed[0]) if crossed.size else len(prefix_logs) - 1
    value = context.propagation(
        selector["wavenumber_m_inv"], selector["direction_n"], prefix_index=prefix,
    )
    return compact_propagation(value), prefix


def analyze_representative(label: str, orientation: tuple[float, float, float]) -> dict[str, Any]:
    print(json.dumps({"stage": "build_baseline_context", "representative": label, "orientation": orientation}))
    coarse_context = build_spectral_context(
        orientation, BASE_FACTOR, BASE_SHEARS,
        evolving_base=True, evolving_operator=True, label=f"{label}_evolving_coarse",
    )
    coarse = coarse_selector(coarse_context)
    refined = refine_selector(coarse_context, coarse)
    print(json.dumps({"stage": "selector_refined", "representative": label,
                      "k": refined["wavenumber_m_inv"], "n": refined["direction_n"],
                      "log_gain": refined["final_log_gain"]}))

    fine_shears = refined_shears(BASE_SHEARS)
    fine_context = build_spectral_context(
        orientation, BASE_FACTOR, fine_shears,
        evolving_base=True, evolving_operator=True, label=f"{label}_evolving_fine",
    )
    target_time = coarse_context.times_s[int(refined["target_prefix_index"])]
    fine_target = int(np.flatnonzero(np.isclose(fine_context.times_s, target_time, rtol=0.0, atol=1.0e-16))[0])
    fine_value = compact_propagation(fine_context.propagation(
        refined["wavenumber_m_inv"], refined["direction_n"], prefix_index=fine_target,
        substeps=2,
    ))
    coarse_log = refined["final_log_gain"]
    history_relative = float(abs(fine_value["final_log_gain"] - coarse_log) / max(abs(fine_value["final_log_gain"]), 1.0e-12))
    causal = shapley_case(fine_context, refined, fine_target)
    print(json.dumps({"stage": "shapley_complete", "representative": label,
                      "pathway": causal["pathway"]["pathway_label"],
                      "efficiency_error": causal["shapley"]["efficiency_error"]}))

    frozen_context = build_spectral_context(
        orientation, BASE_FACTOR, fine_shears,
        evolving_base=True, evolving_operator=False, label=f"{label}_frozen_base_constant_beta",
    )
    constant_context = build_spectral_context(
        orientation, BASE_FACTOR, fine_shears,
        evolving_base=False, evolving_operator=False, label=f"{label}_constant_beta_base",
    )
    frozen_value = compact_propagation(frozen_context.propagation(
        refined["wavenumber_m_inv"], refined["direction_n"], prefix_index=fine_target,
        substeps=2,
    ))
    constant_value = compact_propagation(constant_context.propagation(
        refined["wavenumber_m_inv"], refined["direction_n"], prefix_index=fine_target,
        substeps=2,
    ))

    sensitivity = []
    for factor in SENSITIVITY_FACTORS:
        if factor == BASE_FACTOR:
            context = coarse_context
            selector = refined
        else:
            context = build_spectral_context(
                orientation, factor, BASE_SHEARS,
                evolving_base=True, evolving_operator=True,
                label=f"{label}_line_factor_{factor:g}",
            )
            selector = refined
        selected_value, prefix = evaluate_fixed_selector(context, selector)
        causal_factor = shapley_case(context, selector, prefix)
        sensitivity.append({
            "line_energy_factor": factor,
            "selected_propagation": selected_value,
            "pathway": causal_factor["pathway"],
            "shapley": causal_factor["shapley"],
            "prefix_index": prefix,
            "base_partition_audit": point_partition_audit(context),
            "final_cumulative_storage_fraction": context.history["records"][-1]["cumulative_storage_fraction"],
        })
        print(json.dumps({"stage": "factor_sensitivity", "representative": label,
                          "factor": factor, "pathway": causal_factor["pathway"]["pathway_label"]}))

    labels = [item["pathway"]["pathway_label"] for item in sensitivity]
    baseline_coarse_label = next(
        item["pathway"]["pathway_label"]
        for item in sensitivity
        if item["line_energy_factor"] == BASE_FACTOR
    )
    partition = point_partition_audit(fine_context)
    gates = {
        "power_partition_identity_below_1e_minus_10": partition["maximum_relative_power_partition_residual"] <= 1.0e-10,
        "heat_source_nonnegative": partition["minimum_heat_source_W_m3"] >= -1.0e-8,
        "history_refinement_below_2_percent": history_relative <= 0.02,
        "selector_non_decreasing": refined["coarse_to_refined_log_gain_change"] >= -1.0e-9,
        "selector_k_interior": refined["k_strictly_inside_bounds"],
        "shapley_efficiency_below_1e_minus_10": abs(causal["shapley"]["efficiency_error"]) <= 1.0e-10,
        "pathway_label_robust_over_line_energy_interval": len(set(labels)) == 1,
        "pathway_label_unchanged_under_history_refinement": (
            baseline_coarse_label == causal["pathway"]["pathway_label"]
        ),
    }
    return {
        "label": label,
        "orientation_bunge_deg": list(orientation),
        "coarse_selector": coarse,
        "refined_selector": refined,
        "fine_history_selected_propagation": fine_value,
        "history_refinement_relative_log_gain_change": history_relative,
        "causal_pathway": causal,
        "partition_audit": partition,
        "constant_beta_ablations": {
            "evolving_base_evolving_operator": fine_value,
            "evolving_base_constant_beta_operator": frozen_value,
            "constant_beta_base_constant_beta_operator": constant_value,
            "same_selector_and_target_time": True,
        },
        "line_energy_sensitivity": sensitivity,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "trajectory": fine_context.history["records"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def clean_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    return value


def run_analysis() -> dict[str, Any]:
    atlas = json.loads(ATLAS.read_text(encoding="utf-8")) if ATLAS.exists() else run_atlas()
    require(atlas["status"] == "ATLAS_ACCEPTED", "orientation atlas did not pass its ledger gate")
    selected = atlas["selected_representatives"]
    analyses = {}
    started = time.perf_counter()
    for label, row in selected.items():
        orientation = tuple(float(value) for value in row["orientation_bunge_deg"])
        analyses[label] = analyze_representative(label, orientation)
    all_gates = {
        f"{label}__{name}": passed
        for label, analysis in analyses.items()
        for name, passed in analysis["gates"].items()
    }
    pathway_labels = {
        label: analysis["causal_pathway"]["pathway"]["pathway_label"]
        for label, analysis in analyses.items()
    }
    storage_fractions = [row["storage_fraction_at_shear_0p2"] for row in atlas["rows"]]
    report = {
        "schema": "CP_TI_TEXTURE_PATHWAY_V1",
        "status": "ALL_DECLARED_GATES_PASS" if all(all_gates.values()) else "DECLARED_GAPS_RETAINED",
        "research_question": "whether orientation-dependent slip and dislocation line energy can switch thermal/defect finite-time pathways",
        "atlas_source": ATLAS.relative_to(ROOT).as_posix(),
        "registered_line_energy_factors": list(SENSITIVITY_FACTORS),
        "gain_threshold_log_family": list(LOG_GAIN_THRESHOLDS),
        "analyses": analyses,
        "cross_orientation_conclusion": {
            "pathway_labels": pathway_labels,
            "distinct_pathway_count": len(set(pathway_labels.values())),
            "texture_switches_registered_pathway": len(set(pathway_labels.values())) > 1,
            "atlas_storage_fraction_range": [float(min(storage_fractions)), float(max(storage_fractions))],
            "interpretation": (
                "A one-pathway result falsifies the sufficiency of ordinary slip-mediated dislocation "
                "line energy for the experimental thermal/non-thermal bifurcation; it does not falsify "
                "texture effects when twin-enhanced storage, DRX, or damage is admitted."
            ),
        },
        "gates": all_gates,
        "failed_gates": [name for name, passed in all_gates.items() if not passed],
        "wall_time_s": float(time.perf_counter() - started),
        "claim_boundary": {
            "single_crystal_computational_mechanism": True,
            "experimental_parameter_identification": False,
            "independent_u903_validation": False,
            "ti64_used": False,
            "twinning_drx_damage_in_onset_operator": False,
            "comparison_to_2025_pure_ti_experiment_is_qualitative_falsification_only": True,
        },
    }
    clean = clean_for_json(report)
    RESULT.write_text(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return clean


def _refine_values_to_factor(values: np.ndarray, factor: int) -> np.ndarray:
    require(factor in (1, 2, 4, 8, 16), "unsupported checkpoint refinement factor")
    output = np.asarray(values, dtype=float)
    current = 1
    while current < factor:
        output = refined_shears(output)
        current *= 2
    return output


def run_history_refinement_update() -> dict[str, Any]:
    """Replace the coarse/fine diagnostic by a converged doubled-history audit."""

    report = json.loads(RESULT.read_text(encoding="utf-8"))
    for label, analysis in report["analyses"].items():
        selector = analysis["refined_selector"]
        target_shear = float(selector["final_time_s"] * load_card()["representative_loading"]["macroscopic_shear_rate_s"])
        prefix_values = BASE_SHEARS[BASE_SHEARS <= target_shear + 1.0e-13]
        rows = []
        previous_log = None
        for factor in (2, 4, 8, 16):
            checkpoints = _refine_values_to_factor(prefix_values, factor)
            context = build_spectral_context(
                tuple(float(value) for value in analysis["orientation_bunge_deg"]),
                BASE_FACTOR,
                checkpoints,
                evolving_base=True,
                evolving_operator=True,
                label=f"{label}_history_factor{factor}",
            )
            propagation = compact_propagation(context.propagation(
                selector["wavenumber_m_inv"], selector["direction_n"], substeps=2,
            ))
            relative = None if previous_log is None else float(
                abs(propagation["final_log_gain"] - previous_log)
                / max(abs(propagation["final_log_gain"]), 1.0e-12)
            )
            rows.append({
                "factor": factor,
                "checkpoint_count": len(checkpoints),
                "final_log_gain": propagation["final_log_gain"],
                "final_gain": propagation["final_gain"],
                "relative_log_gain_change_from_previous_factor": relative,
            })
            print(json.dumps({"stage": "history_refinement", "representative": label,
                              "factor": factor, "log_gain": propagation["final_log_gain"],
                              "relative_change": relative}))
            previous_log = propagation["final_log_gain"]
            if factor >= 4 and relative is not None and relative <= 0.02:
                break
        final_change = rows[-1]["relative_log_gain_change_from_previous_factor"]
        passed = bool(final_change is not None and final_change <= 0.02)
        analysis["time_checkpoint_refinement"] = {
            "same_selector": True,
            "same_exponential_midpoint_substeps_per_interval": 2,
            "rows": rows,
            "last_doubling_relative_log_gain_change": final_change,
            "gate_below_2_percent": passed,
        }
        analysis["history_refinement_relative_log_gain_change"] = final_change
        analysis["gates"]["history_refinement_below_2_percent"] = passed
        analysis["failed_gates"] = [name for name, value in analysis["gates"].items() if not value]
    report["gates"] = {
        f"{label}__{name}": passed
        for label, analysis in report["analyses"].items()
        for name, passed in analysis["gates"].items()
    }
    report["failed_gates"] = [name for name, value in report["gates"].items() if not value]
    report["status"] = "ALL_DECLARED_GATES_PASS" if not report["failed_gates"] else "DECLARED_GAPS_RETAINED"
    RESULT.write_text(
        json.dumps(clean_for_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def make_outputs(report: dict[str, Any]) -> None:
    atlas = json.loads(ATLAS.read_text(encoding="utf-8"))
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR / "orientation_atlas.csv", atlas["rows"])
    selector_rows = []
    pathway_rows = []
    trajectory_rows = []
    for label, analysis in report["analyses"].items():
        selector = analysis["refined_selector"]
        selector_rows.append({
            "representative": label,
            "phi1_deg": analysis["orientation_bunge_deg"][0],
            "Phi_deg": analysis["orientation_bunge_deg"][1],
            "phi2_deg": analysis["orientation_bunge_deg"][2],
            "target_time_s": selector["final_time_s"],
            "n1": selector["direction_n"][0], "n2": selector["direction_n"][1], "n3": selector["direction_n"][2],
            "k_m_inv": selector["wavenumber_m_inv"], "wavelength_m": selector["wavelength_m"],
            "log_gain": selector["final_log_gain"],
            "pathway": analysis["causal_pathway"]["pathway"]["pathway_label"],
        })
        for sensitivity in analysis["line_energy_sensitivity"]:
            scores = sensitivity["pathway"]["positive_endpoint_scores"]
            pathway_rows.append({
                "representative": label,
                "line_energy_factor": sensitivity["line_energy_factor"],
                "pathway": sensitivity["pathway"]["pathway_label"],
                "thermal_score": scores["thermal"],
                "dislocation_score": scores["dislocation"],
                "mechanical_score": scores["mechanical_inertia"],
                "plastic_score": scores["plastic_kinematics"],
                "storage_fraction_final": sensitivity["final_cumulative_storage_fraction"],
            })
        for row in analysis["trajectory"]:
            trajectory_rows.append({"representative": label, **row})
    write_csv(TABLE_DIR / "selectors.csv", selector_rows)
    write_csv(TABLE_DIR / "pathway_sensitivity.csv", pathway_rows)
    write_csv(TABLE_DIR / "representative_trajectories.csv", trajectory_rows)

    lines = [
        "# CP-Ti texture-selected pathway analysis V1", "",
        f"Status: **{report['status']}**.", "",
        "The analysis is a single-crystal theoretical/computational test; it is not experimental parameter identification or U903 validation.", "",
        "| representative | Bunge angles (deg) | tc for log G=log(1e4) (us) | wavelength (um) | pathway | failed gates |",
        "|---|---|---:|---:|---|---|",
    ]
    for label, analysis in report["analyses"].items():
        crossings = analysis["coarse_selector"]["threshold_crossing_times_s"]
        tc = crossings[f"log_gain_{HIGH_THRESHOLD:.12g}"]
        lines.append(
            f"| {label} | {analysis['orientation_bunge_deg']} | "
            f"{'--' if tc is None else f'{1e6*tc:.6g}'} | "
            f"{1e6*analysis['refined_selector']['wavelength_m']:.6g} | "
            f"{analysis['causal_pathway']['pathway']['pathway_label']} | {analysis['failed_gates']} |"
        )
    conclusion = report["cross_orientation_conclusion"]
    lines.extend([
        "", "## Cross-orientation decision", "",
        f"- Texture switches the registered pathway: {conclusion['texture_switches_registered_pathway']}.",
        f"- Atlas storage-fraction range at shear 0.2: {conclusion['atlas_storage_fraction_range']}.",
        f"- Pathway labels: {conclusion['pathway_labels']}.",
        "- A negative switch result is retained as a sufficiency test; no storage coefficient was fitted to force agreement with the 2025 experiment.",
        "", "## Failed gates", "", f"{report['failed_gates']}",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    for phi2, marker in ((0.0, "o"), (30.0, "s")):
        subset = [row for row in atlas["rows"] if row["orientation_bunge_deg"][2] == phi2]
        axes[0, 0].scatter(
            [row["orientation_bunge_deg"][1] for row in subset],
            [1e3 * row["storage_fraction_at_shear_0p2"] for row in subset],
            c=[row["orientation_bunge_deg"][0] for row in subset], cmap="viridis",
            marker=marker, label=fr"$\phi_2={phi2:g}^\circ$", edgecolors="black", linewidths=0.3,
        )
    axes[0, 0].set_xlabel(r"c-axis tilt proxy $\Phi$ (deg)")
    axes[0, 0].set_ylabel("cumulative storage fraction (permille)")
    axes[0, 0].set_title("Registered orientation atlas at shear 0.2")
    axes[0, 0].legend(frameon=False)

    colors = {"minimum_storage": "#0072B2", "median_storage": "#009E73", "maximum_storage": "#D55E00"}
    for label, analysis in report["analyses"].items():
        trajectory = analysis["trajectory"]
        axes[0, 1].plot([row["shear"] for row in trajectory],
                        [row["temperature_K"] for row in trajectory], "o-", color=colors[label], label=label)
        envelope = analysis["coarse_selector"]["envelope"]
        axes[1, 0].semilogy([1e6*row["time_s"] for row in envelope],
                            [row["maximum_gain"] for row in envelope], "o-", color=colors[label], label=label)
    axes[0, 1].set_xlabel("macroscopic shear"); axes[0, 1].set_ylabel("temperature (K)")
    axes[0, 1].set_title("Evolving-partition base trajectories")
    axes[1, 0].axhline(1.0e4, color="black", ls="--", lw=1)
    axes[1, 0].set_xlabel("time (microseconds)"); axes[1, 0].set_ylabel("maximum finite-time gain")
    axes[1, 0].set_title("Orientation-resolved gain envelopes")
    axes[1, 0].legend(frameon=False)

    x = np.arange(len(report["analyses"]))
    width = 0.22
    for offset, factor in enumerate(SENSITIVITY_FACTORS):
        thermal = []
        defect = []
        for analysis in report["analyses"].values():
            item = next(row for row in analysis["line_energy_sensitivity"] if row["line_energy_factor"] == factor)
            thermal.append(item["pathway"]["positive_endpoint_scores"]["thermal"])
            defect.append(item["pathway"]["positive_endpoint_scores"]["dislocation"])
        position = x + (offset - 1) * width
        axes[1, 1].bar(position, thermal, width, label=fr"thermal $c_\rho={factor:g}$", alpha=0.85)
        axes[1, 1].scatter(position, defect, marker="D", color="black", s=28,
                           label="dislocation score" if offset == 0 else None, zorder=3)
    axes[1, 1].set_xticks(x, list(report["analyses"]))
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set_ylabel("positive endpoint Shapley score")
    axes[1, 1].set_title("Causal pathway and line-energy sensitivity")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.ravel(): axis.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=240); plt.close(fig)

    # Paper figures are regenerated from the accepted machine-readable result.
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for phi2, marker in ((0.0, "o"), (30.0, "s")):
        subset = [row for row in atlas["rows"] if row["orientation_bunge_deg"][2] == phi2]
        scatter = axis.scatter(
            [row["orientation_bunge_deg"][1] for row in subset],
            [100.0 * row["storage_fraction_at_shear_0p2"] for row in subset],
            c=[row["orientation_bunge_deg"][0] for row in subset], cmap="viridis",
            marker=marker, s=62, label=fr"$\phi_2={phi2:g}^\circ$",
            edgecolors="black", linewidths=0.35,
        )
    colorbar = fig.colorbar(scatter, ax=axis); colorbar.set_label(r"$\phi_1$ (deg)")
    axis.set_xlabel(r"Bunge angle $\Phi$ (deg)")
    axis.set_ylabel("cumulative defect-storage fraction at shear 0.2 (%)")
    axis.legend(frameon=False); axis.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "fig_orientation_atlas.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    for label, analysis in report["analyses"].items():
        trajectory = [row for row in analysis["trajectory"] if row["shear"] <= 0.5 + 1.0e-13]
        partition_rows = [row for row in analysis["partition_audit"]["rows"] if row["shear"] <= 0.5 + 1.0e-13]
        axes[0].plot([row["shear"] for row in trajectory], [row["temperature_K"] for row in trajectory], "o-", color=colors[label], label=label)
        axes[1].plot([row["shear"] for row in trajectory], [100.0*row["cumulative_storage_fraction"] for row in trajectory], "o-", color=colors[label])
        axes[2].plot([row["shear"] for row in partition_rows], [row["beta_eff"] for row in partition_rows], "o-", color=colors[label])
    axes[0].set_ylabel("temperature (K)"); axes[1].set_ylabel("cumulative storage fraction (%)"); axes[2].set_ylabel(r"differential $\beta_{\rm eff}$")
    for axis in axes:
        axis.set_xlabel("macroscopic shear"); axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "fig_partition_trajectories.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for label, analysis in report["analyses"].items():
        envelope = [row for row in analysis["coarse_selector"]["envelope"] if row["time_s"] <= 5.0e-5 + 1.0e-16]
        axes[0].semilogy([1e6*row["time_s"] for row in envelope], [row["maximum_gain"] for row in envelope], "o-", color=colors[label], label=label)
    axes[0].axhline(1.0e4, color="black", ls="--", lw=1)
    axes[0].set_xlabel("time (microseconds)"); axes[0].set_ylabel("coarse-grid gain envelope")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(True, alpha=0.25)
    labels_order = list(report["analyses"])
    axes[1].bar(np.arange(3), [1e6*report["analyses"][label]["refined_selector"]["wavelength_m"] for label in labels_order], color=[colors[label] for label in labels_order])
    axes[1].set_xticks(np.arange(3), [label.replace("_", "\n") for label in labels_order])
    axes[1].set_ylabel(r"screening wavelength $2\pi/k^*$ ($\mu$m)"); axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "fig_gain_selectors.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    edge_x = np.arange(len(COUPLING_EDGES)); bar_width = 0.25
    for index, (label, analysis) in enumerate(report["analyses"].items()):
        attribution = analysis["causal_pathway"]["shapley"]["attribution"]
        axes[0].bar(edge_x+(index-1)*bar_width, [attribution[edge] for edge in COUPLING_EDGES], bar_width, color=colors[label], label=label)
    edge_labels = [edge.replace("mechanical_inertia", "M").replace("plastic_kinematics", "P").replace("dislocation", "D").replace("thermal", "T").replace("__", "--") for edge in COUPLING_EDGES]
    axes[0].set_xticks(edge_x, edge_labels, rotation=30, ha="right"); axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Shapley value of finite-time log gain"); axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(3)
    for offset, factor in enumerate(SENSITIVITY_FACTORS):
        ratios = []
        for analysis in report["analyses"].values():
            item = next(row for row in analysis["line_energy_sensitivity"] if row["line_energy_factor"] == factor)
            ratios.append(item["pathway"]["thermal_to_dislocation_positive_score_ratio"])
        axes[1].bar(x+(offset-1)*bar_width, ratios, bar_width, label=fr"$c_\rho={factor:g}$")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set_xticks(x, [label.replace("_", "\n") for label in labels_order])
    axes[1].set_ylabel("thermal/dislocation positive-score ratio"); axes[1].legend(frameon=False, fontsize=8)
    for axis in axes: axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "fig_causal_pathways.pdf"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    width = 0.24; x = np.arange(3)
    series = (
        ("evolving_base_evolving_operator", "evolving base + evolving partition"),
        ("evolving_base_constant_beta_operator", r"evolving base + frozen $\beta=0.9$"),
        ("constant_beta_base_constant_beta_operator", r"constant-$\beta$ base + operator"),
    )
    for index, (key, title) in enumerate(series):
        values = [report["analyses"][label]["constant_beta_ablations"][key]["final_log_gain"] for label in labels_order]
        axis.bar(x+(index-1)*width, values, width, label=title)
    axis.set_xticks(x, [label.replace("_", "\n") for label in labels_order]); axis.set_ylabel("finite-time log gain at the registered comparison time")
    axis.legend(frameon=False, fontsize=8); axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "fig_constant_beta_ablation.pdf"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--refine-history", action="store_true")
    args = parser.parse_args()
    if not (args.atlas or args.analyze or args.aggregate or args.refine_history):
        args.atlas = args.analyze = args.refine_history = args.aggregate = True
    if args.atlas:
        run_atlas()
    report = None
    if args.analyze:
        report = run_analysis()
    if args.refine_history:
        report = run_history_refinement_update()
    if args.aggregate:
        if report is None:
            report = json.loads(RESULT.read_text(encoding="utf-8"))
        make_outputs(report)
    print(json.dumps({
        "atlas": str(ATLAS), "result": str(RESULT), "summary": str(SUMMARY),
        "figure": str(FIGURE),
        "status": None if report is None else report["status"],
        "failed_gates": None if report is None else report["failed_gates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
