"""Audit W_E/W_O, nonlinear switches, and the periodic 69-state tangent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    build_material_objects,
    load_card,
    local_state92_from_material_state,
    simple_shear,
)
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    N_GENERATOR,
    assemble_dynamic_crystal_operator_v1,
)
from hcp_cp_gnd.lsbrp_metrics_v1 import (  # noqa: E402
    construct_checkpoint_energy_metric,
    construct_observation_metric,
    weighted_propagator_gain,
)
from hcp_cp_gnd.micromorphic import MicromorphicParameters  # noqa: E402
from hcp_cp_gnd.nonlinear_mechanism_switches_v1 import (  # noqa: E402
    evolve_model_to_shears,
    intervention_effect_decomposition,
    model_with_mechanism_disabled,
    switch_manifest,
)
from hcp_cp_gnd.periodic69_nonlinear_v1 import Periodic69CheckpointV1  # noqa: E402
from hcp_cp_gnd.sl3_chart import SL3LocalChart  # noqa: E402
from hcp_cp_gnd.spectral_export import (  # noqa: E402
    ContinuousSpectralPointModel,
    SpectralActiveState62,
    SpectralObserverState,
)
from tools.run_cp_ti_lsbrp_q1_dense_history_v1 import (  # noqa: E402
    CACHE,
    _sha256,
    load_verified_cache,
)


SCHEMA = "CP_TI_LSBRP_PERIODIC69_V1"
Q3_RESULT = ROOT / "05_results/cp_ti_lsbrp_q3_optimal_input_v1.json"
RESULT = ROOT / "05_results/cp_ti_lsbrp_periodic69_v1.json"
SUMMARY = ROOT / "05_results/cp_ti_lsbrp_periodic69_v1.md"
CELLS = 16
MODE = 1


def _spectral_model(model: Any, parameters: Any, card: dict[str, Any]) -> ContinuousSpectralPointModel:
    micro = MicromorphicParameters(
        reference_shear_modulus_Pa=parameters.reference_shear_modulus,
        nye_length_scale_m=1.0e-6,
        penalty_modulus_Pa=parameters.reference_shear_modulus,
        slip_gradient_length_m=0.25e-6,
        burgers_m=parameters.burgers,
    )
    return ContinuousSpectralPointModel(
        model,
        micro,
        conductivity_W_mK=card["thermal"]["conductivity_W_mK_at_300K"],
        parameter_provenance=card["status"],
    )


def _nonlinear_observable(
    spectral: ContinuousSpectralPointModel,
    storage: Any,
    F: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    chart = SL3LocalChart(
        storage.Fp,
        determinant_tolerance=spectral.base_model.parameters.determinant_tolerance,
    )
    active = SpectralActiveState62.from_storage(storage, chart)
    observer = SpectralObserverState.from_storage(storage)
    raw = spectral.evaluate_active_state(
        F,
        storage.temperature_K,
        storage.gamma_signed,
        np.zeros((18, 3)),
        np.zeros(3),
        direction,
        chart,
        active,
        observer,
    )
    return np.r_[raw.heat_source_W_m3, raw.active_rhs]


def _block_norms(vector: np.ndarray) -> dict[str, float]:
    blocks = {
        "heat": slice(0, 1),
        "theta_p": slice(1, 9),
        "rho_mobile": slice(9, 27),
        "rho_dipole": slice(27, 45),
        "gamma_signed": slice(45, 63),
    }
    return {name: float(np.linalg.norm(vector[part])) for name, part in blocks.items()}


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    started = time.perf_counter()
    q3 = json.loads(Q3_RESULT.read_text(encoding="utf-8"))
    payload = load_verified_cache()
    context = payload["contexts"]["factor64"]
    base_storage = context["storages"][0]
    base_point = context["points"][0]
    base_shear = float(context["shears"][0])
    base_time = float(context["times"][0])
    direction = np.asarray(q3["contract"]["direction_n"], dtype=float)
    direction /= np.linalg.norm(direction)
    wavenumber = float(q3["selection"]["selected_k_m_inv"][1])
    domain_length = 2.0 * np.pi * MODE / wavenumber
    scales = np.asarray(payload["coordinate_scales"], dtype=float)
    card = load_card()
    _, parameters, base_model = build_material_objects(card)
    spectral = _spectral_model(base_model, parameters, card)
    operator = assemble_dynamic_crystal_operator_v1(
        base_point,
        wavenumber_m_inv=wavenumber,
        direction_n=direction,
        admission=payload["admission"],
    )

    periodic = Periodic69CheckpointV1(
        spectral,
        simple_shear(base_shear),
        base_storage,
        direction,
        domain_length,
        CELLS,
    )
    base_residual = periodic.residual(periodic.base_field())
    numerical_symbol = periodic.fourier_mode_consistent_tangent(
        MODE,
        coordinate_scales=scales,
        relative_step=2.0e-6,
    )
    reference_symbol = (operator.generator_A * scales[None, :]) / scales[:, None]
    difference = numerical_symbol - reference_symbol
    tangent_relative_error = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(reference_symbol), np.finfo(float).tiny)
    )
    column_scales = np.maximum(
        np.linalg.norm(reference_symbol, axis=0),
        np.linalg.norm(numerical_symbol, axis=0),
    )
    active_columns = column_scales > np.finfo(float).tiny
    column_errors = np.zeros(N_GENERATOR)
    column_errors[active_columns] = (
        np.linalg.norm(difference[:, active_columns], axis=0)
        / column_scales[active_columns]
    )
    random = np.random.default_rng(20260822)
    direction_test = random.normal(size=N_GENERATOR) + 1j * random.normal(size=N_GENERATOR)
    self_action_error = periodic.tangent_action_error(
        numerical_symbol,
        MODE,
        direction_test,
        coordinate_scales=scales,
        relative_step=7.5e-7,
    )

    energy_metric_record: dict[str, Any]
    energy_metric = None
    try:
        energy_metric = construct_checkpoint_energy_metric(
            operator,
            base_point,
            reference_temperature_K=base_storage.temperature_K,
            coordinate_scales=scales,
            spectral_model=spectral,
            base_F_sample=simple_shear(base_shear),
        )
        energy_metric_record = {"constructed": True, **energy_metric.audit()}
    except ValueError as error:
        energy_metric_record = {
            "constructed": False,
            "error": str(error),
            "interpretation": "no diagonal regularization was added",
        }

    observed = np.arange(6, N_GENERATOR, dtype=int)
    H = np.zeros((observed.size, N_GENERATOR))
    H[np.arange(observed.size), observed] = 1.0
    numerical_covariance = np.diag(scales[observed] ** 2)
    observation_metric = construct_observation_metric(
        H,
        numerical_covariance,
        provenance=(
            "NUMERICAL_SURROGATE_ONLY: legacy coordinate scales treated as independent "
            "standard deviations; not DIC/thermometry/EBSD covariance"
        ),
        coordinate_scales=scales,
    )
    observation_record = {
        "numerical_surrogate": observation_metric.audit(),
        "experimental_metric_constructed": False,
        "blocking_data": (
            "registered observation matrix and measured DIC/thermometry/EBSD-TKD "
            "noise covariance are absent"
        ),
    }
    weighted_record: dict[str, Any] | None = None
    if energy_metric is not None:
        gain = weighted_propagator_gain(
            operator.propagator(1.0e-7),
            input_metric=energy_metric,
            output_metric=observation_metric,
        )
        weighted_record = {
            "elapsed_s": 1.0e-7,
            "gain": gain["gain"],
            "input_metric_rank": gain["input_metric_rank"],
            "output_metric_rank": gain["output_metric_rank"],
            "classification": "NUMERICAL_WO_SURROGATE__NOT_EXPERIMENTAL_GAIN",
        }

    baseline_observable = _nonlinear_observable(
        spectral, base_storage, simple_shear(base_shear), direction
    )
    switch_results: list[dict[str, Any]] = []
    for contract in switch_manifest():
        if not contract["supported_in_periodic69"]:
            switch_results.append(
                {
                    **contract,
                    "status": "EXCLUDED_REQUIRES_STATE_EXPANSION",
                }
            )
            continue
        disabled_model = model_with_mechanism_disabled(base_model, contract["name"])
        disabled_spectral = _spectral_model(disabled_model, parameters, card)
        fixed = _nonlinear_observable(
            disabled_spectral, base_storage, simple_shear(base_shear), direction
        )
        disabled_state = evolve_model_to_shears(
            disabled_model,
            [base_shear],
            shear_rate_s_inv=card["representative_loading"]["macroscopic_shear_rate_s"],
            shear_increment=2.0e-3,
        )[0]
        disabled_storage = local_state92_from_material_state(
            disabled_state, disabled_model, simple_shear(base_shear)
        )
        reevolved = _nonlinear_observable(
            disabled_spectral,
            disabled_storage,
            simple_shear(base_shear),
            direction,
        )
        effects = intervention_effect_decomposition(
            baseline_observable, fixed, reevolved
        )
        effect_scale = max(
            float(np.linalg.norm(effects["direct"])),
            float(np.linalg.norm(effects["base_mediated"])),
            float(np.linalg.norm(effects["total"])),
            1.0,
        )
        switch_results.append(
            {
                **contract,
                "status": "FIXED_AND_REEVOLVED_INTERVENTIONS_EXECUTED",
                "direct_block_l2": _block_norms(effects["direct"]),
                "base_mediated_block_l2": _block_norms(effects["base_mediated"]),
                "total_block_l2": _block_norms(effects["total"]),
                "effect_closure_max_abs": float(
                    np.max(np.abs(effects["closure_residual"]), initial=0.0)
                ),
                "effect_closure_relative_l2": float(
                    np.linalg.norm(effects["closure_residual"]) / effect_scale
                ),
                "reevolved_state_change": {
                    "temperature_K": float(
                        disabled_storage.temperature_K - base_storage.temperature_K
                    ),
                    "Fp_frobenius": float(
                        np.linalg.norm(disabled_storage.Fp - base_storage.Fp)
                    ),
                    "rho_mobile_relative_l2": float(
                        np.linalg.norm(
                            disabled_storage.rho_mobile_m2 - base_storage.rho_mobile_m2
                        )
                        / np.linalg.norm(base_storage.rho_mobile_m2)
                    ),
                    "rho_dipole_relative_l2": float(
                        np.linalg.norm(
                            disabled_storage.rho_dipole_m2 - base_storage.rho_dipole_m2
                        )
                        / np.linalg.norm(base_storage.rho_dipole_m2)
                    ),
                },
            }
        )

    gates = {
        "homogeneous_base_residual_below_1e_14": float(np.max(np.abs(base_residual))) < 1.0e-14,
        "independent_tangent_action_error_below_1e_4": self_action_error < 1.0e-4,
        "periodic_tangent_matches_registered_generator_below_2_percent": tangent_relative_error < 0.02,
        "all_supported_switch_effect_decompositions_close_below_1e_12": all(
            row.get("effect_closure_relative_l2", 0.0) < 1.0e-12
            for row in switch_results
            if row["supported_in_periodic69"]
        ),
        "energy_metric_constructed_without_regularization": energy_metric is not None,
        "experimental_observation_metric_available": False,
    }
    implementation_passed = all(
        gates[name]
        for name in (
            "homogeneous_base_residual_below_1e_14",
            "independent_tangent_action_error_below_1e_4",
            "periodic_tangent_matches_registered_generator_below_2_percent",
            "all_supported_switch_effect_decompositions_close_below_1e_12",
        )
    )
    report = {
        "schema": SCHEMA,
        "status": (
            "PERIODIC69_RESIDUAL_AND_TANGENT_GATE_PASSED__EXPERIMENTAL_WO_PENDING"
            if implementation_passed
            else "PERIODIC69_IMPLEMENTED__CONSISTENCY_GATE_FAILED"
        ),
        "classification": (
            "FROZEN_CHECKPOINT_NONLINEAR_69_STATE_CLOSURE__NOT_FULL_GAMMA_ABSOLUTE_HISTORY"
        ),
        "contract": {
            "base_time_s": base_time,
            "base_shear": base_shear,
            "wavenumber_m_inv": wavenumber,
            "direction_n": direction.tolist(),
            "cells": CELLS,
            "mode": MODE,
            "domain_length_m": domain_length,
            "state_count_per_cell": N_GENERATOR,
            "algebraic_micromorphic_states": 18,
            "passive_observer_closure": "Gamma_absolute and accumulated ledgers frozen",
        },
        "metrics": {
            "W_E": energy_metric_record,
            "W_O": observation_record,
            "weighted_gain_diagnostic": weighted_record,
        },
        "periodic_tangent": {
            "method": "central residual differentiation on cosine basis; complex Fourier extraction",
            "relative_step": 2.0e-6,
            "independent_action_relative_step": 7.5e-7,
            "base_residual_max_abs": float(np.max(np.abs(base_residual))),
            "relative_frobenius_error_vs_registered_generator": tangent_relative_error,
            "maximum_column_relative_error": float(np.max(column_errors)),
            "maximum_column_relative_error_index": int(np.argmax(column_errors)),
            "independent_mixed_phase_action_relative_error": self_action_error,
        },
        "mechanism_switch_mapping": {
            "edge_to_mechanism_one_to_one_claim": False,
            "effect_orientation": "baseline_on - mechanism_off",
            "identity": "total = fixed-base direct + re-evolved-base mediated",
            "results": switch_results,
        },
        "gates": gates,
        "provenance": {
            "base_cache": CACHE.relative_to(ROOT).as_posix(),
            "base_cache_sha256": _sha256(CACHE),
            "q3_result": Q3_RESULT.relative_to(ROOT).as_posix(),
            "q3_result_sha256": _sha256_text(Q3_RESULT),
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": _sha256_text(Path(__file__)),
            "residual_source": "src/hcp_cp_gnd/periodic69_nonlinear_v1.py",
            "residual_source_sha256": _sha256_text(
                ROOT / "src/hcp_cp_gnd/periodic69_nonlinear_v1.py"
            ),
        },
        "runtime": {
            "elapsed_s": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    SUMMARY.write_text(
        "\n".join(
            [
                "# LSBRP periodic 69-state audit V1",
                "",
                f"- Status: `{report['status']}`.",
                f"- W_E: constructed={energy_metric is not None}; W_O experimental covariance available=False.",
                f"- Tangent/generator relative error: {tangent_relative_error:.6e}.",
                f"- Independent tangent-action relative error: {self_action_error:.6e}.",
                "- Seven supported nonlinear switches were run in fixed-base and re-evolved-base forms; twinning remains outside the 69-state contract.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": gates,
                "tangent_relative_error": tangent_relative_error,
                "independent_action_error": self_action_error,
                "W_E": energy_metric_record,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
