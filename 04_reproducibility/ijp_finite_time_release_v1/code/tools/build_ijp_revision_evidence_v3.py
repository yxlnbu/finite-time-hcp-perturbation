"""Build the submission-stage V3 evidence requested for the finite-time HCP paper.

This script adds five auditable layers without changing the constitutive equations:

1. direction--wavenumber re-optimization for seven input/output selectors under
   the baseline metric and four pre-registered adverse metric variants;
2. a same-selector, same-metric piecewise-frozen propagator baseline whose
   macro-window refinement converges to the time-ordered reference;
3. singular-value-gap diagnostics with a pre-registered degeneracy threshold;
4. three loading transfers and branch-local sensitivities of two dislocation
   and two gradient parameters; and
5. a bounded comparison with the synchronized stress/temperature landmarks of
   Guo et al., Phys. Rev. Lett. 122 (2019) 015503 for Grade-II CP titanium.

The sphere--wavenumber search remains an anchor-assisted audit.  It is not
labelled as a proof of the global optimum.  Material transfers remain research
baselines rather than batch-specific validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import cos, log, pi, sin
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V01_SRC = ROOT.parent / "HCP_CP_v0.1/src"
for location in (ROOT, SRC, V01_SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from hcp_cp.model import HCPMaterialPoint, orientation_from_bunge  # noqa: E402
from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    build_material_objects,
    heat_capacity_J_kgK,
    load_card,
    local_state92_from_material_state,
    simple_shear,
)
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
)
from hcp_cp_gnd.micromorphic import MicromorphicParameters  # noqa: E402
from hcp_cp_gnd.spectral_export import ContinuousSpectralPointModel  # noqa: E402
from tools.build_ijp_operator_consistency_v2 import (  # noqa: E402
    CONSTITUTIVE,
    FULL,
    log_gain,
    norm_variants,
    rescale_propagator,
    scaled_generator,
    selectors,
)
from tools.run_cp_ti_finite_time_propagator_v2 import (  # noqa: E402
    _admission,
    _coordinate_scales,
)


RESULT = ROOT / "05_results/ijp_revision_evidence_v3.json"
ARRAYS = ROOT / "05_results/ijp_revision_evidence_v3_arrays.npz"
TABLE_DIR = ROOT / "05_results/ijp_revision_evidence_v3"
FIGURE_DIR = ROOT / "06_manuscript/ijp_spectral_hcp/figures"
FIG_REOPT = FIGURE_DIR / "fig10_reoptimized_selector_audit"
FIG_TRANSFER = FIGURE_DIR / "fig11_loading_parameter_validation"
FIG_MATRIX = FIGURE_DIR / "figS01_full_norm_selector_matrix"

STRENGTHENING_ARRAYS = ROOT / "05_results/ijp_strengthening_evidence_v1_arrays.npz"
FIXED_MATRIX = ROOT / "05_results/ijp_operator_consistency_v2/norm_selector_winners.csv"
PRL_LOCAL = ROOT.parent / "reference/PhysRevLett.122.015503.pdf"
PRL_DIGITIZATION = (
    ROOT / "04_reproducibility/literature_digitization/guo2019_prl/landmarks.csv"
)
PRL_EXPECTED_SHA256 = "fd290d4347b8e8576b5d0ce2c8ba2f7498d54b0f8915c33c505238f08e6e4a6f"

BASE_SHEARS = np.asarray([0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1.40])
K_BOUNDS = (3.0e2, 3.0e5)
BOUNDS = ((0.0, 0.5 * pi), (-pi, pi), (log(K_BOUNDS[0]), log(K_BOUNDS[1])))
TINY = np.finfo(float).tiny
SEARCH_QMC_COUNT = 128
TRANSFER_QMC_COUNT = 64
SEARCH_SUBSTEPS = 2
VERIFY_SUBSTEPS = 4
DEGENERACY_RELATIVE_GAP = 0.05
DEGENERACY_RATIO_LIMIT = 1.0 / (1.0 - DEGENERACY_RELATIVE_GAP)

# These four variants were pre-registered from the fixed-branch 11 x 7 screen:
# they stress mechanical input scaling, temperature observation, dislocation
# coordinates, and the complete constitutive block, respectively.
REOPTIMIZED_NORMS = (
    "baseline",
    "mechanical_x0.5",
    "thermal_x0.5",
    "dislocation_x2",
    "all_constitutive_x2",
)

SELECTOR_LABELS = {
    "full_to_full": "full/full",
    "constitutive_to_constitutive": "const./const.",
    "full_to_constitutive": "full→const.",
    "constitutive_to_full": "const.→full",
    "mechanical_to_mechanical": "mech./mech.",
    "mechanical_to_constitutive": "mech.→const.",
    "constitutive_to_temperature": "const.→T",
}

# These selectors measure a constitutive/thermal localization response and are
# expected to possess an interior finite-band optimum on the registered k
# interval. Full/full and mechanical/mechanical additionally see reversible
# high-frequency elastic waves; a boundary maximum for those questions is
# classified, not silently promoted to a localization wavelength.
LOCALIZATION_INTERIOR_SELECTORS = {
    "constitutive_to_constitutive",
    "full_to_constitutive",
    "constitutive_to_full",
    "mechanical_to_constitutive",
    "constitutive_to_temperature",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refined_shears(factor: int) -> np.ndarray:
    require(factor >= 1, "refinement factor must be positive")
    values = [float(BASE_SHEARS[0])]
    for left, right in zip(BASE_SHEARS[:-1], BASE_SHEARS[1:], strict=True):
        values.extend(
            float(left + (right - left) * index / factor)
            for index in range(1, factor + 1)
        )
    return np.asarray(values)


def direction_from_coordinates(value: Any) -> np.ndarray:
    theta, phi = float(value[0]), float(value[1])
    direction = np.asarray(
        [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], dtype=float
    )
    return direction / np.linalg.norm(direction)


def coordinates_from_direction(direction: Any, wavenumber_m_inv: float) -> np.ndarray:
    normal = np.asarray(direction, dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[2] < -1.0e-14:
        normal = -normal
    return np.asarray(
        [
            np.arccos(np.clip(normal[2], -1.0, 1.0)),
            np.arctan2(normal[1], normal[0]),
            np.log(float(wavenumber_m_inv)),
        ]
    )


def direction_angle_degrees(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def qmc_coordinates(count: int, seed: int) -> np.ndarray:
    require(count > 0 and count & (count - 1) == 0, "Sobol count must be a power of two")
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    z = unit[:, 0]
    theta = np.arccos(z)
    phi = -pi + 2.0 * pi * unit[:, 1]
    log_k = BOUNDS[2][0] + unit[:, 2] * (BOUNDS[2][1] - BOUNDS[2][0])
    return np.c_[theta, phi, log_k]


def registered_anchors() -> list[np.ndarray]:
    require(STRENGTHENING_ARRAYS.is_file(), "missing registered branch archive")
    output: list[np.ndarray] = []
    with np.load(STRENGTHENING_ARRAYS) as source:
        for branch in ("onset_x", "onset_y", "terminal_y"):
            output.append(
                coordinates_from_direction(
                    source[f"baseline_{branch}_direction_n"],
                    float(source[f"baseline_{branch}_k_m_inv"][0]),
                )
            )
    for normal in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        for wavenumber in np.geomspace(K_BOUNDS[0], K_BOUNDS[1], 5):
            output.append(coordinates_from_direction(normal, float(wavenumber)))
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for value in output:
        unique[tuple(np.round(value, 10))] = value
    return list(unique.values())


@dataclass
class SpectralContextV3:
    case_id: str
    shears: np.ndarray
    times_s: np.ndarray
    points: list[Any]
    storages: list[Any]
    records: list[dict[str, Any]]
    scales: np.ndarray
    parameter_summary: dict[str, Any]

    def indices_for_factor(self, factor: int) -> np.ndarray:
        targets = refined_shears(factor)
        indices = []
        for target in targets:
            match = np.flatnonzero(np.isclose(self.shears, target, rtol=0.0, atol=1.0e-14))
            require(match.size == 1, f"{self.case_id}: target shear {target} absent")
            indices.append(int(match[0]))
        return np.asarray(indices, dtype=int)

    def generators(self, x: Any, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        raw = np.asarray(x, dtype=float)
        direction = direction_from_coordinates(raw)
        wavenumber = float(np.exp(raw[2]))
        matrices = []
        admission = _admission()
        for index in indices:
            operator = assemble_dynamic_crystal_operator_v1(
                self.points[int(index)],
                wavenumber_m_inv=wavenumber,
                direction_n=direction,
                admission=admission,
            )
            matrices.append(operator.generator_A)
        return np.asarray(matrices), direction, wavenumber

    def propagator(
        self,
        x: Any,
        indices: np.ndarray,
        *,
        substeps: int,
        return_generators: bool = False,
    ) -> dict[str, Any]:
        generators, direction, wavenumber = self.generators(x, indices)
        times = self.times_s[indices]
        scaled = [scaled_generator(value, self.scales) for value in generators]
        phi = np.eye(69, dtype=np.complex128)
        for index in range(len(times) - 1):
            dt = float(times[index + 1] - times[index]) / substeps
            for substep in range(substeps):
                fraction = (substep + 0.5) / substeps
                midpoint = (
                    (1.0 - fraction) * scaled[index]
                    + fraction * scaled[index + 1]
                )
                phi = expm(midpoint * dt) @ phi
        result = {
            "propagator": phi,
            "direction_n": direction,
            "k_m_inv": wavenumber,
            "times_s": times,
        }
        if return_generators:
            result["generators"] = generators
        return result


def build_case_models(
    case_id: str,
    *,
    shear_rate_s_inv: float = 1.0e4,
    initial_temperature_K: float = 293.15,
    rho_mobile_factor: float = 1.0,
    mean_free_path_factor: float = 1.0,
    nye_length_factor: float = 1.0,
    slip_gradient_length_factor: float = 1.0,
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Build the exact fixed-beta/passive-ledger models used by the paper."""
    card = load_card()
    systems, parameters, seed_model = build_material_objects(card)
    parameters = replace(
        parameters,
        # The registered Shomate fit starts at 298 K.  The 293.15 K baseline
        # therefore retains the documented 298.15 K value, as in the material
        # card, while the elevated-temperature case evaluates the fit at T0.
        # T_ref itself remains the material reference temperature, so a warm
        # initial state retains its physical thermal-softening offset.
        heat_capacity=heat_capacity_J_kgK(
            max(float(initial_temperature_K), 298.15), card
        ),
        rho_mobile_0=np.asarray(parameters.rho_mobile_0) * float(rho_mobile_factor),
        mean_free_path_coefficient=(
            float(parameters.mean_free_path_coefficient) * float(mean_free_path_factor)
        ),
    )
    orientation = orientation_from_bunge(*card["representative_loading"]["orientation_bunge_deg"])
    model = HCPMaterialPoint(
        systems,
        parameters,
        orientation,
        seed_model.switches,
    )
    micro = MicromorphicParameters(
        reference_shear_modulus_Pa=parameters.reference_shear_modulus,
        nye_length_scale_m=1.0e-6 * float(nye_length_factor),
        penalty_modulus_Pa=parameters.reference_shear_modulus,
        slip_gradient_length_m=0.25e-6 * float(slip_gradient_length_factor),
        burgers_m=parameters.burgers,
    )
    spectral = ContinuousSpectralPointModel(
        model,
        micro,
        conductivity_W_mK=card["thermal"]["conductivity_W_mK_at_300K"],
        parameter_provenance=(
            f"{card['status']}|V3:{case_id}|rate={shear_rate_s_inv:g}|"
            f"T0={initial_temperature_K:g}|rho_m={rho_mobile_factor:g}|"
            f"KLambda={mean_free_path_factor:g}|ellN={nye_length_factor:g}|"
            f"ellchi={slip_gradient_length_factor:g}"
        ),
    )
    return card, parameters, model, spectral


def build_context(
    case_id: str,
    *,
    factor: int,
    shear_rate_s_inv: float = 1.0e4,
    initial_temperature_K: float = 293.15,
    rho_mobile_factor: float = 1.0,
    mean_free_path_factor: float = 1.0,
    nye_length_factor: float = 1.0,
    slip_gradient_length_factor: float = 1.0,
) -> SpectralContextV3:
    """Build a paper-consistent base path and continuous spectral checkpoints."""

    card, parameters, model, spectral = build_case_models(
        case_id,
        shear_rate_s_inv=shear_rate_s_inv,
        initial_temperature_K=initial_temperature_K,
        rho_mobile_factor=rho_mobile_factor,
        mean_free_path_factor=mean_free_path_factor,
        nye_length_factor=nye_length_factor,
        slip_gradient_length_factor=slip_gradient_length_factor,
    )
    shears = refined_shears(factor)
    state = model.initial_state()
    if not np.isclose(initial_temperature_K, parameters.T_ref, rtol=0.0, atol=1.0e-12):
        state = replace(state, temperature=float(initial_temperature_K))
    gamma = 0.0
    states = []
    records: list[dict[str, Any]] = []
    maximum_energy_error = 0.0
    maximum_partition_error = 0.0
    for target in shears:
        last_result = None
        while gamma < target - 4.0 * np.finfo(float).eps:
            next_gamma = min(gamma + 2.0e-3, float(target))
            last_result = model.advance(
                simple_shear(gamma),
                simple_shear(next_gamma),
                state,
                (next_gamma - gamma) / float(shear_rate_s_inv),
            )
            state = last_result.state
            gamma = next_gamma
            maximum_energy_error = max(
                maximum_energy_error, float(last_result.energy_balance_relative_error)
            )
            maximum_partition_error = max(
                maximum_partition_error, float(last_result.work_partition_relative_error)
            )
        response = model.evaluate(simple_shear(float(target)), state)
        records.append(
            {
                "case_id": case_id,
                "shear": float(target),
                "time_s": float(target / shear_rate_s_inv),
                "cauchy_shear_Pa": float(response.cauchy[0, 1]),
                "first_piola_shear_Pa": float(response.first_piola[0, 1]),
                "temperature_K": float(state.temperature),
                "temperature_rise_K": float(state.temperature - initial_temperature_K),
                "plastic_work_J_m3": float(state.plastic_work_density),
                "generated_heat_J_m3": float(state.heat_density),
                "stored_energy_J_m3": float(state.stored_energy_density),
            }
        )
        states.append(state)

    storages = [
        local_state92_from_material_state(state_value, model, simple_shear(float(shear)))
        for state_value, shear in zip(states, shears, strict=True)
    ]
    require(
        STRENGTHENING_ARRAYS.is_file(),
        "the published baseline coordinate metric is required",
    )
    with np.load(STRENGTHENING_ARRAYS) as published:
        locked_scales = np.asarray(
            published["baseline_coordinate_scales"], dtype=float
        ).copy()
    reference = np.asarray([1.0, 2.0, 3.0])
    reference /= np.linalg.norm(reference)
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
        for storage, shear in zip(storages, shears, strict=True)
    ]
    return SpectralContextV3(
        case_id=case_id,
        shears=shears,
        times_s=np.asarray([row["time_s"] for row in records]),
        points=points,
        storages=storages,
        records=records,
        scales=locked_scales,
        parameter_summary={
            "shear_rate_s_inv": float(shear_rate_s_inv),
            "initial_temperature_K": float(initial_temperature_K),
            "rho_mobile_factor": float(rho_mobile_factor),
            "mean_free_path_factor": float(mean_free_path_factor),
            "nye_length_factor": float(nye_length_factor),
            "slip_gradient_length_factor": float(slip_gradient_length_factor),
            "checkpoint_factor": int(factor),
            "checkpoint_count": int(len(shears)),
            "maximum_energy_balance_relative_error": maximum_energy_error,
            "maximum_work_partition_relative_error": maximum_partition_error,
        },
    )


def objective_score(
    baseline_phi: np.ndarray,
    baseline_scales: np.ndarray,
    active_scales: np.ndarray,
    output_indices: np.ndarray,
    input_indices: np.ndarray,
) -> float:
    active_phi = rescale_propagator(baseline_phi, baseline_scales, active_scales)
    return log_gain(active_phi, output_indices, input_indices)


def evaluate_pool(
    context: SpectralContextV3,
    indices: np.ndarray,
    coordinates: np.ndarray,
    *,
    substeps: int,
) -> list[np.ndarray]:
    propagators = []
    for count, value in enumerate(coordinates, start=1):
        propagators.append(context.propagator(value, indices, substeps=substeps)["propagator"])
        if count % 32 == 0:
            print(
                json.dumps(
                    {
                        "stage": "shared_pool",
                        "case": context.case_id,
                        "completed": count,
                        "total": len(coordinates),
                    }
                ),
                flush=True,
            )
    return propagators


def local_refinement(
    context: SpectralContextV3,
    indices: np.ndarray,
    start: np.ndarray,
    active_scales: np.ndarray,
    output_indices: np.ndarray,
    input_indices: np.ndarray,
    *,
    run_id: str,
    substeps: int,
    maximum_iterations: int = 35,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], tuple[float, dict[str, Any]]] = {}

    def evaluate(raw_value: Any) -> tuple[float, dict[str, Any]]:
        raw = np.asarray(raw_value, dtype=float)
        key = tuple(np.round(raw, 10))
        if key not in cache:
            propagated = context.propagator(raw, indices, substeps=substeps)
            score = objective_score(
                propagated["propagator"],
                context.scales,
                active_scales,
                output_indices,
                input_indices,
            )
            cache[key] = (score, propagated)
            direction = propagated["direction_n"]
            trace.append(
                {
                    "run_id": run_id,
                    "evaluation": len(trace) + 1,
                    "theta_rad": float(raw[0]),
                    "phi_rad": float(raw[1]),
                    "log_k": float(raw[2]),
                    "n1": float(direction[0]),
                    "n2": float(direction[1]),
                    "n3": float(direction[2]),
                    "k_m_inv": float(propagated["k_m_inv"]),
                    "log_gain": float(score),
                }
            )
        return cache[key]

    start_score, start_value = evaluate(start)
    result = minimize(
        lambda value: -evaluate(value)[0],
        np.asarray(start, dtype=float),
        method="Powell",
        bounds=BOUNDS,
        options={
            "xtol": 2.0e-4,
            "ftol": 2.0e-7,
            "maxiter": maximum_iterations,
        },
    )
    optimized_score, optimized = evaluate(result.x)
    if start_score >= optimized_score:
        retained_score, retained = start_score, start_value
        retained_start = True
    else:
        retained_score, retained = optimized_score, optimized
        retained_start = False
    return {
        "run_id": run_id,
        "log_gain": float(retained_score),
        "gain": float(np.exp(retained_score)),
        "direction_n": retained["direction_n"].tolist(),
        "k_m_inv": float(retained["k_m_inv"]),
        "coordinates": coordinates_from_direction(
            retained["direction_n"], retained["k_m_inv"]
        ).tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "retained_start_instead_of_optimizer": retained_start,
    }, trace


def singular_diagnostics(
    propagator: np.ndarray,
    output_indices: np.ndarray,
    input_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    observed = propagator[np.ix_(output_indices, input_indices)]
    left, singular, right_h = np.linalg.svd(observed, full_matrices=False)
    if singular.size >= 2:
        ratio = float(singular[0] / max(singular[1], TINY))
        relative_gap = float((singular[0] - singular[1]) / max(singular[0], TINY))
        degenerate = bool(relative_gap < DEGENERACY_RELATIVE_GAP)
    else:
        ratio = None
        relative_gap = None
        degenerate = None
    return {
        "sigma_1": float(singular[0]),
        "sigma_2": None if singular.size < 2 else float(singular[1]),
        "sigma_1_over_sigma_2": ratio,
        "relative_singular_gap": relative_gap,
        "degeneracy_threshold_relative_gap": DEGENERACY_RELATIVE_GAP,
        "near_degenerate": degenerate,
        "individual_vector_interpretation_authorized": (
            None if degenerate is None else not degenerate
        ),
        "rank_one_observable": bool(singular.size == 1),
    }, {
        "singular_values": singular,
        "input_singular_vector": right_h[0].conj(),
        "output_singular_vector": left[:, 0],
    }


def search_selector_norms(
    search_context: SpectralContextV3,
    reference_context: SpectralContextV3,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Re-optimize all seven selectors for five pre-registered norm cases."""

    selector_map = selectors()
    search_norms_all = norm_variants(search_context.scales)
    reference_norms_all = norm_variants(reference_context.scales)
    search_norms = {key: search_norms_all[key] for key in REOPTIMIZED_NORMS}
    search_grid = search_context.indices_for_factor(2)
    reference_grid = np.arange(len(reference_context.shears), dtype=int)
    onset_search = int(np.argmin(np.abs(search_context.times_s[search_grid] - 1.46e-6)))
    onset_reference = int(np.argmin(np.abs(reference_context.times_s - 1.46e-6)))
    horizons = {
        "onset": {
            "search_indices": search_grid[: onset_search + 1],
            "reference_indices": reference_grid[: onset_reference + 1],
        },
        "terminal": {
            "search_indices": search_grid,
            "reference_indices": reference_grid,
        },
    }
    pool_coordinates = np.vstack(
        (qmc_coordinates(SEARCH_QMC_COUNT, 20260829), np.asarray(registered_anchors()))
    )
    records: dict[str, Any] = {}
    released: dict[str, np.ndarray] = {}
    all_traces: list[dict[str, Any]] = []

    for horizon_id, horizon in horizons.items():
        print(json.dumps({"stage": "selector_pool", "horizon": horizon_id}), flush=True)
        pool_phi = evaluate_pool(
            search_context,
            horizon["search_indices"],
            pool_coordinates,
            substeps=SEARCH_SUBSTEPS,
        )
        for norm_id, active_scales in search_norms.items():
            for selector_id, (output_indices, input_indices) in selector_map.items():
                scores = np.asarray(
                    [
                        objective_score(
                            phi,
                            search_context.scales,
                            active_scales,
                            output_indices,
                            input_indices,
                        )
                        for phi in pool_phi
                    ]
                )
                order = np.argsort(scores)[::-1]
                starts = [pool_coordinates[int(order[0])]]
                for candidate_index in order[1:]:
                    candidate = pool_coordinates[int(candidate_index)]
                    if (
                        direction_angle_degrees(
                            direction_from_coordinates(candidate),
                            direction_from_coordinates(starts[0]),
                        )
                        >= 3.0
                        or abs(float(candidate[2] - starts[0][2])) >= 0.10
                    ):
                        if scores[order[0]] - scores[candidate_index] <= 0.02:
                            starts.append(candidate)
                        break
                runs = []
                for start_index, start in enumerate(starts, start=1):
                    run_id = f"{horizon_id}__{norm_id}__{selector_id}__start{start_index}"
                    run, trace = local_refinement(
                        search_context,
                        horizon["search_indices"],
                        start,
                        active_scales,
                        output_indices,
                        input_indices,
                        run_id=run_id,
                        substeps=SEARCH_SUBSTEPS,
                    )
                    runs.append(run)
                    all_traces.extend(trace)
                winner = max(runs, key=lambda item: item["log_gain"])

                # Search and local refinement use the registered reduced-order
                # context. Every retained point is then evaluated once on the
                # 129-state/four-substep reference. This cleanly separates basin
                # discovery from the high-resolution acceptance calculation.
                winner_coordinates = np.asarray(winner["coordinates"], dtype=float)
                reference_seed = winner

                verified = reference_context.propagator(
                    winner_coordinates,
                    horizon["reference_indices"],
                    substeps=VERIFY_SUBSTEPS,
                    return_generators=(norm_id == "baseline"),
                )
                active_reference_phi = rescale_propagator(
                    verified["propagator"],
                    reference_context.scales,
                    reference_norms_all[norm_id],
                )
                verified_log_gain = log_gain(
                    active_reference_phi, output_indices, input_indices
                )
                singular, vectors = singular_diagnostics(
                    active_reference_phi, output_indices, input_indices
                )
                wavenumber_interior = bool(
                    verified["k_m_inv"] > 1.01 * K_BOUNDS[0]
                    and verified["k_m_inv"] < 0.99 * K_BOUNDS[1]
                )
                boundary_extension: list[dict[str, Any]] = []
                if not wavenumber_interior:
                    for multiplier in (10.0, 100.0):
                        extension_coordinates = coordinates_from_direction(
                            verified["direction_n"],
                            min(float(verified["k_m_inv"] * multiplier), 3.0e7),
                        )
                        extension = search_context.propagator(
                            extension_coordinates,
                            horizon["search_indices"],
                            substeps=SEARCH_SUBSTEPS,
                        )
                        extension_log_gain = objective_score(
                            extension["propagator"],
                            search_context.scales,
                            search_norms[norm_id],
                            output_indices,
                            input_indices,
                        )
                        boundary_extension.append(
                            {
                                "k_m_inv": float(extension["k_m_inv"]),
                                "log_gain": float(extension_log_gain),
                            }
                        )
                key = f"{horizon_id}__{norm_id}__{selector_id}"
                records[key] = {
                    "horizon_id": horizon_id,
                    "time_s": float(verified["times_s"][-1]),
                    "norm_id": norm_id,
                    "selector_id": selector_id,
                    "input_dimension": int(len(input_indices)),
                    "output_dimension": int(len(output_indices)),
                    "search_qmc_count": SEARCH_QMC_COUNT,
                    "search_pool_best_log_gain": float(scores[order[0]]),
                    "search_local_runs": runs,
                    "reference_seed": reference_seed,
                    "direction_n": verified["direction_n"].tolist(),
                    "k_m_inv": float(verified["k_m_inv"]),
                    "reference_log_gain": float(verified_log_gain),
                    "reference_gain": float(np.exp(verified_log_gain)),
                    "wavenumber_interior": wavenumber_interior,
                    "boundary_classification": (
                        "interior_finite_band_candidate"
                        if wavenumber_interior
                        else "compact_domain_boundary_branch_not_a_finite_localization_wavelength"
                    ),
                    "fixed_direction_extended_k_diagnostic": boundary_extension,
                    "singular_diagnostics": singular,
                }
                prefix = key.replace(".", "p")
                released[f"{prefix}__propagator"] = active_reference_phi
                released[f"{prefix}__singular_values"] = vectors["singular_values"]
                released[f"{prefix}__input_singular_vector"] = vectors[
                    "input_singular_vector"
                ]
                released[f"{prefix}__output_singular_vector"] = vectors[
                    "output_singular_vector"
                ]
                if norm_id == "baseline":
                    released[f"{prefix}__generators"] = verified["generators"]
                print(
                    json.dumps(
                        {
                            "stage": "selector_winner",
                            "key": key,
                            "log_gain": verified_log_gain,
                            "k": verified["k_m_inv"],
                        }
                    ),
                    flush=True,
                )

    gates = {
        "all_7_selectors_reoptimized": len(
            {record["selector_id"] for record in records.values()}
        )
        == 7,
        "five_registered_norms_reoptimized": len(
            {record["norm_id"] for record in records.values()}
        )
        == len(REOPTIMIZED_NORMS),
        "both_horizons_reoptimized": {
            record["horizon_id"] for record in records.values()
        }
        == {"onset", "terminal"},
        "all_localization_selector_wavenumbers_interior": all(
            record["wavenumber_interior"]
            for record in records.values()
            if record["selector_id"] in LOCALIZATION_INTERIOR_SELECTORS
        ),
        "all_boundary_cases_explicitly_classified": all(
            record["wavenumber_interior"]
            or (
                record["boundary_classification"]
                == "compact_domain_boundary_branch_not_a_finite_localization_wavelength"
                and len(record["fixed_direction_extended_k_diagnostic"]) == 2
            )
            for record in records.values()
        ),
        "formal_degeneracy_rule_applied": all(
            record["singular_diagnostics"]["degeneracy_threshold_relative_gap"]
            == DEGENERACY_RELATIVE_GAP
            for record in records.values()
        ),
    }
    return {
        "status": "REOPTIMIZATION_AUDIT_PASS" if all(gates.values()) else "REOPTIMIZATION_GATES_OPEN",
        "audit_boundary": (
            "Shared scrambled-Sobol reconnaissance plus registered anchors and bounded "
            "Powell refinement; this retains and challenges basins but is not a proof "
            "of the global optimum."
        ),
        "pre_registered_adverse_norms": list(REOPTIMIZED_NORMS[1:]),
        "degeneracy_rule": {
            "relative_gap_definition": "(sigma1-sigma2)/sigma1",
            "near_degenerate_when_below": DEGENERACY_RELATIVE_GAP,
            "equivalent_sigma1_over_sigma2_limit": DEGENERACY_RATIO_LIMIT,
            "interpretation": (
                "When triggered, only the leading singular subspace may be interpreted; "
                "a single singular vector is not assigned a mechanism."
            ),
        },
        "records": records,
        "gates": gates,
    }, released, all_traces


def interpolate_matrix(times: np.ndarray, matrices: np.ndarray, target: float) -> np.ndarray:
    if target <= times[0]:
        return matrices[0]
    if target >= times[-1]:
        return matrices[-1]
    upper = int(np.searchsorted(times, target, side="right"))
    lower = upper - 1
    fraction = float((target - times[lower]) / (times[upper] - times[lower]))
    return (1.0 - fraction) * matrices[lower] + fraction * matrices[upper]


def piecewise_frozen_baseline(
    reoptimization: dict[str, Any], released: dict[str, np.ndarray]
) -> dict[str, Any]:
    rows = []
    refinements_per_reference_interval = (1, 2, 4, 8, 16)
    selector_map = selectors()
    for key, record in reoptimization["records"].items():
        if record["norm_id"] != "baseline":
            continue
        prefix = key.replace(".", "p")
        generators = released[f"{prefix}__generators"]
        reference_phi = released[f"{prefix}__propagator"]
        # Generator times are the corresponding 129-state prefix. They start at
        # the first admitted 0.005-shear checkpoint, just as the reference map.
        if record["horizon_id"] == "onset":
            full_times = refined_shears(16) / 1.0e4
            times = full_times[: len(generators)]
        else:
            times = refined_shears(16) / 1.0e4
        output_indices, input_indices = selector_map[record["selector_id"]]
        reference_log_gain = log_gain(reference_phi, output_indices, input_indices)
        scaled = np.asarray([scaled_generator(value, np.ones(69)) for value in generators])
        # generators in the archive are dimensional; the released propagator is
        # baseline scaled. Reconstruct the baseline-scaled generator from the
        # context scales encoded by the similarity relation using the first map's
        # singular-vector archive is impossible here, so the caller injects the
        # scales in a dedicated array below.
        scales = released["reference_coordinate_scales"]
        scaled = np.asarray([scaled_generator(value, scales) for value in generators])
        for refinement in refinements_per_reference_interval:
            phi = np.eye(69, dtype=np.complex128)
            for interval in range(len(times) - 1):
                dt = float(times[interval + 1] - times[interval]) / refinement
                for subinterval in range(refinement):
                    fraction = subinterval / refinement
                    frozen = (
                        (1.0 - fraction) * scaled[interval]
                        + fraction * scaled[interval + 1]
                    )
                    phi = expm(frozen * dt) @ phi
            value = log_gain(phi, output_indices, input_indices)
            rows.append(
                {
                    "horizon_id": record["horizon_id"],
                    "selector_id": record["selector_id"],
                    "refinement_per_reference_interval": refinement,
                    "segment_count": int((len(times) - 1) * refinement),
                    "piecewise_frozen_log_gain": float(value),
                    "time_ordered_reference_log_gain": float(reference_log_gain),
                    "absolute_log_gain_error": float(abs(value - reference_log_gain)),
                    "same_input_selector": True,
                    "same_output_selector": True,
                    "same_norm": True,
                    "freeze_rule": (
                        "left_endpoint_generator_in_each_subdivision_of_every_"
                        "registered_nonuniform_reference_interval"
                    ),
                }
            )
    write_csv(TABLE_DIR / "piecewise_frozen_same_selector.csv", rows)
    terminal_errors = [
        row["absolute_log_gain_error"]
        for row in rows
        if row["refinement_per_reference_interval"] == 16
    ]
    return {
        "definition": (
            "Every registered nonuniform base-state interval is divided into r "
            "subintervals and A is frozen at each subinterval's left endpoint; "
            "P_in, P_out and the metric are identical to the time-ordered reference."
        ),
        "refinements_per_reference_interval": list(
            refinements_per_reference_interval
        ),
        "row_count": len(rows),
        "maximum_refinement16_log_gain_error": float(max(terminal_errors)),
        "gate_refinement16_error_below_0p10": bool(max(terminal_errors) <= 0.10),
        "table": (TABLE_DIR / "piecewise_frozen_same_selector.csv").relative_to(ROOT).as_posix(),
    }


def branch_local_transfer_search(
    context: SpectralContextV3,
    starts_by_horizon: dict[str, np.ndarray],
    *,
    qmc_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selector_id = "constitutive_to_constitutive"
    output_indices, input_indices = selectors()[selector_id]
    active_scales = context.scales
    search_grid = context.indices_for_factor(2)
    onset_index = int(np.argmin(np.abs(context.times_s[search_grid] - 1.46e-6 * 1.0e4 / context.parameter_summary["shear_rate_s_inv"])))
    horizons = {
        "onset": search_grid[: onset_index + 1],
        "terminal": search_grid,
    }
    output = {}
    traces: list[dict[str, Any]] = []
    qmc_values = qmc_coordinates(qmc_count, 20260901 + int(context.parameter_summary["shear_rate_s_inv"]))
    anchors = np.asarray(registered_anchors())
    pool = np.vstack((qmc_values, anchors))
    for horizon_id, indices in horizons.items():
        pool_phi = evaluate_pool(context, indices, pool, substeps=SEARCH_SUBSTEPS)
        scores = np.asarray(
            [log_gain(value, output_indices, input_indices) for value in pool_phi]
        )
        candidate = pool[int(np.argmax(scores))]
        continuation = starts_by_horizon[horizon_id]
        starts = [candidate, continuation]
        runs = []
        for start_index, start in enumerate(starts, start=1):
            run, trace = local_refinement(
                context,
                indices,
                start,
                active_scales,
                output_indices,
                input_indices,
                run_id=f"{context.case_id}__{horizon_id}__start{start_index}",
                substeps=SEARCH_SUBSTEPS,
                maximum_iterations=30,
            )
            runs.append(run)
            traces.extend(trace)
        winner = max(runs, key=lambda value: value["log_gain"])
        # Verify once on every factor-4 checkpoint with four midpoint substeps;
        # the optimization itself remains the lower-order basin search above.
        target_time = float(context.times_s[indices[-1]])
        verify_end = int(np.argmin(np.abs(context.times_s - target_time)))
        verify_indices = np.arange(verify_end + 1, dtype=int)
        verified = context.propagator(
            np.asarray(winner["coordinates"]),
            verify_indices,
            substeps=VERIFY_SUBSTEPS,
        )
        verified_log_gain = log_gain(
            verified["propagator"], output_indices, input_indices
        )
        output[horizon_id] = {
            **winner,
            "search_log_gain": winner["log_gain"],
            "log_gain": float(verified_log_gain),
            "gain": float(np.exp(verified_log_gain)),
            "time_s": float(context.times_s[verify_indices[-1]]),
            "shear": float(context.shears[verify_indices[-1]]),
            "qmc_count": qmc_count,
            "audit_boundary": "shared QMC plus baseline-continuation anchors; not a global certificate",
        }
    return output, traces


def loading_transfers(
    baseline_context: SpectralContextV3,
    reoptimization: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_starts = {
        horizon: np.asarray(
            reoptimization["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]["direction_n"]
        )
        for horizon in ("onset", "terminal")
    }
    starts = {
        horizon: coordinates_from_direction(
            baseline_starts[horizon],
            reoptimization["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]["k_m_inv"],
        )
        for horizon in ("onset", "terminal")
    }
    cases = (
        ("rate_5e3_T293", 5.0e3, 293.15),
        ("rate_1p3e4_T293", 1.3e4, 293.15),
        ("rate_1e4_T373", 1.0e4, 373.15),
    )
    contexts = {"baseline_rate_1e4_T293": baseline_context}
    records: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    base_rows = list(baseline_context.records)
    for case_id, rate, temperature in cases:
        print(json.dumps({"stage": "build_loading_case", "case": case_id}), flush=True)
        context = build_context(
            case_id,
            factor=4,
            shear_rate_s_inv=rate,
            initial_temperature_K=temperature,
        )
        contexts[case_id] = context
        selected, case_traces = branch_local_transfer_search(
            context, starts, qmc_count=TRANSFER_QMC_COUNT
        )
        traces.extend(case_traces)
        stresses = np.asarray([row["cauchy_shear_Pa"] for row in context.records])
        peak = int(np.argmax(stresses))
        records[case_id] = {
            "parameters": context.parameter_summary,
            "selection": selected,
            "peak_cauchy_shear_MPa": float(stresses[peak] * 1.0e-6),
            "peak_time_us": float(context.times_s[peak] * 1.0e6),
            "peak_shear": float(context.shears[peak]),
            "temperature_rise_at_peak_K": float(context.records[peak]["temperature_rise_K"]),
            "terminal_temperature_K": float(context.records[-1]["temperature_K"]),
        }
        base_rows.extend(context.records)
    records["baseline_rate_1e4_T293"] = {
        "parameters": baseline_context.parameter_summary,
        "selection": {
            horizon: reoptimization["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]
            for horizon in ("onset", "terminal")
        },
    }
    write_csv(TABLE_DIR / "loading_case_base_histories.csv", base_rows)
    return {
        "status": "THREE_LOADING_TRANSFERS_COMPLETE",
        "added_case_count": len(cases),
        "cases": records,
        "claim_boundary": (
            "Rate/temperature transfers test model robustness; they are not "
            "specimen-calibrated predictions."
        ),
    }, traces, base_rows


def local_parameter_sensitivity(
    reoptimization: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    starts = {
        horizon: coordinates_from_direction(
            reoptimization["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]["direction_n"],
            reoptimization["records"][
                f"{horizon}__baseline__constitutive_to_constitutive"
            ]["k_m_inv"],
        )
        for horizon in ("onset", "terminal")
    }
    variations = (
        ("rho_mobile_0", "minus", 0.8, {"rho_mobile_factor": 0.8}),
        ("rho_mobile_0", "plus", 1.2, {"rho_mobile_factor": 1.2}),
        ("K_Lambda", "minus", 0.8, {"mean_free_path_factor": 0.8}),
        ("K_Lambda", "plus", 1.2, {"mean_free_path_factor": 1.2}),
        ("ell_N", "minus", 0.8, {"nye_length_factor": 0.8}),
        ("ell_N", "plus", 1.2, {"nye_length_factor": 1.2}),
        ("ell_chi", "minus", 0.8, {"slip_gradient_length_factor": 0.8}),
        ("ell_chi", "plus", 1.2, {"slip_gradient_length_factor": 1.2}),
    )
    selector_id = "constitutive_to_constitutive"
    output_indices, input_indices = selectors()[selector_id]
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for parameter, side, factor_value, kwargs in variations:
        case_id = f"sensitivity_{parameter}_{side}20pct"
        print(json.dumps({"stage": "build_sensitivity_case", "case": case_id}), flush=True)
        context = build_context(case_id, factor=4, **kwargs)
        search_grid = context.indices_for_factor(2)
        onset_index = int(np.argmin(np.abs(context.times_s[search_grid] - 1.46e-6)))
        horizons = {
            "onset": search_grid[: onset_index + 1],
            "terminal": search_grid,
        }
        stresses = np.asarray([row["cauchy_shear_Pa"] for row in context.records])
        peak = int(np.argmax(stresses))
        for horizon_id, search_indices in horizons.items():
            local_starts = [
                starts[horizon_id],
                starts[horizon_id] + np.asarray([0.03, -0.03, 0.05]),
            ]
            local_starts = [
                np.asarray(
                    [
                        np.clip(value[0], *BOUNDS[0]),
                        np.clip(value[1], *BOUNDS[1]),
                        np.clip(value[2], *BOUNDS[2]),
                    ]
                )
                for value in local_starts
            ]
            runs = []
            for start_index, start in enumerate(local_starts, start=1):
                run, trace = local_refinement(
                    context,
                    search_indices,
                    start,
                    context.scales,
                    output_indices,
                    input_indices,
                    run_id=f"{case_id}__{horizon_id}__start{start_index}",
                    substeps=SEARCH_SUBSTEPS,
                    maximum_iterations=18,
                )
                runs.append(run)
                traces.extend(trace)
            winner = max(runs, key=lambda value: value["log_gain"])
            target_time = float(context.times_s[search_indices[-1]])
            verify_end = int(np.argmin(np.abs(context.times_s - target_time)))
            verify_indices = np.arange(verify_end + 1, dtype=int)
            verified = context.propagator(
                np.asarray(winner["coordinates"]),
                verify_indices,
                substeps=VERIFY_SUBSTEPS,
            )
            verified_log_gain = log_gain(
                verified["propagator"], output_indices, input_indices
            )
            records.append(
                {
                    "parameter": parameter,
                    "side": side,
                    "factor": factor_value,
                    "horizon_id": horizon_id,
                    "log_gain": float(verified_log_gain),
                    "search_log_gain": winner["log_gain"],
                    "k_m_inv": winner["k_m_inv"],
                    "n1": winner["direction_n"][0],
                    "n2": winner["direction_n"][1],
                    "n3": winner["direction_n"][2],
                    "angle_from_baseline_deg": direction_angle_degrees(
                        winner["direction_n"],
                        reoptimization["records"][
                            f"{horizon_id}__baseline__constitutive_to_constitutive"
                        ]["direction_n"],
                    ),
                    "peak_cauchy_shear_MPa": float(stresses[peak] * 1.0e-6),
                    "terminal_temperature_K": float(context.records[-1]["temperature_K"]),
                }
            )
    write_csv(TABLE_DIR / "local_parameter_sensitivity.csv", records)
    derivatives = []
    for parameter in ("rho_mobile_0", "K_Lambda", "ell_N", "ell_chi"):
        for horizon_id in ("onset", "terminal"):
            pair = {
                row["side"]: row
                for row in records
                if row["parameter"] == parameter and row["horizon_id"] == horizon_id
            }
            denominator = log(1.2) - log(0.8)
            derivatives.append(
                {
                    "parameter": parameter,
                    "horizon_id": horizon_id,
                    "d_log_gain_d_log_parameter": float(
                        (pair["plus"]["log_gain"] - pair["minus"]["log_gain"])
                        / denominator
                    ),
                    "d_log_k_d_log_parameter": float(
                        (log(pair["plus"]["k_m_inv"]) - log(pair["minus"]["k_m_inv"]))
                        / denominator
                    ),
                    "maximum_direction_change_deg": float(
                        max(
                            pair["plus"]["angle_from_baseline_deg"],
                            pair["minus"]["angle_from_baseline_deg"],
                        )
                    ),
                }
            )
    write_csv(TABLE_DIR / "local_parameter_log_derivatives.csv", derivatives)
    return {
        "status": "LOCAL_SENSITIVITY_COMPLETE",
        "perturbation": "symmetric multiplicative +/-20 percent",
        "scope": "branch-local continuation with direction--wavenumber re-optimization",
        "parameters": ["rho_mobile_0", "K_Lambda", "ell_N", "ell_chi"],
        "derivatives": derivatives,
        "claim_boundary": "local branch sensitivity, not a global uncertainty quantification",
    }, traces, records


def literature_comparison(loading: dict[str, Any]) -> dict[str, Any]:
    """Compare model landmarks to one synchronized pure-Ti stress/T dataset."""

    require(
        PRL_DIGITIZATION.is_file(),
        f"missing literature digitization record {PRL_DIGITIZATION}",
    )
    local_source_present = PRL_LOCAL.is_file()
    local_sha256 = (
        sha256(PRL_LOCAL.read_bytes()).hexdigest()
        if local_source_present
        else PRL_EXPECTED_SHA256
    )
    if local_source_present:
        require(
            local_sha256 == PRL_EXPECTED_SHA256,
            "local PRL source SHA-256 differs from the registered extraction source",
        )
    model = loading["cases"]["rate_1p3e4_T293"]
    source = {
        "authors": "Y. Guo et al.",
        "title": "Temperature Rise Associated with Adiabatic Shear Band: Causality Clarified",
        "journal": "Physical Review Letters",
        "volume_article_year": "122, 015503 (2019)",
        "doi": "10.1103/PhysRevLett.122.015503",
        "material": "commercial-purity Grade-II titanium",
        "nominal_shear_rate_s_inv": 1.3e4,
        "local_file": PRL_LOCAL.relative_to(ROOT.parent).as_posix(),
        "local_source_present": local_source_present,
        "local_sha256": local_sha256,
        "digitized_landmarks_file": PRL_DIGITIZATION.relative_to(ROOT).as_posix(),
        "access_date": "2026-08-29",
    }
    experimental = {
        "stress_peak_time_us": 58.5,
        "stress_peak_MPa_figure1_approx": 455.0,
        "temperature_rise_at_peak_stress_K_interval": [50.0, 90.0],
        "asb_first_visible_time_us": 67.5,
        "maximum_temperature_delay_after_asb_us_approx": 30.0,
        "localized_temperature_C_interval": [350.0, 650.0],
        "evidence": (
            "Peak/ASB times and approximate peak stress are read from Fig. 1; the "
            "temperature-rise and localized-temperature intervals are stated in the text."
        ),
    }
    model_peak_temperature_rise = float(model["temperature_rise_at_peak_K"])
    comparison = {
        "model_rate_s_inv": 1.3e4,
        "model_peak_stress_MPa": float(model["peak_cauchy_shear_MPa"]),
        "model_peak_time_us": float(model["peak_time_us"]),
        "model_temperature_rise_at_peak_K": model_peak_temperature_rise,
        "peak_stress_ratio_model_over_experiment": float(
            model["peak_cauchy_shear_MPa"] / experimental["stress_peak_MPa_figure1_approx"]
        ),
        "peak_time_ratio_model_over_experiment": float(
            model["peak_time_us"] / experimental["stress_peak_time_us"]
        ),
        "temperature_rise_at_peak_inside_experimental_interval": bool(
            experimental["temperature_rise_at_peak_stress_K_interval"][0]
            <= model_peak_temperature_rise
            <= experimental["temperature_rise_at_peak_stress_K_interval"][1]
        ),
    }
    return {
        "status": "EXTERNAL_TREND_CHECK_COMPLETE_NOT_CALIBRATION",
        "source": source,
        "experimental_landmarks": experimental,
        "model_comparison": comparison,
        "supported_inference": (
            "Both histories place the stress maximum before the largest temperature rise."
        ),
        "failed_or_unmatched_features": (
            "The homogeneous single-crystal path does not reproduce the specimen-scale "
            "peak time, the measured temperature rise at that peak, or the post-localization "
            "stress collapse; geometry, damage, localization and batch calibration are absent."
        ),
    }


def read_fixed_matrix() -> list[dict[str, Any]]:
    require(FIXED_MATRIX.is_file(), "missing fixed-branch 11 x 7 matrix")
    with FIXED_MATRIX.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["horizon_id"] == "terminal"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.2,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.8,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def make_reoptimization_figure(
    reoptimization: dict[str, Any], piecewise: dict[str, Any]
) -> None:
    del piecewise
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    records = reoptimization["records"]
    selector_ids = list(selectors())
    norm_ids = list(REOPTIMIZED_NORMS)
    for ax, horizon_id, panel in (
        (axes[0, 0], "onset", "(a) Re-optimized onset gain"),
        (axes[0, 1], "terminal", "(b) Re-optimized terminal gain"),
    ):
        matrix = np.asarray(
            [
                [
                    records[f"{horizon_id}__{norm_id}__{selector_id}"]["reference_log_gain"]
                    for selector_id in selector_ids
                ]
                for norm_id in norm_ids
            ]
        )
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(
            range(len(selector_ids)),
            [SELECTOR_LABELS[value] for value in selector_ids],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(range(len(norm_ids)), [value.replace("_", " ") for value in norm_ids])
        ax.set_title(panel, loc="left")
        fig.colorbar(image, ax=ax, label="optimized log gain", fraction=0.046, pad=0.03)

    ax = axes[1, 0]
    for selector_id, color in zip(selector_ids, plt.cm.tab10(np.linspace(0, 0.9, 7)), strict=True):
        ratios = []
        labels = []
        for horizon_id in ("onset", "terminal"):
            diagnostic = records[
                f"{horizon_id}__baseline__{selector_id}"
            ]["singular_diagnostics"]
            ratios.append(
                np.nan
                if diagnostic["sigma_1_over_sigma_2"] is None
                else diagnostic["sigma_1_over_sigma_2"]
            )
            labels.append(horizon_id)
        ax.plot([0, 1], ratios, "o-", color=color, label=SELECTOR_LABELS[selector_id])
    ax.axhline(DEGENERACY_RATIO_LIMIT, color="#b2182b", ls="--", lw=1.2, label="5% gap limit")
    ax.set_xticks([0, 1], ["onset", "terminal"])
    ax.set_yscale("log")
    ax.set_ylabel(r"$sigma_1/\sigma_2$")
    ax.set_title("(c) Formal singular-value separation", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=7.8)

    ax = axes[1, 1]
    rows = list(csv.DictReader((TABLE_DIR / "piecewise_frozen_same_selector.csv").open(encoding="utf-8")))
    for selector_id, color in zip(selector_ids, plt.cm.tab10(np.linspace(0, 0.9, 7)), strict=True):
        selected = [
            row
            for row in rows
            if row["horizon_id"] == "terminal" and row["selector_id"] == selector_id
        ]
        ax.loglog(
            [int(row["segment_count"]) for row in selected],
            [float(row["absolute_log_gain_error"]) for row in selected],
            "o-",
            color=color,
            label=SELECTOR_LABELS[selector_id],
        )
    ax.axhline(0.10, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("piecewise-frozen segments (registered intervals preserved)")
    ax.set_ylabel("absolute log-gain error")
    ax.set_title("(d) Same-selector frozen-window convergence", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    fig.tight_layout()
    save_figure(fig, FIG_REOPT)


def make_transfer_figure(
    loading: dict[str, Any],
    base_rows: list[dict[str, Any]],
    sensitivity: dict[str, Any],
    literature: dict[str, Any],
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    cases = []
    for case_id in dict.fromkeys(row["case_id"] for row in base_rows):
        rows = [row for row in base_rows if row["case_id"] == case_id]
        cases.append((case_id, rows))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(cases)))
    case_labels = {
        "baseline_search": r"$\dot\gamma=1.0\times10^4\ \mathrm{s^{-1}}$ (baseline)",
        "rate_5e3_T293": r"$\dot\gamma=5.0\times10^3\ \mathrm{s^{-1}}$",
        "rate_1p3e4_T293": r"$\dot\gamma=1.3\times10^4\ \mathrm{s^{-1}}$",
        "rate_1e4_T373": r"$\dot\gamma=1.0\times10^4\ \mathrm{s^{-1}},\ T_0=373\ \mathrm{K}$",
    }
    ax = axes[0, 0]
    for (case_id, rows), color in zip(cases, colors, strict=True):
        ax.plot(
            np.asarray([row["time_s"] for row in rows]) * 1.0e6,
            np.asarray([row["cauchy_shear_Pa"] for row in rows]) * 1.0e-6,
            color=color,
            label=case_labels.get(case_id, case_id.replace("_", " ")),
        )
    experiment = literature["experimental_landmarks"]
    ax.errorbar(
        [experiment["stress_peak_time_us"]],
        [experiment["stress_peak_MPa_figure1_approx"]],
        yerr=[15.0],
        fmt="D",
        color="#b2182b",
        label="Guo et al. peak (Fig. 1)",
    )
    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel("Cauchy shear stress [MPa]")
    ax.set_title("(a) Added loading cases and CP-Ti landmark", loc="left")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=7.6)

    ax = axes[0, 1]
    for (case_id, rows), color in zip(cases, colors, strict=True):
        initial = float(rows[0]["temperature_K"] - rows[0]["temperature_rise_K"])
        ax.plot(
            np.asarray([row["time_s"] for row in rows]) * 1.0e6,
            np.asarray([row["temperature_K"] for row in rows]) - initial,
            color=color,
            label=case_labels.get(case_id, case_id.replace("_", " ")),
        )
    interval = experiment["temperature_rise_at_peak_stress_K_interval"]
    ax.errorbar(
        [experiment["stress_peak_time_us"]],
        [0.5 * (interval[0] + interval[1])],
        yerr=[0.5 * (interval[1] - interval[0])],
        fmt="D",
        color="#b2182b",
        label="Guo et al. ΔT at peak",
    )
    ax.axvline(experiment["asb_first_visible_time_us"], color="#1b7837", ls="--", lw=1.1, label="ASB visible")
    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel(r"temperature rise $\Delta T$ [K]")
    ax.set_title("(b) Temperature evolution and external interval", loc="left")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=7.6)

    derivatives = sensitivity["derivatives"]
    parameters = ["rho_mobile_0", "K_Lambda", "ell_N", "ell_chi"]
    x = np.arange(len(parameters))
    width = 0.36
    ax = axes[1, 0]
    onset = [
        next(
            row["d_log_gain_d_log_parameter"]
            for row in derivatives
            if row["parameter"] == parameter and row["horizon_id"] == "onset"
        )
        for parameter in parameters
    ]
    terminal = [
        next(
            row["d_log_gain_d_log_parameter"]
            for row in derivatives
            if row["parameter"] == parameter and row["horizon_id"] == "terminal"
        )
        for parameter in parameters
    ]
    ax.bar(x - width / 2, onset, width, label="onset", color="#67a9cf")
    ax.bar(x + width / 2, terminal, width, label="terminal", color="#b2182b")
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xticks(x, [r"$\rho_{m0}$", r"$K_\Lambda$", r"$\ell_N$", r"$\ell_\chi$"])
    ax.set_ylabel(r"$\partial\log G/\partial\log p$")
    ax.set_title("(c) Branch-local gain sensitivity", loc="left")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.22)

    ax = axes[1, 1]
    onset_k = [
        next(
            row["d_log_k_d_log_parameter"]
            for row in derivatives
            if row["parameter"] == parameter and row["horizon_id"] == "onset"
        )
        for parameter in parameters
    ]
    terminal_k = [
        next(
            row["d_log_k_d_log_parameter"]
            for row in derivatives
            if row["parameter"] == parameter and row["horizon_id"] == "terminal"
        )
        for parameter in parameters
    ]
    ax.bar(x - width / 2, onset_k, width, label="onset", color="#67a9cf")
    ax.bar(x + width / 2, terminal_k, width, label="terminal", color="#b2182b")
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xticks(x, [r"$\rho_{m0}$", r"$K_\Lambda$", r"$\ell_N$", r"$\ell_\chi$"])
    ax.set_ylabel(r"$\partial\log k^*/\partial\log p$")
    ax.set_title("(d) Branch-local wavenumber sensitivity", loc="left")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    save_figure(fig, FIG_TRANSFER)


def make_full_matrix_figure(rows: list[dict[str, Any]]) -> None:
    configure_style()
    selector_ids = list(selectors())
    norm_ids = [
        "baseline",
        "mechanical_x0.5",
        "mechanical_x2",
        "thermal_x0.5",
        "thermal_x2",
        "dislocation_x0.5",
        "dislocation_x2",
        "plastic_kinematic_x0.5",
        "plastic_kinematic_x2",
        "all_constitutive_x0.5",
        "all_constitutive_x2",
    ]
    lookup = {(row["norm_id"], row["selector_id"]): row for row in rows}
    matrix = np.asarray(
        [
            [float(lookup[(norm_id, selector_id)]["winner_log_gain"]) for selector_id in selector_ids]
            for norm_id in norm_ids
        ]
    )
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(selector_ids)),
        [SELECTOR_LABELS[value] for value in selector_ids],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(range(len(norm_ids)), [value.replace("_", " ") for value in norm_ids])
    for row_index, norm_id in enumerate(norm_ids):
        for column_index, selector_id in enumerate(selector_ids):
            source = lookup[(norm_id, selector_id)]
            branch = source["winner_branch"].replace("onset_", "o-").replace("terminal_", "t-")
            color = "white" if matrix[row_index, column_index] < np.nanmedian(matrix) else "black"
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}\n{branch}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
            )
    ax.set_title("Full terminal 11 × 7 fixed-candidate norm–selector screen", loc="left")
    fig.colorbar(image, ax=ax, label="winner log gain", fraction=0.028, pad=0.02)
    fig.tight_layout()
    save_figure(fig, FIG_MATRIX)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="development-only reduced baseline; production manuscript must omit this flag",
    )
    parser.add_argument(
        "--refresh-derived",
        action="store_true",
        help="recompute piecewise-frozen convergence and its figure from released V3 arrays",
    )
    parser.add_argument(
        "--resume-after-reoptimization",
        action="store_true",
        help="reuse the checkpointed re-optimization arrays and run transfer/sensitivity stages",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_derived:
        require(RESULT.is_file() and ARRAYS.is_file(), "V3 release is incomplete")
        report = json.loads(RESULT.read_text(encoding="utf-8"))
        with np.load(ARRAYS) as source:
            released = {key: np.asarray(source[key]) for key in source.files}
        piecewise = piecewise_frozen_baseline(report["reoptimization"], released)
        report["piecewise_frozen_same_selector"] = piecewise
        report["gates"]["piecewise_same_selector_contract_complete"] = piecewise[
            "gate_refinement16_error_below_0p10"
        ]
        report["status"] = (
            "ALL_V3_REVISION_GATES_PASS"
            if all(report["gates"].values())
            else "V3_REVISION_GATES_OPEN"
        )
        make_reoptimization_figure(report["reoptimization"], piecewise)
        RESULT.write_text(
            json.dumps(clean_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": report["status"], "piecewise": piecewise}, indent=2))
        return 0 if all(report["gates"].values()) else 2
    production_factor = 4 if args.quick else 16
    require(not args.quick, "quick mode is intentionally not accepted for evidence release")

    print(json.dumps({"stage": "build_baseline_search_context"}), flush=True)
    baseline_search = build_context("baseline_search", factor=4)
    if args.resume_after_reoptimization:
        require(RESULT.is_file() and ARRAYS.is_file(), "no re-optimization checkpoint exists")
        checkpoint = json.loads(RESULT.read_text(encoding="utf-8"))
        require(
            checkpoint.get("status") in {
                "V3_REOPTIMIZATION_CHECKPOINT",
                "V3_REVISION_GATES_OPEN",
                "ALL_V3_REVISION_GATES_PASS",
            },
            "result file is not a reusable V3 checkpoint",
        )
        reoptimization = checkpoint["reoptimization"]
        piecewise = checkpoint["piecewise_frozen_same_selector"]
        with np.load(ARRAYS) as source:
            released = {key: np.asarray(source[key]) for key in source.files}
    else:
        print(json.dumps({"stage": "build_baseline_reference_context", "factor": production_factor}), flush=True)
        baseline_reference = build_context("baseline_reference", factor=production_factor)
        reoptimization, released, traces = search_selector_norms(
            baseline_search, baseline_reference
        )
        released["reference_coordinate_scales"] = baseline_reference.scales
        write_csv(TABLE_DIR / "selector_norm_optimizer_trace.csv", traces)
        selector_rows = []
        singular_rows = []
        for record in reoptimization["records"].values():
            selector_rows.append(
                {
                    "horizon_id": record["horizon_id"],
                    "time_s": record["time_s"],
                    "norm_id": record["norm_id"],
                    "selector_id": record["selector_id"],
                    "n1": record["direction_n"][0],
                    "n2": record["direction_n"][1],
                    "n3": record["direction_n"][2],
                    "k_m_inv": record["k_m_inv"],
                    "reference_log_gain": record["reference_log_gain"],
                    "reference_gain": record["reference_gain"],
                    "wavenumber_interior": record["wavenumber_interior"],
                    "boundary_classification": record["boundary_classification"],
                }
            )
            singular_rows.append(
                {
                    "horizon_id": record["horizon_id"],
                    "norm_id": record["norm_id"],
                    "selector_id": record["selector_id"],
                    **record["singular_diagnostics"],
                }
            )
        write_csv(TABLE_DIR / "reoptimized_selector_norm_winners.csv", selector_rows)
        write_csv(TABLE_DIR / "singular_value_gap_diagnostics.csv", singular_rows)
        piecewise = piecewise_frozen_baseline(reoptimization, released)
        np.savez_compressed(ARRAYS, **released)
        checkpoint = {
            "schema": "IJP_SUBMISSION_REVISION_EVIDENCE_V3",
            "status": "V3_REOPTIMIZATION_CHECKPOINT",
            "reoptimization": reoptimization,
            "piecewise_frozen_same_selector": piecewise,
            "gates": {
                "reoptimization_pass": reoptimization["status"]
                == "REOPTIMIZATION_AUDIT_PASS",
                "piecewise_same_selector_contract_complete": piecewise[
                    "gate_refinement16_error_below_0p10"
                ],
            },
        }
        RESULT.write_text(
            json.dumps(clean_json(checkpoint), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    loading, loading_traces, base_rows = loading_transfers(
        baseline_search, reoptimization
    )
    write_csv(TABLE_DIR / "loading_optimizer_trace.csv", loading_traces)
    sensitivity, sensitivity_traces, sensitivity_rows = local_parameter_sensitivity(
        reoptimization
    )
    write_csv(TABLE_DIR / "sensitivity_optimizer_trace.csv", sensitivity_traces)
    literature = literature_comparison(loading)
    fixed_rows = read_fixed_matrix()

    np.savez_compressed(ARRAYS, **released)
    make_reoptimization_figure(reoptimization, piecewise)
    make_transfer_figure(loading, base_rows, sensitivity, literature)
    make_full_matrix_figure(fixed_rows)

    gates = {
        "reoptimization_pass": reoptimization["status"] == "REOPTIMIZATION_AUDIT_PASS",
        "piecewise_same_selector_contract_complete": piecewise[
            "gate_refinement16_error_below_0p10"
        ],
        "three_loading_transfers_complete": loading["added_case_count"] == 3,
        "four_local_parameters_complete": len(sensitivity["parameters"]) == 4,
        "pure_titanium_external_comparison_complete": literature[
            "status"
        ]
        == "EXTERNAL_TREND_CHECK_COMPLETE_NOT_CALIBRATION",
        "complete_fixed_11_by_7_matrix": len(fixed_rows) == 77,
    }
    report = {
        "schema": "IJP_SUBMISSION_REVISION_EVIDENCE_V3",
        "status": "ALL_V3_REVISION_GATES_PASS" if all(gates.values()) else "V3_REVISION_GATES_OPEN",
        "reoptimization": reoptimization,
        "piecewise_frozen_same_selector": piecewise,
        "loading_transfers": loading,
        "local_parameter_sensitivity": sensitivity,
        "pure_titanium_literature_comparison": literature,
        "gates": gates,
        "outputs": {
            "arrays": ARRAYS.relative_to(ROOT).as_posix(),
            "tables": TABLE_DIR.relative_to(ROOT).as_posix(),
            "reoptimization_figure": FIG_REOPT.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            "loading_sensitivity_figure": FIG_TRANSFER.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            "supplement_full_matrix_figure": FIG_MATRIX.with_suffix(".pdf").relative_to(ROOT).as_posix(),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.write_text(
        json.dumps(clean_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": report["status"], "gates": gates, "result": str(RESULT)},
            indent=2,
        ),
        flush=True,
    )
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
