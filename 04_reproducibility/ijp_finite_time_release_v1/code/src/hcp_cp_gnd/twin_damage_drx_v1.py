"""Thermodynamic admission kernel for twin, damage, and DRX states.

This module supplies a branchwise-consistent local residual and tangent for the
15-state extension of the slip-only periodic69 system.  Its bundled numerical
fixture verifies equations and units; it is deliberately not a TA2 parameter
set.  Material execution fails closed while the admission card contains null
identification fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

ADMISSION_SCHEMA = "CP_TI_TWIN_DAMAGE_DRX_ADMISSION_V1"
EXTENDED_ACTIVE_SCHEMA = "CP_TI_PERIODIC84_TWIN_DAMAGE_DRX_ACTIVE_V1"
N_TWIN = 12
N_EXTENDED_ACTIVE = 15
R_GAS = 8.31446261815324

STATE_LABELS = tuple(
    [f"twin_fraction::{index + 1}" for index in range(N_TWIN)]
    + ["damage", "drx_fraction", "drx_boundary_area_m_inv"]
)
DRIVE_LABELS = tuple(
    [f"resolved_twin_stress_Pa::{index + 1}" for index in range(N_TWIN)]
    + [
        "positive_elastic_energy_J_m3",
        "damage_history_J_m3",
        "total_dislocation_density_m2",
        "temperature_K",
    ]
)


class MaterialParameterIdentificationOpen(RuntimeError):
    """Raised when an uncalibrated material run is requested."""


def _array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _positive_power(overstress: Array | float, scale: float, exponent: float):
    x = np.asarray(overstress, dtype=np.float64)
    active = x > 0.0
    normalized = np.where(active, x / scale, 0.0)
    value = np.where(active, normalized**exponent, 0.0)
    derivative = np.where(
        active,
        exponent * normalized ** (exponent - 1.0) / scale,
        0.0,
    )
    return value, derivative, active


def _arrhenius_ratio(activation_J_mol: float, temperature_K: float, reference_K: float):
    exponent = -activation_J_mol / R_GAS * (1.0 / temperature_K - 1.0 / reference_K)
    if abs(exponent) > 100.0:
        raise FloatingPointError("Arrhenius ratio left the admitted exponent interval")
    value = float(np.exp(exponent))
    derivative = value * activation_J_mol / (R_GAS * temperature_K**2)
    return value, derivative


@dataclass(frozen=True)
class MechanismParametersV1:
    twin_shear: Array
    twin_self_hardening_Pa: float
    twin_latent_hardening_Pa: float
    twin_gradient_J_m: float
    twin_threshold_Pa: float
    twin_force_scale_Pa: float
    twin_reference_rate_s: float
    twin_rate_exponent: float
    twin_max_total_fraction: float
    damage_fracture_energy_J_m2: float
    damage_length_m: float
    damage_residual_stiffness: float
    damage_threshold_Pa: float
    damage_force_scale_Pa: float
    damage_reference_rate_s: float
    damage_rate_exponent: float
    dislocation_storage_coefficient: float
    shear_modulus_Pa: float
    burgers_m: float
    recrystallized_density_m2: float
    drx_grain_boundary_energy_J_m2: float
    drx_target_grain_size_m: float
    drx_hardening_Pa: float
    drx_gradient_J_m: float
    drx_threshold_Pa: float
    drx_force_scale_Pa: float
    drx_reference_rate_s: float
    drx_rate_exponent: float
    drx_activation_energy_J_mol: float
    reference_temperature_K: float
    classification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "twin_shear", _array(self.twin_shear, (N_TWIN,), "twin_shear"))
        values = np.asarray(
            [
                self.twin_self_hardening_Pa,
                self.twin_latent_hardening_Pa,
                self.twin_gradient_J_m,
                self.twin_threshold_Pa,
                self.twin_force_scale_Pa,
                self.twin_reference_rate_s,
                self.twin_rate_exponent,
                self.twin_max_total_fraction,
                self.damage_fracture_energy_J_m2,
                self.damage_length_m,
                self.damage_residual_stiffness,
                self.damage_threshold_Pa,
                self.damage_force_scale_Pa,
                self.damage_reference_rate_s,
                self.damage_rate_exponent,
                self.dislocation_storage_coefficient,
                self.shear_modulus_Pa,
                self.burgers_m,
                self.recrystallized_density_m2,
                self.drx_grain_boundary_energy_J_m2,
                self.drx_target_grain_size_m,
                self.drx_hardening_Pa,
                self.drx_gradient_J_m,
                self.drx_threshold_Pa,
                self.drx_force_scale_Pa,
                self.drx_reference_rate_s,
                self.drx_rate_exponent,
                self.reference_temperature_K,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all positive mechanism parameters must be finite and positive")
        if not np.isfinite(self.drx_activation_energy_J_mol) or self.drx_activation_energy_J_mol < 0.0:
            raise ValueError("DRX activation energy must be finite and non-negative")
        if not 0.0 < self.twin_max_total_fraction < 1.0:
            raise ValueError("maximum twin fraction must lie in (0,1)")
        if not 0.0 < self.damage_residual_stiffness < 1.0:
            raise ValueError("damage residual stiffness must lie in (0,1)")
        if np.any(self.twin_shear <= 0.0):
            raise ValueError("twin shears must be positive")
        if not self.classification:
            raise ValueError("parameter classification is required")


@dataclass(frozen=True)
class MechanismPointStateV1:
    twin_fraction: Array
    damage: float
    drx_fraction: float
    drx_boundary_area_m_inv: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "twin_fraction", _array(self.twin_fraction, (N_TWIN,), "twin_fraction")
        )
        values = np.asarray(
            [self.damage, self.drx_fraction, self.drx_boundary_area_m_inv], dtype=float
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("mechanism state scalars must be finite")
        if np.any(self.twin_fraction < 0.0):
            raise ValueError("twin fractions must be non-negative")
        if not 0.0 <= self.damage <= 1.0:
            raise ValueError("damage must lie in [0,1]")
        if not 0.0 <= self.drx_fraction <= 1.0:
            raise ValueError("DRX fraction must lie in [0,1]")
        if self.drx_boundary_area_m_inv < 0.0:
            raise ValueError("DRX boundary area density must be non-negative")

    def vector(self) -> Array:
        return np.r_[self.twin_fraction, self.damage, self.drx_fraction, self.drx_boundary_area_m_inv]

    @classmethod
    def from_vector(cls, value: Any) -> "MechanismPointStateV1":
        vector = np.asarray(value, dtype=float)
        if vector.shape != (N_EXTENDED_ACTIVE,):
            raise ValueError("mechanism state vector must have length 15")
        return cls(vector[:N_TWIN], float(vector[12]), float(vector[13]), float(vector[14]))


@dataclass(frozen=True)
class MechanismInputsV1:
    resolved_twin_stress_Pa: Array
    twin_gradient_m_inv: Array
    twin_laplacian_m2: Array
    positive_elastic_energy_J_m3: float
    damage_history_J_m3: float
    damage_gradient_m_inv: float
    damage_laplacian_m2: float
    total_dislocation_density_m2: float
    drx_gradient_m_inv: float
    drx_laplacian_m2: float
    temperature_K: float

    def __post_init__(self) -> None:
        for name in ("resolved_twin_stress_Pa", "twin_gradient_m_inv", "twin_laplacian_m2"):
            object.__setattr__(self, name, _array(getattr(self, name), (N_TWIN,), name))
        values = np.asarray(
            [
                self.positive_elastic_energy_J_m3,
                self.damage_history_J_m3,
                self.damage_gradient_m_inv,
                self.damage_laplacian_m2,
                self.total_dislocation_density_m2,
                self.drx_gradient_m_inv,
                self.drx_laplacian_m2,
                self.temperature_K,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("mechanism inputs must be finite")
        if self.positive_elastic_energy_J_m3 < 0.0 or self.damage_history_J_m3 < 0.0:
            raise ValueError("damage energy and history must be non-negative")
        if self.total_dislocation_density_m2 <= 0.0 or self.temperature_K <= 0.0:
            raise ValueError("density and temperature must be positive")


@dataclass(frozen=True)
class MechanismResponseV1:
    rates: Array
    twin_force_Pa: Array
    damage_force_Pa: float
    drx_force_Pa: float
    free_energy_J_m3: dict[str, float]
    dissipation_W_m3: dict[str, float]
    d_rates_d_state: Array
    d_rates_d_drives: Array
    branch_signature: tuple[int, ...]


@dataclass(frozen=True)
class MechanismAdvanceV1:
    state: MechanismPointStateV1
    response: MechanismResponseV1
    algorithmic_tangent: Array
    iterations: int
    residual_norm: float
    dissipation_increment_J_m3: dict[str, float]
    updated_damage_history_J_m3: float


def load_admission_card(path: str | Path | None = None) -> dict[str, Any]:
    source = (
        Path(__file__).resolve().parents[2] / "config/cp_ti_twin_damage_drx_admission_v1.json"
        if path is None
        else Path(path)
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema") != ADMISSION_SCHEMA:
        raise ValueError("unexpected twin--damage--DRX admission schema")
    return data


def _parameters_from_mapping(raw: dict[str, Any], *, classification: str) -> MechanismParametersV1:
    return MechanismParametersV1(
        twin_shear=np.r_[np.full(6, 0.174), np.full(6, 0.225)],
        classification=classification,
        **{key: raw[key] for key in MechanismParametersV1.__dataclass_fields__ if key not in {"twin_shear", "classification"}},
    )


def verification_parameters(card: dict[str, Any] | None = None) -> MechanismParametersV1:
    data = card or load_admission_card()
    raw = data["verification_fixture"]
    return _parameters_from_mapping(raw, classification=str(raw["classification"]))


def material_parameters(card: dict[str, Any] | None = None) -> MechanismParametersV1:
    data = card or load_admission_card()
    missing = [key for key, value in data["required_material_parameters"].items() if value is None]
    if missing or not data.get("production_long_window_enabled", False):
        raise MaterialParameterIdentificationOpen(
            "material long-window execution is blocked; unidentified fields: " + ", ".join(missing)
        )
    raise MaterialParameterIdentificationOpen(
        "the admitted material-parameter mapping has not yet been promoted to an executable schema"
    )


def initial_mechanism_state() -> MechanismPointStateV1:
    return MechanismPointStateV1(np.zeros(N_TWIN), 0.0, 0.0, 0.0)


def mechanism_response(
    state: MechanismPointStateV1,
    inputs: MechanismInputsV1,
    parameters: MechanismParametersV1,
) -> MechanismResponseV1:
    p = parameters
    eta = np.asarray(state.twin_fraction)
    total_eta = float(np.sum(eta))
    if total_eta > p.twin_max_total_fraction * (1.0 + 1.0e-12):
        raise ValueError("total twin fraction exceeds the admitted maximum")
    availability_tw = max(1.0 - total_eta / p.twin_max_total_fraction, 0.0)
    twin_force = (
        p.twin_shear * inputs.resolved_twin_stress_Pa
        - p.twin_self_hardening_Pa * eta
        - p.twin_latent_hardening_Pa * total_eta
        + p.twin_gradient_J_m * inputs.twin_laplacian_m2
    )
    tw_pow, tw_dpow, tw_active = _positive_power(
        twin_force - p.twin_threshold_Pa,
        p.twin_force_scale_Pa,
        p.twin_rate_exponent,
    )
    twin_base = p.twin_reference_rate_s * tw_pow
    twin_rate = twin_base * availability_tw
    twin_dbdy = p.twin_reference_rate_s * tw_dpow

    history = max(inputs.damage_history_J_m3, inputs.positive_elastic_energy_J_m3)
    damage_force = (
        2.0 * (1.0 - p.damage_residual_stiffness) * (1.0 - state.damage) * history
        - p.damage_fracture_energy_J_m2 * state.damage / p.damage_length_m
        + p.damage_fracture_energy_J_m2 * p.damage_length_m * inputs.damage_laplacian_m2
    )
    d_pow, d_dpow, d_active_array = _positive_power(
        damage_force - p.damage_threshold_Pa,
        p.damage_force_scale_Pa,
        p.damage_rate_exponent,
    )
    d_active = bool(np.asarray(d_active_array))
    damage_mobility = p.damage_reference_rate_s
    damage_base = float(damage_mobility * d_pow)
    damage_dbdy = float(damage_mobility * d_dpow)
    damage_availability = max(1.0 - state.damage, 0.0)
    damage_rate = damage_base * damage_availability

    dislocation_energy = (
        p.dislocation_storage_coefficient
        * p.shear_modulus_Pa
        * p.burgers_m**2
        * inputs.total_dislocation_density_m2
    )
    recrystallized_energy = (
        p.dislocation_storage_coefficient
        * p.shear_modulus_Pa
        * p.burgers_m**2
        * p.recrystallized_density_m2
    )
    boundary_area_target = 3.0 / p.drx_target_grain_size_m
    drx_force = (
        dislocation_energy
        - recrystallized_energy
        - p.drx_grain_boundary_energy_J_m2 * boundary_area_target
        - p.drx_hardening_Pa * state.drx_fraction
        + p.drx_gradient_J_m * inputs.drx_laplacian_m2
    )
    rx_pow, rx_dpow, rx_active_array = _positive_power(
        drx_force - p.drx_threshold_Pa,
        p.drx_force_scale_Pa,
        p.drx_rate_exponent,
    )
    rx_active = bool(np.asarray(rx_active_array))
    arrhenius, darrhenius_dT = _arrhenius_ratio(
        p.drx_activation_energy_J_mol,
        inputs.temperature_K,
        p.reference_temperature_K,
    )
    rx_mobility = p.drx_reference_rate_s * arrhenius
    rx_base = float(rx_mobility * rx_pow)
    rx_dbdy = float(rx_mobility * rx_dpow)
    rx_availability = max(1.0 - state.drx_fraction, 0.0)
    drx_rate = rx_base * rx_availability
    boundary_area_rate = boundary_area_target * drx_rate

    rates = np.r_[twin_rate, damage_rate, drx_rate, boundary_area_rate]
    tangent_state = np.zeros((N_EXTENDED_ACTIVE, N_EXTENDED_ACTIVE))
    tangent_drives = np.zeros((N_EXTENDED_ACTIVE, len(DRIVE_LABELS)))

    if availability_tw > 0.0:
        for i in range(N_TWIN):
            for j in range(N_TWIN):
                dy = -p.twin_latent_hardening_Pa
                if i == j:
                    dy -= p.twin_self_hardening_Pa
                tangent_state[i, j] = (
                    twin_dbdy[i] * dy * availability_tw
                    - twin_base[i] / p.twin_max_total_fraction
                )
            tangent_drives[i, i] = twin_dbdy[i] * p.twin_shear[i] * availability_tw

    if damage_availability > 0.0:
        dy_dd = (
            -2.0 * (1.0 - p.damage_residual_stiffness) * history
            - p.damage_fracture_energy_J_m2 / p.damage_length_m
        )
        tangent_state[12, 12] = damage_dbdy * dy_dd * damage_availability - damage_base
        dh_dpsi = 1.0 if inputs.positive_elastic_energy_J_m3 > inputs.damage_history_J_m3 else 0.0
        dh_dhistory = 1.0 if inputs.damage_history_J_m3 > inputs.positive_elastic_energy_J_m3 else 0.0
        dy_dh = 2.0 * (1.0 - p.damage_residual_stiffness) * (1.0 - state.damage)
        tangent_drives[12, 12] = damage_dbdy * dy_dh * dh_dpsi * damage_availability
        tangent_drives[12, 13] = damage_dbdy * dy_dh * dh_dhistory * damage_availability

    if rx_availability > 0.0:
        tangent_state[13, 13] = (
            rx_dbdy * (-p.drx_hardening_Pa) * rx_availability - rx_base
        )
        denergy_drho = p.dislocation_storage_coefficient * p.shear_modulus_Pa * p.burgers_m**2
        tangent_drives[13, 14] = rx_dbdy * denergy_drho * rx_availability
        tangent_drives[13, 15] = (
            p.drx_reference_rate_s * darrhenius_dT * float(rx_pow) * rx_availability
        )
        tangent_state[14, :] = boundary_area_target * tangent_state[13, :]
        tangent_drives[14, :] = boundary_area_target * tangent_drives[13, :]

    twin_energy = (
        0.5 * p.twin_self_hardening_Pa * float(eta @ eta)
        + 0.5 * p.twin_latent_hardening_Pa * total_eta**2
        + 0.5 * p.twin_gradient_J_m * float(inputs.twin_gradient_m_inv @ inputs.twin_gradient_m_inv)
    )
    degradation = (
        (1.0 - p.damage_residual_stiffness) * (1.0 - state.damage) ** 2
        + p.damage_residual_stiffness
    )
    damage_energy = (
        degradation * inputs.positive_elastic_energy_J_m3
        + 0.5 * p.damage_fracture_energy_J_m2 / p.damage_length_m * state.damage**2
        + 0.5
        * p.damage_fracture_energy_J_m2
        * p.damage_length_m
        * inputs.damage_gradient_m_inv**2
    )
    drx_energy = (
        (1.0 - state.drx_fraction) * dislocation_energy
        + state.drx_fraction * recrystallized_energy
        + p.drx_grain_boundary_energy_J_m2 * state.drx_boundary_area_m_inv
        + 0.5 * p.drx_hardening_Pa * state.drx_fraction**2
        + 0.5 * p.drx_gradient_J_m * inputs.drx_gradient_m_inv**2
    )
    dissipation = {
        "twin": float(twin_force @ twin_rate),
        "damage": float(damage_force * damage_rate),
        "drx": float(drx_force * drx_rate),
    }
    scale = max(1.0, *(abs(value) for value in dissipation.values()))
    if min(dissipation.values()) < -1.0e-12 * scale:
        raise FloatingPointError("a new mechanism produced negative dissipation")
    history_branch = (
        1
        if inputs.positive_elastic_energy_J_m3 > inputs.damage_history_J_m3
        else (-1 if inputs.positive_elastic_energy_J_m3 < inputs.damage_history_J_m3 else 0)
    )
    signature = tuple(tw_active.astype(int).tolist()) + (int(d_active), int(rx_active), history_branch)
    return MechanismResponseV1(
        rates=rates,
        twin_force_Pa=twin_force,
        damage_force_Pa=float(damage_force),
        drx_force_Pa=float(drx_force),
        free_energy_J_m3={
            "twin": float(twin_energy),
            "damage": float(damage_energy),
            "drx": float(drx_energy),
            "total": float(twin_energy + damage_energy + drx_energy),
        },
        dissipation_W_m3=dissipation,
        d_rates_d_state=tangent_state,
        d_rates_d_drives=tangent_drives,
        branch_signature=signature,
    )


def advance_backward_euler(
    state: MechanismPointStateV1,
    inputs: MechanismInputsV1,
    parameters: MechanismParametersV1,
    dt_s: float,
    *,
    tolerance: float = 1.0e-11,
    maximum_iterations: int = 20,
) -> MechanismAdvanceV1:
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time increment must be positive")
    old = state.vector()
    trial = old.copy()
    residual_scale = np.ones(N_EXTENDED_ACTIVE)
    residual_scale[14] = max(
        1.0, abs(old[14]), 3.0 / parameters.drx_target_grain_size_m
    )
    residual_norm = np.inf
    response = mechanism_response(state, inputs, parameters)
    for iteration in range(1, maximum_iterations + 1):
        current = MechanismPointStateV1.from_vector(trial)
        response = mechanism_response(current, inputs, parameters)
        residual = trial - old - dt * response.rates
        residual_norm = float(np.linalg.norm(residual / residual_scale, ord=np.inf))
        if residual_norm <= tolerance:
            break
        jacobian = np.eye(N_EXTENDED_ACTIVE) - dt * response.d_rates_d_state
        increment = np.linalg.solve(jacobian, -residual)
        accepted = False
        for reduction in range(20):
            candidate = trial + increment * (0.5**reduction)
            if (
                np.all(candidate[:N_TWIN] >= 0.0)
                and float(np.sum(candidate[:N_TWIN])) <= parameters.twin_max_total_fraction
                and 0.0 <= candidate[12] <= 1.0
                and 0.0 <= candidate[13] <= 1.0
                and candidate[14] >= 0.0
            ):
                trial = candidate
                accepted = True
                break
        if not accepted:
            raise FloatingPointError("backward-Euler Newton step left the state domain")
    else:
        raise FloatingPointError("backward-Euler mechanism update did not converge")
    updated = MechanismPointStateV1.from_vector(trial)
    response = mechanism_response(updated, inputs, parameters)
    jacobian = np.eye(N_EXTENDED_ACTIVE) - dt * response.d_rates_d_state
    algorithmic = np.linalg.solve(jacobian, dt * response.d_rates_d_drives)
    return MechanismAdvanceV1(
        state=updated,
        response=response,
        algorithmic_tangent=algorithmic,
        iterations=iteration,
        residual_norm=residual_norm,
        dissipation_increment_J_m3={
            name: dt * value for name, value in response.dissipation_W_m3.items()
        },
        updated_damage_history_J_m3=max(
            inputs.damage_history_J_m3, inputs.positive_elastic_energy_J_m3
        ),
    )


__all__ = [
    "ADMISSION_SCHEMA",
    "DRIVE_LABELS",
    "EXTENDED_ACTIVE_SCHEMA",
    "MaterialParameterIdentificationOpen",
    "MechanismAdvanceV1",
    "MechanismInputsV1",
    "MechanismParametersV1",
    "MechanismPointStateV1",
    "MechanismResponseV1",
    "N_EXTENDED_ACTIVE",
    "N_TWIN",
    "STATE_LABELS",
    "advance_backward_euler",
    "initial_mechanism_state",
    "load_admission_card",
    "material_parameters",
    "mechanism_response",
    "verification_parameters",
]
