"""Material parameter schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .crystal import HCPSystems

Array = NDArray[np.float64]


@dataclass(frozen=True)
class MaterialParameters:
    status: str
    c_over_a: float
    T_ref: float
    elastic_C0: Array
    mass_density: float
    heat_capacity: float
    taylor_quinney: float
    burgers: Array
    tau0: Array
    tau_cut: Array
    activation_energy: Array
    reference_velocity: Array
    p: Array
    q: Array
    rho_mobile_0: Array
    rho_dipole_0: Array
    grain_size: float
    mean_free_path_coefficient: float
    reference_shear_modulus: float
    forest_interaction: Array
    taylor_coefficient: float
    dipole_min_distance_burgers: float
    climb_frequency: float
    climb_activation: float
    twin_shear_override: float
    twin_crss: float
    twin_reference_rate: float
    twin_rate_exponent: float
    twin_stress_scale: float
    twin_latent_hardening: float
    twin_max_total_fraction: float
    density_floor: float
    determinant_tolerance: float
    maximum_slip_increment: float
    maximum_twin_increment: float

    def elastic_matrix(self) -> Array:
        """Return the fixed-reference elastic tensor used by v0.1."""

        matrix = self.elastic_C0
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.min(eigenvalues) <= 1.0e7:
            raise ValueError(f"elastic stiffness is not positive definite: {np.min(eigenvalues)}")
        return matrix


def _hcp_stiffness(data: dict[str, Any], prefix: str = "") -> Array:
    c11 = float(data[f"{prefix}C11_Pa"])
    c12 = float(data[f"{prefix}C12_Pa"])
    c13 = float(data[f"{prefix}C13_Pa"])
    c33 = float(data[f"{prefix}C33_Pa"])
    c44 = float(data[f"{prefix}C44_Pa"])
    c66 = 0.5 * (c11 - c12)
    return np.array(
        [
            [c11, c12, c13, 0.0, 0.0, 0.0],
            [c12, c11, c13, 0.0, 0.0, 0.0],
            [c13, c13, c33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c66],
        ],
        dtype=float,
    )


def _expand_family(values: list[float], systems: HCPSystems) -> Array:
    if len(values) != 3:
        raise ValueError("family parameter must contain basal, prismatic and pyramidal values")
    family_index = {"basal<a>": 0, "prismatic<a>": 1, "pyramidal<c+a>": 2}
    return np.asarray([float(values[family_index[name]]) for name in systems.slip_families])


def load_material_parameters(path: str | Path, systems: HCPSystems) -> MaterialParameters:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "hcp-cp-material-0.1":
        raise ValueError("unsupported material schema")
    if raw.get("status") != "VERIFICATION_SEED_NOT_CALIBRATED_FOR_TA2":
        raise ValueError("v0.1 accepts only the explicitly watermarked verification material")

    elastic = raw["elastic"]
    thermal = raw["thermal"]
    slip = raw["slip"]
    dislocation = raw["dislocation"]
    twin = raw["twinning"]
    numerics = raw["numerics"]

    self_value = float(dislocation["forest_self"])
    latent_value = float(dislocation["forest_latent"])
    interaction = np.full((systems.n_slip, systems.n_slip), latent_value)
    np.fill_diagonal(interaction, self_value)

    result = MaterialParameters(
        status=raw["status"],
        c_over_a=float(raw["lattice"]["c_over_a"]),
        T_ref=float(elastic["T_ref_K"]),
        elastic_C0=_hcp_stiffness(elastic),
        mass_density=float(thermal["mass_density_kg_per_m3"]),
        heat_capacity=float(thermal["heat_capacity_J_per_kgK"]),
        taylor_quinney=float(thermal["taylor_quinney"]),
        burgers=_expand_family(slip["burgers_m"], systems),
        tau0=_expand_family(slip["tau0_Pa"], systems),
        tau_cut=_expand_family(slip["tau_cut_Pa"], systems),
        activation_energy=_expand_family(slip["activation_energy_J"], systems),
        reference_velocity=_expand_family(slip["reference_velocity_m_per_s"], systems),
        p=_expand_family(slip["p"], systems),
        q=_expand_family(slip["q"], systems),
        rho_mobile_0=_expand_family(slip["rho_mobile_0_per_m2"], systems),
        rho_dipole_0=_expand_family(slip["rho_dipole_0_per_m2"], systems),
        grain_size=float(dislocation["grain_size_m"]),
        mean_free_path_coefficient=float(dislocation["mean_free_path_coefficient"]),
        reference_shear_modulus=float(dislocation["reference_shear_modulus_Pa"]),
        forest_interaction=interaction,
        taylor_coefficient=float(dislocation["taylor_factor_coefficient"]),
        dipole_min_distance_burgers=float(dislocation["dipole_min_distance_burgers"]),
        climb_frequency=float(dislocation["climb_frequency_per_s"]),
        climb_activation=float(dislocation["climb_activation_J"]),
        twin_shear_override=float(twin["shear"]),
        twin_crss=float(twin["crss_Pa"]),
        twin_reference_rate=float(twin["reference_rate_per_s"]),
        twin_rate_exponent=float(twin["rate_exponent"]),
        twin_stress_scale=float(twin["stress_scale_Pa"]),
        twin_latent_hardening=float(twin["latent_hardening_Pa"]),
        twin_max_total_fraction=float(twin["max_total_fraction"]),
        density_floor=float(numerics["density_floor_per_m2"]),
        determinant_tolerance=float(numerics["determinant_tolerance"]),
        maximum_slip_increment=float(numerics["maximum_slip_increment"]),
        maximum_twin_increment=float(numerics["maximum_twin_increment"]),
    )
    _validate(result, systems)
    return result


def _validate(parameters: MaterialParameters, systems: HCPSystems) -> None:
    scalar_values = np.array(
        [
            parameters.c_over_a,
            parameters.T_ref,
            parameters.mass_density,
            parameters.heat_capacity,
            parameters.taylor_quinney,
            parameters.grain_size,
            parameters.mean_free_path_coefficient,
            parameters.reference_shear_modulus,
            parameters.taylor_coefficient,
            parameters.dipole_min_distance_burgers,
            parameters.climb_frequency,
            parameters.climb_activation,
            parameters.twin_shear_override,
            parameters.twin_crss,
            parameters.twin_reference_rate,
            parameters.twin_rate_exponent,
            parameters.twin_stress_scale,
            parameters.twin_latent_hardening,
            parameters.twin_max_total_fraction,
            parameters.density_floor,
            parameters.determinant_tolerance,
            parameters.maximum_slip_increment,
            parameters.maximum_twin_increment,
        ]
    )
    arrays = (
        parameters.elastic_C0,
        parameters.burgers,
        parameters.tau0,
        parameters.tau_cut,
        parameters.activation_energy,
        parameters.reference_velocity,
        parameters.p,
        parameters.q,
        parameters.rho_mobile_0,
        parameters.rho_dipole_0,
        parameters.forest_interaction,
    )
    if not np.all(np.isfinite(scalar_values)) or any(
        not np.all(np.isfinite(values)) for values in arrays
    ):
        raise ValueError("material parameter contains NaN or Inf")
    positive_arrays = (
        parameters.burgers,
        parameters.tau_cut,
        parameters.activation_energy,
        parameters.reference_velocity,
        parameters.rho_mobile_0,
        parameters.rho_dipole_0,
    )
    if any(np.any(values <= 0.0) for values in positive_arrays):
        raise ValueError("strictly positive slip parameter is non-positive")
    if np.any(parameters.tau0 < 0.0):
        raise ValueError("additive slip resistance must be non-negative")
    if np.any(parameters.forest_interaction < 0.0):
        raise ValueError("forest interaction coefficients must be non-negative")
    if not np.all((parameters.p > 0.0) & (parameters.p <= 1.0)):
        raise ValueError("slip p must lie in (0,1]")
    if not np.all((parameters.q >= 1.0) & (parameters.q <= 2.0)):
        raise ValueError("slip q must lie in [1,2]")
    if not 0.0 <= parameters.taylor_quinney <= 1.0:
        raise ValueError("Taylor-Quinney coefficient must lie in [0,1]")
    if not 0.0 < parameters.twin_max_total_fraction < 1.0:
        raise ValueError("maximum total twin fraction must lie in (0,1)")
    if parameters.dipole_min_distance_burgers <= 0.0:
        raise ValueError("dipole minimum distance must be positive")
    if parameters.climb_frequency < 0.0 or parameters.climb_activation <= 0.0:
        raise ValueError("climb parameters are outside their admissible domain")
    if (
        parameters.c_over_a <= 0.0
        or parameters.T_ref <= 0.0
        or parameters.mass_density <= 0.0
        or parameters.heat_capacity <= 0.0
        or parameters.grain_size <= 0.0
        or parameters.mean_free_path_coefficient <= 0.0
        or parameters.reference_shear_modulus <= 0.0
        or parameters.taylor_coefficient < 0.0
        or parameters.twin_shear_override < 0.0
        or parameters.twin_crss < 0.0
        or parameters.twin_reference_rate < 0.0
        or parameters.twin_rate_exponent <= 0.0
        or parameters.twin_stress_scale <= 0.0
        or parameters.twin_latent_hardening < 0.0
        or parameters.density_floor <= 0.0
        or parameters.determinant_tolerance <= 0.0
        or parameters.maximum_slip_increment <= 0.0
        or parameters.maximum_twin_increment <= 0.0
    ):
        raise ValueError("scalar material/numerical parameter is outside its domain")
    if parameters.elastic_C0.shape != (6, 6) or parameters.forest_interaction.shape != (
        systems.n_slip,
        systems.n_slip,
    ):
        raise ValueError("parameter-array shape mismatch")
    parameters.elastic_matrix()
