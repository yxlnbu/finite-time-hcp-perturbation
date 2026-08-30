"""Versioned 105-real state with twin, damage, and DRX histories.

The V1 STATE92 layout remains untouched.  This V2 schema is an explicit state
expansion for mechanism-admission work and must never be read through the old
ABI by position alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .state_contract import TENSOR_COMPONENT_ORDER

Array = NDArray[np.float64]

STATE105_SCHEMA = "HCP_CP_LOCAL_STATE_105_V2"
N_SLIP = 18
N_TWIN_V2 = 12
N_STATE105 = 105

FP = slice(0, 9)
RHO_MOBILE = slice(9, 27)
RHO_DIPOLE = slice(27, 45)
GAMMA_SIGNED = slice(45, 63)
GAMMA_ABSOLUTE = slice(63, 81)
TWIN_FRACTION = slice(81, 93)
TEMPERATURE = 93
CP_WORK = 94
GENERATED_HEAT = 95
STORED_ENERGY = 96
TIME = 97
DAMAGE = 98
DAMAGE_HISTORY = 99
DRX_FRACTION = 100
DRX_BOUNDARY_AREA = 101
TWIN_DISSIPATION = 102
DAMAGE_DISSIPATION = 103
DRX_DISSIPATION = 104


def _frozen(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalState105:
    Fp: Array
    rho_mobile_m2: Array
    rho_dipole_m2: Array
    gamma_signed: Array
    Gamma_absolute: Array
    twin_fraction: Array
    temperature_K: float
    cp_work_density_J_m3: float
    generated_heat_density_J_m3: float
    stored_energy_density_J_m3: float
    time_s: float
    damage: float
    damage_history_J_m3: float
    drx_fraction: float
    drx_boundary_area_m_inv: float
    twin_dissipation_J_m3: float
    damage_dissipation_J_m3: float
    drx_dissipation_J_m3: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("Fp", (3, 3)),
            ("rho_mobile_m2", (N_SLIP,)),
            ("rho_dipole_m2", (N_SLIP,)),
            ("gamma_signed", (N_SLIP,)),
            ("Gamma_absolute", (N_SLIP,)),
            ("twin_fraction", (N_TWIN_V2,)),
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        scalars = np.array(
            [
                self.temperature_K,
                self.cp_work_density_J_m3,
                self.generated_heat_density_J_m3,
                self.stored_energy_density_J_m3,
                self.time_s,
                self.damage,
                self.damage_history_J_m3,
                self.drx_fraction,
                self.drx_boundary_area_m_inv,
                self.twin_dissipation_J_m3,
                self.damage_dissipation_J_m3,
                self.drx_dissipation_J_m3,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("all STATE105 scalars must be finite")

    def validate(
        self,
        *,
        twin_fraction_limit: float,
        determinant_tolerance: float,
        tolerance: float = 1.0e-12,
    ) -> None:
        controls = np.array(
            [twin_fraction_limit, determinant_tolerance, tolerance], dtype=float
        )
        if np.any(~np.isfinite(controls)) or np.any(controls <= 0.0):
            raise ValueError("STATE105 validation controls must be positive")
        determinant = float(np.linalg.det(self.Fp))
        if determinant <= 0.0 or abs(determinant - 1.0) > determinant_tolerance:
            raise ValueError("Fp violates plastic incompressibility")
        if np.any(self.rho_mobile_m2 <= 0.0) or np.any(self.rho_dipole_m2 <= 0.0):
            raise ValueError("dislocation densities must be strictly positive")
        if np.any(self.Gamma_absolute < -tolerance):
            raise ValueError("absolute accumulated slip must be non-negative")
        if np.any(np.abs(self.gamma_signed) > self.Gamma_absolute + tolerance):
            raise ValueError("signed slip exceeds accumulated total variation")
        if np.any(self.twin_fraction < -tolerance):
            raise ValueError("twin fractions must be non-negative")
        if float(np.sum(self.twin_fraction)) > twin_fraction_limit + tolerance:
            raise ValueError("total twin fraction exceeds the configured limit")
        if not -tolerance <= self.damage <= 1.0 + tolerance:
            raise ValueError("damage must lie in [0,1]")
        if not -tolerance <= self.drx_fraction <= 1.0 + tolerance:
            raise ValueError("DRX fraction must lie in [0,1]")
        if self.drx_boundary_area_m_inv < -tolerance:
            raise ValueError("DRX boundary area density must be non-negative")
        if self.drx_fraction > tolerance and self.drx_boundary_area_m_inv <= 0.0:
            raise ValueError("positive DRX fraction requires positive boundary area")
        if self.temperature_K <= 0.0 or self.time_s < -tolerance:
            raise ValueError("temperature must be positive and time non-negative")
        nonnegative = (
            self.cp_work_density_J_m3,
            self.generated_heat_density_J_m3,
            self.stored_energy_density_J_m3,
            self.damage_history_J_m3,
            self.twin_dissipation_J_m3,
            self.damage_dissipation_J_m3,
            self.drx_dissipation_J_m3,
        )
        if min(nonnegative) < -tolerance:
            raise ValueError("STATE105 histories and ledgers must be non-negative")
        scale = max(abs(self.cp_work_density_J_m3), 1.0)
        residual = (
            self.cp_work_density_J_m3
            - self.generated_heat_density_J_m3
            - self.stored_energy_density_J_m3
        )
        if abs(residual) > tolerance * scale:
            raise ValueError("legacy crystal-plastic work partition is inconsistent")

    def pack(self) -> Array:
        result = np.empty(N_STATE105, dtype=np.float64)
        result[FP] = self.Fp.reshape(-1)
        result[RHO_MOBILE] = self.rho_mobile_m2
        result[RHO_DIPOLE] = self.rho_dipole_m2
        result[GAMMA_SIGNED] = self.gamma_signed
        result[GAMMA_ABSOLUTE] = self.Gamma_absolute
        result[TWIN_FRACTION] = self.twin_fraction
        result[TEMPERATURE] = self.temperature_K
        result[CP_WORK] = self.cp_work_density_J_m3
        result[GENERATED_HEAT] = self.generated_heat_density_J_m3
        result[STORED_ENERGY] = self.stored_energy_density_J_m3
        result[TIME] = self.time_s
        result[DAMAGE] = self.damage
        result[DAMAGE_HISTORY] = self.damage_history_J_m3
        result[DRX_FRACTION] = self.drx_fraction
        result[DRX_BOUNDARY_AREA] = self.drx_boundary_area_m_inv
        result[TWIN_DISSIPATION] = self.twin_dissipation_J_m3
        result[DAMAGE_DISSIPATION] = self.damage_dissipation_J_m3
        result[DRX_DISSIPATION] = self.drx_dissipation_J_m3
        return result

    @classmethod
    def unpack(cls, packed: Any) -> "LocalState105":
        values = np.asarray(packed, dtype=np.float64)
        if values.shape != (N_STATE105,) or np.any(~np.isfinite(values)):
            raise ValueError(f"packed state must be a finite ({N_STATE105},) array")
        return cls(
            Fp=values[FP].reshape(3, 3),
            rho_mobile_m2=values[RHO_MOBILE],
            rho_dipole_m2=values[RHO_DIPOLE],
            gamma_signed=values[GAMMA_SIGNED],
            Gamma_absolute=values[GAMMA_ABSOLUTE],
            twin_fraction=values[TWIN_FRACTION],
            temperature_K=float(values[TEMPERATURE]),
            cp_work_density_J_m3=float(values[CP_WORK]),
            generated_heat_density_J_m3=float(values[GENERATED_HEAT]),
            stored_energy_density_J_m3=float(values[STORED_ENERGY]),
            time_s=float(values[TIME]),
            damage=float(values[DAMAGE]),
            damage_history_J_m3=float(values[DAMAGE_HISTORY]),
            drx_fraction=float(values[DRX_FRACTION]),
            drx_boundary_area_m_inv=float(values[DRX_BOUNDARY_AREA]),
            twin_dissipation_J_m3=float(values[TWIN_DISSIPATION]),
            damage_dissipation_J_m3=float(values[DAMAGE_DISSIPATION]),
            drx_dissipation_J_m3=float(values[DRX_DISSIPATION]),
        )


def initial_local_state105(
    *, rho_mobile_m2: Any, rho_dipole_m2: Any, temperature_K: float
) -> LocalState105:
    return LocalState105(
        Fp=np.eye(3),
        rho_mobile_m2=rho_mobile_m2,
        rho_dipole_m2=rho_dipole_m2,
        gamma_signed=np.zeros(N_SLIP),
        Gamma_absolute=np.zeros(N_SLIP),
        twin_fraction=np.zeros(N_TWIN_V2),
        temperature_K=float(temperature_K),
        cp_work_density_J_m3=0.0,
        generated_heat_density_J_m3=0.0,
        stored_energy_density_J_m3=0.0,
        time_s=0.0,
        damage=0.0,
        damage_history_J_m3=0.0,
        drx_fraction=0.0,
        drx_boundary_area_m_inv=0.0,
        twin_dissipation_J_m3=0.0,
        damage_dissipation_J_m3=0.0,
        drx_dissipation_J_m3=0.0,
    )


def upgrade_state92_to_state105(state92: Any) -> LocalState105:
    """Explicitly lift a V1 snapshot without reinterpreting its six twins.

    The old extension-twin fractions occupy the first six V2 entries.  The six
    contraction-twin variants and all new irreversible histories start at zero.
    """

    twin_v2 = np.r_[np.asarray(state92.twin_fraction, dtype=float), np.zeros(6)]
    return LocalState105(
        Fp=np.asarray(state92.Fp),
        rho_mobile_m2=np.asarray(state92.rho_mobile_m2),
        rho_dipole_m2=np.asarray(state92.rho_dipole_m2),
        gamma_signed=np.asarray(state92.gamma_signed),
        Gamma_absolute=np.asarray(state92.Gamma_absolute),
        twin_fraction=twin_v2,
        temperature_K=float(state92.temperature_K),
        cp_work_density_J_m3=float(state92.cp_work_density_J_m3),
        generated_heat_density_J_m3=float(state92.generated_heat_density_J_m3),
        stored_energy_density_J_m3=float(state92.stored_energy_density_J_m3),
        time_s=float(state92.time_s),
        damage=0.0,
        damage_history_J_m3=0.0,
        drx_fraction=0.0,
        drx_boundary_area_m_inv=0.0,
        twin_dissipation_J_m3=0.0,
        damage_dissipation_J_m3=0.0,
        drx_dissipation_J_m3=0.0,
    )


__all__ = [
    "LocalState105",
    "N_STATE105",
    "N_TWIN_V2",
    "STATE105_SCHEMA",
    "TENSOR_COMPONENT_ORDER",
    "initial_local_state105",
    "upgrade_state92_to_state105",
]
