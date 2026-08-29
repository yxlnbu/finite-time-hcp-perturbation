"""Versioned 92-real local-state contract for HCP_CP v0.2.

The micromorphic slip ``zeta`` is a global nodal unknown and is deliberately
absent from this integration-point state.  The fixed layout is suitable for a
UEL ``SVARS`` block, neutral snapshots, and independent Python/Fortran replay.
It does not prescribe how Abaqus manages committed/trial copies; the UEL layer
must still implement and verify that transaction policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

STATE_SCHEMA = "HCP_CP_LOCAL_STATE_92_V1"
TENSOR_COMPONENT_ORDER = ("11", "12", "13", "21", "22", "23", "31", "32", "33")
N_SLIP = 18
N_TWIN = 6
N_STATE = 92

FP = slice(0, 9)
RHO_MOBILE = slice(9, 27)
RHO_DIPOLE = slice(27, 45)
GAMMA_SIGNED = slice(45, 63)
GAMMA_ABSOLUTE = slice(63, 81)
TWIN_FRACTION = slice(81, 87)
TEMPERATURE = 87
CP_WORK = 88
GENERATED_HEAT = 89
STORED_ENERGY = 90
TIME = 91


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
class LocalState92:
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

    def __post_init__(self) -> None:
        for name, shape in (
            ("Fp", (3, 3)),
            ("rho_mobile_m2", (N_SLIP,)),
            ("rho_dipole_m2", (N_SLIP,)),
            ("gamma_signed", (N_SLIP,)),
            ("Gamma_absolute", (N_SLIP,)),
            ("twin_fraction", (N_TWIN,)),
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        scalars = np.array(
            [
                self.temperature_K,
                self.cp_work_density_J_m3,
                self.generated_heat_density_J_m3,
                self.stored_energy_density_J_m3,
                self.time_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("all local-state scalars must be finite")

    def validate(
        self,
        *,
        twin_fraction_limit: float,
        determinant_tolerance: float,
        tolerance: float = 1.0e-12,
    ) -> None:
        limits = np.array(
            [twin_fraction_limit, determinant_tolerance, tolerance], dtype=np.float64
        )
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("state validation limits must be finite and positive")
        determinant = float(np.linalg.det(self.Fp))
        if (
            not np.isfinite(determinant)
            or determinant <= 0.0
            or abs(determinant - 1.0) > determinant_tolerance
        ):
            raise ValueError("Fp violates plastic incompressibility")
        if np.any(self.rho_mobile_m2 <= 0.0) or np.any(self.rho_dipole_m2 <= 0.0):
            raise ValueError("dislocation densities must be strictly positive")
        if np.any(self.Gamma_absolute < -tolerance):
            raise ValueError("absolute accumulated slip must be non-negative")
        if np.any(np.abs(self.gamma_signed) > self.Gamma_absolute + tolerance):
            raise ValueError("signed slip exceeds its accumulated total variation")
        if np.any(self.twin_fraction < -tolerance):
            raise ValueError("twin fractions must be non-negative")
        if float(np.sum(self.twin_fraction)) > twin_fraction_limit + tolerance:
            raise ValueError("total twin fraction exceeds the configured limit")
        if self.temperature_K <= 0.0 or self.time_s < -tolerance:
            raise ValueError("temperature must be positive and time non-negative")
        if min(
            self.cp_work_density_J_m3,
            self.generated_heat_density_J_m3,
            self.stored_energy_density_J_m3,
        ) < -tolerance:
            raise ValueError("cumulative work/heat/stored ledgers must be non-negative")
        scale = max(abs(self.cp_work_density_J_m3), 1.0)
        residual = (
            self.cp_work_density_J_m3
            - self.generated_heat_density_J_m3
            - self.stored_energy_density_J_m3
        )
        if abs(residual) > tolerance * scale:
            raise ValueError("cumulative work partition is inconsistent")

    def pack(self) -> Array:
        """Pack using explicit row-major ``11,12,13,21,...,33`` for ``Fp``.

        A Fortran implementation must use explicit component loops matching
        this order; a bare Fortran ``reshape`` would use a different memory
        convention and is not part of the contract.
        """

        result = np.empty(N_STATE, dtype=np.float64)
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
        return result

    @classmethod
    def unpack(cls, packed: Array) -> "LocalState92":
        values = np.asarray(packed, dtype=np.float64)
        if values.shape != (N_STATE,) or not np.all(np.isfinite(values)):
            raise ValueError(f"packed state must be a finite ({N_STATE},) array")
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
        )


def initial_local_state(
    *,
    rho_mobile_m2: Array,
    rho_dipole_m2: Array,
    temperature_K: float,
) -> LocalState92:
    return LocalState92(
        Fp=np.eye(3),
        rho_mobile_m2=rho_mobile_m2,
        rho_dipole_m2=rho_dipole_m2,
        gamma_signed=np.zeros(N_SLIP),
        Gamma_absolute=np.zeros(N_SLIP),
        twin_fraction=np.zeros(N_TWIN),
        temperature_K=temperature_K,
        cp_work_density_J_m3=0.0,
        generated_heat_density_J_m3=0.0,
        stored_energy_density_J_m3=0.0,
        time_s=0.0,
    )
