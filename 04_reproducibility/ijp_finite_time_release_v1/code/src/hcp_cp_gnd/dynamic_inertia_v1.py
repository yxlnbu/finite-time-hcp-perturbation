"""Inertial extension of the continuous U903 descriptor.

This versioned module leaves the accepted quasi-static descriptor untouched.
It adds only the reference-density displacement inertia required by the
quadratic pencil

    Q(s, k, n) = K(k, n) + s E(k, n) + s**2 M.

The frozen 84-coordinate order is ``[u(3), T, zeta(18), active(62)]``.  Only
the three displacement coordinates carry second-order inertia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .qs_descriptor import N_QS, U_SLICE, QSDescriptorAssembly


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

DYNAMIC_INERTIA_SCHEMA = "HCP_CP_U903_DYNAMIC_INERTIA_V1"


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_complex_matrix(value: Any, name: str) -> ComplexArray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (N_QS, N_QS):
        raise ValueError(f"{name} must have shape {(N_QS, N_QS)}")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


def continuous_mass_matrix(reference_density_kg_m3: float) -> RealArray:
    """Return the continuous 84-by-84 inertia matrix.

    The displacement amplitudes have units of length, so the momentum balance
    contributes ``rho0 * s**2 * u``.  No fictitious inertia is assigned to
    temperature, micromorphic slips, or local constitutive variables.
    """

    density = _positive_finite(reference_density_kg_m3, "reference density")
    mass = np.zeros((N_QS, N_QS), dtype=np.float64)
    mass[U_SLICE, U_SLICE] = density * np.eye(U_SLICE.stop - U_SLICE.start)
    mass.setflags(write=False)
    return mass


@dataclass(frozen=True)
class DynamicDescriptorAssemblyV1:
    """Quadratic-in-growth-rate extension of one accepted QS assembly."""

    stiffness_K: ComplexArray
    descriptor_E: ComplexArray
    inertia_M: RealArray
    reference_density_kg_m3: float
    schema: str = DYNAMIC_INERTIA_SCHEMA

    def __post_init__(self) -> None:
        stiffness = _finite_complex_matrix(self.stiffness_K, "K").copy()
        descriptor = _finite_complex_matrix(self.descriptor_E, "E").copy()
        inertia = np.asarray(self.inertia_M, dtype=np.float64)
        if inertia.shape != (N_QS, N_QS) or not np.all(np.isfinite(inertia)):
            raise ValueError("M must be a finite 84-by-84 real matrix")
        density = _positive_finite(self.reference_density_kg_m3, "reference density")
        expected = continuous_mass_matrix(density)
        if not np.array_equal(inertia, expected):
            raise ValueError("M must equal rho0*I on u and zero on all other coordinates")
        stiffness.setflags(write=False)
        descriptor.setflags(write=False)
        inertia = inertia.copy()
        inertia.setflags(write=False)
        object.__setattr__(self, "stiffness_K", stiffness)
        object.__setattr__(self, "descriptor_E", descriptor)
        object.__setattr__(self, "inertia_M", inertia)
        object.__setattr__(self, "reference_density_kg_m3", density)

    def pencil(self, growth_rate_s_inv: complex) -> ComplexArray:
        """Evaluate ``K + s E + s**2 M`` without linearizing the singular QEP."""

        growth = complex(growth_rate_s_inv)
        if not np.isfinite(growth.real) or not np.isfinite(growth.imag):
            raise ValueError("growth rate must be finite")
        return self.stiffness_K + growth * self.descriptor_E + growth**2 * self.inertia_M

    def residual(self, growth_rate_s_inv: complex, state: Any) -> ComplexArray:
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (N_QS,):
            raise ValueError("dynamic state must be an 84-vector")
        if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
            raise ValueError("dynamic state must be finite")
        return self.pencil(growth_rate_s_inv) @ vector


def augment_qs_descriptor(
    descriptor: QSDescriptorAssembly,
    *,
    reference_density_kg_m3: float,
) -> DynamicDescriptorAssemblyV1:
    """Add inertia to an accepted QS assembly without modifying it."""

    if not isinstance(descriptor, QSDescriptorAssembly):
        raise TypeError("descriptor must be a QSDescriptorAssembly")
    density = _positive_finite(reference_density_kg_m3, "reference density")
    return DynamicDescriptorAssemblyV1(
        stiffness_K=descriptor.stiffness_K,
        descriptor_E=descriptor.descriptor_E,
        inertia_M=continuous_mass_matrix(density),
        reference_density_kg_m3=density,
    )


def kinetic_energy_density_J_m3(
    reference_density_kg_m3: float,
    velocity_m_s: Any,
) -> float:
    """Return ``0.5*rho0*v.v`` for a material-point velocity."""

    density = _positive_finite(reference_density_kg_m3, "reference density")
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be a finite three-vector")
    return 0.5 * density * float(velocity @ velocity)


@dataclass(frozen=True)
class DynamicEnergyLedgerV1:
    """Element-level dynamic energy publication without double counting heat.

    ``generated_heat_J`` and ``cp_work_J`` are cumulative transfer ledgers;
    they are deliberately excluded from ``stored_total_J`` because the current
    thermal internal energy already contains the accepted temperature state.
    """

    kinetic_J: float
    recoverable_J: float
    passive_storage_J: float
    thermal_internal_J: float
    cp_work_J: float
    generated_heat_J: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.kinetic_J,
                self.recoverable_J,
                self.passive_storage_J,
                self.thermal_internal_J,
                self.cp_work_J,
                self.generated_heat_J,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("dynamic energy ledger entries must be finite")
        if np.any(values[:4] < 0.0):
            raise ValueError("stored dynamic energy entries must be non-negative")

    @property
    def stored_internal_J(self) -> float:
        return self.recoverable_J + self.passive_storage_J + self.thermal_internal_J

    @property
    def stored_total_J(self) -> float:
        return self.kinetic_J + self.stored_internal_J

    @property
    def cp_partition_residual_J(self) -> float:
        return self.cp_work_J - self.generated_heat_J - self.passive_storage_J


__all__ = [
    "DYNAMIC_INERTIA_SCHEMA",
    "DynamicDescriptorAssemblyV1",
    "DynamicEnergyLedgerV1",
    "augment_qs_descriptor",
    "continuous_mass_matrix",
    "kinetic_energy_density_J_m3",
]
