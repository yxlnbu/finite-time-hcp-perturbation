"""Material-point state and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .parameters import MaterialParameters

Array = NDArray[np.float64]


@dataclass
class MaterialState:
    Fp: Array
    rho_mobile: Array
    rho_dipole: Array
    accumulated_slip: Array
    twin_fraction: Array
    temperature: float
    plastic_work_density: float = 0.0
    heat_density: float = 0.0
    stored_energy_density: float = 0.0
    time: float = 0.0

    @classmethod
    def initial(cls, parameters: MaterialParameters, n_twin: int) -> "MaterialState":
        return cls(
            Fp=np.eye(3),
            rho_mobile=parameters.rho_mobile_0.copy(),
            rho_dipole=parameters.rho_dipole_0.copy(),
            accumulated_slip=np.zeros_like(parameters.rho_mobile_0),
            twin_fraction=np.zeros(n_twin),
            temperature=parameters.T_ref,
        )

    def copy(self) -> "MaterialState":
        return replace(
            self,
            Fp=self.Fp.copy(),
            rho_mobile=self.rho_mobile.copy(),
            rho_dipole=self.rho_dipole.copy(),
            accumulated_slip=self.accumulated_slip.copy(),
            twin_fraction=self.twin_fraction.copy(),
        )

    def assert_physical(self, parameters: MaterialParameters, tolerance: float = 1.0e-12) -> None:
        n_slip = parameters.rho_mobile_0.size
        if self.Fp.shape != (3, 3):
            raise FloatingPointError("Fp must have shape (3,3)")
        for name, values in (
            ("rho_mobile", self.rho_mobile),
            ("rho_dipole", self.rho_dipole),
            ("accumulated_slip", self.accumulated_slip),
        ):
            if values.shape != (n_slip,):
                raise FloatingPointError(f"{name} has the wrong shape")
        # v0.1 deliberately fixes the extension-twin basis to six variants.
        if self.twin_fraction.shape != (6,):
            raise FloatingPointError("twin_fraction must have shape (6,) in v0.1")
        arrays = (
            self.Fp,
            self.rho_mobile,
            self.rho_dipole,
            self.accumulated_slip,
            self.twin_fraction,
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise FloatingPointError("material state contains NaN or Inf")
        scalar_values = np.array(
            [
                self.temperature,
                self.plastic_work_density,
                self.heat_density,
                self.stored_energy_density,
                self.time,
            ]
        )
        if not np.all(np.isfinite(scalar_values)):
            raise FloatingPointError("material scalar state contains NaN or Inf")
        if self.temperature <= 0.0:
            raise FloatingPointError("material temperature must be finite and positive")
        for name, values in (
            ("rho_mobile", self.rho_mobile),
            ("rho_dipole", self.rho_dipole),
        ):
            if np.min(values) <= 0.0:
                raise FloatingPointError(f"{name} must remain strictly positive")
        for name, values in (
            ("accumulated_slip", self.accumulated_slip),
            ("twin_fraction", self.twin_fraction),
        ):
            if np.min(values) < -tolerance:
                raise FloatingPointError(f"{name} left its admissible non-negative domain")
        total_twin = float(np.sum(self.twin_fraction))
        if total_twin > parameters.twin_max_total_fraction + tolerance:
            raise FloatingPointError("total twin fraction exceeds the configured upper bound")
        determinant = float(np.linalg.det(self.Fp))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise FloatingPointError("plastic deformation gradient has invalid determinant")
        if abs(determinant - 1.0) > parameters.determinant_tolerance:
            raise FloatingPointError(f"plastic incompressibility failed: det(Fp)={determinant}")
        if (
            self.plastic_work_density < -tolerance
            or self.heat_density < -tolerance
            or self.stored_energy_density < -tolerance
            or self.time < -tolerance
        ):
            raise FloatingPointError("cumulative work/heat became negative")
        ledger_residual = (
            self.plastic_work_density
            - self.heat_density
            - self.stored_energy_density
        )
        if abs(ledger_residual) > 1.0e-10 * max(
            abs(self.plastic_work_density),
            abs(self.heat_density),
            abs(self.stored_energy_density),
            1.0,
        ):
            raise FloatingPointError("cumulative plastic-work partition ledger is inconsistent")

    def flat_internal(self) -> Array:
        return np.concatenate(
            (
                self.Fp.reshape(-1),
                self.rho_mobile,
                self.rho_dipole,
                self.accumulated_slip,
                self.twin_fraction,
                np.array(
                    [
                        self.temperature,
                        self.plastic_work_density,
                        self.heat_density,
                        self.stored_energy_density,
                        self.time,
                    ]
                ),
            )
        )
