"""Energy-conjugate dislocation storage and evolving work-to-heat partition.

The module deliberately leaves the accepted v0.1 material kernel untouched.
It supplies (i) a side-effect-free continuous partition law used by the
spectral export and (ii) a material-point subclass whose accepted substeps use
the same defect-energy increment.  The default law is recovered nowhere by
clipping: inadmissible negative heat production raises an error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np


_V01 = Path(__file__).resolve().parents[3] / "HCP_CP_v0.1" / "src"
if str(_V01) not in sys.path:
    sys.path.insert(0, str(_V01))

from hcp_cp.model import HCPMaterialPoint  # noqa: E402


@dataclass(frozen=True)
class WorkPartitionResultV1:
    cp_power_W_m3: float
    heat_source_W_m3: float
    storage_rate_W_m3: float
    beta_eff: float
    power_partition_residual_W_m3: float


@dataclass(frozen=True)
class DislocationStoredEnergyPartitionV1:
    """Line-energy closure ``psi=c_rho*mu*sum(b^2 rho)``.

    ``line_energy_factor`` is an explicitly declared model coefficient rather
    than an identified material parameter.  The paper contract registers 0.5
    as the baseline and 0.25--1.0 as the sensitivity interval.
    """

    line_energy_factor: float = 0.5
    relative_tolerance: float = 1.0e-10

    def validate(self) -> None:
        values = np.asarray(
            [self.line_energy_factor, self.relative_tolerance], dtype=float
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("partition coefficients must be finite and positive")

    def energy_density_J_m3(
        self,
        rho_mobile_m2: Any,
        rho_dipole_m2: Any,
        *,
        burgers_m: Any,
        shear_modulus_Pa: float,
    ) -> float:
        self.validate()
        mobile = np.asarray(rho_mobile_m2, dtype=float)
        dipole = np.asarray(rho_dipole_m2, dtype=float)
        burgers = np.asarray(burgers_m, dtype=float)
        if (
            mobile.shape != dipole.shape
            or mobile.shape != burgers.shape
            or np.any(~np.isfinite(mobile))
            or np.any(~np.isfinite(dipole))
            or np.any(~np.isfinite(burgers))
            or np.any(mobile < 0.0)
            or np.any(dipole < 0.0)
            or np.any(burgers <= 0.0)
            or not np.isfinite(shear_modulus_Pa)
            or shear_modulus_Pa <= 0.0
        ):
            raise ValueError("invalid dislocation-energy state")
        return float(
            self.line_energy_factor
            * float(shear_modulus_Pa)
            * np.sum(burgers * burgers * (mobile + dipole))
        )

    def partition_rate(
        self,
        *,
        cp_power_W_m3: float,
        rho_mobile_rate_m2_s: Any,
        rho_dipole_rate_m2_s: Any,
        burgers_m: Any,
        shear_modulus_Pa: float,
    ) -> WorkPartitionResultV1:
        self.validate()
        cp_power = float(cp_power_W_m3)
        mobile_rate = np.asarray(rho_mobile_rate_m2_s, dtype=float)
        dipole_rate = np.asarray(rho_dipole_rate_m2_s, dtype=float)
        burgers = np.asarray(burgers_m, dtype=float)
        if (
            mobile_rate.shape != dipole_rate.shape
            or mobile_rate.shape != burgers.shape
            or np.any(~np.isfinite(mobile_rate))
            or np.any(~np.isfinite(dipole_rate))
            or np.any(~np.isfinite(burgers))
            or np.any(burgers <= 0.0)
            or not np.isfinite(cp_power)
            or cp_power < 0.0
            or not np.isfinite(shear_modulus_Pa)
            or shear_modulus_Pa <= 0.0
        ):
            raise ValueError("invalid work-partition rate state")
        storage_rate = float(
            self.line_energy_factor
            * float(shear_modulus_Pa)
            * np.sum(burgers * burgers * (mobile_rate + dipole_rate))
        )
        heat_source = cp_power - storage_rate
        scale = max(abs(cp_power), abs(storage_rate), 1.0)
        if heat_source < -self.relative_tolerance * scale:
            raise FloatingPointError(
                "dislocation storage exceeds crystal-plastic power"
            )
        if heat_source < 0.0:
            heat_source = 0.0
            storage_rate = cp_power
        beta_eff = (
            heat_source / cp_power
            if cp_power > self.relative_tolerance
            else 1.0
        )
        residual = cp_power - heat_source - storage_rate
        return WorkPartitionResultV1(
            cp_power_W_m3=cp_power,
            heat_source_W_m3=float(heat_source),
            storage_rate_W_m3=float(storage_rate),
            beta_eff=float(beta_eff),
            power_partition_residual_W_m3=float(residual),
        )


class EvolvingPartitionHCPMaterialPointV1(HCPMaterialPoint):
    """HCP point using the line-energy increment on every accepted substep."""

    def __init__(self, *args: Any, partition_law: DislocationStoredEnergyPartitionV1,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        partition_law.validate()
        self.partition_law = partition_law

    def _advance_substep(self, F_sample: np.ndarray, state: Any, dt: float):
        response = self.evaluate(F_sample, state)
        mobile, dipole = self._density_update(state, response, dt)
        p = self.parameters
        before = self.partition_law.energy_density_J_m3(
            state.rho_mobile,
            state.rho_dipole,
            burgers_m=p.burgers,
            shear_modulus_Pa=p.reference_shear_modulus,
        )
        after = self.partition_law.energy_density_J_m3(
            mobile,
            dipole,
            burgers_m=p.burgers,
            shear_modulus_Pa=p.reference_shear_modulus,
        )
        storage_increment = after - before
        plastic_increment = dt * float(response.plastic_dissipation)
        scale = max(abs(plastic_increment), abs(storage_increment), 1.0)
        if storage_increment > plastic_increment + self.partition_law.relative_tolerance * scale:
            raise FloatingPointError(
                "accepted substep stores more defect energy than plastic work"
            )
        if plastic_increment <= self.partition_law.relative_tolerance:
            if abs(storage_increment) > self.partition_law.relative_tolerance * scale:
                raise FloatingPointError("defect energy changed without plastic work")
            beta_eff = 1.0
        else:
            beta_eff = 1.0 - storage_increment / plastic_increment
        # A temporary base-class point reuses the independently tested update;
        # only its energy split differs.  beta_eff is intentionally not clipped.
        temporary = HCPMaterialPoint(
            self.systems,
            replace(self.parameters, taylor_quinney=float(beta_eff)),
            self.orientation,
            self.switches,
        )
        updated, accepted_response = temporary._advance_substep(F_sample, state, dt)
        ledger_scale = max(abs(updated.plastic_work_density), 1.0)
        residual = (
            updated.plastic_work_density
            - updated.heat_density
            - updated.stored_energy_density
        )
        if abs(residual) > self.partition_law.relative_tolerance * ledger_scale:
            raise FloatingPointError("evolving partition ledger failed")
        if updated.stored_energy_density < -self.partition_law.relative_tolerance * ledger_scale:
            raise FloatingPointError("cumulative defect-storage ledger became negative")
        return updated, accepted_response


__all__ = [
    "DislocationStoredEnergyPartitionV1",
    "EvolvingPartitionHCPMaterialPointV1",
    "WorkPartitionResultV1",
]
