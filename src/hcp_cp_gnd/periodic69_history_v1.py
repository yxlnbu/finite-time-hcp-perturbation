"""Non-autonomous one-dimensional periodic evolution on the registered base path.

The active perturbation has the registered 69-coordinate order
``[u,v,T,theta_p,rho_m,rho_d,gamma]``.  Eighteen accumulated absolute slips
and the work/heat/storage ledgers are advanced as passive history variables.
At every time-stage the homogeneous registered base is subtracted from the
full nonlinear residual.  Consequently the sampled base path is an exact
zero solution even when interpolation and the material-point time integrator
have a small rate defect.

The local ``SL(3)`` chart is recentered on the interpolated base ``Fp``.  This
is the nonlinear continuation of the registered frozen-generator convention;
it is not a new global additive parametrization of ``Fp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .cp_ti_material_v1 import simple_shear
from .dynamic_crystal_perturbation_v1 import (
    GENERATOR_GAMMA_SLICE,
    GENERATOR_RHO_DIPOLE_SLICE,
    GENERATOR_RHO_MOBILE_SLICE,
    GENERATOR_T_SLICE,
    GENERATOR_THETA_SLICE,
    GENERATOR_U_SLICE,
    GENERATOR_V_SLICE,
    N_GENERATOR,
)
from .periodic69_nonlinear_v1 import Periodic69CheckpointV1
from .sl3_chart import SL3LocalChart
from .spectral_export import (
    ContinuousSpectralPointModel,
    SpectralActiveState62,
    SpectralObserverState,
)
from .state_contract import LocalState92


RealArray = NDArray[np.float64]

PASSIVE_GAMMA_ABSOLUTE_SLICE = slice(0, 18)
PASSIVE_CP_WORK = 18
PASSIVE_GENERATED_HEAT = 19
PASSIVE_STORED_ENERGY = 20
N_PASSIVE_HISTORY = 21


def _finite(value: Any, shape: tuple[int, ...], name: str) -> RealArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class RegisteredBaseFrameV1:
    """Interpolated homogeneous base data for one non-autonomous stage."""

    F_sample: RealArray
    temperature_K: float
    chart: SL3LocalChart
    active: SpectralActiveState62
    observer: SpectralObserverState


@dataclass
class HistoryDiagnosticsV1:
    """Running admissibility and balance audit collected during integration."""

    rhs_evaluations: int = 0
    material_point_evaluations: int = 0
    maximum_power_identity_relative_residual: float = 0.0
    maximum_power_partition_relative_residual: float = 0.0
    minimum_temperature_K: float = np.inf
    minimum_dislocation_density_m2: float = np.inf
    minimum_Gamma_absolute: float = np.inf
    minimum_cp_work_density_J_m3: float = np.inf
    minimum_generated_heat_density_J_m3: float = np.inf
    minimum_stored_energy_density_J_m3: float = np.inf
    maximum_signed_slip_excess: float = 0.0

    def observe_raw(self, raw: Any) -> None:
        power_scale = max(abs(float(raw.cp_power_W_m3)), 1.0)
        self.maximum_power_identity_relative_residual = max(
            self.maximum_power_identity_relative_residual,
            abs(float(raw.power_identity_residual_W_m3)) / power_scale,
        )
        self.maximum_power_partition_relative_residual = max(
            self.maximum_power_partition_relative_residual,
            abs(float(raw.power_partition_residual_W_m3)) / power_scale,
        )
        self.material_point_evaluations += 1

    def observe_state(
        self,
        absolute_q: RealArray,
        Gamma_absolute: RealArray,
        ledgers: RealArray,
    ) -> None:
        self.minimum_temperature_K = min(
            self.minimum_temperature_K,
            float(np.min(absolute_q[:, GENERATOR_T_SLICE])),
        )
        self.minimum_dislocation_density_m2 = min(
            self.minimum_dislocation_density_m2,
            float(np.min(absolute_q[:, GENERATOR_RHO_MOBILE_SLICE])),
            float(np.min(absolute_q[:, GENERATOR_RHO_DIPOLE_SLICE])),
        )
        self.minimum_Gamma_absolute = min(
            self.minimum_Gamma_absolute, float(np.min(Gamma_absolute))
        )
        self.minimum_cp_work_density_J_m3 = min(
            self.minimum_cp_work_density_J_m3, float(np.min(ledgers[:, 0]))
        )
        self.minimum_generated_heat_density_J_m3 = min(
            self.minimum_generated_heat_density_J_m3, float(np.min(ledgers[:, 1]))
        )
        self.minimum_stored_energy_density_J_m3 = min(
            self.minimum_stored_energy_density_J_m3, float(np.min(ledgers[:, 2]))
        )
        excess = np.abs(absolute_q[:, GENERATOR_GAMMA_SLICE]) - Gamma_absolute
        self.maximum_signed_slip_excess = max(
            self.maximum_signed_slip_excess, float(np.max(excess, initial=0.0))
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rhs_evaluations": self.rhs_evaluations,
            "material_point_evaluations": self.material_point_evaluations,
            "maximum_power_identity_relative_residual": self.maximum_power_identity_relative_residual,
            "maximum_power_partition_relative_residual": self.maximum_power_partition_relative_residual,
            "minimum_temperature_K": self.minimum_temperature_K,
            "minimum_dislocation_density_m2": self.minimum_dislocation_density_m2,
            "minimum_Gamma_absolute": self.minimum_Gamma_absolute,
            "minimum_cp_work_density_J_m3": self.minimum_cp_work_density_J_m3,
            "minimum_generated_heat_density_J_m3": self.minimum_generated_heat_density_J_m3,
            "minimum_stored_energy_density_J_m3": self.minimum_stored_energy_density_J_m3,
            "maximum_signed_slip_excess": self.maximum_signed_slip_excess,
        }


@dataclass(frozen=True)
class Periodic69HistoryV1:
    """Defect-corrected nonlinear perturbation evolution along a base history."""

    spectral_model: ContinuousSpectralPointModel
    base_shears: RealArray
    base_times_s: RealArray
    base_storages: Sequence[LocalState92]
    direction_n: RealArray
    domain_length_m: float
    cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.spectral_model, ContinuousSpectralPointModel):
            raise TypeError("spectral_model must be ContinuousSpectralPointModel")
        times = np.asarray(self.base_times_s, dtype=np.float64).copy()
        shears = np.asarray(self.base_shears, dtype=np.float64).copy()
        if (
            times.ndim != 1
            or times.size < 2
            or shears.shape != times.shape
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(shears))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("base time/shear arrays must be matching finite histories")
        storages = tuple(self.base_storages)
        if len(storages) != times.size or not all(
            isinstance(item, LocalState92) for item in storages
        ):
            raise ValueError("one LocalState92 is required at every base time")
        if not np.allclose(
            [item.time_s for item in storages], times, rtol=0.0, atol=2.0e-18
        ):
            raise ValueError("storage times do not match registered base times")
        direction = _finite(self.direction_n, (3,), "direction_n").copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("direction_n must be nonzero")
        direction /= norm
        length = float(self.domain_length_m)
        cells = int(self.cells)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("domain_length_m must be finite and positive")
        if cells < 4 or cells % 2:
            raise ValueError("periodic grid needs an even cell count of at least four")

        determinant_tolerance = self.spectral_model.base_model.parameters.determinant_tolerance
        interval_coordinates = []
        for left, right in zip(storages[:-1], storages[1:], strict=True):
            chart = SL3LocalChart(
                left.Fp, determinant_tolerance=determinant_tolerance
            )
            interval_coordinates.append(chart.coordinates(right.Fp))

        spatial = Periodic69CheckpointV1(
            self.spectral_model,
            simple_shear(float(shears[0])),
            storages[0],
            direction,
            length,
            cells,
        )
        times.setflags(write=False)
        shears.setflags(write=False)
        direction.setflags(write=False)
        coordinates = np.asarray(interval_coordinates, dtype=np.float64)
        coordinates.setflags(write=False)
        object.__setattr__(self, "base_times_s", times)
        object.__setattr__(self, "base_shears", shears)
        object.__setattr__(self, "base_storages", storages)
        object.__setattr__(self, "direction_n", direction)
        object.__setattr__(self, "domain_length_m", length)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "_interval_Fp_coordinates", coordinates)
        object.__setattr__(self, "_spatial", spatial)
        object.__setattr__(self, "_frame_cache", {})
        object.__setattr__(self, "_base_raw_cache", {})
        object.__setattr__(self, "_base_field_cache", {})
        object.__setattr__(self, "_exponential_cache", {})

    @property
    def wave_numbers_m_inv(self) -> RealArray:
        return self._spatial.wave_numbers_m_inv

    def derivative(self, field: RealArray) -> RealArray:
        return self._spatial._derivative(field)

    def solve_micromorphic_slip(self, gamma_signed: RealArray) -> RealArray:
        return self._spatial.solve_micromorphic_slip(gamma_signed)

    def enforce_state92_gamma_bound(
        self,
        interval_index: int,
        fraction: float,
        active_delta: Any,
        passive_delta: Any,
        *,
        reference_gamma_signed: Any | None = None,
        reference_Gamma_absolute: Any | None = None,
    ) -> dict[str, float | int]:
        """Apply the minimum total-variation correction ``Gamma>=abs(gamma)``.

        Exact evolution preserves this inequality when it is true initially.
        A split exponential update of the active state and midpoint quadrature
        of the passive history can leave a small deficit.  This operation adds
        only that componentwise deficit and reports it for refinement audits.
        """

        q = _finite(active_delta, (self.cells, N_GENERATOR), "active_delta")
        h = _finite(
            passive_delta, (self.cells, N_PASSIVE_HISTORY), "passive_delta"
        )
        frame = self.base_frame(interval_index, fraction)
        gamma = (
            frame.active.gamma_signed[None, :]
            + q[:, GENERATOR_GAMMA_SLICE]
        )
        Gamma = (
            frame.observer.Gamma_absolute[None, :]
            + h[:, PASSIVE_GAMMA_ABSOLUTE_SLICE]
        )
        required = np.abs(gamma)
        if (reference_gamma_signed is None) != (reference_Gamma_absolute is None):
            raise ValueError(
                "reference gamma and Gamma must either both be supplied or both omitted"
            )
        if reference_gamma_signed is not None:
            gamma_reference = _finite(
                reference_gamma_signed,
                (self.cells, 18),
                "reference_gamma_signed",
            )
            Gamma_reference = _finite(
                reference_Gamma_absolute,
                (self.cells, 18),
                "reference_Gamma_absolute",
            )
            required = np.maximum(
                required,
                Gamma_reference + np.abs(gamma - gamma_reference),
            )
        correction = np.maximum(required - Gamma, 0.0)
        h[:, PASSIVE_GAMMA_ABSOLUTE_SLICE] += correction
        return {
            "maximum_correction": float(np.max(correction, initial=0.0)),
            "total_correction_l1": float(np.sum(correction)),
            "corrected_component_count": int(np.count_nonzero(correction)),
        }

    def base_frame(self, interval_index: int, fraction: float) -> RegisteredBaseFrameV1:
        index = int(interval_index)
        if index < 0 or index >= len(self.base_times_s) - 1:
            raise IndexError("base interval index is out of range")
        value = float(fraction)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("interval fraction must lie in [0,1]")
        key = (index, value)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        left = self.base_storages[index]
        right = self.base_storages[index + 1]
        determinant_tolerance = self.spectral_model.base_model.parameters.determinant_tolerance
        left_chart = SL3LocalChart(
            left.Fp, determinant_tolerance=determinant_tolerance
        )
        Fp = left_chart.matrix(value * self._interval_Fp_coordinates[index])
        chart = SL3LocalChart(Fp, determinant_tolerance=determinant_tolerance)

        def blend(a: Any, b: Any) -> RealArray:
            return (1.0 - value) * np.asarray(a, dtype=float) + value * np.asarray(b, dtype=float)

        active = SpectralActiveState62(
            theta_p=np.zeros(8),
            rho_mobile_m2=blend(left.rho_mobile_m2, right.rho_mobile_m2),
            rho_dipole_m2=blend(left.rho_dipole_m2, right.rho_dipole_m2),
            gamma_signed=blend(left.gamma_signed, right.gamma_signed),
        )
        cp_work = (1.0 - value) * left.cp_work_density_J_m3 + value * right.cp_work_density_J_m3
        heat = (
            (1.0 - value) * left.generated_heat_density_J_m3
            + value * right.generated_heat_density_J_m3
        )
        # Enforce the exact ledger identity after interpolation; this removes
        # only roundoff because every registered endpoint obeys W=Q+S.
        stored = cp_work - heat
        observer = SpectralObserverState(
            Gamma_absolute=blend(left.Gamma_absolute, right.Gamma_absolute),
            cp_work_density_J_m3=float(cp_work),
            generated_heat_density_J_m3=float(heat),
            passive_storage_density_J_m3=float(stored),
            time_s=float(
                (1.0 - value) * self.base_times_s[index]
                + value * self.base_times_s[index + 1]
            ),
        )
        shear = (1.0 - value) * self.base_shears[index] + value * self.base_shears[index + 1]
        temperature = (
            (1.0 - value) * left.temperature_K + value * right.temperature_K
        )
        frame = RegisteredBaseFrameV1(
            simple_shear(float(shear)), float(temperature), chart, active, observer
        )
        self._frame_cache[key] = frame
        return frame

    def base_generator_field(self, frame: RegisteredBaseFrameV1) -> RealArray:
        key = (frame.observer.time_s, self.cells)
        cached = self._base_field_cache.get(key)
        if cached is not None:
            return cached.copy()
        base = np.zeros(N_GENERATOR, dtype=np.float64)
        base[GENERATOR_T_SLICE] = frame.temperature_K
        base[GENERATOR_THETA_SLICE] = 0.0
        base[GENERATOR_RHO_MOBILE_SLICE] = frame.active.rho_mobile_m2
        base[GENERATOR_RHO_DIPOLE_SLICE] = frame.active.rho_dipole_m2
        base[GENERATOR_GAMMA_SLICE] = frame.active.gamma_signed
        field = np.broadcast_to(base, (self.cells, N_GENERATOR)).copy()
        self._base_field_cache[key] = field
        return field.copy()

    def rhs(
        self,
        interval_index: int,
        fraction: float,
        active_delta: Any,
        passive_delta: Any,
        diagnostics: HistoryDiagnosticsV1 | None = None,
    ) -> tuple[RealArray, RealArray]:
        """Evaluate the non-autonomous active and passive deviation rates."""

        q_delta = _finite(
            active_delta, (self.cells, N_GENERATOR), "active_delta"
        )
        h_delta = _finite(
            passive_delta, (self.cells, N_PASSIVE_HISTORY), "passive_delta"
        )
        frame = self.base_frame(interval_index, fraction)
        base_field = self.base_generator_field(frame)
        absolute_q = base_field + q_delta
        if np.any(absolute_q[:, GENERATOR_T_SLICE] <= 0.0):
            raise ValueError("temperature field left its admissible domain")
        if (
            np.any(absolute_q[:, GENERATOR_RHO_MOBILE_SLICE] <= 0.0)
            or np.any(absolute_q[:, GENERATOR_RHO_DIPOLE_SLICE] <= 0.0)
        ):
            raise ValueError("dislocation density field left its admissible domain")

        Gamma = frame.observer.Gamma_absolute[None, :] + h_delta[:, PASSIVE_GAMMA_ABSOLUTE_SLICE]
        ledgers = np.column_stack(
            (
                frame.observer.cp_work_density_J_m3 + h_delta[:, PASSIVE_CP_WORK],
                frame.observer.generated_heat_density_J_m3 + h_delta[:, PASSIVE_GENERATED_HEAT],
                frame.observer.passive_storage_density_J_m3 + h_delta[:, PASSIVE_STORED_ENERGY],
            )
        )
        if diagnostics is not None:
            diagnostics.rhs_evaluations += 1
            diagnostics.observe_state(absolute_q, Gamma, ledgers)
        if np.any(Gamma < -1.0e-13):
            raise ValueError("accumulated absolute slip became negative")
        ledger_scale = np.maximum(np.abs(ledgers[:, 0]), 1.0)
        if np.any(np.abs(ledgers[:, 0] - ledgers[:, 1] - ledgers[:, 2]) > 2.0e-10 * ledger_scale):
            raise ValueError("passive work/heat/storage ledger lost closure")
        if np.any(ledgers < -1.0e-10):
            raise ValueError("a cumulative energy ledger became negative")

        temperature = absolute_q[:, GENERATOR_T_SLICE][:, 0]
        temperature_gradient_scalar = self.derivative(temperature[:, None])[:, 0]
        displacement_gradient = self.derivative(absolute_q[:, GENERATOR_U_SLICE])
        zeta = self.solve_micromorphic_slip(absolute_q[:, GENERATOR_GAMMA_SLICE])
        zeta_gradient_scalar = self.derivative(zeta)

        base_key = (int(interval_index), float(fraction))
        base_raw = self._base_raw_cache.get(base_key)
        if base_raw is None:
            base_raw = self.spectral_model._evaluate_raw(
                frame.F_sample,
                frame.temperature_K,
                frame.active.gamma_signed,
                np.zeros((18, 3)),
                np.zeros(3),
                self.direction_n,
                frame.chart,
                frame.active,
                frame.observer,
            )
            self._base_raw_cache[base_key] = base_raw
        if diagnostics is not None:
            diagnostics.observe_raw(base_raw)

        first_piola = np.empty((self.cells, 3, 3), dtype=np.float64)
        heat_source = np.empty(self.cells, dtype=np.float64)
        active_rhs = np.empty((self.cells, 62), dtype=np.float64)
        passive_rhs = np.empty((self.cells, N_PASSIVE_HISTORY), dtype=np.float64)
        for cell in range(self.cells):
            observer = SpectralObserverState(
                Gamma_absolute=np.maximum(Gamma[cell], 0.0),
                cp_work_density_J_m3=float(max(ledgers[cell, 0], 0.0)),
                generated_heat_density_J_m3=float(max(ledgers[cell, 1], 0.0)),
                passive_storage_density_J_m3=float(max(ledgers[cell, 2], 0.0)),
                time_s=frame.observer.time_s,
            )
            active = SpectralActiveState62(
                theta_p=absolute_q[cell, GENERATOR_THETA_SLICE],
                rho_mobile_m2=absolute_q[cell, GENERATOR_RHO_MOBILE_SLICE],
                rho_dipole_m2=absolute_q[cell, GENERATOR_RHO_DIPOLE_SLICE],
                gamma_signed=absolute_q[cell, GENERATOR_GAMMA_SLICE],
            )
            F = frame.F_sample + np.outer(
                displacement_gradient[cell], self.direction_n
            )
            raw = self.spectral_model._evaluate_raw(
                F,
                float(temperature[cell]),
                zeta[cell],
                zeta_gradient_scalar[cell, :, None] * self.direction_n[None, :],
                temperature_gradient_scalar[cell] * self.direction_n,
                self.direction_n,
                frame.chart,
                active,
                observer,
            )
            first_piola[cell] = raw.first_piola_Pa
            heat_source[cell] = raw.heat_source_W_m3
            active_rhs[cell] = raw.active_rhs
            passive_rhs[cell, PASSIVE_GAMMA_ABSOLUTE_SLICE] = (
                raw.Gamma_absolute_rate_s_inv - base_raw.Gamma_absolute_rate_s_inv
            )
            passive_rhs[cell, PASSIVE_CP_WORK] = raw.cp_power_W_m3 - base_raw.cp_power_W_m3
            passive_rhs[cell, PASSIVE_GENERATED_HEAT] = (
                raw.heat_source_W_m3 - base_raw.heat_source_W_m3
            )
            passive_rhs[cell, PASSIVE_STORED_ENERGY] = (
                raw.storage_rate_W_m3 - base_raw.storage_rate_W_m3
            )
            if diagnostics is not None:
                diagnostics.observe_raw(raw)

        traction = np.einsum("nij,j->ni", first_piola, self.direction_n)
        density = float(self.spectral_model.base_model.parameters.mass_density)
        heat_capacity = float(self.spectral_model.base_model.parameters.heat_capacity)
        momentum_rhs = self.derivative(traction) / density
        conductivity = np.asarray(self.spectral_model.K0, dtype=float)
        heat_flux_normal = -float(
            self.direction_n @ conductivity @ self.direction_n
        ) * temperature_gradient_scalar
        heat_rhs = (
            heat_source - self.derivative(heat_flux_normal[:, None])[:, 0]
        ) / (density * heat_capacity)

        q_rhs = np.zeros_like(q_delta)
        q_rhs[:, GENERATOR_U_SLICE] = q_delta[:, GENERATOR_V_SLICE]
        q_rhs[:, GENERATOR_V_SLICE] = momentum_rhs
        q_rhs[:, GENERATOR_T_SLICE] = (
            heat_rhs - base_raw.heat_source_W_m3 / (density * heat_capacity)
        )[:, None]
        q_rhs[:, GENERATOR_THETA_SLICE] = active_rhs[:, 0:8] - base_raw.active_rhs[0:8]
        q_rhs[:, GENERATOR_RHO_MOBILE_SLICE] = active_rhs[:, 8:26] - base_raw.active_rhs[8:26]
        q_rhs[:, GENERATOR_RHO_DIPOLE_SLICE] = active_rhs[:, 26:44] - base_raw.active_rhs[26:44]
        q_rhs[:, GENERATOR_GAMMA_SLICE] = active_rhs[:, 44:62] - base_raw.active_rhs[44:62]
        return q_rhs, passive_rhs

    def integrate(
        self,
        initial_active_delta: Any,
        *,
        initial_passive_delta: Any | None = None,
        integration_substeps_per_interval: int = 4,
        end_index: int | None = None,
    ) -> dict[str, Any]:
        """Advance deviations with explicit midpoint substeps."""

        q = _finite(
            initial_active_delta,
            (self.cells, N_GENERATOR),
            "initial_active_delta",
        ).copy()
        h = (
            np.zeros((self.cells, N_PASSIVE_HISTORY), dtype=np.float64)
            if initial_passive_delta is None
            else _finite(
                initial_passive_delta,
                (self.cells, N_PASSIVE_HISTORY),
                "initial_passive_delta",
            ).copy()
        )
        if (
            not isinstance(integration_substeps_per_interval, (int, np.integer))
            or int(integration_substeps_per_interval) < 1
        ):
            raise ValueError("integration substeps must be a positive integer")
        substeps = int(integration_substeps_per_interval)
        stop = len(self.base_times_s) - 1 if end_index is None else int(end_index)
        if stop < 1 or stop >= len(self.base_times_s):
            raise ValueError("end_index must identify a noninitial base checkpoint")
        diagnostics = HistoryDiagnosticsV1()
        self.spectral_model._verify_configuration()
        for interval in range(stop):
            substep_dt = (
                self.base_times_s[interval + 1] - self.base_times_s[interval]
            ) / substeps
            for substep in range(substeps):
                left_fraction = substep / substeps
                midpoint_fraction = (substep + 0.5) / substeps
                q1, h1 = self.rhs(
                    interval, left_fraction, q, h, diagnostics
                )
                q_mid = q + 0.5 * substep_dt * q1
                h_mid = h + 0.5 * substep_dt * h1
                h_mid[:, PASSIVE_STORED_ENERGY] = (
                    h_mid[:, PASSIVE_CP_WORK] - h_mid[:, PASSIVE_GENERATED_HEAT]
                )
                q2, h2 = self.rhs(
                    interval, midpoint_fraction, q_mid, h_mid, diagnostics
                )
                q += substep_dt * q2
                h += substep_dt * h2
                h[:, PASSIVE_STORED_ENERGY] = (
                    h[:, PASSIVE_CP_WORK] - h[:, PASSIVE_GENERATED_HEAT]
                )
                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(h)):
                    raise FloatingPointError("non-autonomous history integration became non-finite")
        self.spectral_model._verify_configuration()
        return {
            "active_delta": q,
            "passive_delta": h,
            "initial_time_s": float(self.base_times_s[0]),
            "final_time_s": float(self.base_times_s[stop]),
            "registered_state_count": len(self.base_times_s),
            "integrated_state_count": stop + 1,
            "integration_substeps_per_interval": substeps,
            "integration_method": "explicit_midpoint_on_linearly_interpolated_defect_corrected_residual",
            "diagnostics": diagnostics.as_dict(),
        }

    def integrate_exponential_midpoint(
        self,
        initial_active_delta: Any,
        *,
        coordinate_scales: Any,
        scaled_generator: Callable[[int, float, int], RealArray],
        retained_nonnegative_modes: Sequence[int],
        initial_passive_delta: Any | None = None,
        integration_substeps_per_interval: int = 1,
        end_index: int | None = None,
        enforce_state92_admissibility: bool = False,
        checkpoint_monitor: Callable[
            [int, float, RealArray, RealArray], dict[str, Any] | None
        ]
        | None = None,
        stop_condition: Callable[[Sequence[dict[str, Any]]], str | None]
        | None = None,
    ) -> dict[str, Any]:
        """Advance a stiff residual with its registered tangent as integrating factor.

        The linear part is the checkpoint generator interpolated at the
        substep midpoint and is propagated exactly with a matrix exponential.
        The nonlinear remainder uses exponential-midpoint quadrature.  Thus,
        as the amplitude tends to zero, the method recovers the same
        exponential-midpoint propagator used by the finite-time linear gate.
        """

        q = _finite(
            initial_active_delta,
            (self.cells, N_GENERATOR),
            "initial_active_delta",
        ).copy()
        h = (
            np.zeros((self.cells, N_PASSIVE_HISTORY), dtype=np.float64)
            if initial_passive_delta is None
            else _finite(
                initial_passive_delta,
                (self.cells, N_PASSIVE_HISTORY),
                "initial_passive_delta",
            ).copy()
        )
        scales = _finite(coordinate_scales, (N_GENERATOR,), "coordinate_scales")
        if np.any(scales <= 0.0):
            raise ValueError("coordinate_scales must be positive")
        if not callable(scaled_generator):
            raise TypeError("scaled_generator must be callable")
        if checkpoint_monitor is not None and not callable(checkpoint_monitor):
            raise TypeError("checkpoint_monitor must be callable")
        if stop_condition is not None and not callable(stop_condition):
            raise TypeError("stop_condition must be callable")
        if stop_condition is not None and checkpoint_monitor is None:
            raise ValueError("stop_condition requires checkpoint_monitor")
        if (
            not isinstance(integration_substeps_per_interval, (int, np.integer))
            or int(integration_substeps_per_interval) < 1
        ):
            raise ValueError("integration substeps must be a positive integer")
        substeps = int(integration_substeps_per_interval)
        stop = len(self.base_times_s) - 1 if end_index is None else int(end_index)
        if stop < 1 or stop >= len(self.base_times_s):
            raise ValueError("end_index must identify a noninitial base checkpoint")
        retained_positive = sorted({int(value) for value in retained_nonnegative_modes})
        if (
            not retained_positive
            or retained_positive[0] < 0
            or retained_positive[-1] > self.cells // 2
        ):
            raise ValueError("retained modes must lie between zero and Nyquist")
        retained_indices = set(retained_positive)
        retained_indices.update(
            self.cells - value for value in retained_positive if 0 < value < self.cells // 2
        )
        diagnostics = HistoryDiagnosticsV1()
        maximum_discarded_rhs_fraction = 0.0
        maximum_fourier_conjugacy_relative_defect = 0.0
        maximum_Gamma_projection = 0.0
        total_Gamma_projection_l1 = 0.0
        Gamma_projection_component_count = 0
        monitor_records: list[dict[str, Any]] = []
        stop_reason: str | None = None
        actual_stop = 0
        self.spectral_model._verify_configuration()

        def enforce(
            interval: int,
            fraction: float,
            active: RealArray,
            passive: RealArray,
            reference_gamma: RealArray | None = None,
            reference_Gamma: RealArray | None = None,
        ) -> None:
            nonlocal maximum_Gamma_projection
            nonlocal total_Gamma_projection_l1
            nonlocal Gamma_projection_component_count
            if not enforce_state92_admissibility:
                return
            audit = self.enforce_state92_gamma_bound(
                interval,
                fraction,
                active,
                passive,
                reference_gamma_signed=reference_gamma,
                reference_Gamma_absolute=reference_Gamma,
            )
            maximum_Gamma_projection = max(
                maximum_Gamma_projection, float(audit["maximum_correction"])
            )
            total_Gamma_projection_l1 += float(audit["total_correction_l1"])
            Gamma_projection_component_count += int(
                audit["corrected_component_count"]
            )

        def observe(index: int, active: RealArray, passive: RealArray) -> None:
            if checkpoint_monitor is None:
                return
            record = checkpoint_monitor(
                int(index), float(self.base_times_s[index]), active, passive
            )
            if record is not None:
                monitor_records.append(dict(record))

        enforce(0, 0.0, q, h)
        observe(0, q, h)

        def generator(interval: int, fraction: float, fft_index: int) -> np.ndarray:
            nonnegative = fft_index if fft_index <= self.cells // 2 else self.cells - fft_index
            value = np.asarray(
                scaled_generator(interval, fraction, nonnegative), dtype=np.complex128
            )
            if value.shape != (N_GENERATOR, N_GENERATOR) or not np.all(np.isfinite(value)):
                raise ValueError("scaled generator callback returned an invalid matrix")
            return value if fft_index <= self.cells // 2 else value.conj()

        def linear_action(
            field_hat: np.ndarray,
            interval: int,
            fraction: float,
        ) -> np.ndarray:
            result = np.zeros_like(field_hat)
            for fft_index in retained_indices:
                result[fft_index] = generator(interval, fraction, fft_index) @ field_hat[fft_index]
            return result

        def exponential_action(
            field_hat: np.ndarray,
            interval: int,
            fraction: float,
            duration: float,
        ) -> np.ndarray:
            result = np.zeros_like(field_hat)
            for fft_index in retained_indices:
                nonnegative = fft_index if fft_index <= self.cells // 2 else self.cells - fft_index
                key = (interval, float(fraction), float(duration), nonnegative)
                propagator = self._exponential_cache.get(key)
                if propagator is None:
                    positive_generator = np.asarray(
                        scaled_generator(interval, fraction, nonnegative),
                        dtype=np.complex128,
                    )
                    propagator = expm(positive_generator * duration)
                    if not np.all(np.isfinite(propagator)):
                        raise FloatingPointError("registered integrating factor became non-finite")
                    self._exponential_cache[key] = propagator
                matrix = propagator if fft_index <= self.cells // 2 else propagator.conj()
                result[fft_index] = matrix @ field_hat[fft_index]
            return result

        def enforce_conjugate_symmetry(field_hat: np.ndarray) -> np.ndarray:
            """Project Fourier coefficients onto the real-field subspace."""

            nonlocal maximum_fourier_conjugacy_relative_defect
            projected = np.asarray(field_hat, dtype=np.complex128).copy()
            scale = max(float(np.max(np.abs(projected), initial=0.0)), 1.0)
            defect = float(np.max(np.abs(projected[0].imag), initial=0.0))
            projected[0] = projected[0].real
            if self.cells % 2 == 0:
                defect = max(
                    defect,
                    float(
                        np.max(
                            np.abs(projected[self.cells // 2].imag), initial=0.0
                        )
                    ),
                )
                projected[self.cells // 2] = projected[self.cells // 2].real
            for index in range(1, (self.cells + 1) // 2):
                partner = self.cells - index
                defect = max(
                    defect,
                    float(
                        np.max(
                            np.abs(projected[partner] - projected[index].conj()),
                            initial=0.0,
                        )
                    ),
                )
                average = 0.5 * (projected[index] + projected[partner].conj())
                projected[index] = average
                projected[partner] = average.conj()
            relative = defect / scale
            maximum_fourier_conjugacy_relative_defect = max(
                maximum_fourier_conjugacy_relative_defect, relative
            )
            if relative > 1.0e-7:
                raise FloatingPointError(
                    "integrating factor Fourier conjugacy defect exceeded 1e-7"
                )
            return projected

        z = q / scales[None, :]
        for interval in range(stop):
            substep_dt = (
                self.base_times_s[interval + 1] - self.base_times_s[interval]
            ) / substeps
            for substep in range(substeps):
                left_fraction = substep / substeps
                midpoint_fraction = (substep + 0.5) / substeps
                q_left = z * scales[None, :]
                enforce(interval, left_fraction, q_left, h)
                left_frame = self.base_frame(interval, left_fraction)
                reference_gamma = (
                    left_frame.active.gamma_signed[None, :]
                    + q_left[:, GENERATOR_GAMMA_SLICE]
                )
                reference_Gamma = (
                    left_frame.observer.Gamma_absolute[None, :]
                    + h[:, PASSIVE_GAMMA_ABSOLUTE_SLICE]
                )
                q_rate_left, h_rate_left = self.rhs(
                    interval, left_fraction, q_left, h, diagnostics
                )
                z_hat = np.fft.fft(z, axis=0)
                rate_left_hat = np.fft.fft(
                    q_rate_left / scales[None, :], axis=0
                )
                nonlinear_left_hat = rate_left_hat - linear_action(
                    z_hat, interval, left_fraction
                )
                stage_hat = exponential_action(
                    z_hat + 0.5 * substep_dt * nonlinear_left_hat,
                    interval,
                    midpoint_fraction,
                    0.5 * substep_dt,
                )
                stage_hat = enforce_conjugate_symmetry(stage_hat)
                z_mid_complex = np.fft.ifft(stage_hat, axis=0)
                imaginary = float(np.max(np.abs(z_mid_complex.imag), initial=0.0))
                if imaginary > 2.0e-10 * max(float(np.max(np.abs(z_mid_complex.real), initial=0.0)), 1.0):
                    raise FloatingPointError("integrating factor lost Fourier conjugate symmetry")
                z_mid = z_mid_complex.real
                h_mid = h + 0.5 * substep_dt * h_rate_left
                h_mid[:, PASSIVE_STORED_ENERGY] = (
                    h_mid[:, PASSIVE_CP_WORK] - h_mid[:, PASSIVE_GENERATED_HEAT]
                )
                enforce(
                    interval,
                    midpoint_fraction,
                    z_mid * scales[None, :],
                    h_mid,
                    reference_gamma,
                    reference_Gamma,
                )
                q_rate_mid, h_rate_mid = self.rhs(
                    interval,
                    midpoint_fraction,
                    z_mid * scales[None, :],
                    h_mid,
                    diagnostics,
                )
                rate_mid_hat = np.fft.fft(
                    q_rate_mid / scales[None, :], axis=0
                )
                nonlinear_mid_hat = rate_mid_hat - linear_action(
                    stage_hat, interval, midpoint_fraction
                )
                total_rhs_energy = float(np.sum(np.abs(rate_mid_hat) ** 2))
                discarded_rhs_energy = float(
                    sum(
                        np.sum(np.abs(rate_mid_hat[index]) ** 2)
                        for index in range(self.cells)
                        if index not in retained_indices
                    )
                )
                if total_rhs_energy > np.finfo(float).tiny:
                    maximum_discarded_rhs_fraction = max(
                        maximum_discarded_rhs_fraction,
                        discarded_rhs_energy / total_rhs_energy,
                    )
                z_new_hat = exponential_action(
                    z_hat, interval, midpoint_fraction, substep_dt
                ) + substep_dt * exponential_action(
                    nonlinear_mid_hat,
                    interval,
                    midpoint_fraction,
                    0.5 * substep_dt,
                )
                z_new_hat = enforce_conjugate_symmetry(z_new_hat)
                z_complex = np.fft.ifft(z_new_hat, axis=0)
                imaginary = float(np.max(np.abs(z_complex.imag), initial=0.0))
                if imaginary > 2.0e-10 * max(float(np.max(np.abs(z_complex.real), initial=0.0)), 1.0):
                    raise FloatingPointError("integrating factor lost Fourier conjugate symmetry")
                z = z_complex.real
                h += substep_dt * h_rate_mid
                h[:, PASSIVE_STORED_ENERGY] = (
                    h[:, PASSIVE_CP_WORK] - h[:, PASSIVE_GENERATED_HEAT]
                )
                right_fraction = (substep + 1.0) / substeps
                enforce(
                    interval,
                    right_fraction,
                    z * scales[None, :],
                    h,
                    reference_gamma,
                    reference_Gamma,
                )
                if not np.all(np.isfinite(z)) or not np.all(np.isfinite(h)):
                    raise FloatingPointError("non-autonomous history integration became non-finite")
            actual_stop = interval + 1
            observe(actual_stop, z * scales[None, :], h)
            if stop_condition is not None:
                candidate = stop_condition(monitor_records)
                if candidate is not None:
                    stop_reason = str(candidate)
                    break
        self.spectral_model._verify_configuration()
        return {
            "active_delta": z * scales[None, :],
            "passive_delta": h,
            "initial_time_s": float(self.base_times_s[0]),
            "final_time_s": float(self.base_times_s[actual_stop]),
            "registered_state_count": len(self.base_times_s),
            "integrated_state_count": actual_stop + 1,
            "requested_end_index": stop,
            "completed_requested_history": actual_stop == stop,
            "stop_reason": stop_reason,
            "integration_substeps_per_interval": substeps,
            "integration_method": "registered_tangent_exponential_midpoint_with_nonlinear_remainder",
            "state92_admissibility_enforced": bool(enforce_state92_admissibility),
            "maximum_Gamma_projection": maximum_Gamma_projection,
            "total_Gamma_projection_l1": total_Gamma_projection_l1,
            "Gamma_projection_component_count": Gamma_projection_component_count,
            "retained_nonnegative_fourier_modes": retained_positive,
            "maximum_discarded_rhs_fourier_energy_fraction": maximum_discarded_rhs_fraction,
            "maximum_fourier_conjugacy_relative_defect_before_projection": maximum_fourier_conjugacy_relative_defect,
            "monitor_records": monitor_records,
            "diagnostics": diagnostics.as_dict(),
        }


__all__ = [
    "HistoryDiagnosticsV1",
    "N_PASSIVE_HISTORY",
    "PASSIVE_CP_WORK",
    "PASSIVE_GAMMA_ABSOLUTE_SLICE",
    "PASSIVE_GENERATED_HEAT",
    "PASSIVE_STORED_ENERGY",
    "Periodic69HistoryV1",
    "RegisteredBaseFrameV1",
]
