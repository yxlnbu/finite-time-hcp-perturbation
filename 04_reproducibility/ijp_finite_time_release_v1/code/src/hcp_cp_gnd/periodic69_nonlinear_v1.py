"""One-dimensional periodic nonlinear residual on the registered 69 states.

The state order is ``[u(3),v(3),T,theta_p(8),rho_m(18),rho_d(18),gamma(18)]``.
Micromorphic slip is not assigned fictitious dynamics: its periodic
microforce equation is solved exactly in Fourier space before the material
response is evaluated.  Passive total-variation and energy ledgers are frozen
at the supplied checkpoint because they are not members of the 69-state
generator contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
from .sl3_chart import SL3LocalChart
from .spectral_export import (
    ContinuousSpectralPointModel,
    SpectralActiveState62,
    SpectralObserverState,
)
from .state_contract import LocalState92


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _finite(value: Any, shape: tuple[int, ...], name: str) -> RealArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class Periodic69CheckpointV1:
    """Frozen-base periodic nonlinear closure and its numerical consistent tangent."""

    spectral_model: ContinuousSpectralPointModel
    base_F_sample: RealArray
    base_storage: LocalState92
    direction_n: RealArray
    domain_length_m: float
    cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.spectral_model, ContinuousSpectralPointModel):
            raise TypeError("spectral_model must be ContinuousSpectralPointModel")
        if not isinstance(self.base_storage, LocalState92):
            raise TypeError("base_storage must be LocalState92")
        F = _finite(self.base_F_sample, (3, 3), "base_F_sample").copy()
        if float(np.linalg.det(F)) <= 0.0:
            raise ValueError("base deformation gradient must have positive determinant")
        direction = _finite(self.direction_n, (3,), "direction_n").copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("direction_n must be nonzero")
        direction /= norm
        length = float(self.domain_length_m)
        cells = int(self.cells)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("domain length must be finite and positive")
        if cells < 4 or cells % 2:
            raise ValueError("periodic grid requires an even cell count of at least four")
        object.__setattr__(self, "base_F_sample", F)
        object.__setattr__(self, "direction_n", direction)
        object.__setattr__(self, "domain_length_m", length)
        object.__setattr__(self, "cells", cells)

    @property
    def spacing_m(self) -> float:
        return self.domain_length_m / self.cells

    @property
    def wave_numbers_m_inv(self) -> RealArray:
        return 2.0 * np.pi * np.fft.fftfreq(self.cells, d=self.spacing_m)

    @property
    def chart(self) -> SL3LocalChart:
        return SL3LocalChart(
            self.base_storage.Fp,
            determinant_tolerance=self.spectral_model.base_model.parameters.determinant_tolerance,
        )

    @property
    def observer(self) -> SpectralObserverState:
        return SpectralObserverState.from_storage(self.base_storage)

    def base_generator_state(self) -> RealArray:
        state = np.zeros(N_GENERATOR, dtype=np.float64)
        state[GENERATOR_T_SLICE] = self.base_storage.temperature_K
        state[GENERATOR_THETA_SLICE] = 0.0
        state[GENERATOR_RHO_MOBILE_SLICE] = self.base_storage.rho_mobile_m2
        state[GENERATOR_RHO_DIPOLE_SLICE] = self.base_storage.rho_dipole_m2
        state[GENERATOR_GAMMA_SLICE] = self.base_storage.gamma_signed
        return state

    def base_field(self) -> RealArray:
        return np.broadcast_to(
            self.base_generator_state(), (self.cells, N_GENERATOR)
        ).copy()

    def _derivative(self, field: RealArray) -> RealArray:
        values = np.asarray(field, dtype=np.float64)
        if values.shape[0] != self.cells:
            raise ValueError("periodic field has the wrong cell count")
        reshape = (self.cells,) + (1,) * (values.ndim - 1)
        derivative_waves = self.wave_numbers_m_inv.copy()
        # On an even real grid the Nyquist coefficient has no signed partner;
        # its collocation derivative is therefore defined as zero.
        derivative_waves[self.cells // 2] = 0.0
        derivative = np.fft.ifft(
            (1j * derivative_waves.reshape(reshape)) * np.fft.fft(values, axis=0),
            axis=0,
        )
        imaginary = float(np.max(np.abs(derivative.imag), initial=0.0))
        scale = max(float(np.max(np.abs(derivative.real), initial=0.0)), 1.0)
        if imaginary > 2.0e-11 * scale:
            raise FloatingPointError("spectral derivative lost real conjugate symmetry")
        return derivative.real

    def solve_micromorphic_slip(self, gamma_signed: Any) -> RealArray:
        """Solve ``H(zeta-gamma)-div(xi)=0`` for every periodic Fourier mode."""

        gamma = _finite(
            gamma_signed, (self.cells, 18), "gamma_signed"
        )
        parameters = self.spectral_model.micromorphic_parameters
        H_chi = float(parameters.penalty_modulus_Pa)
        directions = self.spectral_model.directions_sample_reference
        normals = self.spectral_model.normals_sample_reference
        # The gradient Hessian is state independent.  Evaluate it once through
        # the same micromorphic oracle used by the nonlinear material response.
        from .micromorphic import evaluate_micromorphic_state

        response = evaluate_micromorphic_state(
            np.zeros(18),
            np.zeros(18),
            np.zeros((18, 3)),
            directions,
            normals,
            parameters,
        )
        gradient = np.einsum(
            "aibj,i,j->ab",
            response.gradient_hessian_Pa_m2,
            self.direction_n,
            self.direction_n,
            optimize=True,
        )
        gamma_hat = np.fft.fft(gamma, axis=0)
        zeta_hat = np.empty_like(gamma_hat)
        identity = np.eye(18)
        for index, wave_number in enumerate(self.wave_numbers_m_inv):
            algebraic = H_chi * identity + wave_number**2 * gradient
            zeta_hat[index] = np.linalg.solve(algebraic, H_chi * gamma_hat[index])
        zeta = np.fft.ifft(zeta_hat, axis=0)
        imaginary = float(np.max(np.abs(zeta.imag), initial=0.0))
        scale = max(float(np.max(np.abs(zeta.real), initial=0.0)), 1.0)
        if imaginary > 2.0e-11 * scale:
            raise FloatingPointError("micromorphic solve lost real conjugate symmetry")
        return zeta.real

    def absolute_rhs(self, state_field: Any) -> RealArray:
        """Return the physical PDE right-hand side before base-rate subtraction."""

        state = _finite(
            state_field, (self.cells, N_GENERATOR), "state_field"
        )
        if np.any(state[:, GENERATOR_T_SLICE] <= 0.0):
            raise ValueError("temperature field must remain positive")
        if (
            np.any(state[:, GENERATOR_RHO_MOBILE_SLICE] <= 0.0)
            or np.any(state[:, GENERATOR_RHO_DIPOLE_SLICE] <= 0.0)
        ):
            raise ValueError("dislocation-density fields must remain positive")

        displacement_gradient = self._derivative(state[:, GENERATOR_U_SLICE])
        temperature_gradient_scalar = self._derivative(
            state[:, GENERATOR_T_SLICE]
        )[:, 0]
        zeta = self.solve_micromorphic_slip(state[:, GENERATOR_GAMMA_SLICE])
        zeta_gradient_scalar = self._derivative(zeta)
        first_piola = np.empty((self.cells, 3, 3), dtype=np.float64)
        heat_source = np.empty(self.cells, dtype=np.float64)
        active_rhs = np.empty((self.cells, 62), dtype=np.float64)
        chart = self.chart
        observer = self.observer
        for cell in range(self.cells):
            F = self.base_F_sample + np.outer(
                displacement_gradient[cell], self.direction_n
            )
            active = SpectralActiveState62(
                theta_p=state[cell, GENERATOR_THETA_SLICE],
                rho_mobile_m2=state[cell, GENERATOR_RHO_MOBILE_SLICE],
                rho_dipole_m2=state[cell, GENERATOR_RHO_DIPOLE_SLICE],
                gamma_signed=state[cell, GENERATOR_GAMMA_SLICE],
            )
            raw = self.spectral_model.evaluate_active_state(
                F,
                float(state[cell, GENERATOR_T_SLICE][0]),
                zeta[cell],
                zeta_gradient_scalar[cell, :, None] * self.direction_n[None, :],
                temperature_gradient_scalar[cell] * self.direction_n,
                self.direction_n,
                chart,
                active,
                observer,
            )
            first_piola[cell] = raw.first_piola_Pa
            heat_source[cell] = raw.heat_source_W_m3
            active_rhs[cell] = raw.active_rhs

        traction = np.einsum("nij,j->ni", first_piola, self.direction_n)
        momentum_rhs = self._derivative(traction) / float(
            self.spectral_model.base_model.parameters.mass_density
        )
        conductivity = np.asarray(self.spectral_model.K0, dtype=float)
        heat_flux_normal = -float(self.direction_n @ conductivity @ self.direction_n) * (
            temperature_gradient_scalar
        )
        heat_rhs = (
            heat_source - self._derivative(heat_flux_normal[:, None])[:, 0]
        ) / (
            float(self.spectral_model.base_model.parameters.mass_density)
            * float(self.spectral_model.base_model.parameters.heat_capacity)
        )

        rhs = np.zeros_like(state)
        rhs[:, GENERATOR_U_SLICE] = state[:, GENERATOR_V_SLICE]
        rhs[:, GENERATOR_V_SLICE] = momentum_rhs
        rhs[:, GENERATOR_T_SLICE] = heat_rhs[:, None]
        rhs[:, GENERATOR_THETA_SLICE] = active_rhs[:, 0:8]
        rhs[:, GENERATOR_RHO_MOBILE_SLICE] = active_rhs[:, 8:26]
        rhs[:, GENERATOR_RHO_DIPOLE_SLICE] = active_rhs[:, 26:44]
        rhs[:, GENERATOR_GAMMA_SLICE] = active_rhs[:, 44:62]
        return rhs

    def residual(self, state_field: Any) -> RealArray:
        """Return the perturbation residual with the homogeneous base at zero."""

        state = _finite(
            state_field, (self.cells, N_GENERATOR), "state_field"
        )
        base_rhs = self.absolute_rhs(self.base_field())
        return self.absolute_rhs(state) - base_rhs

    def fourier_mode_consistent_tangent(
        self,
        mode: int,
        *,
        coordinate_scales: Any,
        relative_step: float = 2.0e-6,
    ) -> ComplexArray:
        """Differentiate the residual and return one scaled 69x69 Fourier symbol."""

        mode_index = int(mode)
        if mode_index <= 0 or mode_index >= self.cells // 2:
            raise ValueError("mode must be a positive non-Nyquist Fourier index")
        scales = _finite(coordinate_scales, (N_GENERATOR,), "coordinate_scales")
        if np.any(scales <= 0.0):
            raise ValueError("coordinate scales must be positive")
        step = float(relative_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("relative_step must be finite and positive")
        phase = 2.0 * np.pi * mode_index * np.arange(self.cells) / self.cells
        cosine = np.cos(phase)
        base = self.base_field()
        symbol = np.empty((N_GENERATOR, N_GENERATOR), dtype=np.complex128)
        for column in range(N_GENERATOR):
            perturbation = np.zeros_like(base)
            perturbation[:, column] = step * scales[column] * cosine
            derivative = (
                self.residual(base + perturbation)
                - self.residual(base - perturbation)
            ) / (2.0 * step)
            positive_mode = 2.0 * np.fft.fft(derivative, axis=0)[mode_index] / self.cells
            symbol[:, column] = positive_mode / scales
        return symbol

    def tangent_action_error(
        self,
        symbol: Any,
        mode: int,
        direction_dimensionless: Any,
        *,
        coordinate_scales: Any,
        relative_step: float = 7.5e-7,
    ) -> float:
        """Check a tangent against an independently stepped mixed-phase action."""

        matrix = np.asarray(symbol, dtype=np.complex128)
        if matrix.shape != (N_GENERATOR, N_GENERATOR) or not np.all(np.isfinite(matrix)):
            raise ValueError("symbol must be a finite 69x69 matrix")
        vector = np.asarray(direction_dimensionless, dtype=np.complex128)
        if vector.shape != (N_GENERATOR,) or not np.all(np.isfinite(vector)):
            raise ValueError("direction must be a finite complex 69-vector")
        scales = _finite(coordinate_scales, (N_GENERATOR,), "coordinate_scales")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("direction must be nonzero")
        vector = vector / norm
        phase = 2.0 * np.pi * int(mode) * np.arange(self.cells) / self.cells
        real_field = np.real(np.exp(1j * phase)[:, None] * vector[None, :])
        perturbation = float(relative_step) * real_field * scales[None, :]
        base = self.base_field()
        action_field = (
            self.residual(base + perturbation) - self.residual(base - perturbation)
        ) / (2.0 * float(relative_step))
        measured = 2.0 * np.fft.fft(action_field, axis=0)[int(mode)] / self.cells / scales
        predicted = matrix @ vector
        return float(
            np.linalg.norm(measured - predicted)
            / max(np.linalg.norm(measured), np.linalg.norm(predicted), np.finfo(float).tiny)
        )


__all__ = ["Periodic69CheckpointV1"]
