"""Quadratic energy/observation metrics for LSBRP finite-time gain.

The module never repairs a singular physical metric by adding an arbitrary
diagonal.  A positive-semidefinite metric defines a norm on its positive
eigenspace; its nullspace is returned as part of the scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dynamic_crystal_perturbation_v1 import (
    GENERATOR_GAMMA_SLICE,
    GENERATOR_T_SLICE,
    GENERATOR_THETA_SLICE,
    GENERATOR_U_SLICE,
    GENERATOR_V_SLICE,
    N_GENERATOR,
    DynamicCrystalOperatorV1,
)
from .spectral_export import SpectralPointExport


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticMetricV1:
    """Hermitian positive-semidefinite metric with explicit quotient maps."""

    matrix: ComplexArray
    label: str
    provenance: str
    coordinate_scales: RealArray | None = None
    relative_eigenvalue_tolerance: float = 1.0e-11

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("metric matrix must be square")
        if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError("metric matrix must be nonempty and finite")
        if not self.label.strip() or not self.provenance.strip():
            raise ValueError("metric label and provenance must be nonempty")
        tolerance = float(self.relative_eigenvalue_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("metric eigentolerance must be finite and positive")
        hermitian = 0.5 * (matrix + matrix.conj().T)
        coordinate_scales = (
            np.ones(matrix.shape[0], dtype=np.float64)
            if self.coordinate_scales is None
            else np.asarray(self.coordinate_scales, dtype=np.float64)
        )
        if (
            coordinate_scales.shape != (matrix.shape[0],)
            or np.any(~np.isfinite(coordinate_scales))
            or np.any(coordinate_scales <= 0.0)
        ):
            raise ValueError("metric coordinate scales must be a positive finite vector")
        equilibrated = (
            coordinate_scales[:, None] * hermitian * coordinate_scales[None, :]
        )
        scale = max(float(np.linalg.norm(equilibrated, ord=2)), np.finfo(float).tiny)
        symmetry_error = float(np.linalg.norm(matrix - matrix.conj().T, ord=2))
        matrix_scale = max(float(np.linalg.norm(hermitian, ord=2)), np.finfo(float).tiny)
        if symmetry_error > tolerance * matrix_scale:
            raise ValueError("metric matrix is not Hermitian within tolerance")
        eigenvalues = np.linalg.eigvalsh(equilibrated)
        if float(eigenvalues[0]) < -tolerance * scale:
            raise ValueError(
                f"metric is indefinite: minimum eigenvalue={eigenvalues[0]:.6e}"
            )
        hermitian.setflags(write=False)
        coordinate_scales = coordinate_scales.copy()
        coordinate_scales.setflags(write=False)
        object.__setattr__(self, "matrix", hermitian)
        object.__setattr__(self, "coordinate_scales", coordinate_scales)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def eigensystem(self) -> tuple[RealArray, ComplexArray]:
        equilibrated = (
            self.coordinate_scales[:, None]
            * self.matrix
            * self.coordinate_scales[None, :]
        )
        values, vectors = np.linalg.eigh(equilibrated)
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        positive = values > self.relative_eigenvalue_tolerance * scale
        return values[positive], vectors[:, positive]

    @property
    def rank(self) -> int:
        return int(self.eigensystem()[0].size)

    @property
    def nullity(self) -> int:
        return self.dimension - self.rank

    def state_from_unit_coordinates(self) -> ComplexArray:
        """Return ``B`` with ``q=Bz`` and ``q*Wq=z*z`` on the positive space."""

        values, vectors = self.eigensystem()
        return self.coordinate_scales[:, None] * vectors / np.sqrt(values)[None, :]

    def unit_coordinates_from_state(self) -> ComplexArray:
        """Return ``C`` with ``z=Cq`` on the positive metric eigenspace."""

        values, vectors = self.eigensystem()
        return (
            np.sqrt(values)[:, None]
            * vectors.conj().T
            / self.coordinate_scales[None, :]
        )

    def quadratic_value(self, state: Any) -> float:
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("state has the wrong shape or contains non-finite values")
        value = float(np.vdot(vector, self.matrix @ vector).real)
        scale = max(float(np.linalg.norm(self.matrix)) * float(np.linalg.norm(vector)) ** 2, 1.0)
        if value < -self.relative_eigenvalue_tolerance * scale:
            raise FloatingPointError("metric quadratic form became negative")
        return max(value, 0.0)

    def audit(self) -> dict[str, Any]:
        equilibrated = (
            self.coordinate_scales[:, None]
            * self.matrix
            * self.coordinate_scales[None, :]
        )
        values = np.linalg.eigvalsh(equilibrated)
        positive, _ = self.eigensystem()
        return {
            "label": self.label,
            "provenance": self.provenance,
            "dimension": self.dimension,
            "rank": self.rank,
            "nullity": self.nullity,
            "minimum_eigenvalue": float(values[0]),
            "maximum_eigenvalue": float(values[-1]),
            "minimum_positive_eigenvalue": (
                None if positive.size == 0 else float(positive[0])
            ),
            "full_state_spd": self.rank == self.dimension,
            "positive_semidefinite_within_relative_tolerance": bool(
                values[0]
                >= -self.relative_eigenvalue_tolerance
                * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
            ),
            "rank_basis": "registered_dimensionless_coordinates_q=Dz",
        }


def construct_checkpoint_energy_metric(
    operator: DynamicCrystalOperatorV1,
    point: SpectralPointExport,
    *,
    reference_temperature_K: float,
    coordinate_scales: Any | None = None,
    spectral_model: Any,
    base_F_sample: Any,
) -> QuadraticMetricV1:
    """Construct the recoverable incremental-energy seminorm ``W_E``.

    Included terms are the positive elastic strain increment
    ``delta(Ee):C:delta(Ee)`` jointly mapped from displacement and the local
    SL(3) plastic chart, kinetic energy, thermal availability, and condensed
    micromorphic signed-slip curvature.  The current line-energy storage law
    is affine in dislocation density, so density directions remain an explicit
    nullspace.
    """

    if not isinstance(operator, DynamicCrystalOperatorV1):
        raise TypeError("operator must be a DynamicCrystalOperatorV1")
    if not isinstance(point, SpectralPointExport):
        raise TypeError("point must be a SpectralPointExport")
    required = ("base_model", "directions_sample_reference")
    if any(not hasattr(spectral_model, name) for name in required):
        raise TypeError("spectral_model does not provide the energy-metric contract")
    F_sample = np.asarray(base_F_sample, dtype=np.float64)
    if F_sample.shape != (3, 3) or not np.all(np.isfinite(F_sample)):
        raise ValueError("base_F_sample must be a finite 3x3 matrix")
    temperature = float(reference_temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("reference temperature must be finite and positive")

    metric = np.zeros((N_GENERATOR, N_GENERATOR), dtype=np.complex128)
    model = spectral_model.base_model
    rotation = np.asarray(model.orientation, dtype=np.float64)
    F_crystal = rotation.T @ F_sample @ rotation
    Fp = np.asarray(point.metadata.chart_anchor_Fp, dtype=np.float64)
    inverse_Fp = np.linalg.inv(Fp)
    Fe = F_crystal @ inverse_Fp

    def strain_voigt(delta_Fe: ComplexArray) -> ComplexArray:
        delta_Ee = 0.5 * (Fe.T @ delta_Fe + delta_Fe.T @ Fe)
        return np.asarray(
            [
                delta_Ee[0, 0],
                delta_Ee[1, 1],
                delta_Ee[2, 2],
                2.0 * delta_Ee[1, 2],
                2.0 * delta_Ee[0, 2],
                2.0 * delta_Ee[0, 1],
            ],
            dtype=np.complex128,
        )

    elastic_map = np.empty((6, 11), dtype=np.complex128)
    ik = 1j * operator.wavenumber_m_inv
    for component in range(3):
        delta_F_sample = np.zeros((3, 3), dtype=np.complex128)
        delta_F_sample[component, :] = ik * np.asarray(operator.direction_n)
        delta_F_crystal = rotation.T @ delta_F_sample @ rotation
        elastic_map[:, component] = strain_voigt(delta_F_crystal @ inverse_Fp)
    generators = np.asarray(point.metadata.sl3_generators, dtype=np.float64)
    for chart_index, generator in enumerate(generators):
        # Fp(theta)=exp(theta_A G_A) Fp0, hence
        # delta(Fe)=-Fe G_A at the chart origin.
        elastic_map[:, 3 + chart_index] = strain_voigt(-Fe @ generator)
    stiffness = np.asarray(model.parameters.elastic_matrix(), dtype=np.float64)
    elastic_metric = elastic_map.conj().T @ stiffness @ elastic_map
    elastic_indices = np.r_[
        np.arange(GENERATOR_U_SLICE.start, GENERATOR_U_SLICE.stop),
        np.arange(GENERATOR_THETA_SLICE.start, GENERATOR_THETA_SLICE.stop),
    ]
    metric[np.ix_(elastic_indices, elastic_indices)] = elastic_metric
    density = float(point.reference_density_kg_m3)
    heat_capacity = float(point.specific_heat_J_kgK)
    metric[GENERATOR_V_SLICE, GENERATOR_V_SLICE] = density * np.eye(3)
    metric[GENERATOR_T_SLICE, GENERATOR_T_SLICE] = density * heat_capacity / temperature

    direction = np.asarray(operator.direction_n, dtype=float)
    gradient = np.einsum(
        "aibj,i,j->ab",
        np.asarray(point.gradient_hessian_Pa_m2, dtype=float),
        direction,
        direction,
        optimize=True,
    )
    if point.derivatives is None:
        raise ValueError("energy metric requires a Jacobian-bearing point export")
    penalty = 0.5 * (
        np.asarray(point.derivatives.dpi_dzeta, dtype=float)
        + np.asarray(point.derivatives.dpi_dzeta, dtype=float).T
    )
    algebraic = penalty + operator.wavenumber_m_inv**2 * gradient
    condensed_gamma = penalty - penalty @ np.linalg.solve(algebraic, penalty)
    condensed_gamma = 0.5 * (condensed_gamma + condensed_gamma.T)
    metric[GENERATOR_GAMMA_SLICE, GENERATOR_GAMMA_SLICE] = condensed_gamma
    return QuadraticMetricV1(
        matrix=metric,
        label="W_E",
        provenance=(
            "checkpoint deltaEe:C:deltaEe displacement-SL3 elastic block+kinetic+"
            "thermal-availability+condensed-micromorphic curvature; affine density "
            "storage left null"
        ),
        coordinate_scales=coordinate_scales,
    )


def construct_observation_metric(
    observation_matrix: Any,
    noise_covariance: Any,
    *,
    state_dimension: int = N_GENERATOR,
    provenance: str,
    coordinate_scales: Any | None = None,
) -> QuadraticMetricV1:
    """Construct ``W_O = H* R^{-1} H`` from an observation/noise contract."""

    H = np.asarray(observation_matrix, dtype=np.complex128)
    R = np.asarray(noise_covariance, dtype=np.complex128)
    if H.ndim != 2 or H.shape[1] != int(state_dimension) or H.shape[0] == 0:
        raise ValueError("observation matrix must have shape (m,state_dimension)")
    if R.shape != (H.shape[0], H.shape[0]):
        raise ValueError("noise covariance shape does not match observations")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(R)):
        raise ValueError("observation contract must be finite")
    hermitian_R = 0.5 * (R + R.conj().T)
    scale = max(float(np.linalg.norm(hermitian_R, ord=2)), 1.0)
    if float(np.linalg.norm(R - R.conj().T, ord=2)) > 1.0e-11 * scale:
        raise ValueError("noise covariance must be Hermitian")
    if float(np.min(np.linalg.eigvalsh(hermitian_R))) <= 0.0:
        raise ValueError("noise covariance must be positive definite")
    metric = H.conj().T @ np.linalg.solve(hermitian_R, H)
    return QuadraticMetricV1(
        matrix=metric,
        label="W_O",
        provenance=provenance,
        coordinate_scales=coordinate_scales,
    )


def weighted_propagator_gain(
    propagator: Any,
    *,
    input_metric: QuadraticMetricV1,
    output_metric: QuadraticMetricV1,
) -> dict[str, Any]:
    """Largest gain between two PSD metric quotient spaces."""

    Phi = np.asarray(propagator, dtype=np.complex128)
    shape = (output_metric.dimension, input_metric.dimension)
    if Phi.shape != shape or not np.all(np.isfinite(Phi)):
        raise ValueError(f"propagator must be a finite matrix with shape {shape}")
    if input_metric.rank == 0 or output_metric.rank == 0:
        raise ValueError("input and output metrics must each have positive rank")
    B_in = input_metric.state_from_unit_coordinates()
    C_out = output_metric.unit_coordinates_from_state()
    weighted = C_out @ Phi @ B_in
    left, singular, right_h = np.linalg.svd(weighted, full_matrices=False)
    input_state = B_in @ right_h[0].conj()
    output_state = Phi @ input_state
    return {
        "gain": float(singular[0]),
        "input_state": input_state,
        "output_state": output_state,
        "input_unit_coordinates": right_h[0].conj(),
        "output_unit_coordinates": left[:, 0],
        "singular_values": singular,
        "input_metric_rank": input_metric.rank,
        "output_metric_rank": output_metric.rank,
    }


__all__ = [
    "QuadraticMetricV1",
    "construct_checkpoint_energy_metric",
    "construct_observation_metric",
    "weighted_propagator_gain",
]
