"""Local eight-coordinate chart for isochoric plastic deformation gradients.

The persistent material state stores all nine components of ``Fp``.  A
continuous spectral linearization must not treat those nine entries as
independent because ``det(Fp)=1``.  This module supplies the registered local
chart

``Fp(theta) = exp(sum_A theta[A] G[A]) @ Fp_anchor``

with eight Frobenius-orthonormal traceless generators.  The chart is recentered
at every frozen base state; it does not replace the 92-real storage ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm_frechet

from .local_coupling import _matrix_exponential_3x3

Array = NDArray[np.float64]
N_SL3 = 8


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _frozen(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = _finite_array(value, shape, name).copy()
    result.setflags(write=False)
    return result


def traceless_generators() -> Array:
    """Return the locked Frobenius-orthonormal basis of ``sl(3)``.

    The first six generators are the off-diagonal Cartesian dyads in
    row-major order.  The final two span the traceless diagonal subspace.
    """

    basis = np.zeros((N_SL3, 3, 3), dtype=np.float64)
    for index, (row, column) in enumerate(
        ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    ):
        basis[index, row, column] = 1.0
    basis[6] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    basis[7] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    basis.setflags(write=False)
    return basis


@dataclass(frozen=True)
class SL3LocalChart:
    """Eight-coordinate exponential chart centered at ``anchor_Fp``."""

    anchor_Fp: Array
    basis: Array = field(default_factory=traceless_generators)
    determinant_tolerance: float = 5.0e-10
    inverse_tolerance: float = 2.0e-11
    inverse_maximum_iterations: int = 20

    def __post_init__(self) -> None:
        anchor = _frozen(self.anchor_Fp, (3, 3), "anchor_Fp")
        basis = _frozen(self.basis, (N_SL3, 3, 3), "basis")
        controls = np.array(
            [
                self.determinant_tolerance,
                self.inverse_tolerance,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(controls)) or np.any(controls <= 0.0):
            raise ValueError("SL(3) chart tolerances must be finite and positive")
        if self.inverse_maximum_iterations <= 0:
            raise ValueError("inverse_maximum_iterations must be positive")
        determinant = float(np.linalg.det(anchor))
        if (
            not np.isfinite(determinant)
            or determinant <= 0.0
            or abs(determinant - 1.0) > self.determinant_tolerance
        ):
            raise ValueError("anchor_Fp must lie on SL(3)")
        if not np.allclose(np.trace(basis, axis1=1, axis2=2), 0.0, atol=2.0e-15):
            raise ValueError("every SL(3) generator must be traceless")
        gram = np.einsum("aij,bij->ab", basis, basis)
        if not np.allclose(gram, np.eye(N_SL3), rtol=0.0, atol=2.0e-15):
            raise ValueError("SL(3) generators must be Frobenius orthonormal")
        object.__setattr__(self, "anchor_Fp", anchor)
        object.__setattr__(self, "basis", basis)
        if np.linalg.matrix_rank(self.tangent_matrix(np.zeros(N_SL3))) != N_SL3:
            raise ValueError("SL(3) chart tangent does not have rank eight")

    def matrix(self, theta: Array) -> Array:
        """Map eight local coordinates to a plastic deformation gradient."""

        coordinates = _finite_array(theta, (N_SL3,), "theta")
        generator = np.einsum("a,aij->ij", coordinates, self.basis)
        result = _matrix_exponential_3x3(generator) @ self.anchor_Fp
        determinant = float(np.linalg.det(result))
        if (
            not np.all(np.isfinite(result))
            or determinant <= 0.0
            or abs(determinant - 1.0) > 10.0 * self.determinant_tolerance
        ):
            raise FloatingPointError("SL(3) exponential chart lost isochoricity")
        return result

    def tangent_matrix(self, theta: Array) -> Array:
        """Return ``d vec(Fp) / d theta`` in row-major tensor order."""

        coordinates = _finite_array(theta, (N_SL3,), "theta")
        generator = np.einsum("a,aij->ij", coordinates, self.basis)
        tangent = np.stack(
            [
                (expm_frechet(generator, basis_vector, compute_expm=False) @ self.anchor_Fp).reshape(-1)
                for basis_vector in self.basis
            ],
            axis=1,
        )
        condition = float(np.linalg.cond(tangent))
        if (
            not np.all(np.isfinite(tangent))
            or np.linalg.matrix_rank(tangent) != N_SL3
            or not np.isfinite(condition)
            or condition > 1.0e10
        ):
            raise FloatingPointError("SL(3) chart tangent is singular or non-finite")
        return tangent

    def dual_matrix(self, theta: Array) -> Array:
        """Return the least-squares dual that maps ``vec(Fp_dot)`` to rates."""

        return np.linalg.pinv(self.tangent_matrix(theta), rcond=1.0e-13)

    def coordinate_rate(self, theta: Array, Fp_rate: Array) -> Array:
        """Express an admissible physical ``Fp_rate`` in local coordinates."""

        coordinates = _finite_array(theta, (N_SL3,), "theta")
        physical_rate = _finite_array(Fp_rate, (3, 3), "Fp_rate")
        tangent = self.tangent_matrix(coordinates)
        Fp = self.matrix(coordinates)
        left_velocity = physical_rate @ np.linalg.inv(Fp)
        generator = np.einsum("a,aij->ij", coordinates, self.basis)
        # Bernoulli-series inverse of the left-trivialized exponential
        # differential: A_dot = dexp_A^{-1}(Fp_dot Fp^{-1}).  The registered
        # chart is local (the continuous exporter limits ||theta||), so terms
        # through ad_A^8 are far below double-precision error at its boundary.
        # The exact Fréchet tangent below remains the independent residual
        # check, rather than differentiating through a nested least-squares
        # solve whose symmetry-zero components are cancellation dominated.
        inverse_dexp = left_velocity.copy()
        commutator = left_velocity.copy()
        coefficients = {
            1: -0.5,
            2: 1.0 / 12.0,
            4: -1.0 / 720.0,
            6: 1.0 / 30240.0,
            8: -1.0 / 1209600.0,
        }
        for order in range(1, 9):
            commutator = generator @ commutator - commutator @ generator
            if order in coefficients:
                inverse_dexp += coefficients[order] * commutator
        rate = np.einsum("aij,ij->a", self.basis, inverse_dexp)
        residual = tangent @ rate - physical_rate.reshape(-1)
        rate_scale = max(
            float(np.linalg.norm(physical_rate)),
            np.finfo(np.float64).eps * float(np.linalg.norm(tangent)) * float(np.linalg.norm(rate)),
            np.finfo(np.float64).tiny,
        )
        isochoric_rate = left_velocity
        trace_scale = max(float(np.linalg.norm(isochoric_rate)), np.finfo(np.float64).tiny)
        if (
            float(np.linalg.norm(residual)) > 5.0e-10 * rate_scale
            or abs(float(np.trace(isochoric_rate))) > 5.0e-10 * trace_scale
        ):
            raise FloatingPointError("Fp_rate is not tangent to the registered SL(3) chart")
        return rate

    def coordinates(self, Fp: Array) -> Array:
        """Invert a nearby chart state by a rank-eight Newton solve.

        The inverse is deliberately local.  A target outside the registered
        chart neighborhood fails closed instead of being projected back onto
        ``SL(3)``.
        """

        target = _finite_array(Fp, (3, 3), "Fp")
        determinant = float(np.linalg.det(target))
        if (
            determinant <= 0.0
            or abs(determinant - 1.0) > 10.0 * self.determinant_tolerance
        ):
            raise ValueError("Fp must lie on SL(3) before chart inversion")
        if np.array_equal(target, self.anchor_Fp):
            return np.zeros(N_SL3)
        theta = np.zeros(N_SL3, dtype=np.float64)
        target_scale = max(float(np.linalg.norm(target)), 1.0)
        for _ in range(self.inverse_maximum_iterations):
            residual = (self.matrix(theta) - target).reshape(-1)
            if float(np.linalg.norm(residual)) <= self.inverse_tolerance * target_scale:
                return theta
            tangent = self.tangent_matrix(theta)
            increment, _, _, _ = np.linalg.lstsq(tangent, -residual, rcond=1.0e-13)
            base_norm = float(np.linalg.norm(residual))
            accepted = False
            factor = 1.0
            while factor >= 2.0**-14:
                candidate = theta + factor * increment
                candidate_norm = float(np.linalg.norm(self.matrix(candidate) - target))
                if candidate_norm < base_norm:
                    theta = candidate
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                break
        raise ValueError("Fp lies outside the converged neighborhood of this SL(3) chart")


__all__ = ["N_SL3", "SL3LocalChart", "traceless_generators"]
