"""Full inertial HCP crystal perturbation operator and finite-time propagator.

For one frozen material point and Fourier pair ``(k,n)`` the public 84-state
descriptor is

``(K(k,n) + s E + s**2 M) x = 0``.

The 18 micromorphic-slip rows contain neither first- nor second-time
derivatives.  They are eliminated by an exact Schur complement.  The retained
coordinates are displacement ``u(3)`` and differential state
``d=[T,active62]``.  Introducing velocity gives a nonsingular 69-state
generator for ``[u,v,d]``.  No fictitious inertia is assigned to temperature,
micromorphic slip, or constitutive state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, matrix_balance

from .dynamic_inertia_v1 import augment_qs_descriptor
from .qs_descriptor import (
    ACTIVE_SLICE,
    N_QS,
    QSDescriptorAdmission,
    T_SLICE,
    U_SLICE,
    ZETA_SLICE,
    assemble_qs_descriptor,
)
from .spectral_export import SpectralPointExport


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

DYNAMIC_CRYSTAL_PERTURBATION_SCHEMA_V1 = "HCP_CP_DYNAMIC_CRYSTAL_PERTURBATION_V1"
N_DIFFERENTIAL = 63
N_RETAINED = 66
N_GENERATOR = 69

# Generator order q=[u(3), v(3), T, theta_p(8), rho_m(18), rho_d(18), gamma(18)].
# These slices are deliberately public: finite-time singular vectors are only
# scientifically interpretable when their coordinate partition is frozen.
GENERATOR_U_SLICE = slice(0, 3)
GENERATOR_V_SLICE = slice(3, 6)
GENERATOR_T_SLICE = slice(6, 7)
GENERATOR_THETA_SLICE = slice(7, 15)
GENERATOR_RHO_MOBILE_SLICE = slice(15, 33)
GENERATOR_RHO_DIPOLE_SLICE = slice(33, 51)
GENERATOR_GAMMA_SLICE = slice(51, 69)

Z_INDICES = np.arange(ZETA_SLICE.start, ZETA_SLICE.stop)
Y_INDICES = np.r_[
    np.arange(U_SLICE.start, U_SLICE.stop),
    np.arange(T_SLICE.start, T_SLICE.stop),
    np.arange(ACTIVE_SLICE.start, ACTIVE_SLICE.stop),
]


def _matrix(value: Any, shape: tuple[int, int], name: str) -> ComplexArray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DynamicCrystalOperatorV1:
    wavenumber_m_inv: float
    direction_n: RealArray
    full_K: ComplexArray
    full_E: ComplexArray
    full_M: ComplexArray
    algebraic_Kzz: ComplexArray
    algebraic_Kzy: ComplexArray
    effective_Kyy: ComplexArray
    retained_Eyy: ComplexArray
    retained_Myy: ComplexArray
    generator_A: ComplexArray
    algebraic_condition_number: float
    schema: str = DYNAMIC_CRYSTAL_PERTURBATION_SCHEMA_V1

    def __post_init__(self) -> None:
        k = float(self.wavenumber_m_inv)
        direction = np.asarray(self.direction_n, dtype=np.float64)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("dynamic crystal operator requires finite k>0")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite three-vector")
        if abs(float(np.linalg.norm(direction)) - 1.0) > 2.0e-13:
            raise ValueError("direction must be normalized")
        shapes = {
            "full_K": (N_QS, N_QS),
            "full_E": (N_QS, N_QS),
            "full_M": (N_QS, N_QS),
            "algebraic_Kzz": (18, 18),
            "algebraic_Kzy": (18, N_RETAINED),
            "effective_Kyy": (N_RETAINED, N_RETAINED),
            "retained_Eyy": (N_RETAINED, N_RETAINED),
            "retained_Myy": (N_RETAINED, N_RETAINED),
            "generator_A": (N_GENERATOR, N_GENERATOR),
        }
        for name, shape in shapes.items():
            value = _matrix(getattr(self, name), shape, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "direction_n", direction.copy())

    def eigenpairs(self) -> tuple[ComplexArray, ComplexArray]:
        balanced, transform = matrix_balance(
            self.generator_A, permute=True, scale=True, separate=False
        )
        values, balanced_vectors = np.linalg.eig(balanced)
        vectors = transform @ balanced_vectors
        vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
        order = np.lexsort((values.imag, values.real))
        return values[order], vectors[:, order]

    def reconstruct_full_amplitude(self, generator_state: Any) -> ComplexArray:
        q = np.asarray(generator_state, dtype=np.complex128)
        if q.shape != (N_GENERATOR,) or not np.all(np.isfinite(q)):
            raise ValueError("generator state must be a finite 69-vector")
        retained = np.concatenate((q[0:3], q[6:69]))
        zeta = -np.linalg.solve(self.algebraic_Kzz, self.algebraic_Kzy @ retained)
        full = np.zeros(N_QS, dtype=np.complex128)
        full[Y_INDICES] = retained
        full[Z_INDICES] = zeta
        return full

    def quadratic_residual(self, growth_rate_s_inv: complex, full_state: Any) -> ComplexArray:
        s = complex(growth_rate_s_inv)
        if not np.isfinite(s.real) or not np.isfinite(s.imag):
            raise ValueError("growth rate must be finite")
        x = np.asarray(full_state, dtype=np.complex128)
        if x.shape != (N_QS,) or not np.all(np.isfinite(x)):
            raise ValueError("full state must be a finite 84-vector")
        return (self.full_K + s * self.full_E + s * s * self.full_M) @ x

    def modal_residuals(self) -> RealArray:
        values, vectors = self.eigenpairs()
        residuals = np.empty(values.size)
        for index, growth in enumerate(values):
            full = self.reconstruct_full_amplitude(vectors[:, index])
            residual = self.quadratic_residual(growth, full)
            row_scale = (
                np.abs(self.full_K)
                + abs(growth) * np.abs(self.full_E)
                + abs(growth) ** 2 * np.abs(self.full_M)
            ) @ np.abs(full)
            active = row_scale > np.finfo(float).tiny
            residuals[index] = (
                float(np.max(np.abs(residual[active]) / row_scale[active]))
                if np.any(active)
                else float(np.linalg.norm(residual))
            )
        return residuals

    def generator_residuals(self) -> RealArray:
        """Return normwise backward errors of the 69-state generator modes."""

        values, vectors = self.eigenpairs()
        scale_A = float(np.linalg.norm(self.generator_A, ord=2))
        result = np.empty(values.size)
        for index, growth in enumerate(values):
            vector = vectors[:, index]
            residual = self.generator_A @ vector - growth * vector
            denominator = (scale_A + abs(growth)) * np.linalg.norm(vector)
            result[index] = float(
                np.linalg.norm(residual)
                / max(denominator, np.finfo(float).tiny)
            )
        return result

    def first_order_descriptor_pair(self) -> tuple[ComplexArray, ComplexArray]:
        """Return the 87-state dynamic descriptor pair ``K_1 + s E_1``.

        The descriptor state is ordered as ``[u(3), v(3), d(63), zeta(18)]``.
        It retains the 18 purely algebraic micromorphic coordinates and therefore
        provides an independent generalized-eigenvalue audit of their exact Schur
        elimination.  The finite spectrum must match ``generator_A``; the remaining
        18 generalized eigenvalues are structural infinite roots.
        """

        Kyy = self.full_K[np.ix_(Y_INDICES, Y_INDICES)]
        Kyz = self.full_K[np.ix_(Y_INDICES, Z_INDICES)]
        Kzy = self.full_K[np.ix_(Z_INDICES, Y_INDICES)]
        Kzz = self.full_K[np.ix_(Z_INDICES, Z_INDICES)]

        first_K = np.zeros((87, 87), dtype=np.complex128)
        first_E = np.zeros_like(first_K)

        # Kinematic relation: s u - v = 0.
        first_K[0:3, 3:6] = -np.eye(3)
        first_E[0:3, 0:3] = np.eye(3)

        # Momentum and differential-state rows, in retained order y=[u,d].
        first_K[3:6, 0:3] = Kyy[0:3, 0:3]
        first_K[3:6, 6:69] = Kyy[0:3, 3:66]
        first_K[3:6, 69:87] = Kyz[0:3, :]
        first_E[3:6, 3:6] = self.retained_Myy[0:3, 0:3]

        first_K[6:69, 0:3] = Kyy[3:66, 0:3]
        first_K[6:69, 6:69] = Kyy[3:66, 3:66]
        first_K[6:69, 69:87] = Kyz[3:66, :]
        first_E[6:69, 6:69] = self.retained_Eyy[3:66, 3:66]

        # Purely algebraic micromorphic rows.
        first_K[69:87, 0:3] = Kzy[:, 0:3]
        first_K[69:87, 6:69] = Kzy[:, 3:66]
        first_K[69:87, 69:87] = Kzz

        first_K.setflags(write=False)
        first_E.setflags(write=False)
        return first_K, first_E

    def regularized_first_order_descriptor_pair(
        self,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Return a row-equivalent 87-state pair suitable for QZ classification.

        The physical pair spans displacement, velocity, temperature and density
        coordinates with very different units.  Direct projective classification
        can consequently lose a very slow finite root in binary64.  This routine
        applies nonsingular block-row operations and one block-triangular variable
        transformation: it separates the algebraic columns from the first 69 rows
        and normalizes the invertible dynamic storage and algebraic stiffness blocks.
        Generalized eigenvalues are unchanged.
        """

        first_K, first_E = self.first_order_descriptor_pair()
        lower_K = first_K[69:87, :]
        Kzz = first_K[69:87, 69:87]
        algebraic_normalized = np.linalg.solve(Kzz, lower_K)

        upper_K = first_K[0:69, :] - (
            first_K[0:69, 69:87] @ algebraic_normalized
        )
        # This block is analytically zero after the row operation.  Assigning
        # the structural zero avoids carrying mixed-unit solve roundoff into QZ.
        upper_K[:, 69:87] = 0.0
        dynamic_E = first_E[0:69, 0:69]
        dynamic_normalized = np.linalg.solve(dynamic_E, upper_K)

        regular_K = np.zeros((87, 87), dtype=np.complex128)
        regular_E = np.zeros_like(regular_K)
        regular_K[0:69, :] = dynamic_normalized
        regular_K[69:87, :] = algebraic_normalized
        regular_E[0:69, 0:69] = np.eye(69)
        # With the upper-right block zero and the lower-right block equal to I,
        # the nonsingular change z_new=zeta+C q removes the remaining lower-left
        # block C without altering E.  The pair is then block diagonal in its
        # finite and structural-infinite sectors.
        regular_K[69:87, 0:69] = 0.0
        regular_K.setflags(write=False)
        regular_E.setflags(write=False)
        return regular_K, regular_E

    def admitted_eigenpairs(
        self, maximum_relative_residual: float = 2.0e-10
    ) -> tuple[ComplexArray, ComplexArray, RealArray]:
        """Return modes passing an explicit residual-quality gate.

        Very slow neutral internal modes can be ill-resolved in binary64 when
        the same generator also contains elastic-wave frequencies.  They are
        never silently used for wavelength selection.
        """

        tolerance = float(maximum_relative_residual)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("modal residual tolerance must be positive")
        values, vectors = self.eigenpairs()
        residuals = self.generator_residuals()
        accepted = residuals <= tolerance
        if not np.any(accepted):
            raise FloatingPointError("no dynamic crystal mode passed the residual gate")
        return values[accepted], vectors[:, accepted], residuals[accepted]

    def propagator(self, duration_s: float) -> ComplexArray:
        duration = float(duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("propagation duration must be finite and nonnegative")
        return expm(self.generator_A * duration)


def reduce_dynamic_matrices(
    stiffness_K: Any,
    descriptor_E: Any,
    inertia_M: Any,
    *,
    wavenumber_m_inv: float,
    direction_n: Any,
) -> DynamicCrystalOperatorV1:
    """Eliminate zeta and form the exact first-order 69-state generator."""

    K = _matrix(stiffness_K, (N_QS, N_QS), "K")
    E = _matrix(descriptor_E, (N_QS, N_QS), "E")
    M = _matrix(inertia_M, (N_QS, N_QS), "M")
    direction = np.asarray(direction_n, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction must be a finite nonzero three-vector")
    direction = direction / norm

    if np.count_nonzero(E[Z_INDICES, :]) or np.count_nonzero(M[Z_INDICES, :]):
        raise ValueError("zeta rows must be purely algebraic")
    if np.count_nonzero(E[:, Z_INDICES]) or np.count_nonzero(M[:, Z_INDICES]):
        raise ValueError("zeta columns may not carry time derivatives")

    Kzz = K[np.ix_(Z_INDICES, Z_INDICES)]
    Kzy = K[np.ix_(Z_INDICES, Y_INDICES)]
    Kyz = K[np.ix_(Y_INDICES, Z_INDICES)]
    Kyy = K[np.ix_(Y_INDICES, Y_INDICES)]
    condition = float(np.linalg.cond(Kzz))
    if not np.isfinite(condition) or condition >= 1.0e14:
        raise np.linalg.LinAlgError("micromorphic algebraic block is singular or ill-conditioned")
    correction = np.linalg.solve(Kzz, Kzy)
    Keff = Kyy - Kyz @ correction
    Eyy = E[np.ix_(Y_INDICES, Y_INDICES)]
    Myy = M[np.ix_(Y_INDICES, Y_INDICES)]

    rho_block = Myy[0:3, 0:3]
    Edd = Eyy[3:66, 3:66]
    if np.count_nonzero(Myy[3:, :]) or np.count_nonzero(Myy[:, 3:]):
        raise ValueError("only displacement may carry inertia")
    if np.count_nonzero(Eyy[0:3, :]) or np.count_nonzero(Eyy[:, 0:3]):
        raise ValueError("displacement rows may not carry first-order storage")
    if np.linalg.cond(rho_block) >= 1.0e12 or np.linalg.cond(Edd) >= 1.0e12:
        raise np.linalg.LinAlgError("retained mass/storage block is singular")

    A = np.zeros((N_GENERATOR, N_GENERATOR), dtype=np.complex128)
    # q=[u(3),v(3),d(63)]
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -np.linalg.solve(rho_block, Keff[0:3, 0:3])
    A[3:6, 6:69] = -np.linalg.solve(rho_block, Keff[0:3, 3:66])
    A[6:69, 0:3] = -np.linalg.solve(Edd, Keff[3:66, 0:3])
    A[6:69, 6:69] = -np.linalg.solve(Edd, Keff[3:66, 3:66])

    return DynamicCrystalOperatorV1(
        wavenumber_m_inv=float(wavenumber_m_inv),
        direction_n=direction,
        full_K=K,
        full_E=E,
        full_M=M,
        algebraic_Kzz=Kzz,
        algebraic_Kzy=Kzy,
        effective_Kyy=Keff,
        retained_Eyy=Eyy,
        retained_Myy=Myy,
        generator_A=A,
        algebraic_condition_number=condition,
    )


def assemble_dynamic_crystal_operator_v1(
    point: SpectralPointExport,
    *,
    wavenumber_m_inv: float,
    direction_n: Any,
    admission: QSDescriptorAdmission,
    local_slip_gradient_diffusivity_m2_s: float = 0.0,
) -> DynamicCrystalOperatorV1:
    descriptor = assemble_qs_descriptor(
        point,
        wavenumber_m_inv=float(wavenumber_m_inv),
        direction_n=np.asarray(direction_n, dtype=np.float64),
        admission=admission,
    )
    dynamic = augment_qs_descriptor(
        descriptor,
        reference_density_kg_m3=point.reference_density_kg_m3,
    )
    diffusivity = float(local_slip_gradient_diffusivity_m2_s)
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("local slip-gradient diffusivity must be finite and nonnegative")
    stiffness = np.asarray(dynamic.stiffness_K).copy()
    if diffusivity > 0.0:
        # The local plastic chart theta_p (first eight active coordinates) and
        # gamma_signed (last 18) describe the same evolving plastic distortion
        # at different resolutions.  A direct local-gradient closure must act
        # on both; damping gamma alone leaves a spurious high-k theta branch.
        theta = np.arange(ACTIVE_SLICE.start, ACTIVE_SLICE.start + 8)
        gamma = np.arange(ACTIVE_SLICE.stop - 18, ACTIVE_SLICE.stop)
        for indices in (theta, gamma):
            stiffness[indices, indices] += (
                diffusivity * float(wavenumber_m_inv) ** 2
            )
    return reduce_dynamic_matrices(
        stiffness,
        dynamic.descriptor_E,
        dynamic.inertia_M,
        wavenumber_m_inv=wavenumber_m_inv,
        direction_n=direction_n,
    )


def piecewise_frozen_propagator(
    operators: Iterable[DynamicCrystalOperatorV1],
    durations_s: Iterable[float],
) -> ComplexArray:
    """Return the time-ordered product for a piecewise-frozen base history."""

    operator_list = list(operators)
    duration_list = [float(value) for value in durations_s]
    if not operator_list or len(operator_list) != len(duration_list):
        raise ValueError("operators and durations must have the same nonzero length")
    result = np.eye(N_GENERATOR, dtype=np.complex128)
    for operator, duration in zip(operator_list, duration_list, strict=True):
        if not isinstance(operator, DynamicCrystalOperatorV1):
            raise TypeError("all propagation segments must be dynamic crystal operators")
        result = operator.propagator(duration) @ result
    return result


def maximum_amplification(
    propagator: Any,
    *,
    input_scales: Any | None = None,
    output_scales: Any | None = None,
) -> dict[str, Any]:
    """Compute the largest finite-time gain and its right/left singular vectors."""

    Phi = _matrix(propagator, (N_GENERATOR, N_GENERATOR), "propagator")
    in_scale = np.ones(N_GENERATOR) if input_scales is None else np.asarray(input_scales, dtype=float)
    out_scale = np.ones(N_GENERATOR) if output_scales is None else np.asarray(output_scales, dtype=float)
    if (
        in_scale.shape != (N_GENERATOR,)
        or out_scale.shape != (N_GENERATOR,)
        or np.any(~np.isfinite(in_scale))
        or np.any(~np.isfinite(out_scale))
        or np.any(in_scale <= 0.0)
        or np.any(out_scale <= 0.0)
    ):
        raise ValueError("propagator scales must be positive finite 69-vectors")
    # q=D z.  Hence the dimensionless propagator is D_out^{-1} Phi D_in.
    scaled = (Phi * in_scale[None, :]) / out_scale[:, None]
    left, singular, right_h = np.linalg.svd(scaled, full_matrices=False)
    return {
        "gain": float(singular[0]),
        "input_vector": right_h[0].conj(),
        "output_vector": left[:, 0],
        "all_singular_values": singular,
    }


def generator_mechanism_participation(
    generator_vector: Any,
    *,
    coordinate_scales: Any | None = None,
) -> dict[str, Any]:
    """Partition a generator vector into dimensionless mechanism weights.

    ``coordinate_scales`` has the physical units of the generator coordinates
    and defines ``q = D z``.  The reported weights are squared Euclidean norms
    of ``z`` and therefore must always be interpreted together with the frozen
    scale vector.  Passing ``None`` is appropriate for singular vectors already
    expressed in dimensionless coordinates.
    """

    vector = np.asarray(generator_vector, dtype=np.complex128)
    if vector.shape != (N_GENERATOR,) or not np.all(np.isfinite(vector)):
        raise ValueError("generator vector must be a finite 69-vector")
    scales = (
        np.ones(N_GENERATOR, dtype=np.float64)
        if coordinate_scales is None
        else np.asarray(coordinate_scales, dtype=np.float64)
    )
    if (
        scales.shape != (N_GENERATOR,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("coordinate scales must be a positive finite 69-vector")
    dimensionless = vector / scales
    total = float(np.vdot(dimensionless, dimensionless).real)
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        raise ValueError("generator vector has zero dimensionless norm")

    slices = {
        "displacement": GENERATOR_U_SLICE,
        "velocity_inertia": GENERATOR_V_SLICE,
        "temperature": GENERATOR_T_SLICE,
        "plastic_distortion_chart": GENERATOR_THETA_SLICE,
        "mobile_dislocation_density": GENERATOR_RHO_MOBILE_SLICE,
        "dipole_dislocation_density": GENERATOR_RHO_DIPOLE_SLICE,
        "signed_slip": GENERATOR_GAMMA_SLICE,
    }
    detailed = {
        name: float(np.vdot(dimensionless[part], dimensionless[part]).real / total)
        for name, part in slices.items()
    }
    coarse = {
        "thermal": detailed["temperature"],
        "dislocation": (
            detailed["mobile_dislocation_density"]
            + detailed["dipole_dislocation_density"]
        ),
        "plastic_kinematics": (
            detailed["plastic_distortion_chart"] + detailed["signed_slip"]
        ),
        "mechanical_inertia": detailed["displacement"] + detailed["velocity_inertia"],
    }
    return {
        "normalization": "squared_dimensionless_generator_norm",
        "detailed": detailed,
        "coarse": coarse,
        "dominant_detailed": max(detailed, key=detailed.get),
        "dominant_coarse": max(coarse, key=coarse.get),
    }


def finite_time_amplification_history(
    operators: Iterable[DynamicCrystalOperatorV1],
    times_s: Iterable[float],
    *,
    coordinate_scales: Any | None = None,
    gain_threshold: float = float(np.e),
    input_indices: Any | None = None,
    output_indices: Any | None = None,
    integration_substeps_per_interval: int = 1,
) -> dict[str, Any]:
    """Integrate a non-autonomous frozen history and report prefix gains.

    Adjacent frozen generators are linearly interpolated and integrated with
    second-order exponential midpoint substeps.  For ``m`` substeps on one
    checkpoint interval, the generator is evaluated at fractions
    ``(j+1/2)/m`` and each exponential spans ``dt/m``.  The historical method
    is recovered exactly for ``m=1``.  The state scaling is held
    fixed over the whole history, making singular gains comparable in time.
    ``input_indices`` and ``output_indices`` optionally define a rectangular
    observable map.  This is needed to distinguish localization amplification
    (thermal/constitutive input and output) from reversible elastic-wave gain.
    ``critical_time_s`` is the first log-linear crossing of the explicitly
    reported gain threshold; it is ``None`` when the threshold is not reached.
    """

    operator_list = list(operators)
    times = np.asarray(list(times_s), dtype=np.float64)
    if len(operator_list) < 2 or times.shape != (len(operator_list),):
        raise ValueError("finite-time history needs matching operator/time arrays of length >=2")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("finite-time history times must be finite and strictly increasing")
    if (
        not isinstance(integration_substeps_per_interval, (int, np.integer))
        or int(integration_substeps_per_interval) < 1
    ):
        raise ValueError("integration substeps per interval must be a positive integer")
    substeps = int(integration_substeps_per_interval)
    if not all(isinstance(item, DynamicCrystalOperatorV1) for item in operator_list):
        raise TypeError("all history entries must be dynamic crystal operators")
    reference_k = operator_list[0].wavenumber_m_inv
    reference_n = operator_list[0].direction_n
    for item in operator_list[1:]:
        if not np.isclose(item.wavenumber_m_inv, reference_k, rtol=0.0, atol=0.0):
            raise ValueError("one finite-time history must keep a fixed wavenumber")
        if not np.allclose(item.direction_n, reference_n, rtol=0.0, atol=2.0e-13):
            raise ValueError("one finite-time history must keep a fixed direction")
    threshold = float(gain_threshold)
    if not np.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("gain threshold must be finite and greater than one")
    scales = (
        np.ones(N_GENERATOR, dtype=np.float64)
        if coordinate_scales is None
        else np.asarray(coordinate_scales, dtype=np.float64)
    )
    if (
        scales.shape != (N_GENERATOR,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("coordinate scales must be a positive finite 69-vector")

    def admitted_indices(value: Any | None, name: str) -> np.ndarray:
        if value is None:
            return np.arange(N_GENERATOR, dtype=int)
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
        result = raw.astype(int, copy=False)
        if np.any(result < 0) or np.any(result >= N_GENERATOR) or np.unique(result).size != result.size:
            raise ValueError(f"{name} contains invalid or repeated coordinates")
        return result

    input_coordinates = admitted_indices(input_indices, "input indices")
    output_coordinates = admitted_indices(output_indices, "output indices")

    def observed_matrix(full_propagator: ComplexArray) -> ComplexArray:
        return full_propagator[np.ix_(output_coordinates, input_coordinates)]

    def scaled_generator(operator: DynamicCrystalOperatorV1) -> ComplexArray:
        return (operator.generator_A * scales[None, :]) / scales[:, None]

    Phi = np.eye(N_GENERATOR, dtype=np.complex128)
    prefix: list[dict[str, float]] = [
        {
            "time_s": float(times[0]),
            "elapsed_s": 0.0,
            "maximum_gain": 1.0,
            "log_gain": 0.0,
        }
    ]
    for index in range(len(operator_list) - 1):
        dt = float(times[index + 1] - times[index])
        left_generator = scaled_generator(operator_list[index])
        right_generator = scaled_generator(operator_list[index + 1])
        substep_dt = dt / substeps
        for substep_index in range(substeps):
            fraction = (substep_index + 0.5) / substeps
            midpoint = (1.0 - fraction) * left_generator + fraction * right_generator
            step = expm(midpoint * substep_dt)
            if not np.all(np.isfinite(step)):
                raise FloatingPointError("finite-time exponential produced a non-finite step")
            Phi = step @ Phi
            if not np.all(np.isfinite(Phi)):
                raise FloatingPointError("finite-time propagator overflowed")
        largest = float(np.linalg.svd(observed_matrix(Phi), compute_uv=False)[0])
        prefix.append(
            {
                "time_s": float(times[index + 1]),
                "elapsed_s": float(times[index + 1] - times[0]),
                "maximum_gain": largest,
                "log_gain": float(np.log(largest)),
            }
        )

    observed = observed_matrix(Phi)
    left, singular, right_h = np.linalg.svd(observed, full_matrices=False)
    input_vector = np.zeros(N_GENERATOR, dtype=np.complex128)
    observed_output_vector = np.zeros(N_GENERATOR, dtype=np.complex128)
    input_vector[input_coordinates] = right_h[0].conj()
    observed_output_vector[output_coordinates] = left[:, 0]
    full_output_response = Phi @ input_vector
    target_log = float(np.log(threshold))
    critical_time: float | None = None
    for left, right in zip(prefix[:-1], prefix[1:], strict=True):
        if left["log_gain"] < target_log <= right["log_gain"]:
            denominator = right["log_gain"] - left["log_gain"]
            fraction = (
                0.0 if denominator <= np.finfo(float).tiny
                else (target_log - left["log_gain"]) / denominator
            )
            critical_time = float(
                left["time_s"] + fraction * (right["time_s"] - left["time_s"])
            )
            break
    return {
        "wavenumber_m_inv": float(reference_k),
        "direction_n": np.asarray(reference_n, dtype=float).tolist(),
        "integration": "piecewise_linear_generator_exponential_midpoint_substepped",
        "integration_substeps_per_interval": substeps,
        "initial_time_s": float(times[0]),
        "final_time_s": float(times[-1]),
        "gain_threshold": threshold,
        "input_indices": input_coordinates.tolist(),
        "output_indices": output_coordinates.tolist(),
        "critical_time_s": critical_time,
        "prefix": prefix,
        "final_gain": float(singular[0]),
        "final_log_gain": float(np.log(singular[0])),
        "input_mechanism_participation": generator_mechanism_participation(
            input_vector
        ),
        # Backward-compatible diagnostic: this is the zero-embedded left
        # singular vector of the declared output subspace.  Coordinates omitted
        # by output_indices are identically zero by construction.
        "output_mechanism_participation": generator_mechanism_participation(
            observed_output_vector
        ),
        "output_mechanism_participation_scope": (
            "zero_embedded_observed_output_subspace_singular_vector"
        ),
        # Physically interpretable partition of the complete propagated state
        # generated by the optimal admitted input.  This is the quantity to use
        # when mechanical coordinates are excluded from the observation norm.
        "full_state_output_mechanism_participation": generator_mechanism_participation(
            full_output_response
        ),
        "input_vector_dimensionless": input_vector,
        "output_vector_dimensionless": observed_output_vector,
        "full_state_output_response_dimensionless": full_output_response,
        "propagator_dimensionless": Phi,
    }


__all__ = [
    "DYNAMIC_CRYSTAL_PERTURBATION_SCHEMA_V1",
    "DynamicCrystalOperatorV1",
    "finite_time_amplification_history",
    "generator_mechanism_participation",
    "assemble_dynamic_crystal_operator_v1",
    "maximum_amplification",
    "piecewise_frozen_propagator",
    "reduce_dynamic_matrices",
]
