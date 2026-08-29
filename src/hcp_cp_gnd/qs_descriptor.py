"""Reference assembly of the continuous QS--U903 descriptor pencil.

The module consumes the side-effect-free :mod:`spectral_export` contract.  It
does not use U903 ``AMATRX`` or a backward-Euler endpoint tangent.  The output
is a verification-layer pencil ``K + s E``; generalized-Schur filtering,
equilibration and scientific wave-number scans remain downstream gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .spectral_export import (
    INPUT_COORDINATE_LABELS,
    OUTPUT_COORDINATE_LABELS,
    SPECTRAL_EXPORT_SCHEMA,
    SpectralPointExport,
)


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

N_U = 3
N_T = 1
N_ZETA = 18
N_ACTIVE = 62
N_QS = N_U + N_T + N_ZETA + N_ACTIVE

U_SLICE = slice(0, 3)
T_SLICE = slice(3, 4)
ZETA_SLICE = slice(4, 22)
ACTIVE_SLICE = slice(22, 84)

QS_DESCRIPTOR_SCHEMA = "HCP_CP_QS_U903_CONTINUOUS_DESCRIPTOR_V1"


def _real_frozen(value: Any, shape: tuple[int, ...], name: str) -> RealArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


def _complex_frozen(value: Any, shape: tuple[int, ...], name: str) -> ComplexArray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


def _sha256_payload(*items: bytes) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(len(item).to_bytes(8, "little", signed=False))
        digest.update(item)
    return digest.hexdigest()


@dataclass(frozen=True)
class QSDescriptorAdmission:
    """Explicit pre-run tolerances for a homogeneous verification base state."""

    microforce_abs_tolerance_Pa: float
    power_identity_abs_tolerance_W_m3: float
    power_partition_abs_tolerance_W_m3: float
    minimum_normalized_branch_distance: float
    direction_norm_abs_tolerance: float
    determinant_abs_tolerance: float
    symmetry_abs_tolerance: float
    psd_eigenvalue_abs_tolerance: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.microforce_abs_tolerance_Pa,
                self.power_identity_abs_tolerance_W_m3,
                self.power_partition_abs_tolerance_W_m3,
                self.minimum_normalized_branch_distance,
                self.direction_norm_abs_tolerance,
                self.determinant_abs_tolerance,
                self.symmetry_abs_tolerance,
                self.psd_eigenvalue_abs_tolerance,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("descriptor admission tolerances must be finite and non-negative")
        if self.minimum_normalized_branch_distance <= 0.0:
            raise ValueError("minimum branch distance must be positive")
        if self.direction_norm_abs_tolerance <= 0.0:
            raise ValueError("direction-norm tolerance must be positive")


@dataclass(frozen=True)
class QSDescriptorMetadata:
    schema: str
    track: str
    point_export_schema: str
    point_case_hash: str
    point_source_hash: str
    point_parameter_hash: str
    point_configuration_hash: str
    tensor_flattening: str
    raw_microforce_equation: bool
    omega_chi_observer_only: float
    wavenumber_m_inv: float
    direction_n: RealArray
    directional_conductivity_W_mK: float
    state_labels: tuple[str, ...]
    residual_labels: tuple[str, ...]
    assembly_hash: str
    claim_boundary: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "direction_n", _real_frozen(self.direction_n, (3,), "direction_n")
        )
        if len(self.state_labels) != N_QS or len(self.residual_labels) != N_QS:
            raise ValueError("descriptor metadata must enumerate all 84 rows and columns")
        if self.track != "QS-U903":
            raise ValueError("this module assembles only the QS-U903 track")
        if not self.raw_microforce_equation:
            raise ValueError("the continuous descriptor requires the raw microforce equation")
        scalars = np.array(
            [
                self.omega_chi_observer_only,
                self.wavenumber_m_inv,
                self.directional_conductivity_W_mK,
            ]
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("descriptor metadata contains a non-finite scalar")


@dataclass(frozen=True)
class QSDescriptorAssembly:
    """Unscaled complex pencil ``K + s E`` in the frozen 84-coordinate order."""

    stiffness_K: ComplexArray
    descriptor_E: ComplexArray
    direction_map_B: RealArray
    traction_map_N: RealArray
    directional_gradient_Pa_m2: RealArray
    algebraic_block_unscaled: ComplexArray
    algebraic_singular_values_unscaled: RealArray
    metadata: QSDescriptorMetadata

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stiffness_K", _complex_frozen(self.stiffness_K, (N_QS, N_QS), "K")
        )
        object.__setattr__(
            self, "descriptor_E", _complex_frozen(self.descriptor_E, (N_QS, N_QS), "E")
        )
        object.__setattr__(
            self, "direction_map_B", _real_frozen(self.direction_map_B, (9, 3), "B_n")
        )
        object.__setattr__(
            self, "traction_map_N", _real_frozen(self.traction_map_N, (3, 9), "N_n")
        )
        object.__setattr__(
            self,
            "directional_gradient_Pa_m2",
            _real_frozen(
                self.directional_gradient_Pa_m2,
                (N_ZETA, N_ZETA),
                "H_nabla",
            ),
        )
        object.__setattr__(
            self,
            "algebraic_block_unscaled",
            _complex_frozen(
                self.algebraic_block_unscaled,
                (N_U + N_ZETA, N_U + N_ZETA),
                "C_alg",
            ),
        )
        object.__setattr__(
            self,
            "algebraic_singular_values_unscaled",
            _real_frozen(
                self.algebraic_singular_values_unscaled,
                (N_U + N_ZETA,),
                "unscaled algebraic singular values",
            ),
        )

    def residual(self, growth_rate_s_inv: complex, state: ComplexArray) -> ComplexArray:
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (N_QS,) or not np.all(np.isfinite(vector)):
            raise ValueError("descriptor state must be a finite complex 84-vector")
        growth = complex(growth_rate_s_inv)
        if not np.isfinite(growth.real) or not np.isfinite(growth.imag):
            raise ValueError("growth rate must be finite")
        return self.stiffness_K @ vector + growth * (self.descriptor_E @ vector)


def direction_maps(direction_n: RealArray, *, norm_tolerance: float) -> tuple[RealArray, RealArray]:
    """Return row-major ``B_n`` and traction ``N_n`` without normalizing input."""

    direction = np.asarray(direction_n, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_n must be a finite three-vector")
    norm = float(np.linalg.norm(direction))
    if abs(norm - 1.0) > norm_tolerance:
        raise ValueError("direction_n must already be unit length")
    B = np.zeros((9, 3), dtype=np.float64)
    N = np.zeros((3, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            B[3 * i + j, i] = direction[j]
            N[i, 3 * i + j] = direction[j]
    B.setflags(write=False)
    N.setflags(write=False)
    return B, N


def _labels(point: SpectralPointExport) -> tuple[tuple[str, ...], tuple[str, ...]]:
    state = (
        ("u[0]", "u[1]", "u[2]", "T")
        + tuple(f"zeta[{index}]" for index in range(N_ZETA))
        + tuple(point.metadata.input_coordinate_labels[28:90])
    )
    residual = (
        ("momentum[0]", "momentum[1]", "momentum[2]", "heat")
        + tuple(f"microforce[{index}]" for index in range(N_ZETA))
        + tuple(f"state[{label}]" for label in point.metadata.output_coordinate_labels[28:90])
    )
    return state, residual


def _validate_point_export(
    point: SpectralPointExport,
    admission: QSDescriptorAdmission,
) -> None:
    if not isinstance(point, SpectralPointExport):
        raise TypeError("point must be a SpectralPointExport")
    if point.derivatives is None:
        raise ValueError("descriptor assembly requires the continuous 90-by-90 Jacobian")
    metadata = point.metadata
    if metadata.schema != SPECTRAL_EXPORT_SCHEMA:
        raise ValueError("unexpected continuous spectral export schema")
    if not metadata.raw_microforce_equation:
        raise ValueError("continuous descriptor cannot consume a published-scaled microforce row")
    if metadata.tensor_flattening != "ROW_MAJOR_C_ORDER":
        raise ValueError("descriptor direction maps require row-major tensor flattening")
    if metadata.input_coordinate_labels != INPUT_COORDINATE_LABELS:
        raise ValueError("continuous input-coordinate labels do not match the frozen contract")
    if metadata.output_coordinate_labels != OUTPUT_COORDINATE_LABELS:
        raise ValueError("continuous output-coordinate labels do not match the frozen contract")
    if abs(float(np.linalg.det(metadata.chart_anchor_Fp)) - 1.0) > admission.determinant_abs_tolerance:
        raise ValueError("SL(3) chart anchor does not preserve det(Fp)=1")
    if np.max(np.abs(point.penalty_microstress_Pa)) > admission.microforce_abs_tolerance_Pa:
        raise ValueError("homogeneous base state does not satisfy raw microforce equilibrium")
    if abs(point.power_identity_residual_W_m3) > admission.power_identity_abs_tolerance_W_m3:
        raise ValueError("base-state power identity exceeds the admitted tolerance")
    if abs(point.power_partition_residual_W_m3) > admission.power_partition_abs_tolerance_W_m3:
        raise ValueError("base-state power partition exceeds the admitted tolerance")
    if (
        point.derivatives.minimum_normalized_branch_distance
        <= admission.minimum_normalized_branch_distance
    ):
        raise ValueError("base state is too close to a registered constitutive branch surface")
    if point.reference_density_kg_m3 <= 0.0 or point.specific_heat_J_kgK <= 0.0:
        raise ValueError("reference density and specific heat must be positive")


def assemble_qs_descriptor(
    point: SpectralPointExport,
    *,
    wavenumber_m_inv: float,
    direction_n: RealArray,
    admission: QSDescriptorAdmission,
) -> QSDescriptorAssembly:
    """Assemble the unscaled bulk QS pencil from one admitted point export."""

    _validate_point_export(point, admission)
    k = float(wavenumber_m_inv)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("the ordinary QS descriptor requires finite k > 0")
    B, N = direction_maps(direction_n, norm_tolerance=admission.direction_norm_abs_tolerance)
    direction = np.asarray(direction_n, dtype=np.float64)

    conductivity = np.asarray(point.conductivity_W_mK, dtype=np.float64)
    if np.max(np.abs(conductivity - conductivity.T)) > admission.symmetry_abs_tolerance:
        raise ValueError("reference conductivity must be symmetric")
    conductivity_eigenvalues = np.linalg.eigvalsh(0.5 * (conductivity + conductivity.T))
    if float(np.min(conductivity_eigenvalues)) < -admission.psd_eigenvalue_abs_tolerance:
        raise ValueError("reference conductivity must be positive semidefinite")
    kappa_n = float(direction @ conductivity @ direction)

    full_gradient = np.asarray(point.gradient_hessian_Pa_m2, dtype=np.float64)
    H = np.einsum("aibj,i,j->ab", full_gradient, direction, direction, optimize=True)
    if np.max(np.abs(H - H.T)) > admission.symmetry_abs_tolerance:
        raise ValueError("directional slip-gradient Hessian must be symmetric")
    H_eigenvalues = np.linalg.eigvalsh(0.5 * (H + H.T))
    if float(np.min(H_eigenvalues)) < -admission.psd_eigenvalue_abs_tolerance:
        raise ValueError("directional slip-gradient Hessian must be positive semidefinite")

    derivative = point.derivatives
    assert derivative is not None
    K = np.zeros((N_QS, N_QS), dtype=np.complex128)
    E = np.zeros((N_QS, N_QS), dtype=np.complex128)
    ik = 1j * k
    k2 = k * k

    K[U_SLICE, U_SLICE] = k2 * (N @ derivative.dP_dF @ B)
    K[U_SLICE, T_SLICE] = -ik * (N @ derivative.dP_dT)
    K[U_SLICE, ZETA_SLICE] = -ik * (N @ derivative.dP_dzeta)
    K[U_SLICE, ACTIVE_SLICE] = -ik * (N @ derivative.dP_da)

    K[T_SLICE, U_SLICE] = -ik * (derivative.dq_dF @ B)
    K[T_SLICE, T_SLICE] = k2 * kappa_n - derivative.dq_dT
    K[T_SLICE, ZETA_SLICE] = -derivative.dq_dzeta
    K[T_SLICE, ACTIVE_SLICE] = -derivative.dq_da

    K[ZETA_SLICE, U_SLICE] = ik * (derivative.dpi_dF @ B)
    K[ZETA_SLICE, T_SLICE] = derivative.dpi_dT
    K[ZETA_SLICE, ZETA_SLICE] = derivative.dpi_dzeta + k2 * H
    K[ZETA_SLICE, ACTIVE_SLICE] = derivative.dpi_da

    K[ACTIVE_SLICE, U_SLICE] = -ik * (derivative.dg_dF @ B)
    K[ACTIVE_SLICE, T_SLICE] = -derivative.dg_dT
    K[ACTIVE_SLICE, ZETA_SLICE] = -derivative.dg_dzeta
    K[ACTIVE_SLICE, ACTIVE_SLICE] = -derivative.dg_da

    E[T_SLICE, T_SLICE] = point.reference_density_kg_m3 * point.specific_heat_J_kgK
    E[ACTIVE_SLICE, ACTIVE_SLICE] = np.eye(N_ACTIVE)

    algebraic_indices = np.r_[np.arange(U_SLICE.start, U_SLICE.stop), np.arange(ZETA_SLICE.start, ZETA_SLICE.stop)]
    C_alg = K[np.ix_(algebraic_indices, algebraic_indices)]
    singular_values = np.linalg.svd(C_alg, compute_uv=False)

    state_labels, residual_labels = _labels(point)
    hash_context = json.dumps(
        {
            "schema": QS_DESCRIPTOR_SCHEMA,
            "track": "QS-U903",
            "point_case_hash": point.metadata.case_hash,
            "point_source_hash": point.metadata.source_hash,
            "point_parameter_hash": point.metadata.parameter_hash,
            "point_configuration_hash": point.metadata.configuration_hash,
            "tensor_flattening": point.metadata.tensor_flattening,
            "raw_microforce_equation": True,
            "k_hex": k.hex(),
            "direction_hex": [float(value).hex() for value in direction],
            "state_labels": state_labels,
            "residual_labels": residual_labels,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assembly_hash = _sha256_payload(
        hash_context,
        np.ascontiguousarray(K.astype("<c16", copy=False)).tobytes(),
        np.ascontiguousarray(E.astype("<c16", copy=False)).tobytes(),
    )
    metadata = QSDescriptorMetadata(
        schema=QS_DESCRIPTOR_SCHEMA,
        track="QS-U903",
        point_export_schema=point.metadata.schema,
        point_case_hash=point.metadata.case_hash,
        point_source_hash=point.metadata.source_hash,
        point_parameter_hash=point.metadata.parameter_hash,
        point_configuration_hash=point.metadata.configuration_hash,
        tensor_flattening=point.metadata.tensor_flattening,
        raw_microforce_equation=True,
        omega_chi_observer_only=float(point.metadata.omega_chi),
        wavenumber_m_inv=k,
        direction_n=direction,
        directional_conductivity_W_mK=kappa_n,
        state_labels=state_labels,
        residual_labels=residual_labels,
        assembly_hash=assembly_hash,
        claim_boundary=(
            "REFERENCE_ASSEMBLY_ONLY__UNSCALED_PENCIL__NO_QZ_AUTHORITY__"
            "NO_EIGENVALUE_RESULT__NOT_TA2"
        ),
    )
    return QSDescriptorAssembly(
        stiffness_K=K,
        descriptor_E=E,
        direction_map_B=B,
        traction_map_N=N,
        directional_gradient_Pa_m2=H,
        algebraic_block_unscaled=C_alg,
        algebraic_singular_values_unscaled=singular_values,
        metadata=metadata,
    )


__all__ = [
    "ACTIVE_SLICE",
    "N_QS",
    "QS_DESCRIPTOR_SCHEMA",
    "T_SLICE",
    "U_SLICE",
    "ZETA_SLICE",
    "QSDescriptorAdmission",
    "QSDescriptorAssembly",
    "QSDescriptorMetadata",
    "assemble_qs_descriptor",
    "direction_maps",
]
