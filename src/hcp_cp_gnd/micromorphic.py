"""Energy-consistent micromorphic slip and Nye/GND reference equations.

The module deliberately contains no Abaqus or DAMASK dependency.  It provides
the independent numerical oracle used to verify a later multi-field UEL.  The
primary kinematic variables are 18 *signed* micromorphic slips
``bar_gamma[a]``.  Their gradients generate the Nye tensor

``Alpha = sum_a s_a outer (grad(bar_gamma[a]) cross n_a)``.

The free-energy contribution is

``psi = 1/2 H_chi ||bar_gamma-gamma||^2
       + 1/2 H_chi ell_chi^2 sum_a |grad(bar_gamma[a])|^2
       + 1/2 mu_ref ell_N^2 Alpha:Alpha``.

Here ``gamma`` is the local signed slip supplied by the material-point model.
The current v0.1 material-point state stores absolute accumulated slip only;
therefore coupling to this module requires adding signed slip as a new v0.2
state.  This module does not infer signed slip from accumulated slip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_finite_array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def cross_matrix(vector: Array) -> Array:
    """Return ``C`` such that ``C @ x == vector cross x``."""

    vector = _as_finite_array(vector, (3,), "vector")
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MicromorphicParameters:
    """Verification parameters for the local/gradient energetic coupling.

    ``nye_length_scale_m`` and ``slip_gradient_length_m`` are independent
    model parameters.  Neither is a measured shear-band width.  Values used
    in a manufactured verification case must carry the
    ``NUMERICAL_VERIFICATION_ONLY`` provenance in its case manifest.
    """

    reference_shear_modulus_Pa: float
    nye_length_scale_m: float
    penalty_modulus_Pa: float
    slip_gradient_length_m: float
    burgers_m: Array

    def validate(self, n_slip: int) -> None:
        scalars = np.array(
            [
                self.reference_shear_modulus_Pa,
                self.nye_length_scale_m,
                self.penalty_modulus_Pa,
                self.slip_gradient_length_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ValueError("mu_ref, ell_N, H_chi, and ell_chi must be finite and positive")
        burgers = _as_finite_array(self.burgers_m, (n_slip,), "burgers_m")
        if np.any(burgers <= 0.0):
            raise ValueError("all Burgers-vector magnitudes must be positive")

    @property
    def defect_coefficient_Pa_m2(self) -> float:
        return self.reference_shear_modulus_Pa * self.nye_length_scale_m**2

    @property
    def slip_gradient_coefficient_Pa_m2(self) -> float:
        return self.penalty_modulus_Pa * self.slip_gradient_length_m**2


@dataclass(frozen=True)
class MicromorphicResponse:
    nye_tensor_m_inv: Array
    curl_slip_m_inv: Array
    gnd_density_m2: Array
    higher_order_stress_Pa_m: Array
    gradient_hessian_Pa_m2: Array
    penalty_microstress_Pa: Array
    defect_energy_J_m3: float
    slip_gradient_energy_J_m3: float
    penalty_energy_J_m3: float

    @property
    def total_energy_J_m3(self) -> float:
        return (
            self.defect_energy_J_m3
            + self.slip_gradient_energy_J_m3
            + self.penalty_energy_J_m3
        )


def _validate_geometry(directions: Array, normals: Array) -> tuple[Array, Array]:
    directions = np.asarray(directions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n_slip,3)")
    normals = _as_finite_array(normals, directions.shape, "normals")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions must be finite")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2.0e-12):
        raise ValueError("slip directions must be normalized")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2.0e-12):
        raise ValueError("slip normals must be normalized")
    if not np.allclose(np.einsum("ai,ai->a", directions, normals), 0.0, atol=2.0e-12):
        raise ValueError("each slip direction must lie in its slip plane")
    return directions, normals


def evaluate_micromorphic_state(
    local_signed_slip: Array,
    micromorphic_slip: Array,
    micromorphic_gradient_m_inv: Array,
    directions: Array,
    normals: Array,
    parameters: MicromorphicParameters,
) -> MicromorphicResponse:
    """Evaluate energy, microstress, and exact gradient Hessian at one point."""

    directions, normals = _validate_geometry(directions, normals)
    n_slip = directions.shape[0]
    parameters.validate(n_slip)
    local_signed_slip = _as_finite_array(local_signed_slip, (n_slip,), "local_signed_slip")
    micromorphic_slip = _as_finite_array(
        micromorphic_slip, (n_slip,), "micromorphic_slip"
    )
    gradients = _as_finite_array(
        micromorphic_gradient_m_inv,
        (n_slip, 3),
        "micromorphic_gradient_m_inv",
    )

    # C[a] maps grad(bar_gamma[a]) to grad(bar_gamma[a]) cross n[a].
    operators = np.stack([-cross_matrix(normal) for normal in normals], axis=0)
    curl_slip = np.einsum("aij,aj->ai", operators, gradients)
    nye = np.einsum("ai,aj->ij", directions, curl_slip)
    coefficient = parameters.defect_coefficient_Pa_m2
    slip_gradient_coefficient = parameters.slip_gradient_coefficient_Pa_m2
    higher_order = slip_gradient_coefficient * gradients + coefficient * np.einsum(
        "aji,jk,ak->ai", operators, nye.T, directions
    )
    direction_gram = directions @ directions.T
    hessian = coefficient * np.einsum(
        "ab,aij,bjk->aibk", direction_gram, operators.transpose(0, 2, 1), operators
    )
    hessian += slip_gradient_coefficient * np.einsum(
        "ab,ij->aibj", np.eye(n_slip), np.eye(3)
    )
    mismatch = micromorphic_slip - local_signed_slip
    penalty_microstress = parameters.penalty_modulus_Pa * mismatch
    defect_energy = 0.5 * coefficient * float(np.einsum("ij,ij->", nye, nye))
    slip_gradient_energy = 0.5 * slip_gradient_coefficient * float(
        np.einsum("ai,ai->", gradients, gradients)
    )
    penalty_energy = 0.5 * parameters.penalty_modulus_Pa * float(mismatch @ mismatch)
    gnd_density = np.linalg.norm(curl_slip, axis=1) / np.asarray(parameters.burgers_m)
    return MicromorphicResponse(
        nye_tensor_m_inv=nye,
        curl_slip_m_inv=curl_slip,
        gnd_density_m2=gnd_density,
        higher_order_stress_Pa_m=higher_order,
        gradient_hessian_Pa_m2=hessian,
        penalty_microstress_Pa=penalty_microstress,
        defect_energy_J_m3=defect_energy,
        slip_gradient_energy_J_m3=slip_gradient_energy,
        penalty_energy_J_m3=penalty_energy,
    )


_HEX8_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _hex8_shape(natural: Array) -> tuple[Array, Array]:
    natural = _as_finite_array(natural, (3,), "natural")
    factors = 1.0 + _HEX8_SIGNS * natural[None, :]
    shape = 0.125 * np.prod(factors, axis=1)
    derivatives = np.empty((8, 3), dtype=np.float64)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        derivatives[:, axis] = (
            0.125
            * _HEX8_SIGNS[:, axis]
            * factors[:, other[0]]
            * factors[:, other[1]]
        )
    return shape, derivatives


@dataclass(frozen=True)
class Hex8MicromorphicElement:
    """Full-integration Q1 element for 18 scalar micromorphic slip fields.

    The local signed slips are supplied at the eight 2x2x2 integration points.
    This element is the independent oracle for the later Fortran UEL; it is not
    itself a polycrystal solver and contains no displacement DOFs.
    """

    coordinates_m: Array
    directions: Array
    normals: Array
    parameters: MicromorphicParameters

    def __post_init__(self) -> None:
        coordinates = _as_finite_array(self.coordinates_m, (8, 3), "coordinates_m")
        directions, normals = _validate_geometry(self.directions, self.normals)
        self.parameters.validate(directions.shape[0])
        object.__setattr__(self, "coordinates_m", coordinates)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "normals", normals)

    @property
    def n_slip(self) -> int:
        return self.directions.shape[0]

    @staticmethod
    def integration_point_table() -> tuple[tuple[int, Array], ...]:
        """Return the locked Abaqus C3D8 2x2x2 integration-point ordering.

        IDs 1--8 follow the standard HEX8 sign sequence used for the element
        nodes and by the Abaqus full-integration brick.  Exposing the natural
        coordinates prevents local material-point data from being silently
        permuted at the Python/Fortran/Abaqus boundary.
        """

        point = 1.0 / np.sqrt(3.0)
        return tuple(
            (index, signs.copy() * point)
            for index, signs in enumerate(_HEX8_SIGNS, start=1)
        )

    def integration_data(self) -> tuple[tuple[Array, Array, float], ...]:
        result: list[tuple[Array, Array, float]] = []
        for _, natural in self.integration_point_table():
            shape, dshape_dnatural = _hex8_shape(natural)
            jacobian = self.coordinates_m.T @ dshape_dnatural
            determinant = float(np.linalg.det(jacobian))
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError("HEX8 mapping must have a finite positive Jacobian")
            gradient = dshape_dnatural @ np.linalg.inv(jacobian)
            result.append((shape, gradient, determinant))
        return tuple(result)

    def energy_residual_tangent(
        self,
        nodal_micromorphic_slip: Array,
        local_signed_slip_at_ip: Array,
    ) -> tuple[float, Array, Array]:
        nodal = _as_finite_array(
            nodal_micromorphic_slip,
            (8, self.n_slip),
            "nodal_micromorphic_slip",
        )
        local_ip = _as_finite_array(
            local_signed_slip_at_ip,
            (8, self.n_slip),
            "local_signed_slip_at_ip",
        )
        residual = np.zeros((8, self.n_slip), dtype=np.float64)
        tangent = np.zeros((8, self.n_slip, 8, self.n_slip), dtype=np.float64)
        energy = 0.0
        for ip, (shape, gradient, weight_jacobian) in enumerate(self.integration_data()):
            bar_gamma = shape @ nodal
            grad_bar_gamma = nodal.T @ gradient
            response = evaluate_micromorphic_state(
                local_ip[ip],
                bar_gamma,
                grad_bar_gamma,
                self.directions,
                self.normals,
                self.parameters,
            )
            energy += weight_jacobian * response.total_energy_J_m3
            residual += weight_jacobian * (
                np.einsum("i,a->ia", shape, response.penalty_microstress_Pa)
                + np.einsum("ik,ak->ia", gradient, response.higher_order_stress_Pa_m)
            )
            penalty = self.parameters.penalty_modulus_Pa * np.einsum(
                "i,j,ab->iajb", shape, shape, np.eye(self.n_slip)
            )
            gradient_part = np.einsum(
                "ik,akbl,jl->iajb",
                gradient,
                response.gradient_hessian_Pa_m2,
                gradient,
            )
            tangent += weight_jacobian * (penalty + gradient_part)
        return float(energy), residual, tangent
