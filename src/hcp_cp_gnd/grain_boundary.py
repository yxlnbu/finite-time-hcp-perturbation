"""Energetically consistent micromorphic grain-boundary response.

The interface model is the independent Python oracle for the grain-boundary
part of the later multi-field UEL.  On each side of an oriented interface the
micromorphic plastic distortion is reconstructed from signed slip as

``B_chi = sum_a chi[a] s[a] outer m[a]``.

With the unit interface normal ``N`` directed from the minus to the plus grain,
the interfacial Burgers tensor and surface energy are

``G = [[B_chi]] cross N = (B_chi_plus - B_chi_minus) @ C(N)``

and

``psi_GB = 1/2 k_GB G:G``.

Here ``C(N) @ x == N cross x`` and the tensor cross product is applied row by
row, so that ``row @ C(N) == row cross N``.  ``k_GB`` has units J/m2 and the
returned scalar residuals have the same units.  No slip-system correspondence
between the two grains is assumed or required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .micromorphic import cross_matrix

Array = NDArray[np.float64]


def _as_finite_vector(value: Array, size: int, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_slip_geometry(
    directions: Array,
    normals: Array,
    side: str,
) -> tuple[Array, Array]:
    directions = np.asarray(directions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 1:
        raise ValueError(f"{side}_directions must have shape (n_slip,3), n_slip >= 1")
    if normals.shape != directions.shape:
        raise ValueError(
            f"{side}_normals must have shape {directions.shape}, got {normals.shape}"
        )
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(normals)):
        raise ValueError(f"{side} slip geometry must be finite")
    if not np.allclose(
        np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=2.0e-12
    ):
        raise ValueError(f"{side} slip directions must be normalized")
    if not np.allclose(
        np.linalg.norm(normals, axis=1), 1.0, rtol=0.0, atol=2.0e-12
    ):
        raise ValueError(f"{side} slip normals must be normalized")
    if not np.allclose(
        np.einsum("ai,ai->a", directions, normals),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise ValueError(f"each {side} slip direction must lie in its slip plane")
    return directions, normals


def _plastic_distortion(slip: Array, directions: Array, normals: Array) -> Array:
    return np.einsum("a,ai,aj->ij", slip, directions, normals)


@dataclass(frozen=True)
class GrainBoundaryResponse:
    """Surface energy, exact first derivatives, and exact Hessian.

    The full residual and tangent use the variable order
    ``[chi_minus, chi_plus]``.  The tensor microtractions are derivatives with
    respect to the two independent distortion tensors; consequently
    ``microtraction_minus_B + microtraction_plus_B == 0`` exactly.  Scalar
    residuals are projections of those tensors on the side-specific Schmid
    tensors and therefore need not be pairwise opposite for dissimilar grains.
    """

    distortion_minus: Array
    distortion_plus: Array
    distortion_jump: Array
    interfacial_burgers_tensor: Array
    energy_J_m2: float
    microtraction_minus_B_J_m2: Array
    microtraction_plus_B_J_m2: Array
    residual_minus_J_m2: Array
    residual_plus_J_m2: Array
    tangent_J_m2: Array

    @property
    def residual_J_m2(self) -> Array:
        return np.concatenate((self.residual_minus_J_m2, self.residual_plus_J_m2))


def evaluate_grain_boundary_state(
    slip_minus: Array,
    slip_plus: Array,
    directions_minus: Array,
    normals_minus: Array,
    directions_plus: Array,
    normals_plus: Array,
    interface_normal: Array,
    grain_boundary_modulus_J_m2: float,
) -> GrainBoundaryResponse:
    """Evaluate the quadratic grain-boundary energy and its exact derivatives.

    Parameters
    ----------
    slip_minus, slip_plus
        Signed micromorphic slips on the two sides of the interface.
    directions_*, normals_*
        Unit slip directions and unit plane normals in a common spatial frame.
        The two sides may contain different numbers and orderings of systems.
    interface_normal
        Unit normal directed from the minus side to the plus side.  Reversing
        the normal leaves the energy, residual, and tangent unchanged.
    grain_boundary_modulus_J_m2
        Non-negative surface-energy coefficient ``k_GB``.  Zero recovers the
        microfree interface exactly.
    """

    directions_minus, normals_minus = _validate_slip_geometry(
        directions_minus, normals_minus, "minus"
    )
    directions_plus, normals_plus = _validate_slip_geometry(
        directions_plus, normals_plus, "plus"
    )
    n_minus = directions_minus.shape[0]
    n_plus = directions_plus.shape[0]
    slip_minus = _as_finite_vector(slip_minus, n_minus, "slip_minus")
    slip_plus = _as_finite_vector(slip_plus, n_plus, "slip_plus")
    normal = _as_finite_vector(interface_normal, 3, "interface_normal")
    if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=2.0e-12):
        raise ValueError("interface_normal must be normalized")
    modulus = float(grain_boundary_modulus_J_m2)
    if not np.isfinite(modulus) or modulus < 0.0:
        raise ValueError("grain_boundary_modulus_J_m2 must be finite and non-negative")

    distortion_minus = _plastic_distortion(
        slip_minus, directions_minus, normals_minus
    )
    distortion_plus = _plastic_distortion(slip_plus, directions_plus, normals_plus)
    jump = distortion_plus - distortion_minus
    normal_cross = cross_matrix(normal)
    burgers_tensor = jump @ normal_cross

    schmid_minus = np.einsum("ai,aj->aij", directions_minus, normals_minus)
    schmid_plus = np.einsum("ai,aj->aij", directions_plus, normals_plus)
    sensitivity_minus = -np.einsum("aij,jk->aik", schmid_minus, normal_cross)
    sensitivity_plus = np.einsum("aij,jk->aik", schmid_plus, normal_cross)
    sensitivity = np.concatenate((sensitivity_minus, sensitivity_plus), axis=0)

    residual = modulus * np.einsum("aij,ij->a", sensitivity, burgers_tensor)
    tangent = modulus * np.einsum("aij,bij->ab", sensitivity, sensitivity)
    traction_plus = modulus * burgers_tensor @ normal_cross.T
    traction_minus = -traction_plus
    energy = 0.5 * modulus * float(
        np.einsum("ij,ij->", burgers_tensor, burgers_tensor)
    )

    return GrainBoundaryResponse(
        distortion_minus=distortion_minus,
        distortion_plus=distortion_plus,
        distortion_jump=jump,
        interfacial_burgers_tensor=burgers_tensor,
        energy_J_m2=energy,
        microtraction_minus_B_J_m2=traction_minus,
        microtraction_plus_B_J_m2=traction_plus,
        residual_minus_J_m2=residual[:n_minus],
        residual_plus_J_m2=residual[n_minus:],
        tangent_J_m2=tangent,
    )
