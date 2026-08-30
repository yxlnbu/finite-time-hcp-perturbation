"""HCP slip and extension-twin crystallography.

The implementation constructs systems from Miller--Bravais indices rather than
copying tabulated vectors from another constitutive code.  Vectors are returned
in an orthonormal crystal Cartesian basis with ``x || a1`` and ``z || c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class HCPSystems:
    c_over_a: float
    slip_directions: Array
    slip_normals: Array
    slip_families: tuple[str, ...]
    slip_labels: tuple[str, ...]
    slip_burgers: Array
    twin_directions: Array
    twin_normals: Array
    twin_families: tuple[str, ...]
    twin_labels: tuple[str, ...]
    twin_shear: Array

    @property
    def n_slip(self) -> int:
        return self.slip_directions.shape[0]

    @property
    def n_twin(self) -> int:
        return self.twin_directions.shape[0]

    @property
    def slip_schmid(self) -> Array:
        return np.einsum("ai,aj->aij", self.slip_directions, self.slip_normals)

    @property
    def twin_schmid(self) -> Array:
        return np.einsum("ai,aj->aij", self.twin_directions, self.twin_normals)


def direction_4_to_cart(indices: tuple[int, int, int, int], c_over_a: float) -> Array:
    """Convert ``[u v t w]`` to a Cartesian direction.

    The basal coefficients include the standard factor one third.  This factor
    matters for the relative basal/c-axis length of ``<c+a>`` directions.
    """

    u, v, t, w = indices
    if u + v + t != 0:
        raise ValueError(f"invalid four-index direction {indices}: u+v+t != 0")
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0])
    c = np.array([0.0, 0.0, c_over_a])
    U = (2.0 * u + v) / 3.0
    V = (2.0 * v + u) / 3.0
    vector = U * a1 + V * a2 + (float(w) / 3.0) * c
    return vector / np.linalg.norm(vector)


def plane_4_to_cart(indices: tuple[int, int, int, int], c_over_a: float) -> Array:
    """Convert ``(h k i l)`` to a unit Cartesian plane normal."""

    h, k, i, ell = indices
    if h + k + i != 0:
        raise ValueError(f"invalid four-index plane {indices}: h+k+i != 0")
    direct_basis = np.column_stack(
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0]),
            np.array([0.0, 0.0, c_over_a]),
        )
    )
    reciprocal_basis = np.linalg.inv(direct_basis).T
    # With U=(2u+v)/3, V=(u+2v)/3 and W=w/3, the four-index
    # zone law divided by three is h*U + k*V + ell*W = 0.
    normal = reciprocal_basis @ np.array([float(h), float(k), float(ell)])
    return normal / np.linalg.norm(normal)


def zone_law(plane: tuple[int, int, int, int], direction: tuple[int, int, int, int]) -> int:
    return int(sum(p * d for p, d in zip(plane, direction, strict=True)))


def _canonical_axis(indices: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    for value in indices:
        if value:
            return indices if value > 0 else tuple(-v for v in indices)
    raise ValueError("zero crystallographic direction")


def _unique_axes(candidates: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(set(_canonical_axis(candidate) for candidate in candidates))


def _family_directions(
    basal_pattern: tuple[int, int, int], axial_magnitude: int
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for basal in set(permutations(basal_pattern)):
        for basal_sign, axial_sign in product((-1, 1), repeat=2):
            u, v, t = (basal_sign * value for value in basal)
            w = axial_sign * axial_magnitude
            candidates.append((u, v, t, w))
    return _unique_axes(candidates)


def _six_plane_representatives(ell: int) -> list[tuple[int, int, int, int]]:
    return [
        (1, 0, -1, ell),
        (0, 1, -1, ell),
        (-1, 1, 0, ell),
        (-1, 0, 1, ell),
        (0, -1, 1, ell),
        (1, -1, 0, ell),
    ]


def _select_zone_directions(
    plane: tuple[int, int, int, int],
    candidates: list[tuple[int, int, int, int]],
    expected: int,
) -> list[tuple[int, int, int, int]]:
    selected = [candidate for candidate in candidates if zone_law(plane, candidate) == 0]
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} directions in plane {plane}, found {len(selected)}: {selected}"
        )
    return selected


def build_hcp_systems(
    c_over_a: float = 1.587,
    burgers_a: float = 2.95e-10,
    burgers_ca: float = 5.53e-10,
) -> HCPSystems:
    """Construct 18 slip systems and six ``{10-12}<10-11>`` twins."""

    if c_over_a <= 0.0 or burgers_a <= 0.0 or burgers_ca <= 0.0:
        raise ValueError("lattice ratio and Burgers-vector magnitudes must be positive")

    a_directions = _unique_axes(
        [
            (1, 1, -2, 0),
            (1, -2, 1, 0),
            (-2, 1, 1, 0),
        ]
    )
    ca_directions = _family_directions((1, 1, -2), 3)

    directions: list[Array] = []
    normals: list[Array] = []
    families: list[str] = []
    labels: list[str] = []
    burgers: list[float] = []

    basal_plane = (0, 0, 0, 1)
    for direction in a_directions:
        directions.append(direction_4_to_cart(direction, c_over_a))
        normals.append(plane_4_to_cart(basal_plane, c_over_a))
        families.append("basal<a>")
        labels.append(f"(0001)[{direction[0]} {direction[1]} {direction[2]} 0]")
        burgers.append(burgers_a)

    for plane in _six_plane_representatives(0)[:3]:
        direction = _select_zone_directions(plane, a_directions, expected=1)[0]
        directions.append(direction_4_to_cart(direction, c_over_a))
        normals.append(plane_4_to_cart(plane, c_over_a))
        families.append("prismatic<a>")
        labels.append(f"({plane})[{direction}]")
        burgers.append(burgers_a)

    for plane in _six_plane_representatives(1):
        for direction in _select_zone_directions(plane, ca_directions, expected=2):
            directions.append(direction_4_to_cart(direction, c_over_a))
            normals.append(plane_4_to_cart(plane, c_over_a))
            families.append("pyramidal<c+a>")
            labels.append(f"({plane})[{direction}]")
            burgers.append(burgers_ca)

    twin_directions: list[Array] = []
    twin_normals: list[Array] = []
    twin_labels: list[str] = []
    twin_candidates = []
    for basal in set(permutations((1, 0, -1))):
        if sum(basal) != 0:
            continue
        for axial_sign in (-1, 1):
            twin_candidates.append((*basal, axial_sign))

    for plane in _six_plane_representatives(2):
        in_plane = [candidate for candidate in twin_candidates if zone_law(plane, candidate) == 0]
        positive_c = [candidate for candidate in in_plane if candidate[3] > 0]
        if len(positive_c) != 1:
            raise RuntimeError(f"extension-twin direction is ambiguous for plane {plane}: {in_plane}")
        direction = positive_c[0]
        twin_directions.append(direction_4_to_cart(direction, c_over_a))
        twin_normals.append(plane_4_to_cart(plane, c_over_a))
        twin_labels.append(f"({plane})[{direction}]")

    twin_shear_value = abs((3.0 - c_over_a**2) / (np.sqrt(3.0) * c_over_a))
    result = HCPSystems(
        c_over_a=float(c_over_a),
        slip_directions=np.asarray(directions),
        slip_normals=np.asarray(normals),
        slip_families=tuple(families),
        slip_labels=tuple(labels),
        slip_burgers=np.asarray(burgers),
        twin_directions=np.asarray(twin_directions),
        twin_normals=np.asarray(twin_normals),
        twin_families=tuple("extension{10-12}<10-11>" for _ in twin_labels),
        twin_labels=tuple(twin_labels),
        twin_shear=np.full(len(twin_labels), twin_shear_value),
    )
    _validate_systems(result)
    return result


def _validate_systems(systems: HCPSystems, atol: float = 2.0e-12) -> None:
    if systems.n_slip != 18:
        raise RuntimeError(f"expected 18 slip systems, found {systems.n_slip}")
    if systems.n_twin != 6:
        raise RuntimeError(f"expected 6 twin systems, found {systems.n_twin}")
    for name, directions, normals in (
        ("slip", systems.slip_directions, systems.slip_normals),
        ("twin", systems.twin_directions, systems.twin_normals),
    ):
        if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=atol):
            raise RuntimeError(f"{name} directions are not normalized")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=atol):
            raise RuntimeError(f"{name} normals are not normalized")
        if not np.allclose(np.einsum("ai,ai->a", directions, normals), 0.0, atol=atol):
            raise RuntimeError(f"{name} direction does not lie in its plane")
        schmid = np.einsum("ai,aj->aij", directions, normals)
        if not np.allclose(np.trace(schmid, axis1=1, axis2=2), 0.0, atol=atol):
            raise RuntimeError(f"{name} Schmid tensor is not traceless")
