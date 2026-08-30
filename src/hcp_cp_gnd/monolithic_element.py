"""Reference assembly for the monolithic ``u-T-zeta`` HEX8 element.

This module is an assembly contract, not a replacement constitutive model.  A
caller must supply the *algorithmically condensed* material-point response

``y = [P(9), q, pi(18), xi(18x3)]``

and its derivative with respect to

``x = [F(9), T, zeta(18), Grad(zeta)(18x3)]``.

The component order is row-major for second-order tensors and
``(slip-system, reference-direction)`` for microfield gradients.  The local
element-vector order mirrors the intended Abaqus UEL connectivity: eight
physical nodes with ``[u1,u2,u3,T]`` followed by eight coincident carrier
nodes with 18 signed micromorphic slips.  The total size is therefore 176.

The returned residual is the internal/balance residual.  An Abaqus UEL must
return its negative in ``RHS``.  Committed integration-point states are never
passed directly to the callback: each call receives a deep copy and mutation
of that trial input is rejected.  This gives a testable transaction boundary
before the same rule is implemented in Fortran/UEL storage.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import pickle
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

N_NODE = 8
N_SLIP = 18
N_MAIN_DOF_PER_NODE = 4
N_MICRO_DOF_PER_NODE = N_SLIP
N_MAIN_DOF = N_NODE * N_MAIN_DOF_PER_NODE
N_MICRO_DOF = N_NODE * N_MICRO_DOF_PER_NODE
N_ELEMENT_DOF = N_MAIN_DOF + N_MICRO_DOF
N_POINT_INPUT = 9 + 1 + N_SLIP + 3 * N_SLIP
N_POINT_RESPONSE = N_POINT_INPUT

F_SLICE = slice(0, 9)
T_INDEX = 9
ZETA_SLICE = slice(10, 10 + N_SLIP)
GRAD_ZETA_SLICE = slice(10 + N_SLIP, N_POINT_INPUT)
P_SLICE = F_SLICE
Q_INDEX = T_INDEX
PI_SLICE = ZETA_SLICE
XI_SLICE = GRAD_ZETA_SLICE

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


def _finite(value: Any, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _frozen(value: Any, shape: tuple[int, ...], name: str) -> Array:
    array = _finite(value, shape, name).copy()
    array.setflags(write=False)
    return array


def _fingerprint(value: Any) -> str:
    """Hash a Python reference state for mutation auditing.

    The production Fortran implementation will use an explicit numeric state
    layout and bytewise comparison.  Pickle is intentionally confined to this
    in-process verification oracle.
    """

    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception as exc:  # pragma: no cover - defensive contract boundary
        raise TypeError("committed state must be serializable for transaction audit") from exc
    return sha256(payload).hexdigest()


def _shape_data(natural: Array) -> tuple[Array, Array]:
    natural = _finite(natural, (3,), "natural coordinate")
    factors = 1.0 + _HEX8_SIGNS * natural[None, :]
    shape = 0.125 * np.prod(factors, axis=1)
    derivatives = np.empty((N_NODE, 3), dtype=np.float64)
    for direction in range(3):
        other = [index for index in range(3) if index != direction]
        derivatives[:, direction] = (
            0.125
            * _HEX8_SIGNS[:, direction]
            * factors[:, other[0]]
            * factors[:, other[1]]
        )
    return shape, derivatives


@dataclass(frozen=True)
class MonolithicElementParameters:
    """Constant verification thermal coefficients in the reference frame."""

    volumetric_heat_capacity_J_m3K: float
    conductivity_W_mK: Array

    def __post_init__(self) -> None:
        capacity = float(self.volumetric_heat_capacity_J_m3K)
        conductivity = _frozen(self.conductivity_W_mK, (3, 3), "conductivity_W_mK")
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("volumetric heat capacity must be finite and positive")
        if not np.allclose(conductivity, conductivity.T, atol=1.0e-13):
            raise ValueError("conductivity must be symmetric")
        if float(np.min(np.linalg.eigvalsh(conductivity))) <= 0.0:
            raise ValueError("conductivity must be positive definite")
        object.__setattr__(self, "volumetric_heat_capacity_J_m3K", capacity)
        object.__setattr__(self, "conductivity_W_mK", conductivity)


@dataclass(frozen=True)
class PointFieldState:
    """Fields reconstructed at one locked Abaqus C3D8 integration point."""

    integration_point: int
    natural_coordinates: Array
    deformation_gradient_previous: Array
    deformation_gradient_current: Array
    temperature_previous_K: float
    temperature_current_K: float
    temperature_gradient_previous_K_m: Array
    temperature_gradient_current_K_m: Array
    zeta_previous: Array
    zeta_current: Array
    zeta_gradient_previous_m_inv: Array
    zeta_gradient_current_m_inv: Array
    dt_s: float

    def __post_init__(self) -> None:
        if self.integration_point not in range(1, 9):
            raise ValueError("integration point must be in 1..8")
        for name, shape in (
            ("natural_coordinates", (3,)),
            ("deformation_gradient_previous", (3, 3)),
            ("deformation_gradient_current", (3, 3)),
            ("temperature_gradient_previous_K_m", (3,)),
            ("temperature_gradient_current_K_m", (3,)),
            ("zeta_previous", (N_SLIP,)),
            ("zeta_current", (N_SLIP,)),
            ("zeta_gradient_previous_m_inv", (N_SLIP, 3)),
            ("zeta_gradient_current_m_inv", (N_SLIP, 3)),
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        scalars = np.array(
            [self.temperature_previous_K, self.temperature_current_K, self.dt_s]
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ValueError("temperatures and dt must be finite and positive")
        for name in ("deformation_gradient_previous", "deformation_gradient_current"):
            if float(np.linalg.det(getattr(self, name))) <= 0.0:
                raise ValueError(f"{name} must have positive determinant")

    def current_input_vector(self) -> Array:
        vector = np.empty(N_POINT_INPUT, dtype=np.float64)
        vector[F_SLICE] = self.deformation_gradient_current.reshape(-1)
        vector[T_INDEX] = self.temperature_current_K
        vector[ZETA_SLICE] = self.zeta_current
        vector[GRAD_ZETA_SLICE] = self.zeta_gradient_current_m_inv.reshape(-1)
        return vector


@dataclass(frozen=True)
class CondensedPointResponse:
    """A complete local response after implicit-state condensation.

    ``algorithmic_jacobian`` must already include the derivative of all local
    state variables through their converged integration residual.  Supplying
    an instantaneous or frozen-state derivative under this name is a contract
    violation.
    """

    first_piola_Pa: Array
    heat_source_W_m3: float
    penalty_microstress_Pa: Array
    higher_order_stress_Pa_m: Array
    algorithmic_jacobian: Array
    trial_state: Any
    branch_signature: tuple[str, ...]
    branch_audit: Any | None = None
    converged: bool = True

    def __post_init__(self) -> None:
        for name, shape in (
            ("first_piola_Pa", (3, 3)),
            ("penalty_microstress_Pa", (N_SLIP,)),
            ("higher_order_stress_Pa_m", (N_SLIP, 3)),
            ("algorithmic_jacobian", (N_POINT_RESPONSE, N_POINT_INPUT)),
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        if not np.isfinite(self.heat_source_W_m3):
            raise ValueError("heat source must be finite")
        if not self.converged:
            raise RuntimeError("local implicit update did not converge")
        if not self.branch_signature or not all(
            isinstance(item, str) and item for item in self.branch_signature
        ):
            raise ValueError("a non-empty branch signature is required")

    def response_vector(self) -> Array:
        result = np.empty(N_POINT_RESPONSE, dtype=np.float64)
        result[P_SLICE] = self.first_piola_Pa.reshape(-1)
        result[Q_INDEX] = self.heat_source_W_m3
        result[PI_SLICE] = self.penalty_microstress_Pa
        result[XI_SLICE] = self.higher_order_stress_Pa_m.reshape(-1)
        return result


PointCallback = Callable[[PointFieldState, Any], CondensedPointResponse]


@dataclass(frozen=True)
class MonolithicElementAssembly:
    residual: Array
    tangent: Array
    trial_states: tuple[Any, ...]
    point_fields: tuple[PointFieldState, ...]
    branch_signatures: tuple[tuple[str, ...], ...]
    branch_audits: tuple[Any | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual", _frozen(self.residual, (N_ELEMENT_DOF,), "residual"))
        object.__setattr__(
            self, "tangent", _frozen(self.tangent, (N_ELEMENT_DOF, N_ELEMENT_DOF), "tangent")
        )
        if not (
            len(self.trial_states)
            == len(self.point_fields)
            == len(self.branch_signatures)
            == len(self.branch_audits)
            == 8
        ):
            raise ValueError("assembly must contain eight aligned integration-point records")

    @property
    def rhs_for_abaqus(self) -> Array:
        return -self.residual.copy()

    @property
    def residual_block_norms(self) -> dict[str, float]:
        return {
            "displacement": float(
                np.linalg.norm(self.residual[:N_MAIN_DOF].reshape(N_NODE, 4)[:, :3])
            ),
            "temperature": float(
                np.linalg.norm(self.residual[:N_MAIN_DOF].reshape(N_NODE, 4)[:, 3])
            ),
            "micromorphic": float(np.linalg.norm(self.residual[N_MAIN_DOF:])),
        }


@dataclass(frozen=True)
class Hex8MonolithicAssembler:
    """Total-Lagrangian, quasi-static ``u-T-18 zeta`` reference assembler."""

    physical_coordinates_m: Array
    carrier_coordinates_m: Array
    parameters: MonolithicElementParameters

    def __post_init__(self) -> None:
        physical = _frozen(self.physical_coordinates_m, (N_NODE, 3), "physical_coordinates_m")
        carrier = _frozen(self.carrier_coordinates_m, (N_NODE, 3), "carrier_coordinates_m")
        scale = max(float(np.max(np.abs(physical))), 1.0)
        if not np.allclose(physical, carrier, rtol=0.0, atol=1.0e-13 * scale):
            raise ValueError("physical and microfield carrier nodes must be coincident")
        object.__setattr__(self, "physical_coordinates_m", physical)
        object.__setattr__(self, "carrier_coordinates_m", carrier)
        # Evaluate every mapping now so invalid/inverted elements fail before a trial call.
        self.integration_data()

    @staticmethod
    def integration_point_table() -> tuple[tuple[int, Array], ...]:
        point = 1.0 / np.sqrt(3.0)
        return tuple(
            (index, signs.copy() * point)
            for index, signs in enumerate(_HEX8_SIGNS, start=1)
        )

    def integration_data(self) -> tuple[tuple[int, Array, Array, Array, float], ...]:
        result: list[tuple[int, Array, Array, Array, float]] = []
        for ip, natural in self.integration_point_table():
            shape, derivative_natural = _shape_data(natural)
            jacobian = self.physical_coordinates_m.T @ derivative_natural
            determinant = float(np.linalg.det(jacobian))
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError("HEX8 reference mapping must have a finite positive Jacobian")
            gradient = derivative_natural @ np.linalg.inv(jacobian)
            result.append((ip, natural, shape, gradient, determinant))
        return tuple(result)

    @staticmethod
    def unpack_dofs(vector: Array) -> tuple[Array, Array, Array]:
        vector = _finite(vector, (N_ELEMENT_DOF,), "element vector")
        main = vector[:N_MAIN_DOF].reshape(N_NODE, N_MAIN_DOF_PER_NODE)
        displacement = main[:, :3]
        temperature = main[:, 3]
        zeta = vector[N_MAIN_DOF:].reshape(N_NODE, N_SLIP)
        return displacement, temperature, zeta

    @staticmethod
    def pack_dofs(displacement: Array, temperature: Array, zeta: Array) -> Array:
        displacement = _finite(displacement, (N_NODE, 3), "displacement")
        temperature = _finite(temperature, (N_NODE,), "temperature")
        zeta = _finite(zeta, (N_NODE, N_SLIP), "zeta")
        result = np.empty(N_ELEMENT_DOF, dtype=np.float64)
        main = result[:N_MAIN_DOF].reshape(N_NODE, N_MAIN_DOF_PER_NODE)
        main[:, :3] = displacement
        main[:, 3] = temperature
        result[N_MAIN_DOF:] = zeta.reshape(-1)
        return result

    @staticmethod
    def _input_operator(shape: Array, gradient: Array) -> Array:
        operator = np.zeros((N_POINT_INPUT, N_ELEMENT_DOF), dtype=np.float64)
        for node in range(N_NODE):
            main_base = node * N_MAIN_DOF_PER_NODE
            micro_base = N_MAIN_DOF + node * N_MICRO_DOF_PER_NODE
            for component in range(3):
                for reference in range(3):
                    operator[3 * component + reference, main_base + component] = gradient[
                        node, reference
                    ]
            operator[T_INDEX, main_base + 3] = shape[node]
            for slip in range(N_SLIP):
                operator[ZETA_SLICE.start + slip, micro_base + slip] = shape[node]
                for reference in range(3):
                    row = GRAD_ZETA_SLICE.start + 3 * slip + reference
                    operator[row, micro_base + slip] = gradient[node, reference]
        return operator

    @staticmethod
    def _residual_operator(shape: Array, gradient: Array) -> Array:
        operator = np.zeros((N_ELEMENT_DOF, N_POINT_RESPONSE), dtype=np.float64)
        for node in range(N_NODE):
            main_base = node * N_MAIN_DOF_PER_NODE
            micro_base = N_MAIN_DOF + node * N_MICRO_DOF_PER_NODE
            for component in range(3):
                for reference in range(3):
                    operator[main_base + component, 3 * component + reference] = gradient[
                        node, reference
                    ]
            operator[main_base + 3, Q_INDEX] = -shape[node]
            for slip in range(N_SLIP):
                operator[micro_base + slip, PI_SLICE.start + slip] = shape[node]
                for reference in range(3):
                    column = XI_SLICE.start + 3 * slip + reference
                    operator[micro_base + slip, column] = gradient[node, reference]
        return operator

    def assemble(
        self,
        current_dofs: Array,
        previous_dofs: Array,
        dt_s: float,
        committed_states: tuple[Any, ...],
        point_callback: PointCallback,
    ) -> MonolithicElementAssembly:
        """Assemble residual/tangent without committing any local state."""

        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if len(committed_states) != 8:
            raise ValueError("exactly eight committed integration-point states are required")
        current = _finite(current_dofs, (N_ELEMENT_DOF,), "current_dofs")
        previous = _finite(previous_dofs, (N_ELEMENT_DOF,), "previous_dofs")
        u, temperature, zeta = self.unpack_dofs(current)
        u_previous, temperature_previous, zeta_previous = self.unpack_dofs(previous)
        residual = np.zeros(N_ELEMENT_DOF, dtype=np.float64)
        tangent = np.zeros((N_ELEMENT_DOF, N_ELEMENT_DOF), dtype=np.float64)
        trial_states: list[Any] = []
        point_fields: list[PointFieldState] = []
        signatures: list[tuple[str, ...]] = []
        branch_audits: list[Any | None] = []
        identity = np.eye(3)
        capacity = self.parameters.volumetric_heat_capacity_J_m3K
        conductivity = self.parameters.conductivity_W_mK

        caller_hashes = tuple(_fingerprint(state) for state in committed_states)
        pending_error: BaseException | None = None
        try:
          for ip, natural, shape, gradient, weight in self.integration_data():
            F = identity + u.T @ gradient
            F_previous = identity + u_previous.T @ gradient
            T = float(shape @ temperature)
            T_previous = float(shape @ temperature_previous)
            grad_T = temperature @ gradient
            grad_T_previous = temperature_previous @ gradient
            zeta_ip = shape @ zeta
            zeta_previous_ip = shape @ zeta_previous
            grad_zeta = zeta.T @ gradient
            grad_zeta_previous = zeta_previous.T @ gradient
            fields = PointFieldState(
                integration_point=ip,
                natural_coordinates=natural,
                deformation_gradient_previous=F_previous,
                deformation_gradient_current=F,
                temperature_previous_K=T_previous,
                temperature_current_K=T,
                temperature_gradient_previous_K_m=grad_T_previous,
                temperature_gradient_current_K_m=grad_T,
                zeta_previous=zeta_previous_ip,
                zeta_current=zeta_ip,
                zeta_gradient_previous_m_inv=grad_zeta_previous,
                zeta_gradient_current_m_inv=grad_zeta,
                dt_s=dt_s,
            )
            callback_state = deepcopy(committed_states[ip - 1])
            callback_hash = _fingerprint(callback_state)
            response = point_callback(fields, callback_state)
            if not isinstance(response, CondensedPointResponse):
                raise TypeError("point callback must return CondensedPointResponse")
            if _fingerprint(callback_state) != callback_hash:
                raise RuntimeError("point callback mutated its committed-state argument")
            input_operator = self._input_operator(shape, gradient)
            residual_operator = self._residual_operator(shape, gradient)
            residual += weight * (residual_operator @ response.response_vector())
            tangent += weight * (
                residual_operator @ response.algorithmic_jacobian @ input_operator
            )

            # Backward-Euler heat capacity plus reference-frame conduction.
            thermal_residual = (
                shape * capacity * (T - T_previous) / dt_s
                + gradient @ (conductivity @ grad_T)
            )
            thermal_tangent = (
                capacity / dt_s * np.outer(shape, shape)
                + gradient @ conductivity @ gradient.T
            )
            temperature_dofs = np.arange(3, N_MAIN_DOF, N_MAIN_DOF_PER_NODE)
            residual[temperature_dofs] += weight * thermal_residual
            tangent[np.ix_(temperature_dofs, temperature_dofs)] += weight * thermal_tangent

            trial_states.append(deepcopy(response.trial_state))
            point_fields.append(fields)
            signatures.append(response.branch_signature)
            branch_audits.append(response.branch_audit)
        except BaseException as error:
            pending_error = error
        finally:
            changed = tuple(
                index + 1
                for index, (state, expected) in enumerate(
                    zip(committed_states, caller_hashes)
                )
                if _fingerprint(state) != expected
            )
            if changed:
                pollution = RuntimeError(
                    "caller committed state changed during trial assembly at "
                    f"integration point(s) {changed}"
                )
                if pending_error is not None:
                    raise pollution from pending_error
                raise pollution
        if pending_error is not None:
            raise pending_error

        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent)):
            raise FloatingPointError("assembled residual/tangent is non-finite")
        return MonolithicElementAssembly(
            residual=residual,
            tangent=tangent,
            trial_states=tuple(trial_states),
            point_fields=tuple(point_fields),
            branch_signatures=tuple(signatures),
            branch_audits=tuple(branch_audits),
        )
