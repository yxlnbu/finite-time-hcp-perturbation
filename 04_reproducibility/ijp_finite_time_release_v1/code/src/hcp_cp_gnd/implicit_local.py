"""Implicit material-point reference update and static condensation.

This module is the constitutive reference layer for the future monolithic
``u-T-zeta`` UEL.  It composes, but never modifies, the frozen HCP CP v0.1
material point.  The endpoint update is a backward-Euler solve for the real
v0.1 state variables that are local to an integration point:

* 18 signed slip increments,
* 18 mobile and 18 dipole dislocation densities, and
* six extension-twin fractions.

``Fp`` is reconstructed with the same exponential-map kinematics used by
v0.1, while accumulated slip and the three irreversible thermal/work ledgers
are reconstructed from the converged local solution.  ``zeta`` remains an
external global field and is never duplicated into the 92-real local state.
The committed state is immutable, so a failed trial or an Abaqus cutback
cannot pollute it.

The Jacobians in this file are deliberately transparent finite-difference
reference Jacobians.  Static condensation is exact with respect to those
discrete Jacobians,

``du/dy = -R_u^{-1} R_y`` and
``dg/dy = G_y + G_u du/dy``,

where ``y = [F_np1, T_np1, zeta_np1]`` and
``g = [P, qdot, pi, Dcp]``.  This is an executable oracle for a later analytic
or AD Fortran implementation; it is not yet an Abaqus ``AMATRX``/``DDSDDE``
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .local_coupling import LocalCouplingParameters, _matrix_exponential_3x3
from .state_contract import LocalState92, initial_local_state

Array = NDArray[np.float64]
K_B = 1.380649e-23


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


def _state_bytes(state: LocalState92) -> bytes:
    """Canonical byte fingerprint through the public 92-real row-major contract."""

    return state.pack().astype("<f8", copy=False).tobytes()


def _copy_state(state: LocalState92) -> LocalState92:
    return LocalState92.unpack(state.pack())


class LocalConvergenceError(RuntimeError):
    """The local Newton solve failed without changing the committed state."""


class LocalCutbackRequired(LocalConvergenceError):
    """The converged/requested local increment exceeds the v0.1 bounds."""

    def __init__(self, message: str, suggested_factor: float) -> None:
        super().__init__(message)
        if not np.isfinite(suggested_factor) or not 0.0 < suggested_factor < 1.0:
            raise ValueError("suggested_factor must lie strictly between zero and one")
        self.suggested_factor = float(suggested_factor)


class LocalNondifferentiableError(LocalConvergenceError):
    """A centered derivative probe crossed or approached a constitutive switch."""


@dataclass(frozen=True)
class ImplicitSolverOptions:
    """Numerical controls for the executable reference oracle."""

    residual_tolerance: float = 2.0e-9
    maximum_iterations: int = 24
    finite_difference_step: float = 2.0e-6
    minimum_line_search_factor: float = 2.0**-16
    cutback_safety: float = 0.8
    maximum_local_jacobian_condition: float = 1.0e12

    def validate(self) -> None:
        values = np.array(
            [
                self.residual_tolerance,
                self.finite_difference_step,
                self.minimum_line_search_factor,
                self.cutback_safety,
                self.maximum_local_jacobian_condition,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all implicit solver controls must be finite and positive")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.minimum_line_search_factor >= 1.0:
            raise ValueError("minimum_line_search_factor must be smaller than one")
        if self.cutback_safety >= 1.0:
            raise ValueError("cutback_safety must be smaller than one")
        if self.maximum_local_jacobian_condition <= 1.0:
            raise ValueError("maximum local Jacobian condition must exceed one")


@dataclass(frozen=True)
class LocalUnknownLayout:
    """Slices and scales for the 60 dimensionless Newton coordinates."""

    n_slip: int
    n_twin: int
    slip_scale: float
    twin_scale: float

    @property
    def slip(self) -> slice:
        return slice(0, self.n_slip)

    @property
    def log_rho_mobile(self) -> slice:
        return slice(self.n_slip, 2 * self.n_slip)

    @property
    def log_rho_dipole(self) -> slice:
        return slice(2 * self.n_slip, 3 * self.n_slip)

    @property
    def twin(self) -> slice:
        return slice(3 * self.n_slip, 3 * self.n_slip + self.n_twin)

    @property
    def size(self) -> int:
        return 3 * self.n_slip + self.n_twin


@dataclass(frozen=True)
class CondensedLocalDerivatives:
    """Endpoint derivative matrix and its uncondensed audit factors.

    Rows follow ``[P(9), qdot(1), pi(18), Dcp(1)]``.  Columns follow
    ``[F_np1(9), T_np1(1), zeta_np1(18)]``.  Matrix flattening is C order.
    All derivatives are with respect to physical SI-valued inputs, while the
    local residual Jacobian is with respect to dimensionless Newton
    coordinates.
    """

    condensed: Array
    local_residual_jacobian: Array
    external_residual_jacobian: Array
    local_coordinate_sensitivity: Array
    partial_output_local: Array
    partial_output_external: Array
    schur_residual: Array
    local_jacobian_condition_2: float
    local_jacobian_minimum_singular_value: float
    differentiation_method: str = "CENTRAL_FINITE_DIFFERENCE_BRANCH_CHECKED_REFERENCE"
    abaqus_production_tangent: bool = False

    def __post_init__(self) -> None:
        shapes = {
            "condensed": (29, 28),
            "local_residual_jacobian": (60, 60),
            "external_residual_jacobian": (60, 28),
            "local_coordinate_sensitivity": (60, 28),
            "partial_output_local": (29, 60),
            "partial_output_external": (29, 28),
            "schur_residual": (60, 28),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        conditioning = np.array(
            [
                self.local_jacobian_condition_2,
                self.local_jacobian_minimum_singular_value,
            ]
        )
        if not np.all(np.isfinite(conditioning)) or np.any(conditioning <= 0.0):
            raise ValueError("local Jacobian conditioning metrics must be positive and finite")
        if (
            self.differentiation_method
            != "CENTRAL_FINITE_DIFFERENCE_BRANCH_CHECKED_REFERENCE"
        ):
            raise ValueError("unexpected differentiation method")
        if self.abaqus_production_tangent:
            raise ValueError("the Python reference must not claim an Abaqus production tangent")

    @property
    def dP_dF(self) -> Array:
        return self.condensed[0:9, 0:9]

    @property
    def dP_dT(self) -> Array:
        return self.condensed[0:9, 9:10]

    @property
    def dP_dzeta(self) -> Array:
        return self.condensed[0:9, 10:28]

    @property
    def dq_dF(self) -> Array:
        return self.condensed[9:10, 0:9]

    @property
    def dq_dT(self) -> Array:
        return self.condensed[9:10, 9:10]

    @property
    def dq_dzeta(self) -> Array:
        return self.condensed[9:10, 10:28]


@dataclass(frozen=True)
class LocalBranchAudit:
    """Piecewise constitutive branch contract at one converged endpoint.

    ``categories`` are the machine-comparable branch identifiers used by
    centered-difference audits.  ``signed_distances`` retain the corresponding
    pre-clamp branch functions and ``distance_scales`` make their distance from
    a switch auditable without mixing Pa, m and dimensionless quantities.
    """

    labels: tuple[str, ...]
    categories: tuple[int, ...]
    signed_distances: Array
    distance_scales: Array

    def __post_init__(self) -> None:
        count = len(self.labels)
        if len(self.categories) != count or count == 0:
            raise ValueError("branch labels and categories must be non-empty and aligned")
        object.__setattr__(
            self,
            "signed_distances",
            _frozen(self.signed_distances, (count,), "branch signed distances"),
        )
        object.__setattr__(
            self,
            "distance_scales",
            _frozen(self.distance_scales, (count,), "branch distance scales"),
        )
        if np.any(self.distance_scales <= 0.0):
            raise ValueError("branch distance scales must be positive")

    @property
    def normalized_distances(self) -> Array:
        result = np.abs(self.signed_distances) / self.distance_scales
        result.setflags(write=False)
        return result

    @property
    def category_tokens(self) -> tuple[str, ...]:
        return tuple(
            f"{label}={category}" for label, category in zip(self.labels, self.categories)
        )

    def compatible_with(
        self,
        plus: "LocalBranchAudit",
        minus: "LocalBranchAudit",
        *,
        normalized_distance_floor: float = 1.0e-10,
    ) -> bool:
        """Whether a centered pair stays safely inside the base active set."""

        if (
            self.labels != plus.labels
            or self.labels != minus.labels
        ):
            return False
        for index, label in enumerate(self.labels):
            if label.startswith("switch.") or label.endswith(".dissipation"):
                continue
            if (
                self.categories[index] != plus.categories[index]
                or self.categories[index] != minus.categories[index]
            ):
                return False
            scale = max(
                float(self.distance_scales[index]),
                float(plus.distance_scales[index]),
                float(minus.distance_scales[index]),
            )
            minimum_distance = min(
                abs(float(self.signed_distances[index])),
                abs(float(plus.signed_distances[index])),
                abs(float(minus.signed_distances[index])),
            ) / scale
            if minimum_distance <= normalized_distance_floor:
                return False
        return True


@dataclass(frozen=True)
class ImplicitLocalResponse:
    """Converged endpoint observation for the monolithic local contract."""

    first_piola: Array
    heat_source: float
    penalty_microstress: Array
    cp_dissipation_rate: float
    mechanical_slip_power_rate: float
    microforce_exchange_rate: float
    twin_power_rate: float
    driving_power_residual: float
    mechanical_resolved_slip: Array
    effective_resolved_slip: Array
    resolved_twin: Array
    slip_rate: Array
    twin_rate: Array
    slip_resistance: Array
    mean_free_path: Array
    Lp: Array
    local_residual: Array
    local_coordinates: Array
    newton_iterations: int
    local_residual_norm: float
    derivatives: CondensedLocalDerivatives | None
    branch_audit: LocalBranchAudit
    endpoint_scheme: str = "BACKWARD_EULER_EXPONENTIAL_FP"
    state_update_scope: str = "HCP_CP_LOCAL_STATE_92_V1_FULL"

    def __post_init__(self) -> None:
        for name, shape in (
            ("first_piola", (3, 3)),
            ("penalty_microstress", (18,)),
            ("mechanical_resolved_slip", (18,)),
            ("effective_resolved_slip", (18,)),
            ("resolved_twin", (6,)),
            ("slip_rate", (18,)),
            ("twin_rate", (6,)),
            ("slip_resistance", (18,)),
            ("mean_free_path", (18,)),
            ("Lp", (3, 3)),
            ("local_residual", (60,)),
            ("local_coordinates", (60,)),
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        scalars = np.array(
            [
                self.heat_source,
                self.cp_dissipation_rate,
                self.mechanical_slip_power_rate,
                self.microforce_exchange_rate,
                self.twin_power_rate,
                self.driving_power_residual,
                self.local_residual_norm,
            ]
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("implicit response contains a non-finite scalar")
        if self.heat_source < -1.0e-7 or self.cp_dissipation_rate < -1.0e-7:
            raise ValueError("implicit response violates non-negative dissipation")
        if self.newton_iterations < 0:
            raise ValueError("newton_iterations must be non-negative")

    @property
    def q(self) -> float:
        """Thermal source ``beta*Dcp`` in W/m^3."""

        return self.heat_source

    @property
    def pi(self) -> Array:
        """Penalty microstress ``H_chi*(zeta-gamma_signed)`` in Pa."""

        return self.penalty_microstress

    @property
    def Dcp(self) -> float:
        """Crystal-plastic dissipation rate in W/m^3."""

        return self.cp_dissipation_rate


@dataclass(frozen=True)
class ImplicitLocalTransaction:
    """Pure trial result with explicit accept/reject operations."""

    committed_state: LocalState92
    trial_state: LocalState92
    response: ImplicitLocalResponse
    dt: float
    committed_fingerprint: bytes
    trial_fingerprint: bytes

    def commit(self) -> LocalState92:
        if _state_bytes(self.committed_state) != self.committed_fingerprint:
            raise RuntimeError("committed state changed after the trial was constructed")
        if _state_bytes(self.trial_state) != self.trial_fingerprint:
            raise RuntimeError("trial state changed after the transaction was constructed")
        return _copy_state(self.trial_state)

    def rollback(self) -> LocalState92:
        if _state_bytes(self.committed_state) != self.committed_fingerprint:
            raise RuntimeError("committed state changed after the trial was constructed")
        if _state_bytes(self.trial_state) != self.trial_fingerprint:
            raise RuntimeError("trial state changed after the transaction was constructed")
        return _copy_state(self.committed_state)


@dataclass(frozen=True)
class _ExternalPoint:
    F: Array
    temperature: float
    zeta: Array


@dataclass(frozen=True)
class _LocalFields:
    dgamma: Array
    rho_mobile: Array
    rho_dipole: Array
    twin_increment: Array
    signed_slip: Array
    accumulated_slip: Array
    twin_fraction: Array
    Fp: Array
    mechanical: Any
    effective_tau: Array
    penalty_microstress: Array
    slip_rate: Array
    twin_increment_target: Array
    twin_rate: Array
    slip_resistance: Array
    mean_free_path: Array
    rho_mobile_target: Array
    rho_dipole_target: Array
    Lp: Array


class ImplicitMicromorphicHCPMaterialPoint:
    """Backward-Euler local integrator for the v0.1 + micromorphic model."""

    def __init__(
        self,
        base_model: Any,
        parameters: LocalCouplingParameters,
        options: ImplicitSolverOptions | None = None,
    ) -> None:
        parameters.validate()
        required = (
            "initial_state",
            "evaluate",
            "slip_rates_from_resolved",
            "twin_rates_from_resolved",
            "systems",
            "parameters",
            "switches",
        )
        missing = [name for name in required if not hasattr(base_model, name)]
        if missing:
            raise TypeError(f"base_model lacks the v0.1 bridge contract: {missing}")
        self.base_model = base_model
        self.parameters = parameters
        self.options = options or ImplicitSolverOptions()
        self.options.validate()
        if int(base_model.systems.n_slip) != 18 or int(base_model.systems.n_twin) != 6:
            raise ValueError("the v0.2 contract requires exactly 18 slip and six twin systems")
        self.layout = LocalUnknownLayout(
            n_slip=18,
            n_twin=6,
            slip_scale=float(base_model.parameters.maximum_slip_increment),
            twin_scale=float(base_model.parameters.maximum_twin_increment),
        )

    @property
    def n_slip(self) -> int:
        return self.layout.n_slip

    @property
    def n_twin(self) -> int:
        return self.layout.n_twin

    def initial_state(self) -> LocalState92:
        base = self.base_model.initial_state()
        state = initial_local_state(
            rho_mobile_m2=base.rho_mobile,
            rho_dipole_m2=base.rho_dipole,
            temperature_K=float(base.temperature),
        )
        self._validate_state(state)
        return state

    def _validate_state(self, state: LocalState92) -> None:
        if not isinstance(state, LocalState92):
            raise TypeError("state_n must implement the HCP_CP_LOCAL_STATE_92_V1 contract")
        state.validate(
            twin_fraction_limit=self.base_model.parameters.twin_max_total_fraction,
            determinant_tolerance=self.base_model.parameters.determinant_tolerance,
            tolerance=1.0e-10,
        )

    def _to_base_state(
        self,
        state_n: LocalState92,
        Fp: Array,
        rho_mobile: Array,
        rho_dipole: Array,
        accumulated_slip: Array,
        twin_fraction: Array,
        temperature: float,
    ) -> Any:
        prototype = self.base_model.initial_state()
        return replace(
            prototype,
            Fp=np.asarray(Fp).copy(),
            rho_mobile=np.asarray(rho_mobile).copy(),
            rho_dipole=np.asarray(rho_dipole).copy(),
            accumulated_slip=np.asarray(accumulated_slip).copy(),
            twin_fraction=np.asarray(twin_fraction).copy(),
            temperature=float(temperature),
            plastic_work_density=float(state_n.cp_work_density_J_m3),
            heat_density=float(state_n.generated_heat_density_J_m3),
            stored_energy_density=float(state_n.stored_energy_density_J_m3),
            time=float(state_n.time_s),
        )

    def _decode(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> _LocalFields:
        layout = self.layout
        u = _finite_array(coordinates, (layout.size,), "local coordinates")
        dgamma = layout.slip_scale * u[layout.slip]
        log_mobile = u[layout.log_rho_mobile]
        log_dipole = u[layout.log_rho_dipole]
        if np.any(np.abs(log_mobile) >= 80.0) or np.any(np.abs(log_dipole) >= 80.0):
            raise FloatingPointError(
                "log-density coordinate reached the guarded exponential boundary"
            )
        rho_mobile = state_n.rho_mobile_m2 * np.exp(log_mobile)
        rho_dipole = state_n.rho_dipole_m2 * np.exp(log_dipole)
        twin_increment = layout.twin_scale * u[layout.twin]
        signed_slip = state_n.gamma_signed + dgamma
        accumulated_slip = state_n.Gamma_absolute + np.abs(dgamma)
        twin_fraction = state_n.twin_fraction + twin_increment

        twin_shear = self._twin_shear()
        plastic_increment = np.einsum(
            "a,aij->ij", dgamma, self.base_model.systems.slip_schmid
        )
        plastic_increment += np.einsum(
            "a,a,aij->ij",
            twin_shear,
            twin_increment,
            self.base_model.systems.twin_schmid,
        )
        Fp = _matrix_exponential_3x3(plastic_increment) @ state_n.Fp
        candidate = self._to_base_state(
            state_n,
            Fp,
            rho_mobile,
            rho_dipole,
            accumulated_slip,
            twin_fraction,
            endpoint.temperature,
        )
        mechanical = self.base_model.evaluate(endpoint.F, candidate)
        tau = np.asarray(mechanical.resolved_slip)
        penalty = self.parameters.H_chi * (endpoint.zeta - signed_slip)
        effective_tau = tau + penalty
        slip_rate, resistance, mean_free_path = self.base_model.slip_rates_from_resolved(
            effective_tau, candidate
        )

        raw_twin_rate = self.base_model.twin_rates_from_resolved(
            np.asarray(mechanical.resolved_twin), candidate
        )
        available = max(
            self.base_model.parameters.twin_max_total_fraction
            - float(np.sum(state_n.twin_fraction)),
            0.0,
        )
        total_rate = float(np.sum(raw_twin_rate))
        if total_rate > 0.0 and available > 0.0:
            total_increment = available * (1.0 - np.exp(-total_rate * dt / available))
            twin_target = total_increment * raw_twin_rate / total_rate
        else:
            twin_target = np.zeros(self.n_twin)
        twin_rate = twin_increment / dt

        mobile_target, dipole_target = self._density_targets(
            state_n,
            slip_rate,
            effective_tau,
            mean_free_path,
            rho_mobile,
            endpoint.temperature,
            dt,
        )
        Lp = np.einsum(
            "a,aij->ij", dgamma / dt, self.base_model.systems.slip_schmid
        )
        Lp += np.einsum(
            "a,a,aij->ij",
            twin_shear,
            twin_rate,
            self.base_model.systems.twin_schmid,
        )
        return _LocalFields(
            dgamma=dgamma,
            rho_mobile=rho_mobile,
            rho_dipole=rho_dipole,
            twin_increment=twin_increment,
            signed_slip=signed_slip,
            accumulated_slip=accumulated_slip,
            twin_fraction=twin_fraction,
            Fp=Fp,
            mechanical=mechanical,
            effective_tau=effective_tau,
            penalty_microstress=penalty,
            slip_rate=np.asarray(slip_rate),
            twin_increment_target=twin_target,
            twin_rate=twin_rate,
            slip_resistance=np.asarray(resistance),
            mean_free_path=np.asarray(mean_free_path),
            rho_mobile_target=mobile_target,
            rho_dipole_target=dipole_target,
            Lp=Lp,
        )

    def _branch_audit(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        fields: _LocalFields,
        dt: float,
    ) -> LocalBranchAudit:
        """Return every piecewise branch used by the local constitutive map."""

        p = self.base_model.parameters
        labels: list[str] = []
        categories: list[int] = []
        distances: list[float] = []
        scales: list[float] = []

        def add(label: str, category: int, distance: float, scale: float) -> None:
            if not np.isfinite(distance) or not np.isfinite(scale) or scale <= 0.0:
                raise FloatingPointError(f"invalid branch audit value for {label}")
            labels.append(label)
            categories.append(int(category))
            distances.append(float(distance))
            scales.append(float(scale))

        raw_forest = p.forest_interaction @ (
            fields.rho_mobile + fields.rho_dipole
        )
        effective_overstress = np.abs(fields.effective_tau) - fields.slip_resistance
        active_overstress = np.maximum(effective_overstress, 0.0)
        normalized = np.minimum(active_overstress / p.tau_cut, 1.0)
        barrier = np.power(
            np.maximum(1.0 - np.power(normalized, p.p), 0.0), p.q
        )
        candidate_temperature = float(
            state_n.temperature_K
            if not self.base_model.switches.thermal_softening
            else endpoint.temperature
        )
        raw_exponent = -p.activation_energy * barrier / (K_B * candidate_temperature)

        d_check = p.dipole_min_distance_burgers * p.burgers
        tau_abs = np.abs(fields.effective_tau)
        raw_d_hat = np.empty_like(tau_abs)
        nonzero = tau_abs > 0.0
        raw_d_hat[nonzero] = (
            3.0
            * p.reference_shear_modulus
            * p.burgers[nonzero]
            / (16.0 * np.pi * tau_abs[nonzero])
        )
        raw_d_hat[~nonzero] = fields.mean_free_path[~nonzero] + np.maximum(
            fields.mean_free_path[~nonzero], d_check[~nonzero]
        )

        for index in range(self.n_slip):
            tag = f"slip{index + 1:02d}"
            slip_scale = max(
                abs(float(fields.effective_tau[index])),
                abs(float(fields.slip_resistance[index])),
                1.0,
            )
            yield_distance = float(effective_overstress[index])
            add(f"{tag}.yield", int(yield_distance > 0.0), yield_distance, slip_scale)
            if yield_distance > 0.0:
                direction_distance = float(fields.effective_tau[index])
                direction_category = 1 if direction_distance > 0.0 else -1
                direction_scale = slip_scale
            else:
                # Newton noise in an inactive dgamma is not a constitutive
                # direction branch.  Its stable active set is certified by
                # the negative yield margin instead.
                direction_distance = yield_distance
                direction_category = 0
                direction_scale = slip_scale
            add(
                f"{tag}.rate_direction",
                direction_category,
                direction_distance,
                direction_scale,
            )
            cap_distance = float(active_overstress[index] - p.tau_cut[index])
            add(
                f"{tag}.barrier_cap",
                int(cap_distance >= 0.0),
                cap_distance,
                max(float(p.tau_cut[index]), 1.0),
            )
            exponent_distance = float(raw_exponent[index] + 700.0)
            add(
                f"{tag}.arrhenius_floor",
                int(exponent_distance <= 0.0),
                exponent_distance,
                700.0,
            )
            forest_distance = float(raw_forest[index] - p.density_floor)
            add(
                f"{tag}.forest_floor",
                int(forest_distance <= 0.0),
                forest_distance,
                max(abs(float(raw_forest[index])), float(p.density_floor), 1.0),
            )
            lower_distance = float(raw_d_hat[index] - d_check[index])
            upper_distance = float(fields.mean_free_path[index] - raw_d_hat[index])
            length_scale = max(
                float(fields.mean_free_path[index]), float(d_check[index]), 1.0e-30
            )
            add(
                f"{tag}.dhat_lower",
                int(lower_distance <= 0.0),
                lower_distance,
                length_scale,
            )
            add(
                f"{tag}.dhat_upper",
                int(upper_distance <= 0.0),
                upper_distance,
                length_scale,
            )
            slip_power = float(fields.effective_tau[index] * fields.dgamma[index] / dt)
            if yield_distance > 0.0:
                power_category = int(slip_power > 0.0)
                power_distance = slip_power
                power_scale = max(abs(slip_power), slip_scale / dt * self.layout.slip_scale, 1.0)
            else:
                power_category = 0
                power_distance = yield_distance
                power_scale = slip_scale
            add(
                f"{tag}.dissipation",
                power_category,
                power_distance,
                power_scale,
            )

        twin_total = float(np.sum(fields.twin_fraction))
        twin_threshold = p.twin_crss + p.twin_latent_hardening * twin_total
        total_margin = float(p.twin_max_total_fraction - twin_total)
        committed_available = float(
            p.twin_max_total_fraction - np.sum(state_n.twin_fraction)
        )
        for index in range(self.n_twin):
            tag = f"twin{index + 1:02d}"
            overstress = float(fields.mechanical.resolved_twin[index] - twin_threshold)
            add(
                f"{tag}.overstress",
                int(overstress > 0.0),
                overstress,
                max(abs(float(twin_threshold)), 1.0),
            )
            variant_margin = float(
                p.twin_max_total_fraction - fields.twin_fraction[index]
            )
            add(
                f"{tag}.variant_available",
                int(variant_margin > 0.0),
                variant_margin,
                float(p.twin_max_total_fraction),
            )
            twin_power = float(
                fields.mechanical.resolved_twin[index]
                * self._twin_shear()[index]
                * fields.twin_increment[index]
                / dt
            )
            if (
                overstress > 0.0
                and variant_margin > 0.0
                and total_margin > 0.0
                and committed_available > 0.0
            ):
                twin_power_category = int(twin_power > 0.0)
                twin_power_distance = twin_power
                twin_power_scale = max(
                    abs(twin_power),
                    abs(float(twin_threshold)) * self.layout.twin_scale / dt,
                    1.0,
                )
            else:
                twin_power_category = 0
                twin_power_distance = overstress
                twin_power_scale = max(abs(float(twin_threshold)), 1.0)
            add(
                f"{tag}.dissipation",
                twin_power_category,
                twin_power_distance,
                twin_power_scale,
            )
        add(
            "twin.total_available",
            int(total_margin > 0.0),
            total_margin,
            float(p.twin_max_total_fraction),
        )
        add(
            "twin.committed_available",
            int(committed_available > 0.0),
            committed_available,
            float(p.twin_max_total_fraction),
        )

        slip_increment_margin = float(
            p.maximum_slip_increment
            - np.max(np.abs(fields.dgamma), initial=0.0)
        )
        twin_increment_margin = float(
            p.maximum_twin_increment
            - np.max(np.abs(fields.twin_increment), initial=0.0)
        )
        add(
            "increment.slip_admissible",
            int(slip_increment_margin >= 0.0),
            slip_increment_margin,
            float(p.maximum_slip_increment),
        )
        add(
            "increment.twin_admissible",
            int(twin_increment_margin >= 0.0),
            twin_increment_margin,
            float(p.maximum_twin_increment),
        )

        log_coordinates = np.concatenate(
            (
                coordinates[self.layout.log_rho_mobile],
                coordinates[self.layout.log_rho_dipole],
            )
        )
        for index, value in enumerate(log_coordinates):
            add(
                f"logrho{index + 1:02d}.lower_guard",
                int(value <= -80.0),
                float(value + 80.0),
                80.0,
            )
            add(
                f"logrho{index + 1:02d}.upper_guard",
                int(value >= 80.0),
                float(80.0 - value),
                80.0,
            )

        for name in (
            "slip",
            "twinning",
            "multiplication",
            "dipole_formation",
            "recovery_glide",
            "recovery_climb",
            "thermal_softening",
            "adiabatic_heating",
        ):
            enabled = int(bool(getattr(self.base_model.switches, name)))
            add(f"switch.{name}", enabled, 1.0 if enabled else -1.0, 1.0)

        return LocalBranchAudit(
            labels=tuple(labels),
            categories=tuple(categories),
            signed_distances=np.asarray(distances),
            distance_scales=np.asarray(scales),
        )

    def _twin_shear(self) -> Array:
        p = self.base_model.parameters
        return np.asarray(
            self.base_model.systems.twin_shear
            if p.twin_shear_override <= 0.0
            else np.full(self.n_twin, p.twin_shear_override)
        )

    def _density_targets(
        self,
        state_n: LocalState92,
        slip_rate: Array,
        effective_tau: Array,
        mean_free_path: Array,
        rho_mobile_candidate: Array,
        temperature: float,
        dt: float,
    ) -> tuple[Array, Array]:
        p = self.base_model.parameters
        switches = self.base_model.switches
        absolute_rate = np.abs(slip_rate)
        kinetic_temperature = temperature if switches.thermal_softening else p.T_ref
        source = (
            absolute_rate / (p.burgers * mean_free_path)
            if switches.multiplication
            else np.zeros_like(absolute_rate)
        )
        d_check = p.dipole_min_distance_burgers * p.burgers
        tau_abs = np.abs(effective_tau)
        raw_d_hat = np.full_like(tau_abs, np.inf)
        active = tau_abs > 0.0
        raw_d_hat[active] = (
            3.0
            * p.reference_shear_modulus
            * p.burgers[active]
            / (16.0 * np.pi * tau_abs[active])
        )
        d_hat = np.minimum(np.maximum(raw_d_hat, d_check), mean_free_path)
        formation = (
            2.0 * np.maximum(d_hat - d_check, 0.0) / p.burgers * absolute_rate
            if switches.dipole_formation
            else np.zeros_like(absolute_rate)
        )
        glide = (
            2.0 * d_check / p.burgers * absolute_rate
            if switches.recovery_glide
            else np.zeros_like(absolute_rate)
        )
        climb = (
            p.climb_frequency * np.exp(-p.climb_activation / (K_B * kinetic_temperature))
            if switches.recovery_climb
            else 0.0
        )
        mobile = (state_n.rho_mobile_m2 + dt * source) / (
            1.0 + dt * (formation + glide)
        )
        dipole = (state_n.rho_dipole_m2 + dt * formation * rho_mobile_candidate) / (
            1.0 + dt * (glide + climb)
        )
        if np.any(mobile <= 0.0) or np.any(dipole <= 0.0):
            raise FloatingPointError("implicit density target became non-positive")
        return mobile, dipole

    def _residual_and_fields(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> tuple[Array, _LocalFields]:
        fields = self._decode(coordinates, state_n, endpoint, dt)
        layout = self.layout
        residual = np.empty(layout.size)
        residual[layout.slip] = (
            fields.dgamma - dt * fields.slip_rate
        ) / layout.slip_scale
        residual[layout.log_rho_mobile] = np.log(
            fields.rho_mobile / fields.rho_mobile_target
        )
        residual[layout.log_rho_dipole] = np.log(
            fields.rho_dipole / fields.rho_dipole_target
        )
        residual[layout.twin] = (
            fields.twin_increment - fields.twin_increment_target
        ) / layout.twin_scale
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("implicit local residual became non-finite")
        return residual, fields

    def _central_jacobian(
        self,
        function: Callable[[Array], Array],
        point: Array,
    ) -> Array:
        point = np.asarray(point, dtype=np.float64)
        base = np.asarray(function(point), dtype=np.float64)
        jacobian = np.empty((base.size, point.size))
        h0 = self.options.finite_difference_step
        for column in range(point.size):
            step = h0 * max(1.0, abs(float(point[column])))
            plus = point.copy()
            minus = point.copy()
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * step)
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("finite-difference Jacobian became non-finite")
        return jacobian

    def _branch_checked_central_jacobian(
        self,
        evaluator: Callable[[Array], tuple[Array, LocalBranchAudit]],
        point: Array,
        *,
        label: str,
        maximum_halvings: int = 10,
    ) -> Array:
        """Central Jacobian that rejects every unresolved active-set crossing."""

        point = np.asarray(point, dtype=np.float64)
        base, base_audit = evaluator(point)
        base = np.asarray(base, dtype=np.float64)
        jacobian = np.empty((base.size, point.size))
        h0 = self.options.finite_difference_step
        for column in range(point.size):
            step = h0 * max(1.0, abs(float(point[column])))
            accepted = False
            for _ in range(maximum_halvings + 1):
                plus_point = point.copy()
                minus_point = point.copy()
                plus_point[column] += step
                minus_point[column] -= step
                try:
                    plus, plus_audit = evaluator(plus_point)
                    minus, minus_audit = evaluator(minus_point)
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    step *= 0.5
                    continue
                if base_audit.compatible_with(plus_audit, minus_audit):
                    jacobian[:, column] = (
                        np.asarray(plus) - np.asarray(minus)
                    ) / (2.0 * step)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                raise LocalNondifferentiableError(
                    f"{label} column {column} cannot be differentiated without "
                    "crossing or approaching a constitutive branch"
                )
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError(
                f"branch-checked {label} Jacobian became non-finite"
            )
        return jacobian

    def _residual_and_branch(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> tuple[Array, LocalBranchAudit]:
        residual, fields = self._residual_and_fields(
            coordinates, state_n, endpoint, dt
        )
        return residual, self._branch_audit(
            coordinates, state_n, endpoint, fields, dt
        )

    def _output_and_branch(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> tuple[Array, LocalBranchAudit]:
        fields = self._decode(coordinates, state_n, endpoint, dt)
        return self._output_vector(coordinates, state_n, endpoint, dt), self._branch_audit(
            coordinates, state_n, endpoint, fields, dt
        )

    def _initial_guess(
        self,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> Array:
        u = np.zeros(self.layout.size)
        _, fields = self._residual_and_fields(u, state_n, endpoint, dt)
        predicted_slip = dt * fields.slip_rate
        predicted_twin = fields.twin_increment_target
        p = self.base_model.parameters
        ratios = np.array(
            [
                np.max(np.abs(predicted_slip), initial=0.0) / p.maximum_slip_increment,
                np.max(np.abs(predicted_twin), initial=0.0) / p.maximum_twin_increment,
            ]
        )
        if np.max(ratios) > 4.0:
            factor = max(1.0e-6, self.options.cutback_safety / float(np.max(ratios)))
            raise LocalCutbackRequired(
                "explicit local predictor exceeds four times the v0.1 increment bound",
                factor,
            )
        u[self.layout.slip] = predicted_slip / self.layout.slip_scale
        u[self.layout.log_rho_mobile] = np.log(
            fields.rho_mobile_target / state_n.rho_mobile_m2
        )
        u[self.layout.log_rho_dipole] = np.log(
            fields.rho_dipole_target / state_n.rho_dipole_m2
        )
        u[self.layout.twin] = predicted_twin / self.layout.twin_scale
        return u

    def _solve(
        self,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> tuple[Array, Array, _LocalFields, int, Array]:
        u = self._initial_guess(state_n, endpoint, dt)
        last_jacobian: Array | None = None
        for iteration in range(self.options.maximum_iterations + 1):
            residual, fields = self._residual_and_fields(u, state_n, endpoint, dt)
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if norm <= self.options.residual_tolerance:
                if last_jacobian is None:
                    last_jacobian = self._central_jacobian(
                        lambda probe: self._residual_and_fields(
                            probe, state_n, endpoint, dt
                        )[0],
                        u,
                    )
                return u, residual, fields, iteration, last_jacobian
            if iteration == self.options.maximum_iterations:
                break
            jacobian = self._central_jacobian(
                lambda probe: self._residual_and_fields(probe, state_n, endpoint, dt)[0],
                u,
            )
            last_jacobian = jacobian
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -residual, rcond=1.0e-12)[0]
            alpha = 1.0
            accepted = False
            while alpha >= self.options.minimum_line_search_factor:
                candidate = u + alpha * step
                try:
                    candidate_residual, _ = self._residual_and_fields(
                        candidate, state_n, endpoint, dt
                    )
                    candidate_norm = float(
                        np.linalg.norm(candidate_residual, ord=np.inf)
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    candidate_norm = np.inf
                if candidate_norm < norm * (1.0 - 1.0e-4 * alpha):
                    u = candidate
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                raise LocalConvergenceError(
                    f"local Newton line search failed at residual {norm:.6e}"
                )
        final_norm = float(
            np.linalg.norm(
                self._residual_and_fields(u, state_n, endpoint, dt)[0], ord=np.inf
            )
        )
        raise LocalConvergenceError(
            f"local Newton did not converge in {self.options.maximum_iterations} iterations; "
            f"residual={final_norm:.6e}"
        )

    def _state_from_fields(
        self,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        fields: _LocalFields,
        dt: float,
    ) -> tuple[LocalState92, float, float, tuple[float, float, float, float]]:
        twin_shear = self._twin_shear()
        slip_rate = fields.dgamma / dt
        twin_rate = fields.twin_increment / dt
        tau = np.asarray(fields.mechanical.resolved_slip)
        tau_twin = np.asarray(fields.mechanical.resolved_twin)
        mechanical_slip_terms = tau * slip_rate
        microforce_terms = fields.penalty_microstress * slip_rate
        twin_terms = tau_twin * twin_shear * twin_rate
        slip_driving_terms = fields.effective_tau * slip_rate
        term_scale = max(
            1.0,
            float(np.max(np.abs(slip_driving_terms), initial=0.0)),
            float(np.max(np.abs(twin_terms), initial=0.0)),
        )
        if np.any(slip_driving_terms < -1.0e-12 * term_scale):
            raise FloatingPointError("a slip system produced negative CP dissipation")
        if np.any(twin_terms < -1.0e-12 * term_scale):
            raise FloatingPointError("a twin system produced negative CP dissipation")
        dcp = float(
            np.sum(slip_driving_terms) + np.sum(twin_terms)
        )
        scale = max(
            1.0,
            float(np.max(np.abs(fields.effective_tau * slip_rate), initial=0.0)),
            float(np.max(np.abs(twin_terms), initial=0.0)),
        )
        if dcp < -1.0e-11 * scale:
            raise FloatingPointError("implicit endpoint produced negative CP dissipation")
        if dcp < 0.0:
            raise FloatingPointError(
                "implicit endpoint produced a negative roundoff-scale CP dissipation"
            )
        beta = self.base_model.parameters.taylor_quinney
        qdot = beta * dcp
        mechanical_power = float(np.sum(mechanical_slip_terms))
        microforce_power = float(np.sum(microforce_terms))
        twin_power = float(np.sum(twin_terms))
        driving_residual = dcp - mechanical_power - microforce_power - twin_power
        if abs(driving_residual) > 2.0e-11 * max(abs(dcp), scale):
            raise FloatingPointError("implicit driving-power diagnostic is inconsistent")
        trial = LocalState92(
            Fp=fields.Fp,
            rho_mobile_m2=fields.rho_mobile,
            rho_dipole_m2=fields.rho_dipole,
            Gamma_absolute=fields.accumulated_slip,
            gamma_signed=fields.signed_slip,
            twin_fraction=fields.twin_fraction,
            temperature_K=float(endpoint.temperature),
            cp_work_density_J_m3=state_n.cp_work_density_J_m3 + dt * dcp,
            generated_heat_density_J_m3=(
                state_n.generated_heat_density_J_m3 + dt * qdot
            ),
            stored_energy_density_J_m3=(
                state_n.stored_energy_density_J_m3 + dt * (1.0 - beta) * dcp
            ),
            time_s=state_n.time_s + dt,
        )
        self._validate_state(trial)
        return trial, qdot, dcp, (
            mechanical_power,
            microforce_power,
            twin_power,
            driving_residual,
        )

    def _output_vector(
        self,
        coordinates: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> Array:
        fields = self._decode(coordinates, state_n, endpoint, dt)
        slip_rate = fields.dgamma / dt
        twin_rate = fields.twin_increment / dt
        twin_terms = (
            np.asarray(fields.mechanical.resolved_twin)
            * self._twin_shear()
            * twin_rate
        )
        dcp = float(
            np.sum(fields.effective_tau * slip_rate) + np.sum(twin_terms)
        )
        # Partial derivative probes do not satisfy the local residual and can
        # therefore have a tiny negative trial power.  Do not project or run
        # state admissibility checks here: the unmodified scalar is required
        # for a mathematically valid G_u finite difference.  Positivity is
        # enforced only for the converged state in _state_from_fields.
        qdot = self.base_model.parameters.taylor_quinney * dcp
        return np.concatenate(
            (
                np.asarray(fields.mechanical.first_piola).reshape(-1),
                np.array([qdot]),
                fields.penalty_microstress,
                np.array([dcp]),
            )
        )

    def _external_scale(self, endpoint: _ExternalPoint) -> Array:
        return np.concatenate(
            (
                np.ones(9),
                np.array([max(abs(endpoint.temperature), 300.0)]),
                np.full(self.n_slip, self.layout.slip_scale),
            )
        )

    def _perturb_endpoint(
        self,
        endpoint: _ExternalPoint,
        scaled_delta: Array,
        scale: Array,
    ) -> _ExternalPoint:
        physical = np.concatenate(
            (endpoint.F.reshape(-1), np.array([endpoint.temperature]), endpoint.zeta)
        ) + scale * scaled_delta
        return _ExternalPoint(
            F=physical[0:9].reshape(3, 3),
            temperature=float(physical[9]),
            zeta=physical[10:28],
        )

    def _condense(
        self,
        coordinates: Array,
        local_jacobian: Array,
        state_n: LocalState92,
        endpoint: _ExternalPoint,
        dt: float,
    ) -> CondensedLocalDerivatives:
        singular_values = np.linalg.svd(local_jacobian, compute_uv=False)
        minimum_singular_value = float(np.min(singular_values))
        condition_2 = (
            float(np.max(singular_values) / minimum_singular_value)
            if minimum_singular_value > 0.0
            else float("inf")
        )
        if (
            minimum_singular_value <= 0.0
            or not np.isfinite(condition_2)
            or condition_2 > self.options.maximum_local_jacobian_condition
        ):
            raise LocalConvergenceError(
                "local residual Jacobian is too ill-conditioned for static condensation: "
                f"cond2={condition_2:.6e}"
            )
        external_scale = self._external_scale(endpoint)
        zero = np.zeros(28)
        residual_external_scaled = self._branch_checked_central_jacobian(
            lambda delta: self._residual_and_branch(
                coordinates,
                state_n,
                self._perturb_endpoint(endpoint, delta, external_scale),
                dt,
            ),
            zero,
            label="external residual",
        )
        output_local = self._branch_checked_central_jacobian(
            lambda probe: self._output_and_branch(
                probe, state_n, endpoint, dt
            ),
            coordinates,
            label="local output",
        )
        output_external_scaled = self._branch_checked_central_jacobian(
            lambda delta: self._output_and_branch(
                coordinates,
                state_n,
                self._perturb_endpoint(endpoint, delta, external_scale),
                dt,
            ),
            zero,
            label="external output",
        )
        try:
            sensitivity_scaled = -np.linalg.solve(
                local_jacobian, residual_external_scaled
            )
        except np.linalg.LinAlgError as error:
            raise LocalConvergenceError(
                "local residual Jacobian is singular during static condensation"
            ) from error
        condensed_scaled = output_external_scaled + output_local @ sensitivity_scaled
        inverse_scale = 1.0 / external_scale
        residual_external = residual_external_scaled * inverse_scale[np.newaxis, :]
        sensitivity = sensitivity_scaled * inverse_scale[np.newaxis, :]
        output_external = output_external_scaled * inverse_scale[np.newaxis, :]
        condensed = condensed_scaled * inverse_scale[np.newaxis, :]
        schur_residual = local_jacobian @ sensitivity + residual_external
        return CondensedLocalDerivatives(
            condensed=condensed,
            local_residual_jacobian=local_jacobian,
            external_residual_jacobian=residual_external,
            local_coordinate_sensitivity=sensitivity,
            partial_output_local=output_local,
            partial_output_external=output_external,
            schur_residual=schur_residual,
            local_jacobian_condition_2=condition_2,
            local_jacobian_minimum_singular_value=minimum_singular_value,
        )

    def trial_step(
        self,
        F_n: Array,
        F_np1: Array,
        T_n: float,
        T_np1: float,
        zeta_n: Array,
        zeta_np1: Array,
        state_n: LocalState92,
        dt: float,
        *,
        compute_tangent: bool = True,
    ) -> ImplicitLocalTransaction:
        """Return a pure endpoint trial and its condensed consistent derivatives.

        ``F_n``, ``T_n``, ``zeta_n``, and ``state_n`` are held fixed in the
        returned endpoint tangent.  ``T_np1`` is an independent thermal-field
        unknown; this routine returns its heat source and does not overwrite it
        with an adiabatic temperature predictor.
        """

        F_n = _finite_array(F_n, (3, 3), "F_n")
        F_np1 = _finite_array(F_np1, (3, 3), "F_np1")
        zeta_n = _finite_array(zeta_n, (self.n_slip,), "zeta_n")
        zeta_np1 = _finite_array(zeta_np1, (self.n_slip,), "zeta_np1")
        scalars = np.array([T_n, T_np1, dt], dtype=np.float64)
        if not np.all(np.isfinite(scalars)) or T_n <= 0.0 or T_np1 <= 0.0 or dt <= 0.0:
            raise ValueError("T_n, T_np1, and dt must be finite and positive")
        if np.linalg.det(F_n) <= 0.0 or np.linalg.det(F_np1) <= 0.0:
            raise ValueError("F_n and F_np1 must have positive determinants")
        self._validate_state(state_n)
        fingerprint = _state_bytes(state_n)
        if not np.isclose(state_n.temperature_K, T_n, rtol=2.0e-13, atol=2.0e-13):
            raise ValueError("T_n does not match the committed state temperature")

        endpoint = _ExternalPoint(F=F_np1.copy(), temperature=float(T_np1), zeta=zeta_np1.copy())
        coordinates, residual, fields, iterations, local_jacobian = self._solve(
            state_n, endpoint, float(dt)
        )
        p = self.base_model.parameters
        slip_ratio = float(np.max(np.abs(fields.dgamma), initial=0.0)) / p.maximum_slip_increment
        twin_ratio = float(np.max(np.abs(fields.twin_increment), initial=0.0)) / p.maximum_twin_increment
        maximum_ratio = max(slip_ratio, twin_ratio)
        if maximum_ratio > 1.0 + 1.0e-10:
            factor = max(1.0e-6, self.options.cutback_safety / maximum_ratio)
            raise LocalCutbackRequired(
                "converged local increment exceeds a v0.1 admissibility bound",
                factor,
            )
        if np.any(fields.twin_fraction < -1.0e-10) or (
            float(np.sum(fields.twin_fraction))
            > p.twin_max_total_fraction + 1.0e-10
        ):
            raise LocalCutbackRequired(
                "converged twin fraction left its admissible interval", 0.5
            )
        trial, qdot, dcp, power_diagnostics = self._state_from_fields(
            state_n, endpoint, fields, float(dt)
        )
        derivatives = None
        if compute_tangent:
            branch_checked_local_jacobian = self._branch_checked_central_jacobian(
                lambda probe: self._residual_and_branch(
                    probe, state_n, endpoint, float(dt)
                ),
                coordinates,
                label="local residual",
            )
            derivatives = self._condense(
                coordinates,
                branch_checked_local_jacobian,
                state_n,
                endpoint,
                float(dt),
            )
        response = ImplicitLocalResponse(
            first_piola=fields.mechanical.first_piola,
            heat_source=qdot,
            penalty_microstress=fields.penalty_microstress,
            cp_dissipation_rate=dcp,
            mechanical_slip_power_rate=power_diagnostics[0],
            microforce_exchange_rate=power_diagnostics[1],
            twin_power_rate=power_diagnostics[2],
            driving_power_residual=power_diagnostics[3],
            mechanical_resolved_slip=fields.mechanical.resolved_slip,
            effective_resolved_slip=fields.effective_tau,
            resolved_twin=fields.mechanical.resolved_twin,
            slip_rate=fields.dgamma / dt,
            twin_rate=fields.twin_increment / dt,
            slip_resistance=fields.slip_resistance,
            mean_free_path=fields.mean_free_path,
            Lp=fields.Lp,
            local_residual=residual,
            local_coordinates=coordinates,
            newton_iterations=iterations,
            local_residual_norm=float(np.linalg.norm(residual, ord=np.inf)),
            derivatives=derivatives,
            branch_audit=self._branch_audit(
                coordinates, state_n, endpoint, fields, float(dt)
            ),
        )
        if _state_bytes(state_n) != fingerprint:
            raise RuntimeError("committed state was mutated while constructing a trial")
        return ImplicitLocalTransaction(
            committed_state=state_n,
            trial_state=trial,
            response=response,
            dt=float(dt),
            committed_fingerprint=fingerprint,
            trial_fingerprint=_state_bytes(trial),
        )
