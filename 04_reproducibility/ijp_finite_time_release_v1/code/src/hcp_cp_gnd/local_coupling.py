"""Transactional material-point bridge between HCP CP v0.1 and a micro-slip field.

This module is deliberately a *local* prototype.  It accepts the 18 prescribed
micromorphic slips ``zeta`` at one integration point and augments the mechanical
resolved shear stress with the energetic penalty microstress

``tau_hat = tau + H_chi * (zeta - gamma_signed)``.

The v0.1 material-point object is supplied by composition (duck typing); its
source tree is never imported or modified here.  The bridge copies only the
state-update equations that must use ``tau_hat``.  It does not provide the
coupled displacement/microfield residual or a consistent implicit UEL tangent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
K_B = 1.380649e-23


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _frozen_array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = _finite_array(value, shape, name).copy()
    result.setflags(write=False)
    return result


def _matrix_exponential_3x3(matrix: Array) -> Array:
    """Return exp(matrix) with the order-13 scaling/squaring Padé formula.

    Keeping this small implementation local avoids adding a new package
    dependency to the v0.2 reference kernel.  It is used only for the 3x3
    plastic velocity-gradient exponential.
    """

    matrix = _finite_array(matrix, (3, 3), "matrix")
    identity = np.eye(3)
    coefficients = np.array(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ]
    )
    norm_one = float(np.linalg.norm(matrix, 1))
    theta_13 = 5.371920351148152
    scaling = 0 if norm_one <= theta_13 else int(np.ceil(np.log2(norm_one / theta_13)))
    scaled = matrix / (2.0**scaling)
    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = scaled @ (
        a6 @ (coefficients[13] * a6 + coefficients[11] * a4 + coefficients[9] * a2)
        + coefficients[7] * a6
        + coefficients[5] * a4
        + coefficients[3] * a2
        + coefficients[1] * identity
    )
    v = (
        a6 @ (coefficients[12] * a6 + coefficients[10] * a4 + coefficients[8] * a2)
        + coefficients[6] * a6
        + coefficients[4] * a4
        + coefficients[2] * a2
        + coefficients[0] * identity
    )
    result = np.linalg.solve(v - u, v + u)
    for _ in range(scaling):
        result = result @ result
    return result


@dataclass(frozen=True)
class LocalCouplingParameters:
    """Locked v0.2 micromorphic parameter names and SI-unit contract.

    ``mu_ref`` and ``H_chi`` are in Pa; ``ell_N`` and ``ell_chi`` are in m.
    Only ``H_chi`` enters this zero-gradient material-point bridge.  Keeping
    all four locked fields here prevents the local bridge, Nye kernel, and UEL
    from silently adopting incompatible parameter cards.
    """

    mu_ref: float
    ell_N: float
    H_chi: float
    ell_chi: float

    def validate(self) -> None:
        values = np.array([self.mu_ref, self.ell_N, self.H_chi, self.ell_chi])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("mu_ref, ell_N, H_chi, and ell_chi must be finite and positive")


@dataclass(frozen=True)
class CoupledMaterialState:
    """Immutable state value for a material-point transaction.

    ``accumulated_slip`` is the total variation used by dislocation hardening;
    ``signed_slip`` is the signed kinematic quantity coupled to ``zeta``.
    The irreversible work partition and the recoverable penalty energy are
    intentionally stored in separate fields.
    """

    Fp: Array
    rho_mobile: Array
    rho_dipole: Array
    accumulated_slip: Array
    signed_slip: Array
    twin_fraction: Array
    zeta: Array
    temperature: float
    cp_dissipation_density: float = 0.0
    heat_density: float = 0.0
    stored_irreversible_density: float = 0.0
    mechanical_slip_work_density: float = 0.0
    microforce_exchange_density: float = 0.0
    twin_work_density: float = 0.0
    micromorphic_energy_density: float = 0.0
    time: float = 0.0

    def __post_init__(self) -> None:
        n_slip = np.asarray(self.signed_slip).size
        n_twin = np.asarray(self.twin_fraction).size
        for name, shape in (
            ("Fp", (3, 3)),
            ("rho_mobile", (n_slip,)),
            ("rho_dipole", (n_slip,)),
            ("accumulated_slip", (n_slip,)),
            ("signed_slip", (n_slip,)),
            ("twin_fraction", (n_twin,)),
            ("zeta", (n_slip,)),
        ):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), shape, name))

    def copy(self) -> "CoupledMaterialState":
        return replace(
            self,
            Fp=self.Fp.copy(),
            rho_mobile=self.rho_mobile.copy(),
            rho_dipole=self.rho_dipole.copy(),
            accumulated_slip=self.accumulated_slip.copy(),
            signed_slip=self.signed_slip.copy(),
            twin_fraction=self.twin_fraction.copy(),
            zeta=self.zeta.copy(),
        )

    def assert_physical(self, model: "MicromorphicHCPMaterialPoint", tolerance: float = 1.0e-10) -> None:
        n_slip = model.n_slip
        n_twin = model.n_twin
        for name, values, shape in (
            ("Fp", self.Fp, (3, 3)),
            ("rho_mobile", self.rho_mobile, (n_slip,)),
            ("rho_dipole", self.rho_dipole, (n_slip,)),
            ("accumulated_slip", self.accumulated_slip, (n_slip,)),
            ("signed_slip", self.signed_slip, (n_slip,)),
            ("twin_fraction", self.twin_fraction, (n_twin,)),
            ("zeta", self.zeta, (n_slip,)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise FloatingPointError(f"{name} is not a finite {shape} array")
        scalars = np.array(
            [
                self.temperature,
                self.cp_dissipation_density,
                self.heat_density,
                self.stored_irreversible_density,
                self.mechanical_slip_work_density,
                self.microforce_exchange_density,
                self.twin_work_density,
                self.micromorphic_energy_density,
                self.time,
            ]
        )
        if not np.all(np.isfinite(scalars)) or self.temperature <= 0.0:
            raise FloatingPointError("state scalar is non-finite or temperature is non-positive")
        if np.any(self.rho_mobile <= 0.0) or np.any(self.rho_dipole <= 0.0):
            raise FloatingPointError("dislocation densities must be strictly positive")
        if np.any(self.accumulated_slip < -tolerance):
            raise FloatingPointError("accumulated slip must be non-negative")
        if np.any(np.abs(self.signed_slip) > self.accumulated_slip + tolerance):
            raise FloatingPointError("signed slip exceeds its accumulated total variation")
        if np.any(self.twin_fraction < -tolerance):
            raise FloatingPointError("twin fraction must be non-negative")
        if float(np.sum(self.twin_fraction)) > model.base_model.parameters.twin_max_total_fraction + tolerance:
            raise FloatingPointError("total twin fraction exceeds its configured bound")
        determinant = float(np.linalg.det(self.Fp))
        if (
            not np.isfinite(determinant)
            or determinant <= 0.0
            or abs(determinant - 1.0) > model.base_model.parameters.determinant_tolerance
        ):
            raise FloatingPointError("plastic incompressibility is invalid")
        if min(
            self.cp_dissipation_density,
            self.heat_density,
            self.stored_irreversible_density,
            self.twin_work_density,
            self.micromorphic_energy_density,
            self.time,
        ) < -tolerance:
            raise FloatingPointError("non-negative state ledger became negative")
        scale = max(abs(self.cp_dissipation_density), 1.0)
        partition_residual = (
            self.cp_dissipation_density
            - self.heat_density
            - self.stored_irreversible_density
        )
        driving_residual = (
            self.cp_dissipation_density
            - self.mechanical_slip_work_density
            - self.microforce_exchange_density
            - self.twin_work_density
        )
        if abs(partition_residual) > tolerance * scale:
            raise FloatingPointError("irreversible heat/storage partition is inconsistent")
        if abs(driving_residual) > tolerance * scale:
            raise FloatingPointError("effective/macro/micro/twin work ledger is inconsistent")
        expected_energy = 0.5 * model.parameters.H_chi * float(
            (self.zeta - self.signed_slip) @ (self.zeta - self.signed_slip)
        )
        if not np.isclose(
            self.micromorphic_energy_density,
            expected_energy,
            rtol=tolerance,
            atol=tolerance * max(expected_energy, 1.0),
        ):
            raise FloatingPointError("recoverable micromorphic penalty energy is inconsistent")


@dataclass(frozen=True)
class LocalCouplingResponse:
    """Observation returned by one frozen-kinetics local substep."""

    first_piola: Array
    cauchy: Array
    mandel: Array
    resolved_slip: Array
    penalty_microstress: Array
    effective_resolved_slip: Array
    resolved_twin: Array
    slip_rate: Array
    twin_rate: Array
    slip_resistance: Array
    mean_free_path: Array
    Lp: Array
    mechanical_slip_power_terms: Array
    microforce_power_terms: Array
    effective_slip_power_terms: Array
    twin_power_terms: Array
    cp_dissipation_rate: float
    heat_generation_rate: float
    stored_irreversible_rate: float
    mandel_power_residual: float
    driving_power_residual: float
    recoverable_energy_before: float
    recoverable_energy_after: float
    macroscopic_frame: str = "S0_SAMPLE_REFERENCE"
    system_frame: str = "CRYSTAL_INTERMEDIATE"

    def __post_init__(self) -> None:
        n_slip = np.asarray(self.slip_rate).size
        n_twin = np.asarray(self.twin_rate).size
        for name, shape in (
            ("first_piola", (3, 3)),
            ("cauchy", (3, 3)),
            ("mandel", (3, 3)),
            ("resolved_slip", (n_slip,)),
            ("penalty_microstress", (n_slip,)),
            ("effective_resolved_slip", (n_slip,)),
            ("resolved_twin", (n_twin,)),
            ("slip_rate", (n_slip,)),
            ("twin_rate", (n_twin,)),
            ("slip_resistance", (n_slip,)),
            ("mean_free_path", (n_slip,)),
            ("Lp", (3, 3)),
            ("mechanical_slip_power_terms", (n_slip,)),
            ("microforce_power_terms", (n_slip,)),
            ("effective_slip_power_terms", (n_slip,)),
            ("twin_power_terms", (n_twin,)),
        ):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), shape, name))
        if self.macroscopic_frame != "S0_SAMPLE_REFERENCE":
            raise ValueError("macroscopic tensor frame must be S0_SAMPLE_REFERENCE")
        if self.system_frame != "CRYSTAL_INTERMEDIATE":
            raise ValueError("slip/twin system frame must be CRYSTAL_INTERMEDIATE")

    @property
    def recoverable_energy_increment(self) -> float:
        return self.recoverable_energy_after - self.recoverable_energy_before


@dataclass(frozen=True)
class LocalStepTransaction:
    """A pure trial result: commit/rollback return values and mutate nothing."""

    committed_state: CoupledMaterialState
    trial_state: CoupledMaterialState
    response: LocalCouplingResponse
    dt: float

    def commit(self) -> CoupledMaterialState:
        return self.trial_state.copy()

    def rollback(self) -> CoupledMaterialState:
        return self.committed_state.copy()


class MicromorphicHCPMaterialPoint:
    """Compose a frozen HCP CP v0.1 point with prescribed local ``zeta``."""

    def __init__(self, base_model: Any, parameters: LocalCouplingParameters) -> None:
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

    @property
    def n_slip(self) -> int:
        return int(self.base_model.systems.n_slip)

    @property
    def n_twin(self) -> int:
        return int(self.base_model.systems.n_twin)

    def initial_state(self, zeta: Array | None = None) -> CoupledMaterialState:
        base = self.base_model.initial_state()
        prescribed = (
            np.zeros(self.n_slip, dtype=np.float64)
            if zeta is None
            else _finite_array(zeta, (self.n_slip,), "zeta")
        )
        signed = np.zeros(self.n_slip, dtype=np.float64)
        energy = 0.5 * self.parameters.H_chi * float(prescribed @ prescribed)
        state = CoupledMaterialState(
            Fp=base.Fp,
            rho_mobile=base.rho_mobile,
            rho_dipole=base.rho_dipole,
            accumulated_slip=base.accumulated_slip,
            signed_slip=signed,
            twin_fraction=base.twin_fraction,
            zeta=prescribed,
            temperature=float(base.temperature),
            cp_dissipation_density=float(base.plastic_work_density),
            heat_density=float(base.heat_density),
            stored_irreversible_density=float(base.stored_energy_density),
            mechanical_slip_work_density=float(base.plastic_work_density),
            microforce_exchange_density=0.0,
            twin_work_density=0.0,
            micromorphic_energy_density=energy,
            time=float(base.time),
        )
        state.assert_physical(self)
        return state

    def _to_base_state(self, state: CoupledMaterialState) -> Any:
        prototype = self.base_model.initial_state()
        return replace(
            prototype,
            Fp=state.Fp.copy(),
            rho_mobile=state.rho_mobile.copy(),
            rho_dipole=state.rho_dipole.copy(),
            accumulated_slip=state.accumulated_slip.copy(),
            twin_fraction=state.twin_fraction.copy(),
            temperature=float(state.temperature),
            plastic_work_density=float(state.cp_dissipation_density),
            heat_density=float(state.heat_density),
            stored_energy_density=float(state.stored_irreversible_density),
            time=float(state.time),
        )

    def _density_update(
        self,
        state: CoupledMaterialState,
        slip_rate: Array,
        effective_tau: Array,
        mean_free_path: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        p = self.base_model.parameters
        switches = self.base_model.switches
        abs_rate = np.abs(slip_rate)
        temperature = state.temperature if switches.thermal_softening else p.T_ref
        source = (
            abs_rate / (p.burgers * mean_free_path)
            if switches.multiplication
            else np.zeros_like(abs_rate)
        )
        d_check = p.dipole_min_distance_burgers * p.burgers
        tau_abs = np.abs(effective_tau)
        raw_d_hat = np.full_like(tau_abs, np.inf)
        nonzero = tau_abs > 0.0
        raw_d_hat[nonzero] = (
            3.0
            * p.reference_shear_modulus
            * p.burgers[nonzero]
            / (16.0 * np.pi * tau_abs[nonzero])
        )
        d_hat = np.minimum(np.maximum(raw_d_hat, d_check), mean_free_path)
        formation = (
            2.0 * np.maximum(d_hat - d_check, 0.0) / p.burgers * abs_rate
            if switches.dipole_formation
            else np.zeros_like(abs_rate)
        )
        glide = (
            2.0 * d_check / p.burgers * abs_rate
            if switches.recovery_glide
            else np.zeros_like(abs_rate)
        )
        climb = (
            p.climb_frequency * np.exp(-p.climb_activation / (K_B * temperature))
            if switches.recovery_climb
            else 0.0
        )
        mobile = (state.rho_mobile + dt * source) / (1.0 + dt * (formation + glide))
        dipole = (state.rho_dipole + dt * formation * mobile) / (
            1.0 + dt * (glide + climb)
        )
        return mobile, dipole

    def trial_step(
        self,
        F_sample: Array,
        committed_state: CoupledMaterialState,
        dt: float,
        zeta: Array,
    ) -> LocalStepTransaction:
        """Build one frozen-rate trial transaction without mutating input state."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        F_sample = _finite_array(F_sample, (3, 3), "F_sample")
        prescribed = _finite_array(zeta, (self.n_slip,), "zeta")
        committed_state.assert_physical(self)
        committed_snapshot = committed_state.copy()
        base_state = self._to_base_state(committed_state)
        mechanical = self.base_model.evaluate(F_sample, base_state)

        tau = np.asarray(mechanical.resolved_slip, dtype=np.float64)
        penalty_microstress = self.parameters.H_chi * (
            prescribed - committed_state.signed_slip
        )
        effective_tau = tau + penalty_microstress
        slip_rate, resistance, mean_free_path = self.base_model.slip_rates_from_resolved(
            effective_tau, base_state
        )
        tau_twin = np.asarray(mechanical.resolved_twin, dtype=np.float64)
        raw_twin_rate = self.base_model.twin_rates_from_resolved(tau_twin, base_state)

        max_slip_increment = dt * float(np.max(np.abs(slip_rate), initial=0.0))
        max_twin_increment = dt * float(np.max(np.abs(raw_twin_rate), initial=0.0))
        p = self.base_model.parameters
        if max_slip_increment > p.maximum_slip_increment * (1.0 + 1.0e-12):
            raise RuntimeError("trial slip increment exceeds the v0.1 accepted bound")
        if max_twin_increment > p.maximum_twin_increment * (1.0 + 1.0e-12):
            raise RuntimeError("trial twin increment exceeds the v0.1 accepted bound")

        mobile, dipole = self._density_update(
            committed_state, slip_rate, effective_tau, mean_free_path, dt
        )
        total_twin_start = float(np.sum(committed_state.twin_fraction))
        available = p.twin_max_total_fraction - total_twin_start
        total_rate = float(np.sum(raw_twin_rate))
        if total_rate > 0.0 and available > 0.0:
            effective_frequency = total_rate / available
            total_increment = available * (1.0 - np.exp(-effective_frequency * dt))
            twin_increment = total_increment * raw_twin_rate / total_rate
        else:
            twin_increment = np.zeros_like(committed_state.twin_fraction)
        effective_twin_rate = twin_increment / dt
        twin_shear = (
            self.base_model.systems.twin_shear
            if p.twin_shear_override <= 0.0
            else np.full(self.n_twin, p.twin_shear_override)
        )

        Lp = np.einsum("a,aij->ij", slip_rate, self.base_model.systems.slip_schmid)
        Lp += np.einsum(
            "a,a,aij->ij",
            twin_shear,
            effective_twin_rate,
            self.base_model.systems.twin_schmid,
        )
        mechanical_slip_terms = tau * slip_rate
        microforce_terms = penalty_microstress * slip_rate
        effective_slip_terms = effective_tau * slip_rate
        twin_terms = tau_twin * twin_shear * effective_twin_rate
        power_scale = max(
            1.0,
            float(np.max(np.abs(effective_slip_terms), initial=0.0)),
            float(np.max(np.abs(twin_terms), initial=0.0)),
        )
        if np.min(effective_slip_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("effective slip driving force produced negative dissipation")
        if np.min(twin_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("polar twin system produced negative dissipation")
        dcp = float(np.sum(effective_slip_terms) + np.sum(twin_terms))
        if dcp < -1.0e-12 * power_scale:
            raise FloatingPointError("total crystal-plastic dissipation is negative")
        mechanical_system_power = float(np.sum(mechanical_slip_terms) + np.sum(twin_terms))
        microforce_power = float(np.sum(microforce_terms))
        mandel_power = float(np.sum(mechanical.mandel * Lp))
        mandel_residual = mandel_power - mechanical_system_power
        driving_residual = dcp - mechanical_system_power - microforce_power
        if abs(mandel_residual) > 1.0e-12 * max(abs(mandel_power), abs(mechanical_system_power), 1.0):
            raise FloatingPointError("M:Lp does not equal the mechanical system power")
        if abs(driving_residual) > 1.0e-12 * max(abs(dcp), abs(mechanical_system_power), abs(microforce_power), 1.0):
            raise FloatingPointError("effective driving-power ledger is inconsistent")

        Fp = _matrix_exponential_3x3(dt * Lp) @ committed_state.Fp
        determinant = float(np.linalg.det(Fp))
        if (
            not np.isfinite(determinant)
            or determinant <= 0.0
            or abs(determinant - 1.0) > p.determinant_tolerance
        ):
            raise FloatingPointError("plastic deformation-gradient update is invalid")

        signed_slip = committed_state.signed_slip + dt * slip_rate
        accumulated_slip = committed_state.accumulated_slip + dt * np.abs(slip_rate)
        twin_fraction = committed_state.twin_fraction + twin_increment
        recoverable_before = committed_state.micromorphic_energy_density
        mismatch_after = prescribed - signed_slip
        recoverable_after = 0.5 * self.parameters.H_chi * float(
            mismatch_after @ mismatch_after
        )
        dcp_increment = dt * dcp
        heat_increment = p.taylor_quinney * dcp_increment
        stored_increment = (1.0 - p.taylor_quinney) * dcp_increment
        deposited_heat = heat_increment if self.base_model.switches.adiabatic_heating else 0.0
        temperature = committed_state.temperature + deposited_heat / (
            p.mass_density * p.heat_capacity
        )

        trial_state = CoupledMaterialState(
            Fp=Fp,
            rho_mobile=mobile,
            rho_dipole=dipole,
            accumulated_slip=accumulated_slip,
            signed_slip=signed_slip,
            twin_fraction=twin_fraction,
            zeta=prescribed,
            temperature=float(temperature),
            cp_dissipation_density=committed_state.cp_dissipation_density + dcp_increment,
            heat_density=committed_state.heat_density + heat_increment,
            stored_irreversible_density=(
                committed_state.stored_irreversible_density + stored_increment
            ),
            mechanical_slip_work_density=(
                committed_state.mechanical_slip_work_density
                + dt * float(np.sum(mechanical_slip_terms))
            ),
            microforce_exchange_density=(
                committed_state.microforce_exchange_density + dt * microforce_power
            ),
            twin_work_density=(
                committed_state.twin_work_density + dt * float(np.sum(twin_terms))
            ),
            micromorphic_energy_density=recoverable_after,
            time=committed_state.time + dt,
        )
        trial_state.assert_physical(self)
        response = LocalCouplingResponse(
            first_piola=mechanical.first_piola,
            cauchy=mechanical.cauchy,
            mandel=mechanical.mandel,
            resolved_slip=tau,
            penalty_microstress=penalty_microstress,
            effective_resolved_slip=effective_tau,
            resolved_twin=tau_twin,
            slip_rate=slip_rate,
            twin_rate=effective_twin_rate,
            slip_resistance=resistance,
            mean_free_path=mean_free_path,
            Lp=Lp,
            mechanical_slip_power_terms=mechanical_slip_terms,
            microforce_power_terms=microforce_terms,
            effective_slip_power_terms=effective_slip_terms,
            twin_power_terms=twin_terms,
            cp_dissipation_rate=dcp,
            heat_generation_rate=p.taylor_quinney * dcp,
            stored_irreversible_rate=(1.0 - p.taylor_quinney) * dcp,
            mandel_power_residual=mandel_residual,
            driving_power_residual=driving_residual,
            recoverable_energy_before=recoverable_before,
            recoverable_energy_after=recoverable_after,
        )
        return LocalStepTransaction(
            committed_state=committed_snapshot,
            trial_state=trial_state,
            response=response,
            dt=float(dt),
        )
