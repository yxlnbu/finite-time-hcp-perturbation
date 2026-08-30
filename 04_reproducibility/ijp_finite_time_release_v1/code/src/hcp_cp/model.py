"""Independent finite-strain HCP material-point constitutive model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, polar

from .crystal import HCPSystems
from .parameters import MaterialParameters
from .state import MaterialState

Array = NDArray[np.float64]
K_B = 1.380649e-23


@dataclass(frozen=True)
class MechanismSwitches:
    slip: bool = True
    twinning: bool = True
    multiplication: bool = True
    dipole_formation: bool = True
    recovery_glide: bool = True
    recovery_climb: bool = True
    thermal_softening: bool = True
    adiabatic_heating: bool = True


@dataclass(frozen=True)
class ConstitutiveResponse:
    """Constitutive observation with an explicit mixed-frame contract.

    ``first_piola`` and ``cauchy`` are in the sample frame.  Elastic ``S``,
    Mandel stress, ``Lp``, Schmid projections and all system quantities are in
    the crystal/intermediate frame.  No caller may infer a frame from array
    shape alone.
    """

    first_piola: Array
    cauchy: Array
    second_piola_elastic: Array
    mandel: Array
    resolved_slip: Array
    resolved_twin: Array
    slip_rate: Array
    twin_rate: Array
    slip_resistance: Array
    forest_density: Array
    mean_free_path: Array
    Lp: Array
    plastic_dissipation: float  # legacy field name: total M:Lp plastic power, not beta*M:Lp
    slip_power_terms: Array
    twin_power_terms: Array
    mandel_power_residual: float
    elastic_rotation: Array


@dataclass(frozen=True)
class StepResult:
    state: MaterialState
    response: ConstitutiveResponse
    substeps: int
    energy_balance_relative_error: float
    work_partition_relative_error: float
    maximum_accepted_slip_increment: float
    maximum_accepted_twin_increment: float


def orientation_from_bunge(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> Array:
    """Return the active crystal-to-sample rotation for Bunge angles.

    The conventional Bunge matrix is passive (sample components to crystal
    components).  Its transpose is returned here because the rest of this API
    stores ``Q_cs`` and transforms ``F_c = Q_cs.T @ F_s @ Q_cs``.  Stating and
    testing this convention prevents silent inversion of nonzero EBSD angles.
    """

    phi1, Phi, phi2 = np.deg2rad([phi1_deg, Phi_deg, phi2_deg])
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    passive_sample_to_crystal = np.array(
        [
            [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
            [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
            [s1 * s, -c1 * s, c],
        ]
    )
    return passive_sample_to_crystal.T


def _strain_to_voigt(tensor: Array) -> Array:
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            2.0 * tensor[1, 2],
            2.0 * tensor[0, 2],
            2.0 * tensor[0, 1],
        ]
    )


def _stress_from_voigt(vector: Array) -> Array:
    return np.array(
        [
            [vector[0], vector[5], vector[4]],
            [vector[5], vector[1], vector[3]],
            [vector[4], vector[3], vector[2]],
        ]
    )


class HCPMaterialPoint:
    """Finite-strain HCP material point in an explicit crystal reference frame."""

    def __init__(
        self,
        systems: HCPSystems,
        parameters: MaterialParameters,
        orientation_crystal_to_sample: Array | None = None,
        switches: MechanismSwitches | None = None,
    ) -> None:
        self.systems = systems
        self.parameters = parameters
        self.orientation = (
            np.eye(3)
            if orientation_crystal_to_sample is None
            else np.asarray(orientation_crystal_to_sample, dtype=float)
        )
        if not np.allclose(self.orientation.T @ self.orientation, np.eye(3), atol=2.0e-12):
            raise ValueError("orientation matrix is not orthogonal")
        if np.linalg.det(self.orientation) < 0.0:
            raise ValueError("orientation matrix is improper")
        if not np.isclose(
            self.systems.c_over_a,
            self.parameters.c_over_a,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("slip/twin geometry c/a does not match the material card")
        if not np.allclose(
            self.systems.slip_burgers,
            self.parameters.burgers,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("slip-system Burgers vectors do not match the material card")
        self.switches = switches or MechanismSwitches()

    def initial_state(self) -> MaterialState:
        return MaterialState.initial(self.parameters, self.systems.n_twin)

    def branch_signature(self, F_sample: Array, state: MaterialState) -> tuple[int, ...]:
        """Return all piecewise constitutive branches used by tangent audits."""

        response = self.evaluate(F_sample, state)
        p = self.parameters
        slip = np.where(np.abs(response.slip_rate) > 0.0, np.sign(response.slip_rate), 0.0)
        twin = (response.twin_rate > 0.0).astype(int)
        raw_forest = p.forest_interaction @ (state.rho_mobile + state.rho_dipole)
        forest_floor = (raw_forest <= p.density_floor).astype(int)
        mfp_grain = np.isclose(
            response.mean_free_path, p.grain_size, rtol=0.0, atol=1.0e-16
        ).astype(int)
        effective = np.maximum(
            np.abs(response.resolved_slip) - response.slip_resistance, 0.0
        )
        activation_cap = (effective >= p.tau_cut).astype(int)
        dcheck = p.dipole_min_distance_burgers * p.burgers
        raw = np.full_like(effective, np.inf)
        nonzero = np.abs(response.resolved_slip) > 0.0
        raw[nonzero] = (
            3.0
            * self._shear_modulus(state)
            * p.burgers[nonzero]
            / (16.0 * np.pi * np.abs(response.resolved_slip[nonzero]))
        )
        dhat = np.where(
            raw <= dcheck,
            -1,
            np.where(raw >= response.mean_free_path, 1, 0),
        )
        return tuple(
            np.concatenate(
                (
                    slip.astype(int),
                    twin,
                    forest_floor,
                    mfp_grain,
                    activation_cap,
                    dhat,
                )
            ).tolist()
        )

    def _to_crystal(self, F_sample: Array) -> Array:
        return self.orientation.T @ np.asarray(F_sample, dtype=float) @ self.orientation

    def _to_sample_tensor(self, tensor_crystal: Array) -> Array:
        return self.orientation @ tensor_crystal @ self.orientation.T

    def stress_response(self, F_sample: Array, state: MaterialState) -> tuple[Array, ...]:
        F = self._to_crystal(F_sample)
        determinant = float(np.linalg.det(F))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise ValueError("deformation gradient must have a positive finite determinant")
        inv_Fp = np.linalg.inv(state.Fp)
        Fe = F @ inv_Fp
        Ce = Fe.T @ Fe
        Ee = 0.5 * (Ce - np.eye(3))
        # v0.1 keeps the elastic free energy isothermal at T_ref. Temperature
        # acts through mobility and recovery, avoiding
        # an unaccounted thermoelastic energy term in the heat balance.
        stiffness = self.parameters.elastic_matrix()
        S = _stress_from_voigt(stiffness @ _strain_to_voigt(Ee))
        P = Fe @ S @ inv_Fp.T
        cauchy = Fe @ S @ Fe.T / determinant
        mandel = Ce @ S
        rotation, _ = polar(Fe)
        return F, Fe, S, P, cauchy, mandel, rotation

    def elastic_energy_density(self, F_sample: Array, state: MaterialState) -> float:
        """Reference-volume elastic strain energy for the v0.1 St-Venant law."""

        F = self._to_crystal(F_sample)
        Fe = F @ np.linalg.inv(state.Fp)
        Ee = 0.5 * (Fe.T @ Fe - np.eye(3))
        stiffness = self.parameters.elastic_matrix()
        strain = _strain_to_voigt(Ee)
        return float(0.5 * strain @ stiffness @ strain)

    def _thermal_temperature(self, state: MaterialState) -> float:
        return state.temperature if self.switches.thermal_softening else self.parameters.T_ref

    def _shear_modulus(self, state: MaterialState) -> float:
        # This is an explicit, fixed T_ref parameter used only by the Taylor
        # hardening and dipole-distance laws.  It is not inferred with the
        # cubic-crystal Voigt expression, which is not valid for a general HCP
        # stiffness.  Temperature feedback remains confined to kinetics and
        # recovery so the v0.1 energy ledger stays closed.
        del state
        return self.parameters.reference_shear_modulus

    def slip_rates_from_resolved(
        self, resolved_shear: Array, state: MaterialState
    ) -> tuple[Array, Array, Array]:
        """Evaluate the 18 independent slip kinetics for a supplied tau vector.

        This narrow, side-effect-free entry point is intentional: the
        crystallographic projection and the kinetic activation can be tested
        independently, so a duplicated or inactive system cannot pass merely
        because a full stress state excites another system.
        """

        p = self.parameters
        tau = np.asarray(resolved_shear, dtype=float)
        if tau.shape != (self.systems.n_slip,) or not np.all(np.isfinite(tau)):
            raise ValueError("resolved slip stress must be a finite 18-vector")
        forest_density = p.forest_interaction @ (state.rho_mobile + state.rho_dipole)
        forest_density = np.maximum(forest_density, p.density_floor)
        twin_hardening = p.twin_latent_hardening * float(np.sum(state.twin_fraction))
        resistance = (
            p.tau0
            + p.taylor_coefficient
            * self._shear_modulus(state)
            * p.burgers
            * np.sqrt(forest_density)
            + twin_hardening
        )
        mean_free_path = 1.0 / (
            1.0 / p.grain_size
            + np.sqrt(forest_density) / p.mean_free_path_coefficient
        )
        if not self.switches.slip:
            return np.zeros_like(tau), resistance, mean_free_path

        effective = np.maximum(np.abs(tau) - resistance, 0.0)
        normalized = np.minimum(effective / p.tau_cut, 1.0)
        barrier_shape = np.power(np.maximum(1.0 - np.power(normalized, p.p), 0.0), p.q)
        temperature = self._thermal_temperature(state)
        exponent = -p.activation_energy * barrier_shape / (K_B * temperature)
        prefactor = state.rho_mobile * p.burgers * p.reference_velocity
        rates = prefactor * np.exp(np.clip(exponent, -700.0, 0.0)) * np.sign(tau)
        rates[effective <= 0.0] = 0.0
        return rates, resistance, mean_free_path

    def twin_rates_from_resolved(self, resolved_shear: Array, state: MaterialState) -> Array:
        """Evaluate the six polar extension-twin kinetics for supplied stresses."""

        p = self.parameters
        tau = np.asarray(resolved_shear, dtype=float)
        if tau.shape != (self.systems.n_twin,) or not np.all(np.isfinite(tau)):
            raise ValueError("resolved twin stress must be a finite 6-vector")
        if not self.switches.twinning:
            return np.zeros_like(tau)
        total = float(np.sum(state.twin_fraction))
        remaining = max(1.0 - total / p.twin_max_total_fraction, 0.0)
        threshold = p.twin_crss + p.twin_latent_hardening * total
        overstress = np.maximum(tau - threshold, 0.0)
        drive = np.power(overstress / p.twin_stress_scale, p.twin_rate_exponent)
        rate = p.twin_reference_rate * np.tanh(drive) * remaining
        rate[state.twin_fraction >= p.twin_max_total_fraction] = 0.0
        return rate

    def _slip_kinetics(
        self, mandel: Array, state: MaterialState
    ) -> tuple[Array, Array, Array, Array, Array]:
        p = self.parameters
        tau = np.einsum("ai,ij,aj->a", self.systems.slip_directions, mandel, self.systems.slip_normals)
        rates, resistance, mean_free_path = self.slip_rates_from_resolved(tau, state)
        forest_density = p.forest_interaction @ (state.rho_mobile + state.rho_dipole)
        forest_density = np.maximum(forest_density, p.density_floor)
        return tau, rates, resistance, mean_free_path, forest_density

    def _twin_kinetics(self, mandel: Array, state: MaterialState) -> tuple[Array, Array]:
        p = self.parameters
        tau = np.einsum("ai,ij,aj->a", self.systems.twin_directions, mandel, self.systems.twin_normals)
        return tau, self.twin_rates_from_resolved(tau, state)

    def evaluate(self, F_sample: Array, state: MaterialState) -> ConstitutiveResponse:
        _, _, S, P, cauchy, mandel, rotation = self.stress_response(F_sample, state)
        tau_slip, slip_rate, resistance, mean_free_path, forest_density = self._slip_kinetics(
            mandel, state
        )
        tau_twin, twin_rate = self._twin_kinetics(mandel, state)
        twin_shear = (
            self.systems.twin_shear
            if self.parameters.twin_shear_override <= 0.0
            else np.full(self.systems.n_twin, self.parameters.twin_shear_override)
        )
        Lp = np.einsum("a,aij->ij", slip_rate, self.systems.slip_schmid)
        Lp += np.einsum("a,a,aij->ij", twin_shear, twin_rate, self.systems.twin_schmid)
        slip_power_terms = tau_slip * slip_rate
        twin_power_terms = tau_twin * twin_shear * twin_rate
        power_scale = max(
            1.0,
            float(np.max(np.abs(slip_power_terms), initial=0.0)),
            float(np.max(np.abs(twin_power_terms), initial=0.0)),
        )
        if np.min(slip_power_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("a slip system produced negative plastic power")
        if np.min(twin_power_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("a polar twin system produced negative plastic power")
        dissipation = float(np.sum(slip_power_terms) + np.sum(twin_power_terms))
        mandel_power = float(np.sum(mandel * Lp))
        power_residual = mandel_power - dissipation
        if abs(power_residual) > 1.0e-12 * max(abs(mandel_power), abs(dissipation), 1.0):
            raise FloatingPointError("M:Lp does not equal the resolved-system plastic power")
        return ConstitutiveResponse(
            first_piola=self._to_sample_tensor(P),
            cauchy=self._to_sample_tensor(cauchy),
            second_piola_elastic=S,
            mandel=mandel,
            resolved_slip=tau_slip,
            resolved_twin=tau_twin,
            slip_rate=slip_rate,
            twin_rate=twin_rate,
            slip_resistance=resistance,
            forest_density=forest_density,
            mean_free_path=mean_free_path,
            Lp=Lp,
            plastic_dissipation=dissipation,
            slip_power_terms=slip_power_terms,
            twin_power_terms=twin_power_terms,
            mandel_power_residual=power_residual,
            elastic_rotation=self.orientation @ rotation @ self.orientation.T,
        )

    def _density_update(
        self, state: MaterialState, response: ConstitutiveResponse, dt: float
    ) -> tuple[Array, Array]:
        p = self.parameters
        abs_rate = np.abs(response.slip_rate)
        temperature = self._thermal_temperature(state)
        source = (
            abs_rate / (p.burgers * response.mean_free_path)
            if self.switches.multiplication
            else np.zeros_like(abs_rate)
        )
        d_check = p.dipole_min_distance_burgers * p.burgers
        tau_abs = np.abs(response.resolved_slip)
        raw_d_hat = np.full_like(tau_abs, np.inf)
        nonzero = tau_abs > 0.0
        raw_d_hat[nonzero] = (
            3.0
            * self._shear_modulus(state)
            * p.burgers[nonzero]
            / (16.0 * np.pi * tau_abs[nonzero])
        )
        # This is a constitutive branch, not post-update state clipping: the
        # capture distance is defined between the core distance and Lambda.
        d_hat = np.minimum(np.maximum(raw_d_hat, d_check), response.mean_free_path)
        formation = (
            2.0 * np.maximum(d_hat - d_check, 0.0) / p.burgers * abs_rate
            if self.switches.dipole_formation
            else np.zeros_like(abs_rate)
        )
        glide = (
            2.0 * d_check / p.burgers * abs_rate
            if self.switches.recovery_glide
            else np.zeros_like(abs_rate)
        )
        climb = (
            p.climb_frequency * np.exp(-p.climb_activation / (K_B * temperature))
            if self.switches.recovery_climb
            else 0.0
        )

        # Backward-Euler production--destruction update.  The formation term
        # transfers mobile density into the dipole population and every
        # denominator is positive, so admissibility is preserved by design.
        mobile = (state.rho_mobile + dt * source) / (1.0 + dt * (formation + glide))
        dipole = (state.rho_dipole + dt * formation * mobile) / (
            1.0 + dt * (glide + climb)
        )
        return mobile, dipole

    def _advance_substep(
        self, F_sample: Array, state: MaterialState, dt: float
    ) -> tuple[MaterialState, ConstitutiveResponse]:
        response = self.evaluate(F_sample, state)
        mobile, dipole = self._density_update(state, response, dt)

        total_twin_start = float(np.sum(state.twin_fraction))
        available = self.parameters.twin_max_total_fraction - total_twin_start
        total_rate = float(np.sum(response.twin_rate))
        if total_rate > 0.0 and available > 0.0:
            # Exact available-fraction update for frozen base rates.
            effective_frequency = total_rate / available
            total_increment = available * (1.0 - np.exp(-effective_frequency * dt))
            twin_increment = total_increment * response.twin_rate / total_rate
        else:
            twin_increment = np.zeros_like(state.twin_fraction)
        twin_fraction = state.twin_fraction + twin_increment
        effective_twin_rate = twin_increment / dt
        twin_shear = (
            self.systems.twin_shear
            if self.parameters.twin_shear_override <= 0.0
            else np.full(self.systems.n_twin, self.parameters.twin_shear_override)
        )
        Lp = np.einsum("a,aij->ij", response.slip_rate, self.systems.slip_schmid)
        Lp += np.einsum(
            "a,a,aij->ij", twin_shear, effective_twin_rate, self.systems.twin_schmid
        )
        slip_power_terms = response.resolved_slip * response.slip_rate
        twin_power_terms = response.resolved_twin * twin_shear * effective_twin_rate
        power_scale = max(
            1.0,
            float(np.max(np.abs(slip_power_terms), initial=0.0)),
            float(np.max(np.abs(twin_power_terms), initial=0.0)),
        )
        if np.min(slip_power_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("a substep slip system produced negative plastic power")
        if np.min(twin_power_terms, initial=0.0) < -1.0e-12 * power_scale:
            raise FloatingPointError("a substep twin system produced negative plastic power")
        dissipation = float(np.sum(slip_power_terms) + np.sum(twin_power_terms))
        mandel_power = float(np.sum(response.mandel * Lp))
        power_residual = mandel_power - dissipation
        if abs(power_residual) > 1.0e-12 * max(abs(mandel_power), abs(dissipation), 1.0):
            raise FloatingPointError("substep M:Lp does not equal resolved-system power")
        response = replace(
            response,
            twin_rate=effective_twin_rate,
            Lp=Lp,
            plastic_dissipation=dissipation,
            slip_power_terms=slip_power_terms,
            twin_power_terms=twin_power_terms,
            mandel_power_residual=power_residual,
        )

        increment = expm(dt * response.Lp)
        Fp = increment @ state.Fp
        determinant = float(np.linalg.det(Fp))
        if determinant <= 0.0 or not np.isfinite(determinant):
            raise FloatingPointError("plastic deformation gradient became singular")
        # Every slip/twin generator is traceless; exp(dt*Lp) therefore has
        # determinant one.  Do not conceal a bad generator or integrator with
        # post-update volume normalization.
        if abs(determinant - 1.0) > self.parameters.determinant_tolerance:
            raise FloatingPointError(
                f"plastic incompressibility failed before any correction: det(Fp)={determinant}"
            )

        plastic_work = state.plastic_work_density + dt * response.plastic_dissipation
        generated_heat_increment = (
            dt * self.parameters.taylor_quinney * response.plastic_dissipation
        )
        stored_increment = (
            dt * (1.0 - self.parameters.taylor_quinney) * response.plastic_dissipation
        )
        heat = state.heat_density + generated_heat_increment
        stored = state.stored_energy_density + stored_increment
        deposited_heat_increment = generated_heat_increment if self.switches.adiabatic_heating else 0.0
        temperature = state.temperature + deposited_heat_increment / (
            self.parameters.mass_density * self.parameters.heat_capacity
        )
        updated = MaterialState(
            Fp=Fp,
            rho_mobile=mobile,
            rho_dipole=dipole,
            accumulated_slip=state.accumulated_slip + dt * np.abs(response.slip_rate),
            twin_fraction=twin_fraction,
            temperature=float(temperature),
            plastic_work_density=float(plastic_work),
            heat_density=float(heat),
            stored_energy_density=float(stored),
            time=state.time + dt,
        )
        updated.assert_physical(self.parameters)
        return updated, response

    def _finish_step(
        self,
        F_end_sample: Array,
        current: MaterialState,
        substeps: int,
        start_temperature: float,
        start_heat: float,
        start_stored: float,
        start_work: float,
        maximum_accepted_slip_increment: float,
        maximum_accepted_twin_increment: float,
    ) -> StepResult:
        response = self.evaluate(F_end_sample, current)
        deposited_heat = (
            current.heat_density - start_heat if self.switches.adiabatic_heating else 0.0
        )
        heat_residual = (
            self.parameters.mass_density
            * self.parameters.heat_capacity
            * (current.temperature - start_temperature)
            - deposited_heat
        )
        # A mixed scale prevents a few ulps of temperature subtraction in an
        # almost elastic step from being reported as an O(1) relative error.
        heat_scale = max(
            abs(deposited_heat),
            self.parameters.mass_density * self.parameters.heat_capacity * 1.0e-8,
        )
        energy_error = abs(heat_residual) / heat_scale
        delta_work = current.plastic_work_density - start_work
        partition_residual = delta_work - (current.heat_density - start_heat) - (
            current.stored_energy_density - start_stored
        )
        partition_error = abs(partition_residual) / max(abs(delta_work), 1.0)
        return StepResult(
            state=current,
            response=response,
            substeps=substeps,
            energy_balance_relative_error=float(energy_error),
            work_partition_relative_error=float(partition_error),
            maximum_accepted_slip_increment=float(maximum_accepted_slip_increment),
            maximum_accepted_twin_increment=float(maximum_accepted_twin_increment),
        )

    def advance_fixed(
        self,
        F_start_sample: Array,
        F_end_sample: Array,
        state: MaterialState,
        dt: float,
        *,
        substeps: int,
    ) -> StepResult:
        """Advance with an explicitly fixed midpoint substep count.

        This deterministic path is used by independent tangent and temporal
        convergence audits.  The production driver retains adaptive substeps.
        """

        if not np.isfinite(dt) or dt <= 0.0 or substeps <= 0:
            raise ValueError("dt and fixed substep count must be positive")
        state.assert_physical(self.parameters)
        start_temperature = state.temperature
        start_heat = state.heat_density
        start_stored = state.stored_energy_density
        start_work = state.plastic_work_density
        current = state.copy()
        sub_dt = dt / substeps
        max_slip_increment = 0.0
        max_twin_increment = 0.0
        for index in range(substeps):
            midpoint_fraction = (index + 0.5) / substeps
            F_midpoint = (
                (1.0 - midpoint_fraction) * F_start_sample
                + midpoint_fraction * F_end_sample
            )
            current, response = self._advance_substep(F_midpoint, current, sub_dt)
            max_slip_increment = max(
                max_slip_increment,
                sub_dt * float(np.max(np.abs(response.slip_rate), initial=0.0)),
            )
            max_twin_increment = max(
                max_twin_increment,
                sub_dt * float(np.max(np.abs(response.twin_rate), initial=0.0)),
            )
        return self._finish_step(
            F_end_sample,
            current,
            substeps,
            start_temperature,
            start_heat,
            start_stored,
            start_work,
            max_slip_increment,
            max_twin_increment,
        )

    def advance(
        self,
        F_start_sample: Array,
        F_end_sample: Array,
        state: MaterialState,
        dt: float,
    ) -> StepResult:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("time increment must be positive and finite")
        state.assert_physical(self.parameters)
        start_temperature = state.temperature
        start_heat = state.heat_density
        start_stored = state.stored_energy_density
        start_work = state.plastic_work_density
        current = state.copy()
        elapsed = 0.0
        substeps = 0
        max_slip_increment = 0.0
        max_twin_increment = 0.0
        while elapsed < dt * (1.0 - 2.0e-15):
            allowed = dt - elapsed
            # Rates can be zero at the substep start and enormous at its
            # midpoint.  Re-evaluate the actual midpoint and backtrack until
            # the requested slip/twin increments satisfy both configured
            # bounds.  This prevents a legal large external step from bypassing
            # adaptivity merely because the initial state was elastic.
            while True:
                midpoint_fraction = (elapsed + 0.5 * allowed) / dt
                F_midpoint = (
                    (1.0 - midpoint_fraction) * F_start_sample
                    + midpoint_fraction * F_end_sample
                )
                trial = self.evaluate(F_midpoint, current)
                slip_increment = allowed * float(
                    np.max(np.abs(trial.slip_rate), initial=0.0)
                )
                twin_increment = allowed * float(
                    np.max(np.abs(trial.twin_rate), initial=0.0)
                )
                scale = max(
                    slip_increment / self.parameters.maximum_slip_increment,
                    twin_increment / self.parameters.maximum_twin_increment,
                    1.0,
                )
                if scale <= 1.0 + 1.0e-12:
                    break
                allowed *= max(0.1, 0.8 / scale)
                if allowed <= np.spacing(max(state.time + elapsed, 1.0)):
                    raise RuntimeError("adaptive constitutive substep underflow")
            F_midpoint = (1.0 - midpoint_fraction) * F_start_sample + midpoint_fraction * F_end_sample
            current, response = self._advance_substep(F_midpoint, current, allowed)
            accepted_slip_increment = allowed * float(
                np.max(np.abs(response.slip_rate), initial=0.0)
            )
            accepted_twin_increment = allowed * float(
                np.max(np.abs(response.twin_rate), initial=0.0)
            )
            if (
                accepted_slip_increment
                > self.parameters.maximum_slip_increment * (1.0 + 1.0e-12)
                or accepted_twin_increment
                > self.parameters.maximum_twin_increment * (1.0 + 1.0e-12)
            ):
                raise RuntimeError("accepted constitutive substep exceeded its increment bound")
            max_slip_increment = max(max_slip_increment, accepted_slip_increment)
            max_twin_increment = max(max_twin_increment, accepted_twin_increment)
            elapsed += allowed
            substeps += 1
            if substeps > 2_000_000:
                raise RuntimeError("adaptive constitutive integration exceeded the substep limit")

        return self._finish_step(
            F_end_sample,
            current,
            substeps,
            start_temperature,
            start_heat,
            start_stored,
            start_work,
            max_slip_increment,
            max_twin_increment,
        )
