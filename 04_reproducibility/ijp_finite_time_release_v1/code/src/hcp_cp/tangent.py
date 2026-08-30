"""Automatic-differentiation and independent finite-difference tangent audit.

The production constitutive update is NumPy/SciPy.  This module deliberately
re-expresses one fixed midpoint substep with JAX, so the reported automatic
Jacobian and the finite-difference reference do not share a differentiation
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm
import numpy as np
from numpy.typing import NDArray

from .model import HCPMaterialPoint, K_B
from .state import MaterialState

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TangentAudit:
    automatic: Array
    finite_difference: Array
    finite_difference_by_step: dict[float, Array]
    relative_frobenius_error: float
    maximum_column_relative_error: float
    finite_difference_platform_error: float
    block_relative_errors: dict[str, float]
    block_absolute_errors: dict[str, float]
    block_finite_difference_platform_errors: dict[str, float]
    branch_signature: tuple[int, ...]
    state_hash_unchanged: bool


def _state_core_vector(first_piola: Array, state: MaterialState) -> Array:
    return np.concatenate(
        (
            np.asarray(first_piola).reshape(-1),
            np.array([state.temperature]),
            state.Fp.reshape(-1),
            state.rho_mobile,
            state.rho_dipole,
            state.accumulated_slip,
            state.twin_fraction,
        )
    )


def _state_hash(state: MaterialState) -> bytes:
    return state.flat_internal().astype("<f8", copy=False).tobytes()


def _branch_signature(model: HCPMaterialPoint, F: Array, state: MaterialState) -> tuple[int, ...]:
    return model.branch_signature(F, state)


def _jax_one_substep(model: HCPMaterialPoint, F_start: Array, state: MaterialState, dt: float):
    """Return a pure JAX function mapping F_end(flat) to P/state at n+1."""

    p = model.parameters
    sw = model.switches
    orientation = jnp.asarray(model.orientation)
    F_start_j = jnp.asarray(F_start)
    Fp0 = jnp.asarray(state.Fp)
    rho_m0 = jnp.asarray(state.rho_mobile)
    rho_d0 = jnp.asarray(state.rho_dipole)
    gamma0 = jnp.asarray(state.accumulated_slip)
    twin0 = jnp.asarray(state.twin_fraction)
    T0 = jnp.asarray(state.temperature)
    C = jnp.asarray(p.elastic_C0)
    slip_s = jnp.asarray(model.systems.slip_directions)
    slip_n = jnp.asarray(model.systems.slip_normals)
    slip_P = jnp.asarray(model.systems.slip_schmid)
    twin_s = jnp.asarray(model.systems.twin_directions)
    twin_n = jnp.asarray(model.systems.twin_normals)
    twin_P = jnp.asarray(model.systems.twin_schmid)
    twin_shear = jnp.asarray(
        model.systems.twin_shear
        if p.twin_shear_override <= 0.0
        else np.full(model.systems.n_twin, p.twin_shear_override)
    )
    burgers = jnp.asarray(p.burgers)
    tau0 = jnp.asarray(p.tau0)
    tau_cut = jnp.asarray(p.tau_cut)
    activation = jnp.asarray(p.activation_energy)
    velocity = jnp.asarray(p.reference_velocity)
    pexp = jnp.asarray(p.p)
    qexp = jnp.asarray(p.q)
    interaction = jnp.asarray(p.forest_interaction)
    shear_modulus = float(p.reference_shear_modulus)

    def stress(F_sample, Fp):
        F = orientation.T @ F_sample @ orientation
        inv_Fp = jnp.linalg.inv(Fp)
        Fe = F @ inv_Fp
        Ce = Fe.T @ Fe
        Ee = 0.5 * (Ce - jnp.eye(3))
        strain = jnp.array(
            [Ee[0, 0], Ee[1, 1], Ee[2, 2], 2.0 * Ee[1, 2], 2.0 * Ee[0, 2], 2.0 * Ee[0, 1]]
        )
        sv = C @ strain
        S = jnp.array(
            [[sv[0], sv[5], sv[4]], [sv[5], sv[1], sv[3]], [sv[4], sv[3], sv[2]]]
        )
        Pcr = Fe @ S @ inv_Fp.T
        M = Ce @ S
        return orientation @ Pcr @ orientation.T, M

    def mapping(F_end_flat):
        F_end = F_end_flat.reshape((3, 3))
        F_mid = 0.5 * (F_start_j + F_end)
        _, M = stress(F_mid, Fp0)
        tau_s = jnp.einsum("ai,ij,aj->a", slip_s, M, slip_n)
        forest = interaction @ (rho_m0 + rho_d0)
        forest = jnp.maximum(forest, p.density_floor)
        resistance = (
            tau0
            + p.taylor_coefficient * shear_modulus * burgers * jnp.sqrt(forest)
            + p.twin_latent_hardening * jnp.sum(twin0)
        )
        mfp = 1.0 / (
            1.0 / p.grain_size + jnp.sqrt(forest) / p.mean_free_path_coefficient
        )
        effective = jnp.maximum(jnp.abs(tau_s) - resistance, 0.0)
        normalized = jnp.minimum(effective / tau_cut, 1.0)
        barrier = jnp.maximum(1.0 - normalized**pexp, 0.0) ** qexp
        kinetic_T = T0 if sw.thermal_softening else p.T_ref
        slip_rate = rho_m0 * burgers * velocity * jnp.exp(
            jnp.clip(-activation * barrier / (K_B * kinetic_T), -700.0, 0.0)
        ) * jnp.sign(tau_s)
        slip_rate = jnp.where((effective > 0.0) & sw.slip, slip_rate, 0.0)

        tau_tw = jnp.einsum("ai,ij,aj->a", twin_s, M, twin_n)
        total_tw = jnp.sum(twin0)
        available = p.twin_max_total_fraction - total_tw
        remaining = jnp.maximum(1.0 - total_tw / p.twin_max_total_fraction, 0.0)
        threshold = p.twin_crss + p.twin_latent_hardening * total_tw
        drive = (jnp.maximum(tau_tw - threshold, 0.0) / p.twin_stress_scale) ** p.twin_rate_exponent
        twin_rate = p.twin_reference_rate * jnp.tanh(drive) * remaining
        twin_rate = jnp.where(sw.twinning, twin_rate, 0.0)
        total_rate = jnp.sum(twin_rate)
        safe_total = jnp.maximum(total_rate, jnp.finfo(jnp.float64).tiny)
        effective_frequency = safe_total / jnp.maximum(available, jnp.finfo(jnp.float64).tiny)
        total_increment = available * (1.0 - jnp.exp(-effective_frequency * dt))
        twin_increment = jnp.where(
            (total_rate > 0.0) & (available > 0.0),
            total_increment * twin_rate / safe_total,
            jnp.zeros_like(twin_rate),
        )
        twin_rate_eff = twin_increment / dt

        dcheck = p.dipole_min_distance_burgers * burgers
        tau_abs = jnp.abs(tau_s)
        raw_dhat = jnp.where(
            tau_abs > 0.0,
            3.0 * shear_modulus * burgers / (16.0 * jnp.pi * jnp.maximum(tau_abs, 1.0e-300)),
            jnp.inf,
        )
        dhat = jnp.minimum(jnp.maximum(raw_dhat, dcheck), mfp)
        source = jnp.where(sw.multiplication, jnp.abs(slip_rate) / (burgers * mfp), 0.0)
        formation = jnp.where(
            sw.dipole_formation,
            2.0 * jnp.maximum(dhat - dcheck, 0.0) / burgers * jnp.abs(slip_rate),
            0.0,
        )
        glide = jnp.where(
            sw.recovery_glide, 2.0 * dcheck / burgers * jnp.abs(slip_rate), 0.0
        )
        climb = jnp.where(
            sw.recovery_climb,
            p.climb_frequency * jnp.exp(-p.climb_activation / (K_B * kinetic_T)),
            0.0,
        )
        rho_m = (rho_m0 + dt * source) / (1.0 + dt * (formation + glide))
        rho_d = (rho_d0 + dt * formation * rho_m) / (1.0 + dt * (glide + climb))

        Lp = jnp.einsum("a,aij->ij", slip_rate, slip_P)
        Lp = Lp + jnp.einsum("a,a,aij->ij", twin_shear, twin_rate_eff, twin_P)
        Fp = jax_expm(dt * Lp) @ Fp0
        dissipation = jnp.dot(tau_s, slip_rate) + jnp.dot(tau_tw, twin_shear * twin_rate_eff)
        temperature = T0 + jnp.where(
            sw.adiabatic_heating,
            dt * p.taylor_quinney * dissipation / (p.mass_density * p.heat_capacity),
            0.0,
        )
        gamma = gamma0 + dt * jnp.abs(slip_rate)
        twin = twin0 + twin_increment
        P_end, _ = stress(F_end, Fp)
        return jnp.concatenate(
            (P_end.reshape(-1), jnp.atleast_1d(temperature), Fp.reshape(-1), rho_m, rho_d, gamma, twin)
        )

    return mapping


def audit_algorithmic_tangent(
    model: HCPMaterialPoint,
    F_start: Array,
    F_end: Array,
    state: MaterialState,
    dt: float,
    *,
    fd_steps: tuple[float, ...] = (1.0e-6, 3.0e-7, 1.0e-7),
) -> TangentAudit:
    """Compare a JAX algorithmic Jacobian to independent centered differences."""

    if len(fd_steps) < 2 or any(step <= 0.0 for step in fd_steps):
        raise ValueError("at least two positive finite-difference steps are required")
    before = _state_hash(state)
    mapping = _jax_one_substep(model, np.asarray(F_start), state, float(dt))
    automatic = np.asarray(jax.jacfwd(mapping)(jnp.asarray(F_end).reshape(-1)))
    base_signature = _branch_signature(model, 0.5 * (F_start + F_end), state)
    finite_by_step: dict[float, Array] = {}
    for step in fd_steps:
        columns = []
        for component in range(9):
            direction = np.zeros(9)
            direction[component] = step
            plus_F = np.asarray(F_end).reshape(-1) + direction
            minus_F = np.asarray(F_end).reshape(-1) - direction
            plus_F = plus_F.reshape(3, 3)
            minus_F = minus_F.reshape(3, 3)
            if _branch_signature(model, 0.5 * (F_start + plus_F), state) != base_signature:
                raise RuntimeError("positive finite-difference probe crossed a constitutive branch")
            if _branch_signature(model, 0.5 * (F_start + minus_F), state) != base_signature:
                raise RuntimeError("negative finite-difference probe crossed a constitutive branch")
            plus = model.advance_fixed(F_start, plus_F, state, dt, substeps=1)
            minus = model.advance_fixed(F_start, minus_F, state, dt, substeps=1)
            columns.append(
                (
                    _state_core_vector(plus.response.first_piola, plus.state)
                    - _state_core_vector(minus.response.first_piola, minus.state)
                )
                / (2.0 * step)
            )
        finite_by_step[step] = np.column_stack(columns)
    finite = finite_by_step[min(fd_steps)]
    norm = max(float(np.linalg.norm(finite)), 1.0)
    rel = float(np.linalg.norm(automatic - finite) / norm)
    column_errors = [
        float(np.linalg.norm(automatic[:, index] - finite[:, index]))
        / max(float(np.linalg.norm(finite[:, index])), 1.0)
        for index in range(9)
    ]
    ordered = sorted(fd_steps, reverse=True)
    platform = float(
        np.linalg.norm(finite_by_step[ordered[-1]] - finite_by_step[ordered[-2]])
        / max(np.linalg.norm(finite_by_step[ordered[-1]]), 1.0)
    )
    n_slip = model.systems.n_slip
    n_twin = model.systems.n_twin
    block_slices = {
        "first_piola": slice(0, 9),
        "temperature": slice(9, 10),
        "Fp": slice(10, 19),
        "rho_mobile": slice(19, 19 + n_slip),
        "rho_dipole": slice(19 + n_slip, 19 + 2 * n_slip),
        "accumulated_slip": slice(19 + 2 * n_slip, 19 + 3 * n_slip),
        "twin_fraction": slice(19 + 3 * n_slip, 19 + 3 * n_slip + n_twin),
    }
    block_errors = {}
    block_absolute = {}
    block_platform = {}
    for name, block in block_slices.items():
        absolute = float(np.linalg.norm(automatic[block] - finite[block]))
        block_absolute[name] = absolute
        block_errors[name] = absolute / max(float(np.linalg.norm(finite[block])), 1.0e-30)
        block_platform[name] = float(
            np.linalg.norm(
                finite_by_step[ordered[-1]][block]
                - finite_by_step[ordered[-2]][block]
            )
        ) / max(float(np.linalg.norm(finite_by_step[ordered[-1]][block])), 1.0e-30)
    return TangentAudit(
        automatic=automatic,
        finite_difference=finite,
        finite_difference_by_step=finite_by_step,
        relative_frobenius_error=rel,
        maximum_column_relative_error=max(column_errors),
        finite_difference_platform_error=platform,
        block_relative_errors=block_errors,
        block_absolute_errors=block_absolute,
        block_finite_difference_platform_errors=block_platform,
        branch_signature=base_signature,
        state_hash_unchanged=before == _state_hash(state),
    )
