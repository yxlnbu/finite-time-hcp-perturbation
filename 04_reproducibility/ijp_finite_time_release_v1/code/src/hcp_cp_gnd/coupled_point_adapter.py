"""Compose the implicit HCP update with the micromorphic gradient response.

The adapter is the single integration-point callback consumed by
``Hex8MonolithicAssembler``.  It does not own committed history.  A call
receives one :class:`LocalState92`, performs a pure implicit trial, evaluates
the reference-frame gradient/Nye energy at the converged signed slip, and
returns the complete

``[P(9), q, pi(18), xi(18x3)]``

response and its derivative with respect to

``[F(9), T, zeta(18), Grad(zeta)(18x3)]``.

The local ``P/q/pi`` block is the statically condensed backward-Euler
derivative from ``implicit_local``.  The higher-order stress has the exact
analytic gradient Hessian from ``micromorphic``.  Crystal slip directions and
normals are actively rotated into the sample reference frame before they are
combined with ``Grad(zeta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .implicit_local import ImplicitMicromorphicHCPMaterialPoint, LocalBranchAudit
from .gate01_energy_audit import BackwardEulerSourcesJ
from .micromorphic import MicromorphicParameters, evaluate_micromorphic_state
from .monolithic_element import (
    CondensedPointResponse,
    GRAD_ZETA_SLICE,
    N_POINT_INPUT,
    PointFieldState,
)
from .state_contract import LocalState92


GATE01_POINT_SOURCE_SCHEMA = "HCP_CP_GATE01_POINT_SOURCE_PROBE_V1"


def _same_parameter(left: float, right: float, name: str) -> None:
    if not np.isclose(left, right, rtol=2.0e-15, atol=0.0):
        raise ValueError(f"local and gradient kernels disagree on {name}")


@dataclass(frozen=True)
class Gate01PointSourceProbe:
    """Observer-only Gate-01 source rates at one converged endpoint.

    The three rates come directly from the local ``n+1`` response.  The
    cumulative STATE92 increments are carried only as independent comparison
    quantities and are never used by :meth:`backward_euler_sources`.
    """

    integration_point: int
    cp_power_W_m3: float
    heat_source_W_m3: float
    passive_storage_rate_W_m3: float
    beta: float
    dt_s: float
    committed_time_s: float
    trial_time_s: float
    delta_w_cp_state_J_m3: float
    delta_q_heat_state_J_m3: float
    delta_u_s_state_J_m3: float
    branch_audit: LocalBranchAudit
    schema: str = GATE01_POINT_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GATE01_POINT_SOURCE_SCHEMA:
            raise ValueError("wrong Gate-01 point-source schema")
        if not isinstance(self.integration_point, int) or self.integration_point < 1:
            raise ValueError("integration_point must be a positive integer")
        if not isinstance(self.branch_audit, LocalBranchAudit):
            raise TypeError("branch_audit must be LocalBranchAudit")
        names = (
            "cp_power_W_m3",
            "heat_source_W_m3",
            "passive_storage_rate_W_m3",
            "beta",
            "dt_s",
            "committed_time_s",
            "trial_time_s",
            "delta_w_cp_state_J_m3",
            "delta_q_heat_state_J_m3",
            "delta_u_s_state_J_m3",
        )
        for name in names:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        clock_scale = max(abs(self.trial_time_s), abs(self.committed_time_s), self.dt_s, 1.0)
        if abs(self.trial_time_s - self.committed_time_s - self.dt_s) > (
            128.0 * np.finfo(np.float64).eps * clock_scale
        ):
            raise ValueError("trial and committed clocks do not span dt_s")
        power_scale = max(
            abs(self.cp_power_W_m3),
            abs(self.heat_source_W_m3),
            abs(self.passive_storage_rate_W_m3),
            1.0,
        )
        if self.cp_power_W_m3 < -1.0e-12 * power_scale:
            raise ValueError("cp_power_W_m3 violates the Gate-01 non-negative branch")
        if abs(self.heat_source_W_m3 - self.beta * self.cp_power_W_m3) > (
            1.0e-10 * power_scale
        ):
            raise ValueError("heat source is inconsistent with beta*Pcp")
        if abs(
            self.passive_storage_rate_W_m3
            - (1.0 - self.beta) * self.cp_power_W_m3
        ) > 1.0e-10 * power_scale:
            raise ValueError("passive storage rate is inconsistent with (1-beta)*Pcp")

    @property
    def power_partition_residual_W_m3(self) -> float:
        return (
            self.cp_power_W_m3
            - self.heat_source_W_m3
            - self.passive_storage_rate_W_m3
        )

    @property
    def state_source_residuals_J_m3(self) -> tuple[float, float, float]:
        """STATE92 increment minus independently evaluated BE source."""

        return (
            self.delta_w_cp_state_J_m3 - self.dt_s * self.cp_power_W_m3,
            self.delta_q_heat_state_J_m3 - self.dt_s * self.heat_source_W_m3,
            self.delta_u_s_state_J_m3
            - self.dt_s * self.passive_storage_rate_W_m3,
        )

    @property
    def branch_signature(self) -> tuple[str, ...]:
        return (
            f"schema={self.schema}",
            "scheme=BACKWARD_EULER_EXPONENTIAL_FP",
            "state=HCP_CP_LOCAL_STATE_92_V1",
            f"ip={self.integration_point}",
            *self.branch_audit.category_tokens,
        )

    def backward_euler_sources(self, reference_weight_m3: float) -> BackwardEulerSourcesJ:
        """Return ``h*dV0`` source increments without reading STATE92 ledgers."""

        weight = float(reference_weight_m3)
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("reference_weight_m3 must be finite and positive")
        factor = self.dt_s * weight
        return BackwardEulerSourcesJ(
            delta_w_cp_be_J=factor * self.cp_power_W_m3,
            delta_q_gen_be_J=factor * self.heat_source_W_m3,
            delta_u_s_src_be_J=factor * self.passive_storage_rate_W_m3,
        )


@dataclass(frozen=True)
class CondensedMicromorphicPointAdapter:
    """Pure point-level bridge for the monolithic reference assembler."""

    local_model: ImplicitMicromorphicHCPMaterialPoint
    micromorphic_parameters: MicromorphicParameters

    def __post_init__(self) -> None:
        local = self.local_model.parameters
        gradient = self.micromorphic_parameters
        gradient.validate(18)
        _same_parameter(local.mu_ref, gradient.reference_shear_modulus_Pa, "mu_ref")
        _same_parameter(local.ell_N, gradient.nye_length_scale_m, "ell_N")
        _same_parameter(local.H_chi, gradient.penalty_modulus_Pa, "H_chi")
        _same_parameter(local.ell_chi, gradient.slip_gradient_length_m, "ell_chi")
        if not np.allclose(
            gradient.burgers_m,
            self.local_model.base_model.systems.slip_burgers,
            rtol=2.0e-15,
            atol=0.0,
        ):
            raise ValueError("gradient and local kernels disagree on Burgers vectors")

    @property
    def directions_sample_reference(self) -> np.ndarray:
        rotation = np.asarray(self.local_model.base_model.orientation, dtype=np.float64)
        crystal = np.asarray(self.local_model.base_model.systems.slip_directions)
        return (rotation @ crystal.T).T

    @property
    def normals_sample_reference(self) -> np.ndarray:
        rotation = np.asarray(self.local_model.base_model.orientation, dtype=np.float64)
        crystal = np.asarray(self.local_model.base_model.systems.slip_normals)
        return (rotation @ crystal.T).T

    def __call__(
        self,
        fields: PointFieldState,
        committed_state: LocalState92,
    ) -> CondensedPointResponse:
        transaction, local, micro = self._evaluate_components(
            fields, committed_state, compute_tangent=True
        )
        if local.derivatives is None:  # pragma: no cover - defensive contract
            raise RuntimeError("the monolithic callback requires condensed derivatives")

        jacobian = np.zeros((N_POINT_INPUT, N_POINT_INPUT), dtype=np.float64)
        # Rows 0:28 are [P(9), q(1), pi(18)]; columns 0:28 are [F,T,zeta].
        jacobian[:28, :28] = local.derivatives.condensed[:28, :]
        jacobian[GRAD_ZETA_SLICE, GRAD_ZETA_SLICE] = (
            micro.gradient_hessian_Pa_m2.reshape(54, 54)
        )

        branch_signature = (
            "scheme=BACKWARD_EULER_EXPONENTIAL_FP",
            "state=HCP_CP_LOCAL_STATE_92_V1",
            *local.branch_audit.category_tokens,
            "gradient_active="
            + str(int(np.linalg.norm(fields.zeta_gradient_current_m_inv) > 0.0)),
            f"local_iterations={local.newton_iterations}",
        )
        return CondensedPointResponse(
            first_piola_Pa=local.first_piola,
            heat_source_W_m3=local.q,
            penalty_microstress_Pa=local.pi,
            higher_order_stress_Pa_m=micro.higher_order_stress_Pa_m,
            algorithmic_jacobian=jacobian,
            trial_state=transaction.trial_state,
            branch_signature=branch_signature,
            branch_audit=local.branch_audit,
        )

    def _evaluate_components(
        self,
        fields: PointFieldState,
        committed_state: LocalState92,
        *,
        compute_tangent: bool,
    ):
        if not isinstance(committed_state, LocalState92):
            raise TypeError("the coupled point adapter requires LocalState92")
        transaction = self.local_model.trial_step(
            fields.deformation_gradient_previous,
            fields.deformation_gradient_current,
            fields.temperature_previous_K,
            fields.temperature_current_K,
            fields.zeta_previous,
            fields.zeta_current,
            committed_state,
            fields.dt_s,
            compute_tangent=compute_tangent,
        )
        local = transaction.response
        micro = evaluate_micromorphic_state(
            transaction.trial_state.gamma_signed,
            fields.zeta_current,
            fields.zeta_gradient_current_m_inv,
            self.directions_sample_reference,
            self.normals_sample_reference,
            self.micromorphic_parameters,
        )
        scale = max(
            float(np.linalg.norm(local.pi)),
            float(np.linalg.norm(micro.penalty_microstress_Pa)),
            1.0,
        )
        if np.linalg.norm(local.pi - micro.penalty_microstress_Pa) > 2.0e-12 * scale:
            raise RuntimeError("local and micromorphic penalty stresses are inconsistent")
        return transaction, local, micro

    def probe_response_vector(
        self,
        fields: PointFieldState,
        committed_state: LocalState92,
    ) -> tuple[np.ndarray, LocalBranchAudit]:
        """Return only ``[P,q,pi,xi]`` plus its branch audit for FD probes."""

        _, local, micro = self._evaluate_components(
            fields, committed_state, compute_tangent=False
        )
        vector = np.concatenate(
            (
                local.first_piola.reshape(-1),
                np.array([local.q]),
                local.pi,
                micro.higher_order_stress_Pa_m.reshape(-1),
            )
        )
        vector.setflags(write=False)
        return vector, local.branch_audit

    def probe_gate01_sources(
        self,
        fields: PointFieldState,
        committed_state: LocalState92,
    ) -> Gate01PointSourceProbe:
        """Return Gate-01 endpoint rates and observer-ledger cross-checks."""

        transaction, local, _ = self._evaluate_components(
            fields, committed_state, compute_tangent=False
        )
        trial = transaction.trial_state
        beta = float(self.local_model.base_model.parameters.taylor_quinney)
        cp_power = float(local.cp_dissipation_rate)
        heat_source = float(local.q)
        passive_storage_rate = (1.0 - beta) * cp_power
        return Gate01PointSourceProbe(
            integration_point=int(fields.integration_point),
            cp_power_W_m3=cp_power,
            heat_source_W_m3=heat_source,
            passive_storage_rate_W_m3=passive_storage_rate,
            beta=beta,
            dt_s=float(fields.dt_s),
            committed_time_s=float(committed_state.time_s),
            trial_time_s=float(trial.time_s),
            delta_w_cp_state_J_m3=(
                trial.cp_work_density_J_m3 - committed_state.cp_work_density_J_m3
            ),
            delta_q_heat_state_J_m3=(
                trial.generated_heat_density_J_m3
                - committed_state.generated_heat_density_J_m3
            ),
            delta_u_s_state_J_m3=(
                trial.stored_energy_density_J_m3
                - committed_state.stored_energy_density_J_m3
            ),
            branch_audit=local.branch_audit,
        )
