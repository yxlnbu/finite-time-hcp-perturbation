"""Pure-Python Gate-01/THERMO-02 incremental energy ledger.

All scalar energies returned by this module are reference-configuration
integrals in joules.  The exchange extractor consumes raw Abaqus ``RHS``
blocks and applies the frozen sign convention ``RESIDUAL_R = -RHS`` itself.
Micromorphic blocks remain element-local until their non-physical
``omega_chi`` row scaling has been removed.

The endpoint mapping is deliberately atomic:

``ENERGY(2) = Psi_rec + U_s`` and ``ENERGY(4) = Q_heat``.

Neither their sum nor ``Q_heat`` is used as the physical total internal
energy, which is reconstructed independently as
``Psi_rec + U_s + E_th + Psi_GB``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite, log2
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

GATE01_ENERGY_AUDIT_SCHEMA = "HCP_CP_GATE01_THERMO02_LEDGER_V1"


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: ArrayLike, name: str, *, allow_empty: bool = True) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EndpointEnergyJ:
    """Accepted endpoint ledgers, each integrated over the reference domain.

    Field suffixes are units: every field is in J.  Reference offsets and a
    signed thermal reference energy are permitted, so validation requires
    finiteness rather than non-negativity.
    """

    psi_rec_J: float
    u_s_J: float
    e_th_J: float
    psi_gb_J: float
    w_cp_J: float
    q_heat_J: float

    def __post_init__(self) -> None:
        for name in (
            "psi_rec_J",
            "u_s_J",
            "e_th_J",
            "psi_gb_J",
            "w_cp_J",
            "q_heat_J",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

    @property
    def e_tot_J(self) -> float:
        """Physical total internal energy, reconstructed independently."""

        return fsum((self.psi_rec_J, self.u_s_J, self.e_th_J, self.psi_gb_J))

    @property
    def abaqus_energy_2_J(self) -> float:
        """Frozen atomic mapping for ``ENERGY(2)``."""

        return self.psi_rec_J + self.u_s_J

    @property
    def abaqus_energy_4_J(self) -> float:
        """Frozen atomic mapping for ``ENERGY(4)``."""

        return self.q_heat_J


@dataclass(frozen=True)
class MicromorphicElementRhs:
    """One unassembled element contribution to the published zeta rows.

    ``rhs_zeta_published_W`` has the artificial ``omega_chi`` row scaling and
    therefore has units W.  ``delta_zeta_1`` is dimensionless and
    ``omega_chi_s_inv`` has units s^-1.  Division by the element's own scale
    restores a J-valued generalized force before taking work.
    """

    element_label: str
    rhs_zeta_published_W: Array
    delta_zeta_1: Array
    omega_chi_s_inv: float

    def __post_init__(self) -> None:
        if not isinstance(self.element_label, str) or not self.element_label.strip():
            raise ValueError("element_label must be a non-empty string")
        rhs = _finite_vector(
            self.rhs_zeta_published_W,
            "rhs_zeta_published_W",
            allow_empty=False,
        )
        increment = _finite_vector(
            self.delta_zeta_1, "delta_zeta_1", allow_empty=False
        )
        if rhs.shape != increment.shape:
            raise ValueError(
                "rhs_zeta_published_W and delta_zeta_1 must have the same shape"
            )
        omega = _finite_scalar(self.omega_chi_s_inv, "omega_chi_s_inv")
        if omega <= 0.0:
            raise ValueError("omega_chi_s_inv must be positive")
        object.__setattr__(self, "rhs_zeta_published_W", rhs)
        object.__setattr__(self, "delta_zeta_1", increment)
        object.__setattr__(self, "omega_chi_s_inv", omega)


@dataclass(frozen=True)
class ElementMicromorphicWorkJ:
    """Recovered physical micromorphic work for one element, in J."""

    element_label: str
    omega_chi_s_inv: float
    delta_w_zeta_J: float

    def __post_init__(self) -> None:
        if not isinstance(self.element_label, str) or not self.element_label.strip():
            raise ValueError("element_label must be a non-empty string")
        omega = _finite_scalar(self.omega_chi_s_inv, "omega_chi_s_inv")
        if omega <= 0.0:
            raise ValueError("omega_chi_s_inv must be positive")
        object.__setattr__(self, "omega_chi_s_inv", omega)
        object.__setattr__(
            self,
            "delta_w_zeta_J",
            _finite_scalar(self.delta_w_zeta_J, "delta_w_zeta_J"),
        )


@dataclass(frozen=True)
class BoundaryExchangeJ:
    """Accepted-step boundary exchanges, with heat positive into the body."""

    delta_w_u_J: float
    micromorphic_by_element: tuple[ElementMicromorphicWorkJ, ...]
    delta_q_boundary_J: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "delta_w_u_J", _finite_scalar(self.delta_w_u_J, "delta_w_u_J")
        )
        blocks = tuple(self.micromorphic_by_element)
        if not all(isinstance(block, ElementMicromorphicWorkJ) for block in blocks):
            raise TypeError(
                "micromorphic_by_element must contain ElementMicromorphicWorkJ"
            )
        labels = [block.element_label for block in blocks]
        if len(labels) != len(set(labels)):
            raise ValueError("micromorphic element labels must be unique")
        object.__setattr__(self, "micromorphic_by_element", blocks)
        object.__setattr__(
            self,
            "delta_q_boundary_J",
            _finite_scalar(self.delta_q_boundary_J, "delta_q_boundary_J"),
        )

    @property
    def delta_w_zeta_J(self) -> float:
        return fsum(block.delta_w_zeta_J for block in self.micromorphic_by_element)

    @property
    def delta_w_ext_J(self) -> float:
        return self.delta_w_u_J + self.delta_w_zeta_J


@dataclass(frozen=True)
class BackwardEulerSourcesJ:
    """Independently integrated endpoint source terms for one accepted step.

    Every field is in J and must be formed from the converged ``n+1`` material
    response and the accepted step duration.  These quantities must not be
    reconstructed from differences of the cumulative STATE92 observer ledgers.
    """

    delta_w_cp_be_J: float
    delta_q_gen_be_J: float
    delta_u_s_src_be_J: float

    def __post_init__(self) -> None:
        for name in (
            "delta_w_cp_be_J",
            "delta_q_gen_be_J",
            "delta_u_s_src_be_J",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))


def extract_boundary_exchange_from_rhs(
    *,
    rhs_u_N: ArrayLike,
    delta_u_m: ArrayLike,
    micromorphic_elements: Iterable[MicromorphicElementRhs] = (),
    rhs_temperature_W: ArrayLike = (),
    dt_s: float,
) -> BoundaryExchangeJ:
    """Recover accepted-step physical exchanges from raw Abaqus ``RHS``.

    Only entries in the caller-selected displacement and temperature reaction
    sets are passed here.  This function applies ``RESIDUAL_R = -RHS`` to all
    three equation blocks.  Temperature rows are rate-like: their sum is
    multiplied by the accepted step duration, never by a temperature change.

    Micromorphic contributions must be supplied before assembly.  Dividing an
    already assembled zeta residual by one scale is invalid when neighboring
    elements have different ``omega_chi`` values.
    """

    rhs_u = _finite_vector(rhs_u_N, "rhs_u_N")
    delta_u = _finite_vector(delta_u_m, "delta_u_m")
    if rhs_u.shape != delta_u.shape:
        raise ValueError("rhs_u_N and delta_u_m must have the same shape")

    rhs_temperature = _finite_vector(rhs_temperature_W, "rhs_temperature_W")
    dt = _finite_scalar(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")

    blocks = tuple(micromorphic_elements)
    if not all(isinstance(block, MicromorphicElementRhs) for block in blocks):
        raise TypeError(
            "micromorphic_elements must contain MicromorphicElementRhs"
        )
    labels = [block.element_label for block in blocks]
    if len(labels) != len(set(labels)):
        raise ValueError("micromorphic element labels must be unique")

    # Frozen convention: physical audit residuals are the negative Abaqus RHS.
    residual_u_N = -rhs_u
    delta_w_u_J = float(np.dot(residual_u_N, delta_u))

    element_works: list[ElementMicromorphicWorkJ] = []
    for block in blocks:
        residual_zeta_published_W = -block.rhs_zeta_published_W
        residual_zeta_raw_J = (
            residual_zeta_published_W / block.omega_chi_s_inv
        )
        delta_w_zeta_J = float(np.dot(residual_zeta_raw_J, block.delta_zeta_1))
        element_works.append(
            ElementMicromorphicWorkJ(
                element_label=block.element_label,
                omega_chi_s_inv=block.omega_chi_s_inv,
                delta_w_zeta_J=delta_w_zeta_J,
            )
        )

    residual_temperature_W = -rhs_temperature
    delta_q_boundary_J = dt * fsum(float(value) for value in residual_temperature_W)
    return BoundaryExchangeJ(
        delta_w_u_J=delta_w_u_J,
        micromorphic_by_element=tuple(element_works),
        delta_q_boundary_J=delta_q_boundary_J,
    )


@dataclass(frozen=True)
class Gate01IncrementAuditJ:
    """Independent THERMO-02 ledger for one accepted time increment.

    The three ``r_*_state_source_J`` fields use the fixed sign
    ``cumulative STATE92 increment - independently integrated BE source``.
    Every stored quantity has units J.
    """

    delta_psi_rec_J: float
    delta_u_s_J: float
    delta_e_th_J: float
    delta_psi_gb_J: float
    delta_w_cp_state_J: float
    delta_q_heat_state_J: float
    delta_w_cp_be_J: float
    delta_q_gen_be_J: float
    delta_u_s_src_be_J: float
    delta_w_u_J: float
    delta_w_zeta_J: float
    delta_q_boundary_J: float
    r_mech_J: float
    r_store_J: float
    r_heat_J: float
    r_total_J: float
    r_beta_J: float
    r_w_cp_state_source_J: float
    r_q_heat_state_source_J: float
    r_u_s_state_source_J: float
    r_identity_J: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

    @property
    def identity_rhs_J(self) -> float:
        """Right side of ``Rtotal = -Rmech+Rstore+Rheat-Rbeta``."""

        return -self.r_mech_J + self.r_store_J + self.r_heat_J - self.r_beta_J

    @property
    def s_mech_J(self) -> float:
        """Per-step mechanical scale from absolute physical terms."""

        return fsum(
            abs(value)
            for value in (
                self.delta_w_u_J,
                self.delta_w_zeta_J,
                self.delta_psi_rec_J,
                self.delta_psi_gb_J,
                self.delta_w_cp_be_J,
            )
        )

    @property
    def s_total_J(self) -> float:
        """Per-step total-energy scale without cancellation in ``delta(E)``."""

        return fsum(
            abs(value)
            for value in (
                self.delta_psi_rec_J,
                self.delta_u_s_J,
                self.delta_e_th_J,
                self.delta_psi_gb_J,
                self.delta_w_u_J,
                self.delta_w_zeta_J,
                self.delta_q_boundary_J,
            )
        )

    @property
    def s_store_J(self) -> float:
        """Own physical scale for ``Rstore`` and the U_s STATE/source check."""

        return abs(self.delta_u_s_J) + abs(self.delta_u_s_src_be_J)

    @property
    def s_heat_J(self) -> float:
        """Own physical scale for the reassembled thermal balance."""

        return fsum(
            abs(value)
            for value in (
                self.delta_e_th_J,
                self.delta_q_boundary_J,
                self.delta_q_gen_be_J,
            )
        )

    @property
    def s_beta_J(self) -> float:
        """Own physical scale for ``Pcp=q+u_s_dot`` partition work."""

        return fsum(
            abs(value)
            for value in (
                self.delta_w_cp_be_J,
                self.delta_q_gen_be_J,
                self.delta_u_s_src_be_J,
            )
        )

    @property
    def s_w_cp_state_source_J(self) -> float:
        """Own scale for cumulative Wcp increment versus the V2 BE source."""

        return abs(self.delta_w_cp_state_J) + abs(self.delta_w_cp_be_J)

    @property
    def s_q_heat_state_source_J(self) -> float:
        """Own scale for cumulative Qheat increment versus the V2 BE source."""

        return abs(self.delta_q_heat_state_J) + abs(self.delta_q_gen_be_J)

    @property
    def s_u_s_state_source_J(self) -> float:
        """Own scale for cumulative U_s increment versus the V2 BE source."""

        return self.s_store_J

    @property
    def s_identity_J(self) -> float:
        """Own algebraic scale for the five-term residual identity.

        The identity residual is formed from ``Rtotal + Rmech - Rstore -
        Rheat + Rbeta``.  Its scale is therefore the absolute sum of those
        five independently reported ledger residuals, rather than an unrelated
        mechanical or total-energy scale.
        """

        return fsum(
            abs(value)
            for value in (
                self.r_total_J,
                self.r_mech_J,
                self.r_store_J,
                self.r_heat_J,
                self.r_beta_J,
            )
        )


def audit_gate01_increment(
    *,
    before: EndpointEnergyJ,
    after: EndpointEnergyJ,
    exchange: BoundaryExchangeJ,
    sources: BackwardEulerSourcesJ,
) -> Gate01IncrementAuditJ:
    """Evaluate the frozen Gate-01 residuals for one accepted BE step.

    ``sources`` is mandatory and independent of the cumulative endpoint
    ledgers.  The latter are used only for the three STATE92/source comparisons
    required by E02; they never substitute for the converged endpoint source
    integration in ``Rmech``, ``Rstore``, ``Rheat``, ``Rbeta``, or the identity.
    """

    if not isinstance(before, EndpointEnergyJ) or not isinstance(after, EndpointEnergyJ):
        raise TypeError("before and after must be EndpointEnergyJ")
    if not isinstance(exchange, BoundaryExchangeJ):
        raise TypeError("exchange must be BoundaryExchangeJ")
    if not isinstance(sources, BackwardEulerSourcesJ):
        raise TypeError("sources must be BackwardEulerSourcesJ")

    delta_psi_rec_J = after.psi_rec_J - before.psi_rec_J
    delta_u_s_J = after.u_s_J - before.u_s_J
    delta_e_th_J = after.e_th_J - before.e_th_J
    delta_psi_gb_J = after.psi_gb_J - before.psi_gb_J
    delta_w_cp_state_J = after.w_cp_J - before.w_cp_J
    delta_q_heat_state_J = after.q_heat_J - before.q_heat_J

    r_mech_J = (
        exchange.delta_w_ext_J
        - delta_psi_rec_J
        - delta_psi_gb_J
        - sources.delta_w_cp_be_J
    )
    r_store_J = delta_u_s_J - sources.delta_u_s_src_be_J
    r_heat_J = (
        delta_e_th_J - exchange.delta_q_boundary_J - sources.delta_q_gen_be_J
    )
    r_total_J = (
        after.e_tot_J
        - before.e_tot_J
        - exchange.delta_w_u_J
        - exchange.delta_w_zeta_J
        - exchange.delta_q_boundary_J
    )
    r_beta_J = (
        sources.delta_w_cp_be_J
        - sources.delta_q_gen_be_J
        - sources.delta_u_s_src_be_J
    )
    r_w_cp_state_source_J = delta_w_cp_state_J - sources.delta_w_cp_be_J
    r_q_heat_state_source_J = (
        delta_q_heat_state_J - sources.delta_q_gen_be_J
    )
    r_u_s_state_source_J = delta_u_s_J - sources.delta_u_s_src_be_J
    identity_rhs_J = -r_mech_J + r_store_J + r_heat_J - r_beta_J
    r_identity_J = r_total_J - identity_rhs_J

    return Gate01IncrementAuditJ(
        delta_psi_rec_J=delta_psi_rec_J,
        delta_u_s_J=delta_u_s_J,
        delta_e_th_J=delta_e_th_J,
        delta_psi_gb_J=delta_psi_gb_J,
        delta_w_cp_state_J=delta_w_cp_state_J,
        delta_q_heat_state_J=delta_q_heat_state_J,
        delta_w_cp_be_J=sources.delta_w_cp_be_J,
        delta_q_gen_be_J=sources.delta_q_gen_be_J,
        delta_u_s_src_be_J=sources.delta_u_s_src_be_J,
        delta_w_u_J=exchange.delta_w_u_J,
        delta_w_zeta_J=exchange.delta_w_zeta_J,
        delta_q_boundary_J=exchange.delta_q_boundary_J,
        r_mech_J=r_mech_J,
        r_store_J=r_store_J,
        r_heat_J=r_heat_J,
        r_total_J=r_total_J,
        r_beta_J=r_beta_J,
        r_w_cp_state_source_J=r_w_cp_state_source_J,
        r_q_heat_state_source_J=r_q_heat_state_source_J,
        r_u_s_state_source_J=r_u_s_state_source_J,
        r_identity_J=r_identity_J,
    )


class TimeOrderTarget(str, Enum):
    """The only residuals permitted by Gate-01 section 7.4."""

    MECH = "mech"
    TOTAL = "total"


class TimeOrderStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ORDER_NOT_IDENTIFIABLE = "ORDER_NOT_IDENTIFIABLE"


@dataclass(frozen=True)
class TimeOrderLevelMetricsJ:
    """One time-level aggregate; ``epsilon`` is dimensionless."""

    label: str
    scale_sum_J: float
    residual_abs_sum_J: float
    epsilon: float

    def __post_init__(self) -> None:
        if self.label not in {"DT", "DT2", "DT4"}:
            raise ValueError("label must be DT, DT2, or DT4")
        for name in ("scale_sum_J", "residual_abs_sum_J", "epsilon"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Thermo02TimeOrderGateJ:
    """DT/DT2/DT4 result using one common, pre-data-independent scale."""

    target: TimeOrderTarget
    dt: TimeOrderLevelMetricsJ
    dt2: TimeOrderLevelMetricsJ
    dt4: TimeOrderLevelMetricsJ
    shared_scale_J: float
    normalization_scale_J: float
    absolute_floor_J: float
    signal_floor: float
    signal_threshold_J: float
    status: TimeOrderStatus
    absolute_gate_passed: bool | None
    monotone: bool | None
    p12: float | None
    p24: float | None

    @property
    def passed(self) -> bool:
        """True only for an identifiable first-order PASS."""

        return self.status is TimeOrderStatus.PASS


def _time_level_sums(
    steps: Iterable[Gate01IncrementAuditJ],
    *,
    target: TimeOrderTarget,
) -> tuple[float, float]:
    accepted_steps = tuple(steps)
    if not accepted_steps:
        raise ValueError("each time level must contain at least one accepted step")
    if not all(isinstance(step, Gate01IncrementAuditJ) for step in accepted_steps):
        raise TypeError("time levels must contain Gate01IncrementAuditJ")
    if target is TimeOrderTarget.MECH:
        scales = (step.s_mech_J for step in accepted_steps)
        residuals = (abs(step.r_mech_J) for step in accepted_steps)
    else:
        scales = (step.s_total_J for step in accepted_steps)
        residuals = (abs(step.r_total_J) for step in accepted_steps)
    return fsum(scales), fsum(residuals)


def evaluate_thermo02_time_order(
    *,
    target: TimeOrderTarget,
    dt: Iterable[Gate01IncrementAuditJ],
    dt2: Iterable[Gate01IncrementAuditJ],
    dt4: Iterable[Gate01IncrementAuditJ],
    absolute_floor_J: float,
    signal_floor: float,
) -> Thermo02TimeOrderGateJ:
    """Apply the frozen section-7 first-order gate to three time levels.

    The denominator is always
    ``max(max_h(sum_n(S_X,n)), absolute_floor_J)`` and is shared by all three
    levels.  A level at or below either the absolute or relative signal floor
    yields ``ORDER_NOT_IDENTIFIABLE``.  Its absolute-gate result is reported
    separately and can never turn that status into ``PASS``.

    ``target`` must be a :class:`TimeOrderTarget`; there is intentionally no
    heat, storage, or beta option because those residuals have algebraic or
    solver-tolerance gates rather than a mandatory observed order.
    """

    if not isinstance(target, TimeOrderTarget):
        raise TypeError("target must be TimeOrderTarget.MECH or TimeOrderTarget.TOTAL")
    absolute_floor = _finite_scalar(absolute_floor_J, "absolute_floor_J")
    relative_signal_floor = _finite_scalar(signal_floor, "signal_floor")
    if absolute_floor <= 0.0:
        raise ValueError("absolute_floor_J must be positive")
    if relative_signal_floor <= 0.0:
        raise ValueError("signal_floor must be positive")

    raw_levels = (
        ("DT",) + _time_level_sums(dt, target=target),
        ("DT2",) + _time_level_sums(dt2, target=target),
        ("DT4",) + _time_level_sums(dt4, target=target),
    )
    shared_scale_J = max(item[1] for item in raw_levels)
    normalization_scale_J = max(shared_scale_J, absolute_floor)
    metrics = tuple(
        TimeOrderLevelMetricsJ(
            label=label,
            scale_sum_J=scale_sum_J,
            residual_abs_sum_J=residual_abs_sum_J,
            epsilon=residual_abs_sum_J / normalization_scale_J,
        )
        for label, scale_sum_J, residual_abs_sum_J in raw_levels
    )
    signal_threshold_J = max(
        absolute_floor, relative_signal_floor * normalization_scale_J
    )
    identifiable = shared_scale_J > absolute_floor and all(
        level.residual_abs_sum_J > signal_threshold_J for level in metrics
    )

    if not identifiable:
        return Thermo02TimeOrderGateJ(
            target=target,
            dt=metrics[0],
            dt2=metrics[1],
            dt4=metrics[2],
            shared_scale_J=shared_scale_J,
            normalization_scale_J=normalization_scale_J,
            absolute_floor_J=absolute_floor,
            signal_floor=relative_signal_floor,
            signal_threshold_J=signal_threshold_J,
            status=TimeOrderStatus.ORDER_NOT_IDENTIFIABLE,
            absolute_gate_passed=all(
                level.residual_abs_sum_J <= absolute_floor for level in metrics
            ),
            monotone=None,
            p12=None,
            p24=None,
        )

    epsilon_dt, epsilon_dt2, epsilon_dt4 = (
        level.epsilon for level in metrics
    )
    monotone = epsilon_dt2 < epsilon_dt and epsilon_dt4 < epsilon_dt2
    p12 = log2(epsilon_dt / epsilon_dt2)
    p24 = log2(epsilon_dt2 / epsilon_dt4)
    order_in_band = 0.8 <= p12 <= 1.2 and 0.8 <= p24 <= 1.2
    status = TimeOrderStatus.PASS if monotone and order_in_band else TimeOrderStatus.FAIL
    return Thermo02TimeOrderGateJ(
        target=target,
        dt=metrics[0],
        dt2=metrics[1],
        dt4=metrics[2],
        shared_scale_J=shared_scale_J,
        normalization_scale_J=normalization_scale_J,
        absolute_floor_J=absolute_floor,
        signal_floor=relative_signal_floor,
        signal_threshold_J=signal_threshold_J,
        status=status,
        absolute_gate_passed=None,
        monotone=monotone,
        p12=p12,
        p24=p24,
    )
