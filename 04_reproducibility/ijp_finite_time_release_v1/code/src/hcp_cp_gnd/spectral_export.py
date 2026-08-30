"""Side-effect-free continuous material-point export for spectral stability.

This module is intentionally separate from the backward-Euler local residual
and its endpoint Schur complement.  It evaluates the frozen continuous
constitutive vector field and, when requested, a branch-preserving reference
Jacobian with the locked ordering

``x = [F(9), T, zeta(18), a(62)]`` and
``y = [P(9), q, pi(18), g_a(62)]``.

The first-paper specialization is slip-only: twinning must be disabled and
the stored twin fractions must be exactly zero.  The legacy 92-real state
remains the persistent storage ABI; passive accumulators are observers and do
not enter the 62-dimensional active Jacobian.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .branch_audit import (
    SpectralAdmissibilityError,
    SpectralBranchAudit,
    SpectralNondifferentiableError,
)
from .micromorphic import (
    MicromorphicParameters,
    MicromorphicResponse,
    evaluate_micromorphic_state,
)
from .sl3_chart import N_SL3, SL3LocalChart
from .state_contract import LocalState92

Array = NDArray[np.float64]
N_SLIP = 18
N_ACTIVE = 62
N_FIELD = 28
N_SPECTRAL = 90
SPECTRAL_EXPORT_SCHEMA = "HCP_CP_SPECTRAL_POINT_EXPORT_V1"
SPECTRAL_STATE_SCHEMA = "HCP_CP_ACTIVE_STATE_62_V1"
SPECTRAL_CHART_SCHEMA = "LEFT_EXPONENTIAL_SL3_LOCAL_CHART_V1"
K_B = 1.380649e-23


def _tensor_labels(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{row}{column}" for row in range(1, 4) for column in range(1, 4))


def _system_labels(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{alpha:02d}" for alpha in range(1, N_SLIP + 1))


INPUT_COORDINATE_LABELS = (
    _tensor_labels("F")
    + ("T",)
    + _system_labels("zeta")
    + tuple(f"theta{coordinate:02d}" for coordinate in range(1, N_SL3 + 1))
    + _system_labels("rho_mobile")
    + _system_labels("rho_dipole")
    + _system_labels("gamma_signed")
)
OUTPUT_COORDINATE_LABELS = (
    _tensor_labels("P")
    + ("q",)
    + _system_labels("pi")
    + tuple(f"theta_dot{coordinate:02d}" for coordinate in range(1, N_SL3 + 1))
    + _system_labels("rho_mobile_dot")
    + _system_labels("rho_dipole_dot")
    + _system_labels("gamma_dot")
)


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


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _state_fingerprint(state: LocalState92) -> bytes:
    return state.pack().astype("<f8", copy=False).tobytes()


@dataclass(frozen=True)
class SpectralActiveState62:
    """Physical active state expressed in the registered local ``SL(3)`` chart."""

    theta_p: Array
    rho_mobile_m2: Array
    rho_dipole_m2: Array
    gamma_signed: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_p", _frozen(self.theta_p, (N_SL3,), "theta_p"))
        object.__setattr__(
            self,
            "rho_mobile_m2",
            _frozen(self.rho_mobile_m2, (N_SLIP,), "rho_mobile_m2"),
        )
        object.__setattr__(
            self,
            "rho_dipole_m2",
            _frozen(self.rho_dipole_m2, (N_SLIP,), "rho_dipole_m2"),
        )
        object.__setattr__(
            self, "gamma_signed", _frozen(self.gamma_signed, (N_SLIP,), "gamma_signed")
        )
        if np.any(self.rho_mobile_m2 <= 0.0) or np.any(self.rho_dipole_m2 <= 0.0):
            raise ValueError("active dislocation densities must be strictly positive")

    def pack(self) -> Array:
        result = np.concatenate(
            (self.theta_p, self.rho_mobile_m2, self.rho_dipole_m2, self.gamma_signed)
        )
        if result.shape != (N_ACTIVE,):  # pragma: no cover - defensive contract
            raise RuntimeError("active state did not pack to 62 coordinates")
        return result

    @classmethod
    def unpack(cls, packed: Array) -> "SpectralActiveState62":
        values = _finite_array(packed, (N_ACTIVE,), "packed active state")
        return cls(
            theta_p=values[0:8],
            rho_mobile_m2=values[8:26],
            rho_dipole_m2=values[26:44],
            gamma_signed=values[44:62],
        )

    @classmethod
    def from_storage(
        cls, state: LocalState92, chart: SL3LocalChart
    ) -> "SpectralActiveState62":
        return cls(
            theta_p=chart.coordinates(state.Fp),
            rho_mobile_m2=state.rho_mobile_m2,
            rho_dipole_m2=state.rho_dipole_m2,
            gamma_signed=state.gamma_signed,
        )


@dataclass(frozen=True)
class SpectralObserverState:
    """Passive checkpoint values excluded from the active spectrum."""

    Gamma_absolute: Array
    cp_work_density_J_m3: float
    generated_heat_density_J_m3: float
    passive_storage_density_J_m3: float
    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "Gamma_absolute",
            _frozen(self.Gamma_absolute, (N_SLIP,), "Gamma_absolute"),
        )
        scalars = np.array(
            [
                self.cp_work_density_J_m3,
                self.generated_heat_density_J_m3,
                self.passive_storage_density_J_m3,
                self.time_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
            raise ValueError("observer ledgers and time must be finite and non-negative")
        if np.any(self.Gamma_absolute < 0.0):
            raise ValueError("Gamma_absolute must be non-negative")
        scale = max(abs(self.cp_work_density_J_m3), 1.0)
        partition = (
            self.cp_work_density_J_m3
            - self.generated_heat_density_J_m3
            - self.passive_storage_density_J_m3
        )
        if abs(partition) > 1.0e-10 * scale:
            raise ValueError("observer power/heat/storage partition is inconsistent")

    @classmethod
    def from_storage(cls, state: LocalState92) -> "SpectralObserverState":
        return cls(
            Gamma_absolute=state.Gamma_absolute,
            cp_work_density_J_m3=state.cp_work_density_J_m3,
            generated_heat_density_J_m3=state.generated_heat_density_J_m3,
            passive_storage_density_J_m3=state.stored_energy_density_J_m3,
            time_s=state.time_s,
        )

    @property
    def cp_power_density_J_m3(self) -> float:
        """Legacy observer alias; this ledger is accumulated work, not power."""

        return self.cp_work_density_J_m3


@dataclass(frozen=True)
class SpectralDerivativeOptions:
    relative_step: float = 2.0e-6
    maximum_halvings: int = 12
    normalized_branch_distance_floor: float = 1.0e-10
    slip_coordinate_scale: float = 2.0e-4

    def validate(self) -> None:
        values = np.array(
            [
                self.relative_step,
                self.normalized_branch_distance_floor,
                self.slip_coordinate_scale,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("spectral derivative controls must be finite and positive")
        if self.maximum_halvings < 0:
            raise ValueError("maximum_halvings must be non-negative")


@dataclass(frozen=True)
class QuantityContract:
    name: str
    units: str
    configuration: str
    basis: str
    role: str


_QUANTITY_CONTRACT = (
    QuantityContract("F", "1", "sample reference", "row-major Cartesian", "input"),
    QuantityContract("T", "K", "material point", "scalar", "input"),
    QuantityContract("zeta", "1", "sample reference", "18-system order", "input"),
    QuantityContract(
        "Grad_zeta", "m^-1", "sample reference", "18-system x Cartesian", "input"
    ),
    QuantityContract("P", "Pa", "sample reference", "row-major Cartesian", "response"),
    QuantityContract("tau", "Pa", "crystal intermediate", "18-system order", "response"),
    QuantityContract("tau_hat", "Pa", "crystal intermediate", "18-system order", "response"),
    QuantityContract("gamma_dot", "s^-1", "crystal intermediate", "18-system order", "rate"),
    QuantityContract("Fp_dot", "s^-1", "crystal intermediate", "row-major Cartesian", "rate"),
    QuantityContract("rho_dot", "m^-2 s^-1", "material point", "18-system order", "rate"),
    QuantityContract("Pcp", "W m^-3", "reference volume", "scalar", "power"),
    QuantityContract("q", "W m^-3", "reference volume", "scalar", "heat source"),
    QuantityContract("u_s_dot", "W m^-3", "reference volume", "scalar", "storage"),
    QuantityContract(
        "D_mat", "W m^-3", "reference volume", "scalar", "material dissipation"
    ),
    QuantityContract("pi", "Pa", "sample reference", "18-system order", "microstress"),
    QuantityContract("xi", "Pa m", "sample reference", "18-system x Cartesian", "microstress"),
    QuantityContract("K0", "W m^-1 K^-1", "sample reference", "Cartesian", "conductivity"),
    QuantityContract("j_q", "W m^-2", "sample reference", "Cartesian", "heat flux"),
    QuantityContract(
        "entropy_production", "W m^-3 K^-1", "reference volume", "scalar", "rate"
    ),
    QuantityContract("psi_e", "J m^-3", "reference volume", "scalar", "energy"),
    QuantityContract("psi_chi", "J m^-3", "reference volume", "scalar", "energy"),
    QuantityContract("e_th", "J m^-3", "reference volume", "scalar", "energy"),
    QuantityContract("e_int", "J m^-3", "reference volume", "scalar", "energy"),
    QuantityContract("Gamma_abs", "1", "material point", "18-system order", "observer"),
    QuantityContract("W_cp", "J m^-3", "reference volume", "scalar", "observer"),
    QuantityContract("Q_heat", "J m^-3", "reference volume", "scalar", "observer"),
    QuantityContract("U_s", "J m^-3", "reference volume", "scalar", "observer"),
    QuantityContract("rho0", "kg m^-3", "reference volume", "scalar", "parameter"),
    QuantityContract("c", "J kg^-1 K^-1", "material point", "scalar", "parameter"),
)


@dataclass(frozen=True)
class CoordinateBlock:
    name: str
    start: int
    stop: int
    units: str
    configuration: str
    basis: str


_INPUT_LAYOUT = (
    CoordinateBlock("F", 0, 9, "1", "sample reference", "row-major Cartesian"),
    CoordinateBlock("T", 9, 10, "K", "material point", "scalar"),
    CoordinateBlock("zeta", 10, 28, "1", "sample reference", "18-system order"),
    CoordinateBlock("theta_p", 28, 36, "1", "crystal intermediate", "SL(3) generator order"),
    CoordinateBlock("rho_mobile", 36, 54, "m^-2", "material point", "18-system order"),
    CoordinateBlock("rho_dipole", 54, 72, "m^-2", "material point", "18-system order"),
    CoordinateBlock("gamma_signed", 72, 90, "1", "material point", "18-system order"),
)
_OUTPUT_LAYOUT = (
    CoordinateBlock("P", 0, 9, "Pa", "sample reference", "row-major Cartesian"),
    CoordinateBlock("q", 9, 10, "W m^-3", "reference volume", "scalar"),
    CoordinateBlock("pi", 10, 28, "Pa", "sample reference", "18-system order"),
    CoordinateBlock("theta_p_dot", 28, 36, "s^-1", "crystal intermediate", "SL(3) generator order"),
    CoordinateBlock("rho_mobile_dot", 36, 54, "m^-2 s^-1", "material point", "18-system order"),
    CoordinateBlock("rho_dipole_dot", 54, 72, "m^-2 s^-1", "material point", "18-system order"),
    CoordinateBlock("gamma_dot", 72, 90, "s^-1", "material point", "18-system order"),
)
_ACTIVE_STATE_LAYOUT = (
    CoordinateBlock("theta_p", 0, 8, "1", "crystal intermediate", "SL(3) generator order"),
    CoordinateBlock("rho_mobile", 8, 26, "m^-2", "material point", "18-system order"),
    CoordinateBlock("rho_dipole", 26, 44, "m^-2", "material point", "18-system order"),
    CoordinateBlock("gamma_signed", 44, 62, "1", "material point", "18-system order"),
)


@dataclass(frozen=True)
class SpectralExportMetadata:
    schema: str
    state_schema: str
    track: str
    parameter_status: str
    parameter_provenance: str
    source_hash: str
    source_dependencies: tuple[tuple[str, str], ...]
    parameter_hash: str
    configuration_hash: str
    case_hash: str
    raw_microforce_equation: bool
    omega_chi: float
    mechanism_switches: tuple[str, ...]
    input_coordinate_labels: tuple[str, ...]
    output_coordinate_labels: tuple[str, ...]
    input_layout: tuple[CoordinateBlock, ...]
    output_layout: tuple[CoordinateBlock, ...]
    active_state_layout: tuple[CoordinateBlock, ...]
    chart_schema: str
    chart_anchor_Fp: Array
    sl3_generators: Array
    sl3_generator_hash: str
    tensor_flattening: str
    declared_validity: str
    quantity_contract: tuple[QuantityContract, ...] = _QUANTITY_CONTRACT

    def __post_init__(self) -> None:
        if len(self.input_coordinate_labels) != N_SPECTRAL:
            raise ValueError("spectral input labels must enumerate all 90 columns")
        if len(self.output_coordinate_labels) != N_SPECTRAL:
            raise ValueError("spectral output labels must enumerate all 90 rows")
        object.__setattr__(
            self, "chart_anchor_Fp", _frozen(self.chart_anchor_Fp, (3, 3), "chart_anchor_Fp")
        )
        object.__setattr__(
            self,
            "sl3_generators",
            _frozen(self.sl3_generators, (N_SL3, 3, 3), "sl3_generators"),
        )


@dataclass(frozen=True)
class SpectralPointDerivatives:
    """Packed continuous Jacobian with locked 90-by-90 ordering."""

    packed: Array
    stencil: str
    central_columns: int
    one_sided_columns: int
    minimum_normalized_branch_distance: float
    column_methods: tuple[str, ...]
    final_relative_steps: Array
    final_physical_steps: Array
    chain_scales: Array
    minimum_normalized_branch_distances: Array
    limiting_branch_labels: tuple[str, ...]
    admissibility_boundary_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "packed", _frozen(self.packed, (N_SPECTRAL, N_SPECTRAL), "packed"))
        for name in (
            "final_relative_steps",
            "final_physical_steps",
            "chain_scales",
            "minimum_normalized_branch_distances",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name), (N_SPECTRAL,), name))
        if len(self.column_methods) != N_SPECTRAL:
            raise ValueError("column_methods must enumerate all 90 derivative columns")
        if len(self.limiting_branch_labels) != N_SPECTRAL:
            raise ValueError("limiting_branch_labels must enumerate all 90 columns")
        if len(self.admissibility_boundary_labels) != N_SPECTRAL:
            raise ValueError("admissibility_boundary_labels must enumerate all 90 columns")
        if any(method not in {"central", "one-sided-plus", "one-sided-minus"} for method in self.column_methods):
            raise ValueError("unknown continuous derivative method")
        for method, boundary in zip(
            self.column_methods, self.admissibility_boundary_labels
        ):
            if method == "central" and boundary:
                raise ValueError("central columns may not report a one-sided boundary")
            if method != "central" and not boundary.startswith("admissibility."):
                raise ValueError("one-sided columns must report an admissibility boundary")
        if np.any(self.final_relative_steps <= 0.0) or np.any(self.final_physical_steps <= 0.0):
            raise ValueError("final derivative steps must be positive")
        if np.any(self.chain_scales <= 0.0):
            raise ValueError("derivative chain scales must be positive")
        if np.any(self.minimum_normalized_branch_distances <= 0.0):
            raise ValueError("per-column branch distances must be positive")
        if not np.isclose(
            self.minimum_normalized_branch_distance,
            float(np.min(self.minimum_normalized_branch_distances)),
            rtol=5.0e-15,
            atol=0.0,
        ):
            raise ValueError("global branch distance does not match the column audit")
        if self.central_columns < 0 or self.one_sided_columns < 0:
            raise ValueError("derivative stencil counts must be non-negative")
        if self.central_columns + self.one_sided_columns != N_SPECTRAL:
            raise ValueError("the continuous Jacobian must account for all 90 columns")
        if (
            not np.isfinite(self.minimum_normalized_branch_distance)
            or self.minimum_normalized_branch_distance <= 0.0
        ):
            raise ValueError("minimum branch distance must be positive and finite")

    @property
    def field_blocks(self) -> Array:
        return self.packed[:, :N_FIELD]

    @property
    def state_blocks(self) -> Array:
        return self.packed[:, N_FIELD:]

    @property
    def dP_dF(self) -> Array:
        return self.packed[0:9, 0:9]

    @property
    def dP_dT(self) -> Array:
        return self.packed[0:9, 9:10]

    @property
    def dP_dzeta(self) -> Array:
        return self.packed[0:9, 10:28]

    @property
    def dP_da(self) -> Array:
        return self.packed[0:9, 28:90]

    @property
    def dq_dF(self) -> Array:
        return self.packed[9:10, 0:9]

    @property
    def dq_dT(self) -> Array:
        return self.packed[9:10, 9:10]

    @property
    def dq_dzeta(self) -> Array:
        return self.packed[9:10, 10:28]

    @property
    def dq_da(self) -> Array:
        return self.packed[9:10, 28:90]

    @property
    def dpi_dF(self) -> Array:
        return self.packed[10:28, 0:9]

    @property
    def dpi_dT(self) -> Array:
        return self.packed[10:28, 9:10]

    @property
    def dpi_dzeta(self) -> Array:
        return self.packed[10:28, 10:28]

    @property
    def dpi_da(self) -> Array:
        return self.packed[10:28, 28:90]

    @property
    def dg_dF(self) -> Array:
        return self.packed[28:90, 0:9]

    @property
    def dg_dT(self) -> Array:
        return self.packed[28:90, 9:10]

    @property
    def dg_dzeta(self) -> Array:
        return self.packed[28:90, 10:28]

    @property
    def dg_da(self) -> Array:
        return self.packed[28:90, 28:90]


@dataclass(frozen=True)
class _RawSpectralResponse:
    first_piola_Pa: Array
    resolved_slip_Pa: Array
    effective_resolved_slip_Pa: Array
    slip_rate_s_inv: Array
    plastic_velocity_gradient_s_inv: Array
    Fp_rate_s_inv: Array
    rho_mobile_rate_m2_s: Array
    rho_dipole_rate_m2_s: Array
    gamma_signed_rate_s_inv: Array
    Gamma_absolute_rate_s_inv: Array
    active_rhs: Array
    cp_power_W_m3: float
    heat_source_W_m3: float
    storage_rate_W_m3: float
    material_dissipation_W_m3: float
    mechanical_power_W_m3: float
    microforce_power_W_m3: float
    power_identity_residual_W_m3: float
    power_partition_residual_W_m3: float
    penalty_microstress_Pa: Array
    higher_order_stress_Pa_m: Array
    gradient_hessian_Pa_m2: Array
    directional_gradient_Pa_m2: Array
    heat_flux_W_m2: Array
    material_entropy_production_W_m3K: float
    fourier_entropy_production_W_m3K: float
    total_entropy_production_W_m3K: float
    elastic_energy_J_m3: float
    micromorphic_energy_J_m3: float
    thermal_internal_energy_J_m3: float
    total_internal_energy_J_m3: float
    forest_density_m2: Array
    slip_resistance_Pa: Array
    mean_free_path_m: Array
    branch_audit: SpectralBranchAudit

    def observable_vector(self) -> Array:
        return np.concatenate(
            (
                self.first_piola_Pa.reshape(-1),
                np.array([self.heat_source_W_m3]),
                self.penalty_microstress_Pa,
                self.active_rhs,
            )
        )


@dataclass(frozen=True)
class SpectralPointExport:
    first_piola_Pa: Array
    resolved_slip_Pa: Array
    effective_resolved_slip_Pa: Array
    slip_rate_s_inv: Array
    plastic_velocity_gradient_s_inv: Array
    Fp_rate_s_inv: Array
    rho_mobile_rate_m2_s: Array
    rho_dipole_rate_m2_s: Array
    gamma_signed_rate_s_inv: Array
    Gamma_absolute_rate_s_inv: Array
    active_rhs: Array
    cp_power_W_m3: float
    heat_source_W_m3: float
    storage_rate_W_m3: float
    material_dissipation_W_m3: float
    mechanical_power_W_m3: float
    microforce_power_W_m3: float
    power_identity_residual_W_m3: float
    power_partition_residual_W_m3: float
    penalty_microstress_Pa: Array
    higher_order_stress_Pa_m: Array
    gradient_hessian_Pa_m2: Array
    directional_gradient_Pa_m2: Array
    heat_flux_W_m2: Array
    material_entropy_production_W_m3K: float
    fourier_entropy_production_W_m3K: float
    total_entropy_production_W_m3K: float
    elastic_energy_J_m3: float
    micromorphic_energy_J_m3: float
    thermal_internal_energy_J_m3: float
    total_internal_energy_J_m3: float
    forest_density_m2: Array
    slip_resistance_Pa: Array
    mean_free_path_m: Array
    reference_density_kg_m3: float
    specific_heat_J_kgK: float
    conductivity_W_mK: Array
    Gamma_absolute: Array
    cp_work_density_J_m3: float
    generated_heat_density_J_m3: float
    passive_storage_density_J_m3: float
    time_s: float
    branch_audit: SpectralBranchAudit
    derivatives: SpectralPointDerivatives | None
    metadata: SpectralExportMetadata

    def __post_init__(self) -> None:
        shapes = {
            "first_piola_Pa": (3, 3),
            "resolved_slip_Pa": (N_SLIP,),
            "effective_resolved_slip_Pa": (N_SLIP,),
            "slip_rate_s_inv": (N_SLIP,),
            "plastic_velocity_gradient_s_inv": (3, 3),
            "Fp_rate_s_inv": (3, 3),
            "rho_mobile_rate_m2_s": (N_SLIP,),
            "rho_dipole_rate_m2_s": (N_SLIP,),
            "gamma_signed_rate_s_inv": (N_SLIP,),
            "Gamma_absolute_rate_s_inv": (N_SLIP,),
            "active_rhs": (N_ACTIVE,),
            "penalty_microstress_Pa": (N_SLIP,),
            "higher_order_stress_Pa_m": (N_SLIP, 3),
            "gradient_hessian_Pa_m2": (N_SLIP, 3, N_SLIP, 3),
            "directional_gradient_Pa_m2": (N_SLIP, N_SLIP),
            "heat_flux_W_m2": (3,),
            "forest_density_m2": (N_SLIP,),
            "slip_resistance_Pa": (N_SLIP,),
            "mean_free_path_m": (N_SLIP,),
            "conductivity_W_mK": (3, 3),
            "Gamma_absolute": (N_SLIP,),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _frozen(getattr(self, name), shape, name))
        scalars = np.array(
            [
                self.cp_power_W_m3,
                self.heat_source_W_m3,
                self.storage_rate_W_m3,
                self.material_dissipation_W_m3,
                self.mechanical_power_W_m3,
                self.microforce_power_W_m3,
                self.power_identity_residual_W_m3,
                self.power_partition_residual_W_m3,
                self.material_entropy_production_W_m3K,
                self.fourier_entropy_production_W_m3K,
                self.total_entropy_production_W_m3K,
                self.elastic_energy_J_m3,
                self.micromorphic_energy_J_m3,
                self.thermal_internal_energy_J_m3,
                self.total_internal_energy_J_m3,
                self.reference_density_kg_m3,
                self.specific_heat_J_kgK,
                self.cp_work_density_J_m3,
                self.generated_heat_density_J_m3,
                self.passive_storage_density_J_m3,
                self.time_s,
            ]
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("spectral export contains a non-finite scalar")

    @property
    def cp_power_density_J_m3(self) -> float:
        """Legacy observer alias; the stored quantity is accumulated CP work."""

        return self.cp_work_density_J_m3


class ContinuousSpectralPointModel:
    """Continuous slip-only spectral wrapper around the pure HCP material point."""

    def __init__(
        self,
        base_model: Any,
        micromorphic_parameters: MicromorphicParameters,
        *,
        conductivity_W_mK: float,
        parameter_provenance: str,
        omega_chi: float = 1.0,
        power_partition_law: Any | None = None,
        derivative_options: SpectralDerivativeOptions | None = None,
    ) -> None:
        required = (
            "initial_state",
            "evaluate",
            "elastic_energy_density",
            "slip_rates_from_resolved",
            "systems",
            "parameters",
            "switches",
            "orientation",
        )
        missing = [name for name in required if not hasattr(base_model, name)]
        if missing:
            raise TypeError(f"base_model lacks the continuous export contract: {missing}")
        if int(base_model.systems.n_slip) != N_SLIP:
            raise ValueError("the spectral export requires exactly 18 slip systems")
        if bool(base_model.switches.twinning):
            raise ValueError("the 62-state first-paper export requires twinning=False")
        if not isinstance(parameter_provenance, str) or not parameter_provenance.strip():
            raise ValueError("parameter_provenance must be a non-empty string")
        scalars = np.array([conductivity_W_mK, omega_chi], dtype=np.float64)
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ValueError("conductivity and omega_chi must be finite and positive")
        self.base_model = base_model
        self.micromorphic_parameters = micromorphic_parameters
        self.micromorphic_parameters.validate(N_SLIP)
        if not np.allclose(
            np.asarray(micromorphic_parameters.burgers_m),
            np.asarray(base_model.parameters.burgers),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("micromorphic and material-point Burgers vectors differ")
        self.conductivity_W_mK = float(conductivity_W_mK)
        self.parameter_provenance = parameter_provenance.strip()
        self.omega_chi = float(omega_chi)
        if power_partition_law is not None:
            if not hasattr(power_partition_law, "partition_rate"):
                raise TypeError("power_partition_law must provide partition_rate")
            if hasattr(power_partition_law, "validate"):
                power_partition_law.validate()
        self.power_partition_law = power_partition_law
        self.derivative_options = derivative_options or SpectralDerivativeOptions(
            slip_coordinate_scale=float(base_model.parameters.maximum_slip_increment)
        )
        self.derivative_options.validate()
        rotation = np.asarray(base_model.orientation, dtype=np.float64)
        self.directions_sample_reference = _frozen(
            (rotation @ np.asarray(base_model.systems.slip_directions).T).T,
            (N_SLIP, 3),
            "directions_sample_reference",
        )
        self.normals_sample_reference = _frozen(
            (rotation @ np.asarray(base_model.systems.slip_normals).T).T,
            (N_SLIP, 3),
            "normals_sample_reference",
        )
        self.K0 = _frozen(
            np.eye(3) * self.conductivity_W_mK, (3, 3), "K0"
        )
        switches = _canonical(base_model.switches)
        self.mechanism_switches = tuple(
            f"{name}={int(bool(value))}" for name, value in sorted(switches.items())
        )
        self.parameter_hash = _hash_payload(
            {"material": base_model.parameters, "micromorphic": micromorphic_parameters}
        )
        self.configuration_hash = _hash_payload(self._configuration_payload())
        dependency_paths: list[tuple[str, Path]] = [
            ("hcp_cp_gnd.spectral_export", Path(__file__)),
            ("hcp_cp_gnd.branch_audit", Path(__file__).with_name("branch_audit.py")),
            ("hcp_cp_gnd.sl3_chart", Path(__file__).with_name("sl3_chart.py")),
            ("hcp_cp_gnd.micromorphic", Path(__file__).with_name("micromorphic.py")),
            ("hcp_cp_gnd.local_coupling", Path(__file__).with_name("local_coupling.py")),
            ("hcp_cp_gnd.state_contract", Path(__file__).with_name("state_contract.py")),
        ]
        for source_object in (
            type(base_model),
            type(base_model.parameters),
            type(base_model.systems),
        ):
            source_path = inspect.getsourcefile(source_object)
            if source_path is not None:
                dependency_paths.append((source_object.__module__, Path(source_path)))
        unique_paths = tuple(
            dict.fromkeys((label, path.resolve()) for label, path in dependency_paths)
        )
        self.source_dependencies = tuple(
            (label, sha256(path.read_bytes()).hexdigest())
            for label, path in unique_paths
        )
        self.source_hash = _hash_payload(self.source_dependencies)

    def _configuration_payload(self) -> dict[str, Any]:
        """Return every mutable input that can change a continuous response."""

        return {
            "material": self.base_model.parameters,
            "micromorphic": self.micromorphic_parameters,
            "systems": self.base_model.systems,
            "orientation": np.asarray(self.base_model.orientation),
            "switches": self.base_model.switches,
            "conductivity_W_mK": self.conductivity_W_mK,
            "K0": np.asarray(self.K0),
            "omega_chi": self.omega_chi,
            "power_partition_law": self.power_partition_law,
            "derivative_options": self.derivative_options,
        }

    def _verify_configuration(self) -> None:
        current = _hash_payload(self._configuration_payload())
        if current != self.configuration_hash:
            raise RuntimeError(
                "continuous spectral model configuration changed after construction"
            )

    def _material_state(
        self,
        chart: SL3LocalChart,
        active: SpectralActiveState62,
        observer: SpectralObserverState,
        temperature_K: float,
    ) -> Any:
        prototype = self.base_model.initial_state()
        return replace(
            prototype,
            Fp=chart.matrix(active.theta_p),
            rho_mobile=active.rho_mobile_m2.copy(),
            rho_dipole=active.rho_dipole_m2.copy(),
            accumulated_slip=observer.Gamma_absolute.copy(),
            twin_fraction=np.zeros(6),
            temperature=float(temperature_K),
            plastic_work_density=float(observer.cp_work_density_J_m3),
            heat_density=float(observer.generated_heat_density_J_m3),
            stored_energy_density=float(observer.passive_storage_density_J_m3),
            time=float(observer.time_s),
        )

    @staticmethod
    def _same_branch(
        base: SpectralBranchAudit,
        probe: SpectralBranchAudit,
        floor: float,
    ) -> bool:
        if base.labels != probe.labels:
            return False
        for index, label in enumerate(base.labels):
            if (
                label.startswith("switch.")
                or label.startswith("admissibility.")
                or label.endswith(".dissipation")
            ):
                continue
            if base.categories[index] != probe.categories[index]:
                return False
            scale = max(
                float(base.distance_scales[index]),
                float(probe.distance_scales[index]),
            )
            if min(
                abs(float(base.signed_distances[index])),
                abs(float(probe.signed_distances[index])),
            ) / scale <= floor:
                return False
        return True

    @staticmethod
    def _limiting_branch(
        *audits: SpectralBranchAudit,
    ) -> tuple[float, str]:
        if not audits:
            raise ValueError("at least one branch audit is required")
        labels = audits[0].labels
        if any(audit.labels != labels for audit in audits[1:]):
            raise ValueError("branch audits use different label registries")
        minimum = np.inf
        limiting = ""
        for index, label in enumerate(labels):
            if (
                label.startswith("switch.")
                or label.startswith("admissibility.")
                or label.endswith(".dissipation")
            ):
                continue
            scale = max(float(audit.distance_scales[index]) for audit in audits)
            distance = min(
                abs(float(audit.signed_distances[index])) for audit in audits
            ) / scale
            if distance < minimum:
                minimum = distance
                limiting = label
        if not np.isfinite(minimum) or minimum <= 0.0 or not limiting:
            raise FloatingPointError("no positive constitutive branch margin was found")
        return float(minimum), limiting

    def _branch_audit(
        self,
        temperature_K: float,
        active: SpectralActiveState62,
        tau_hat: Array,
        slip_rate: Array,
        raw_forest: Array,
        resistance: Array,
        mean_free_path: Array,
        raw_exponent: Array,
    ) -> SpectralBranchAudit:
        p = self.base_model.parameters
        labels: list[str] = []
        categories: list[int] = []
        distances: list[float] = []
        scales: list[float] = []

        def append(label: str, category: int, distance: float, scale: float) -> None:
            if not np.isfinite(distance) or not np.isfinite(scale) or scale <= 0.0:
                raise FloatingPointError(f"invalid spectral branch audit entry {label}")
            labels.append(label)
            categories.append(int(category))
            distances.append(float(distance))
            scales.append(float(scale))

        append(
            "admissibility.temperature_positive",
            1,
            temperature_K,
            max(float(p.T_ref), 1.0),
        )
        for name, value in sorted(_canonical(self.base_model.switches).items()):
            append(f"switch.{name}", int(bool(value)), 1.0, 1.0)
        append("switch.first_paper_twin_fraction_zero", 1, 1.0, 1.0)

        effective = np.maximum(np.abs(tau_hat) - resistance, 0.0)
        d_check = p.dipole_min_distance_burgers * p.burgers
        tau_lower = 3.0 * p.reference_shear_modulus * p.burgers / (
            16.0 * np.pi * d_check
        )
        tau_upper = 3.0 * p.reference_shear_modulus * p.burgers / (
            16.0 * np.pi * mean_free_path
        )
        for alpha in range(N_SLIP):
            prefix = f"slip{alpha + 1:02d}"
            stress_scale = max(
                abs(float(tau_hat[alpha])),
                abs(float(resistance[alpha])),
                abs(float(p.tau_cut[alpha])),
                1.0,
            )
            forest_margin = float(raw_forest[alpha] - p.density_floor)
            append(
                f"{prefix}.forest_floor",
                int(forest_margin > 0.0),
                forest_margin,
                max(abs(float(raw_forest[alpha])), float(p.density_floor), 1.0),
            )
            yield_margin = float(abs(tau_hat[alpha]) - resistance[alpha])
            active_category = (
                int(np.sign(tau_hat[alpha])) if effective[alpha] > 0.0 else 0
            )
            append(f"{prefix}.yield", active_category, yield_margin, stress_scale)
            sign_distance = (
                abs(float(tau_hat[alpha]))
                if effective[alpha] > 0.0
                else max(-yield_margin, 1.0)
            )
            append(f"{prefix}.rate_sign", active_category, sign_distance, stress_scale)
            cap_margin = float(p.tau_cut[alpha] - effective[alpha])
            append(
                f"{prefix}.barrier_cap",
                int(cap_margin <= 0.0),
                cap_margin,
                max(float(p.tau_cut[alpha]), 1.0),
            )
            lower_clip_margin = float(raw_exponent[alpha] + 700.0)
            append(
                f"{prefix}.arrhenius_lower_clip",
                int(lower_clip_margin <= 0.0),
                lower_clip_margin,
                700.0,
            )
            append(
                f"switch.{prefix}.arrhenius_upper_clip",
                int(raw_exponent[alpha] >= 0.0),
                -float(raw_exponent[alpha]),
                max(abs(float(raw_exponent[alpha])), 1.0),
            )
            lower_margin = float(abs(tau_hat[alpha]) - tau_lower[alpha])
            upper_margin = float(tau_upper[alpha] - abs(tau_hat[alpha]))
            append(
                f"{prefix}.dhat_lower",
                int(lower_margin >= 0.0),
                lower_margin,
                max(float(tau_lower[alpha]), 1.0),
            )
            append(
                f"{prefix}.dhat_upper",
                int(upper_margin >= 0.0),
                upper_margin,
                max(float(tau_upper[alpha]), 1.0),
            )
            append(
                f"{prefix}.mobile_density_guard",
                1,
                float(active.rho_mobile_m2[alpha]),
                float(active.rho_mobile_m2[alpha]),
            )
            append(
                f"{prefix}.dipole_density_guard",
                1,
                float(active.rho_dipole_m2[alpha]),
                float(active.rho_dipole_m2[alpha]),
            )
            power = float(tau_hat[alpha] * slip_rate[alpha])
            append(
                f"{prefix}.dissipation",
                int(power >= 0.0),
                power,
                max(abs(power), 1.0),
            )
        return SpectralBranchAudit(
            labels=tuple(labels),
            categories=tuple(categories),
            signed_distances=np.asarray(distances),
            distance_scales=np.asarray(scales),
        )

    def _evaluate_raw(
        self,
        F_sample: Array,
        temperature_K: float,
        zeta: Array,
        grad_zeta_m_inv: Array,
        temperature_gradient_K_m: Array,
        direction_n: Array,
        chart: SL3LocalChart,
        active: SpectralActiveState62,
        observer: SpectralObserverState,
    ) -> _RawSpectralResponse:
        F = _finite_array(F_sample, (3, 3), "F_sample")
        micromorphic_slip = _finite_array(zeta, (N_SLIP,), "zeta")
        gradient = _finite_array(grad_zeta_m_inv, (N_SLIP, 3), "grad_zeta_m_inv")
        temperature_gradient = _finite_array(
            temperature_gradient_K_m, (3,), "temperature_gradient_K_m"
        )
        direction = _finite_array(direction_n, (3,), "direction_n")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("direction_n must be non-zero")
        direction = direction / norm
        if not np.isfinite(temperature_K):
            raise ValueError("temperature_K must be finite")
        if temperature_K <= 0.0:
            raise SpectralAdmissibilityError(
                "temperature_K must be positive",
                boundary_label="admissibility.temperature_positive",
            )
        if np.linalg.det(F) <= 0.0:
            raise SpectralAdmissibilityError(
                "F_sample must have positive determinant",
                boundary_label="admissibility.detF_positive",
            )

        material_state = self._material_state(chart, active, observer, temperature_K)
        Fp = np.asarray(material_state.Fp)
        mechanical = self.base_model.evaluate(F, material_state)
        micro: MicromorphicResponse = evaluate_micromorphic_state(
            active.gamma_signed,
            micromorphic_slip,
            gradient,
            self.directions_sample_reference,
            self.normals_sample_reference,
            self.micromorphic_parameters,
        )
        tau = np.asarray(mechanical.resolved_slip, dtype=np.float64)
        pi = np.asarray(micro.penalty_microstress_Pa, dtype=np.float64)
        tau_hat = tau + pi
        slip_rate, resistance, mean_free_path = self.base_model.slip_rates_from_resolved(
            tau_hat, material_state
        )
        slip_rate = np.asarray(slip_rate, dtype=np.float64)
        raw_forest = self.base_model.parameters.forest_interaction @ (
            active.rho_mobile_m2 + active.rho_dipole_m2
        )
        forest_density = np.maximum(raw_forest, self.base_model.parameters.density_floor)
        Lp = np.einsum("a,aij->ij", slip_rate, self.base_model.systems.slip_schmid)
        Fp_rate = Lp @ Fp

        p = self.base_model.parameters
        abs_rate = np.abs(slip_rate)
        source = (
            abs_rate / (p.burgers * mean_free_path)
            if self.base_model.switches.multiplication
            else np.zeros(N_SLIP)
        )
        d_check = p.dipole_min_distance_burgers * p.burgers
        raw_d_hat = np.full(N_SLIP, np.inf)
        nonzero = np.abs(tau_hat) > 0.0
        raw_d_hat[nonzero] = (
            3.0
            * p.reference_shear_modulus
            * p.burgers[nonzero]
            / (16.0 * np.pi * np.abs(tau_hat[nonzero]))
        )
        d_hat = np.minimum(np.maximum(raw_d_hat, d_check), mean_free_path)
        formation = (
            2.0 * np.maximum(d_hat - d_check, 0.0) / p.burgers * abs_rate
            if self.base_model.switches.dipole_formation
            else np.zeros(N_SLIP)
        )
        glide = (
            2.0 * d_check / p.burgers * abs_rate
            if self.base_model.switches.recovery_glide
            else np.zeros(N_SLIP)
        )
        kinetic_temperature = (
            float(temperature_K)
            if self.base_model.switches.thermal_softening
            else float(p.T_ref)
        )
        climb = (
            p.climb_frequency
            * np.exp(-p.climb_activation / (K_B * kinetic_temperature))
            if self.base_model.switches.recovery_climb
            else 0.0
        )
        rho_mobile_rate = source - (formation + glide) * active.rho_mobile_m2
        rho_dipole_rate = (
            formation * active.rho_mobile_m2
            - (glide + climb) * active.rho_dipole_m2
        )
        theta_rate = chart.coordinate_rate(active.theta_p, Fp_rate)
        active_rhs = np.concatenate(
            (theta_rate, rho_mobile_rate, rho_dipole_rate, slip_rate)
        )

        system_power = tau_hat * slip_rate
        power_scale = max(float(np.max(np.abs(system_power), initial=0.0)), 1.0)
        if np.any(system_power < -1.0e-12 * power_scale):
            raise FloatingPointError("a slip system produced negative generalized CP power")
        cp_power = float(np.sum(system_power))
        if cp_power < -1.0e-12 * power_scale:
            raise FloatingPointError("continuous generalized CP power is negative")
        if self.power_partition_law is None:
            beta = float(p.taylor_quinney)
            heat_source = beta * cp_power
            storage_rate = (1.0 - beta) * cp_power
        else:
            partition = self.power_partition_law.partition_rate(
                cp_power_W_m3=cp_power,
                rho_mobile_rate_m2_s=rho_mobile_rate,
                rho_dipole_rate_m2_s=rho_dipole_rate,
                burgers_m=p.burgers,
                shear_modulus_Pa=p.reference_shear_modulus,
            )
            beta = float(partition.beta_eff)
            heat_source = float(partition.heat_source_W_m3)
            storage_rate = float(partition.storage_rate_W_m3)
        material_dissipation = heat_source
        mechanical_power = float(np.sum(tau * slip_rate))
        microforce_power = float(np.sum(pi * slip_rate))
        power_identity_residual = cp_power - mechanical_power - microforce_power
        power_partition_residual = cp_power - heat_source - storage_rate

        effective = np.maximum(np.abs(tau_hat) - resistance, 0.0)
        normalized = np.minimum(effective / p.tau_cut, 1.0)
        barrier_shape = np.power(
            np.maximum(1.0 - np.power(normalized, p.p), 0.0), p.q
        )
        raw_exponent = -p.activation_energy * barrier_shape / (
            K_B * kinetic_temperature
        )
        branch_audit = self._branch_audit(
            temperature_K,
            active,
            tau_hat,
            slip_rate,
            raw_forest,
            resistance,
            mean_free_path,
            raw_exponent,
        )

        directional_gradient = np.einsum(
            "i,aibj,j->ab", direction, micro.gradient_hessian_Pa_m2, direction
        )
        heat_flux = -self.K0 @ temperature_gradient
        material_entropy = material_dissipation / temperature_K
        fourier_entropy = float(
            temperature_gradient @ self.K0 @ temperature_gradient / temperature_K**2
        )
        elastic_energy = float(self.base_model.elastic_energy_density(F, material_state))
        micromorphic_energy = float(micro.total_energy_J_m3)
        thermal_energy = float(
            p.mass_density * p.heat_capacity * (temperature_K - p.T_ref)
        )
        total_energy = (
            elastic_energy
            + micromorphic_energy
            + thermal_energy
            + observer.passive_storage_density_J_m3
        )
        return _RawSpectralResponse(
            first_piola_Pa=np.asarray(mechanical.first_piola),
            resolved_slip_Pa=tau,
            effective_resolved_slip_Pa=tau_hat,
            slip_rate_s_inv=slip_rate,
            plastic_velocity_gradient_s_inv=Lp,
            Fp_rate_s_inv=Fp_rate,
            rho_mobile_rate_m2_s=rho_mobile_rate,
            rho_dipole_rate_m2_s=rho_dipole_rate,
            gamma_signed_rate_s_inv=slip_rate,
            Gamma_absolute_rate_s_inv=abs_rate,
            active_rhs=active_rhs,
            cp_power_W_m3=cp_power,
            heat_source_W_m3=heat_source,
            storage_rate_W_m3=storage_rate,
            material_dissipation_W_m3=material_dissipation,
            mechanical_power_W_m3=mechanical_power,
            microforce_power_W_m3=microforce_power,
            power_identity_residual_W_m3=power_identity_residual,
            power_partition_residual_W_m3=power_partition_residual,
            penalty_microstress_Pa=pi,
            higher_order_stress_Pa_m=micro.higher_order_stress_Pa_m,
            gradient_hessian_Pa_m2=micro.gradient_hessian_Pa_m2,
            directional_gradient_Pa_m2=directional_gradient,
            heat_flux_W_m2=heat_flux,
            material_entropy_production_W_m3K=material_entropy,
            fourier_entropy_production_W_m3K=fourier_entropy,
            total_entropy_production_W_m3K=material_entropy + fourier_entropy,
            elastic_energy_J_m3=elastic_energy,
            micromorphic_energy_J_m3=micromorphic_energy,
            thermal_internal_energy_J_m3=thermal_energy,
            total_internal_energy_J_m3=total_energy,
            forest_density_m2=forest_density,
            slip_resistance_Pa=resistance,
            mean_free_path_m=mean_free_path,
            branch_audit=branch_audit,
        )

    def evaluate_active_state(
        self,
        F_sample: Array,
        temperature_K: float,
        zeta: Array,
        grad_zeta_m_inv: Array,
        temperature_gradient_K_m: Array,
        direction_n: Array,
        chart: SL3LocalChart,
        active: SpectralActiveState62,
        observer: SpectralObserverState,
    ) -> _RawSpectralResponse:
        """Evaluate the continuous 62-state closure at an explicit active state.

        This is the side-effect-free nonlinear counterpart of :meth:`export`.
        It deliberately keeps ``Gamma_absolute`` and the accumulated energy
        ledgers in ``observer``: those quantities are not coordinates of the
        registered 69-state dynamic generator.  Consequently this interface is
        a frozen-checkpoint nonlinear closure, not a replacement for the full
        material-point history integrator.
        """

        self._verify_configuration()
        if not isinstance(chart, SL3LocalChart):
            raise TypeError("chart must be an SL3LocalChart")
        if not isinstance(active, SpectralActiveState62):
            raise TypeError("active must be a SpectralActiveState62")
        if not isinstance(observer, SpectralObserverState):
            raise TypeError("observer must be a SpectralObserverState")
        response = self._evaluate_raw(
            F_sample,
            temperature_K,
            zeta,
            grad_zeta_m_inv,
            temperature_gradient_K_m,
            direction_n,
            chart,
            active,
            observer,
        )
        self._verify_configuration()
        return response

    def _differentiate_column(
        self,
        base: _RawSpectralResponse,
        evaluator: Callable[[float], _RawSpectralResponse],
        chain_scale: float,
        label: str,
    ) -> tuple[Array, str, float, float, float, str, str]:
        options = self.derivative_options
        step = options.relative_step
        base_vector = base.observable_vector()
        one_sided_candidate: (
            tuple[Array, str, float, float, float, str, str] | None
        ) = None
        constitutive_mismatch_seen = False
        for _ in range(options.maximum_halvings + 1):
            plus: _RawSpectralResponse | None = None
            minus: _RawSpectralResponse | None = None
            plus_outside_domain = False
            minus_outside_domain = False
            plus_boundary = ""
            minus_boundary = ""
            try:
                plus = evaluator(step)
            except SpectralAdmissibilityError as error:
                plus_outside_domain = True
                plus_boundary = error.boundary_label
            try:
                minus = evaluator(-step)
            except SpectralAdmissibilityError as error:
                minus_outside_domain = True
                minus_boundary = error.boundary_label
            if (
                plus is not None
                and minus is not None
                and base.branch_audit.compatible_with(
                    plus.branch_audit,
                    minus.branch_audit,
                    normalized_distance_floor=options.normalized_branch_distance_floor,
                )
            ):
                derivative = (
                    plus.observable_vector() - minus.observable_vector()
                ) / (2.0 * step * chain_scale)
                margin, limiting = self._limiting_branch(
                    base.branch_audit, plus.branch_audit, minus.branch_audit
                )
                return (
                    derivative,
                    "central",
                    step,
                    step * chain_scale,
                    margin,
                    limiting,
                    "",
                )

            plus_ok = plus is not None and self._same_branch(
                base.branch_audit,
                plus.branch_audit,
                options.normalized_branch_distance_floor,
            )
            minus_ok = minus is not None and self._same_branch(
                base.branch_audit,
                minus.branch_audit,
                options.normalized_branch_distance_floor,
            )
            if plus is not None and minus is not None and not (plus_ok and minus_ok):
                constitutive_mismatch_seen = True
            if plus_ok and minus_outside_domain:
                try:
                    outer = evaluator(2.0 * step)
                except SpectralAdmissibilityError:
                    outer = None
                if outer is not None and self._same_branch(
                    base.branch_audit,
                    outer.branch_audit,
                    options.normalized_branch_distance_floor,
                ):
                    derivative = (
                        -3.0 * base_vector
                        + 4.0 * plus.observable_vector()
                        - outer.observable_vector()
                    ) / (2.0 * step * chain_scale)
                    margin, limiting = self._limiting_branch(
                        base.branch_audit, plus.branch_audit, outer.branch_audit
                    )
                    one_sided_candidate = (
                        derivative,
                        "one-sided-plus",
                        step,
                        step * chain_scale,
                        margin,
                        limiting,
                        minus_boundary,
                    )
            if minus_ok and plus_outside_domain:
                try:
                    outer = evaluator(-2.0 * step)
                except SpectralAdmissibilityError:
                    outer = None
                if outer is not None and self._same_branch(
                    base.branch_audit,
                    outer.branch_audit,
                    options.normalized_branch_distance_floor,
                ):
                    derivative = (
                        3.0 * base_vector
                        - 4.0 * minus.observable_vector()
                        + outer.observable_vector()
                    ) / (2.0 * step * chain_scale)
                    margin, limiting = self._limiting_branch(
                        base.branch_audit, minus.branch_audit, outer.branch_audit
                    )
                    one_sided_candidate = (
                        derivative,
                        "one-sided-minus",
                        step,
                        step * chain_scale,
                        margin,
                        limiting,
                        plus_boundary,
                    )
            step *= 0.5
        if one_sided_candidate is not None and not constitutive_mismatch_seen:
            return one_sided_candidate
        raise SpectralNondifferentiableError(
            f"continuous spectral column {label} crosses or approaches a branch"
        )

    def _jacobian(
        self,
        base: _RawSpectralResponse,
        F: Array,
        temperature_K: float,
        zeta: Array,
        gradient: Array,
        temperature_gradient: Array,
        direction: Array,
        chart: SL3LocalChart,
        active: SpectralActiveState62,
        observer: SpectralObserverState,
    ) -> SpectralPointDerivatives:
        matrix = np.empty((N_SPECTRAL, N_SPECTRAL), dtype=np.float64)
        methods = [""] * N_SPECTRAL
        relative_steps = np.empty(N_SPECTRAL, dtype=np.float64)
        physical_steps = np.empty(N_SPECTRAL, dtype=np.float64)
        chain_scales = np.empty(N_SPECTRAL, dtype=np.float64)
        branch_distances = np.empty(N_SPECTRAL, dtype=np.float64)
        limiting_labels = [""] * N_SPECTRAL
        boundary_labels = [""] * N_SPECTRAL

        def assign(
            column: int,
            result: tuple[Array, str, float, float, float, str, str],
            chain_scale: float,
        ) -> None:
            derivative, method, relative, physical, margin, limiting, boundary = result
            matrix[:, column] = derivative
            methods[column] = method
            relative_steps[column] = relative
            physical_steps[column] = physical
            chain_scales[column] = chain_scale
            branch_distances[column] = margin
            limiting_labels[column] = limiting
            boundary_labels[column] = boundary

        def raw(
            F_probe: Array = F,
            T_probe: float = temperature_K,
            zeta_probe: Array = zeta,
            active_probe: SpectralActiveState62 = active,
        ) -> _RawSpectralResponse:
            return self._evaluate_raw(
                F_probe,
                T_probe,
                zeta_probe,
                gradient,
                temperature_gradient,
                direction,
                chart,
                active_probe,
                observer,
            )

        for column in range(9):
            scale = max(1.0, abs(float(F.reshape(-1)[column])))

            def evaluate(delta: float, column: int = column, scale: float = scale):
                value = F.copy()
                value.reshape(-1)[column] += delta * scale
                return raw(F_probe=value)

            assign(
                column,
                self._differentiate_column(base, evaluate, scale, f"F[{column}]"),
                scale,
            )

        temperature_scale = max(abs(float(temperature_K)), float(self.base_model.parameters.T_ref))
        assign(
            9,
            self._differentiate_column(
                base,
                lambda delta: raw(T_probe=temperature_K + delta * temperature_scale),
                temperature_scale,
                "T",
            ),
            temperature_scale,
        )

        slip_scale = self.derivative_options.slip_coordinate_scale
        for alpha in range(N_SLIP):

            def evaluate(delta: float, alpha: int = alpha):
                value = zeta.copy()
                value[alpha] += delta * slip_scale
                return raw(zeta_probe=value)

            assign(
                10 + alpha,
                self._differentiate_column(
                    base, evaluate, slip_scale, f"zeta[{alpha}]"
                ),
                slip_scale,
            )

        for coordinate in range(N_SL3):

            def evaluate(delta: float, coordinate: int = coordinate):
                value = active.pack()
                value[coordinate] += delta
                return raw(active_probe=SpectralActiveState62.unpack(value))

            assign(
                28 + coordinate,
                self._differentiate_column(
                    base, evaluate, 1.0, f"theta[{coordinate}]"
                ),
                1.0,
            )

        for alpha in range(N_SLIP):
            density = float(active.rho_mobile_m2[alpha])

            def evaluate(delta: float, alpha: int = alpha):
                value = active.pack()
                value[8 + alpha] *= np.exp(delta)
                return raw(active_probe=SpectralActiveState62.unpack(value))

            assign(
                36 + alpha,
                self._differentiate_column(
                    base, evaluate, density, f"log_rho_mobile[{alpha}]"
                ),
                density,
            )

        for alpha in range(N_SLIP):
            density = float(active.rho_dipole_m2[alpha])

            def evaluate(delta: float, alpha: int = alpha):
                value = active.pack()
                value[26 + alpha] *= np.exp(delta)
                return raw(active_probe=SpectralActiveState62.unpack(value))

            assign(
                54 + alpha,
                self._differentiate_column(
                    base, evaluate, density, f"log_rho_dipole[{alpha}]"
                ),
                density,
            )

        for alpha in range(N_SLIP):

            def evaluate(delta: float, alpha: int = alpha):
                value = active.pack()
                value[44 + alpha] += delta * slip_scale
                return raw(active_probe=SpectralActiveState62.unpack(value))

            assign(
                72 + alpha,
                self._differentiate_column(
                    base, evaluate, slip_scale, f"gamma_signed[{alpha}]"
                ),
                slip_scale,
            )

        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError("continuous spectral Jacobian is non-finite")
        one_sided = sum(method != "central" for method in methods)
        return SpectralPointDerivatives(
            packed=matrix,
            stencil="BRANCH_PRESERVING_CENTRAL_OR_SECOND_ORDER_ONE_SIDED_FD",
            central_columns=N_SPECTRAL - one_sided,
            one_sided_columns=one_sided,
            minimum_normalized_branch_distance=float(np.min(branch_distances)),
            column_methods=tuple(methods),
            final_relative_steps=relative_steps,
            final_physical_steps=physical_steps,
            chain_scales=chain_scales,
            minimum_normalized_branch_distances=branch_distances,
            limiting_branch_labels=tuple(limiting_labels),
            admissibility_boundary_labels=tuple(boundary_labels),
        )

    def export(
        self,
        F_sample: Array,
        temperature_K: float,
        zeta: Array,
        grad_zeta_m_inv: Array,
        state92: LocalState92,
        *,
        direction_n: Array | None = None,
        temperature_gradient_K_m: Array | None = None,
        compute_jacobian: bool = True,
    ) -> SpectralPointExport:
        """Export a continuous response without modifying the storage state."""

        self._verify_configuration()
        if not isinstance(state92, LocalState92):
            raise TypeError("state92 must implement HCP_CP_LOCAL_STATE_92_V1")
        if np.any(state92.twin_fraction != 0.0):
            raise ValueError("the 62-state export requires zero stored twin fractions")
        if not np.isclose(
            state92.temperature_K, temperature_K, rtol=2.0e-13, atol=2.0e-13
        ):
            raise ValueError("temperature_K must match the frozen storage checkpoint")
        fingerprint = _state_fingerprint(state92)
        F = _finite_array(F_sample, (3, 3), "F_sample").copy()
        zeta_value = _finite_array(zeta, (N_SLIP,), "zeta").copy()
        gradient = _finite_array(
            grad_zeta_m_inv, (N_SLIP, 3), "grad_zeta_m_inv"
        ).copy()
        direction = (
            np.array([1.0, 0.0, 0.0])
            if direction_n is None
            else _finite_array(direction_n, (3,), "direction_n").copy()
        )
        temperature_gradient = (
            np.zeros(3)
            if temperature_gradient_K_m is None
            else _finite_array(
                temperature_gradient_K_m, (3,), "temperature_gradient_K_m"
            ).copy()
        )
        chart = SL3LocalChart(
            state92.Fp,
            determinant_tolerance=self.base_model.parameters.determinant_tolerance,
        )
        active = SpectralActiveState62.from_storage(state92, chart)
        observer = SpectralObserverState.from_storage(state92)
        if np.any(np.abs(active.gamma_signed) > observer.Gamma_absolute + 1.0e-10):
            raise ValueError("frozen signed slip exceeds its passive total variation")
        raw = self._evaluate_raw(
            F,
            float(temperature_K),
            zeta_value,
            gradient,
            temperature_gradient,
            direction,
            chart,
            active,
            observer,
        )
        derivatives = (
            self._jacobian(
                raw,
                F,
                float(temperature_K),
                zeta_value,
                gradient,
                temperature_gradient,
                direction,
                chart,
                active,
                observer,
            )
            if compute_jacobian
            else None
        )
        case_hash = _hash_payload(
            {
                "F": F,
                "T": temperature_K,
                "zeta": zeta_value,
                "grad_zeta": gradient,
                "temperature_gradient": temperature_gradient,
                "direction": direction / np.linalg.norm(direction),
                "state92": state92.pack(),
                "parameter_hash": self.parameter_hash,
                "configuration_hash": self.configuration_hash,
            }
        )
        metadata = SpectralExportMetadata(
            schema=SPECTRAL_EXPORT_SCHEMA,
            state_schema=SPECTRAL_STATE_SCHEMA,
            track="CONTINUOUS_CONSTITUTIVE_REFERENCE_DYN_QS_NEUTRAL",
            parameter_status=str(self.base_model.parameters.status),
            parameter_provenance=self.parameter_provenance,
            source_hash=self.source_hash,
            source_dependencies=self.source_dependencies,
            parameter_hash=self.parameter_hash,
            configuration_hash=self.configuration_hash,
            case_hash=case_hash,
            raw_microforce_equation=True,
            omega_chi=self.omega_chi,
            mechanism_switches=self.mechanism_switches,
            input_coordinate_labels=INPUT_COORDINATE_LABELS,
            output_coordinate_labels=OUTPUT_COORDINATE_LABELS,
            input_layout=_INPUT_LAYOUT,
            output_layout=_OUTPUT_LAYOUT,
            active_state_layout=_ACTIVE_STATE_LAYOUT,
            chart_schema=SPECTRAL_CHART_SCHEMA,
            chart_anchor_Fp=chart.anchor_Fp,
            sl3_generators=chart.basis,
            sl3_generator_hash=_hash_payload(chart.basis),
            tensor_flattening="ROW_MAJOR_C_ORDER",
            declared_validity=(
                "FROZEN_TIME_LOCAL_CONTINUOUS_SLIP_ONLY__TWINS_OFF__"
                "POSITIVE_TEMPERATURE_AND_DETF__SMOOTH_CONSTITUTIVE_BRANCH"
            ),
        )
        result = SpectralPointExport(
            **{
                name: getattr(raw, name)
                for name in _RawSpectralResponse.__dataclass_fields__
                if name != "branch_audit"
            },
            reference_density_kg_m3=float(self.base_model.parameters.mass_density),
            specific_heat_J_kgK=float(self.base_model.parameters.heat_capacity),
            conductivity_W_mK=self.K0,
            Gamma_absolute=observer.Gamma_absolute,
            cp_work_density_J_m3=observer.cp_work_density_J_m3,
            generated_heat_density_J_m3=observer.generated_heat_density_J_m3,
            passive_storage_density_J_m3=observer.passive_storage_density_J_m3,
            time_s=observer.time_s,
            branch_audit=raw.branch_audit,
            derivatives=derivatives,
            metadata=metadata,
        )
        if _state_fingerprint(state92) != fingerprint:
            raise RuntimeError("continuous spectral export mutated the storage checkpoint")
        self._verify_configuration()
        return result


def spectral_point_export(
    base_model: Any,
    micromorphic_parameters: MicromorphicParameters,
    F_sample: Array,
    temperature_K: float,
    zeta: Array,
    grad_zeta_m_inv: Array,
    state92: LocalState92,
    *,
    conductivity_W_mK: float,
    parameter_provenance: str,
    omega_chi: float = 1.0,
    direction_n: Array | None = None,
    temperature_gradient_K_m: Array | None = None,
    compute_jacobian: bool = True,
) -> SpectralPointExport:
    """Functional convenience wrapper for :class:`ContinuousSpectralPointModel`."""

    model = ContinuousSpectralPointModel(
        base_model,
        micromorphic_parameters,
        conductivity_W_mK=conductivity_W_mK,
        parameter_provenance=parameter_provenance,
        omega_chi=omega_chi,
    )
    return model.export(
        F_sample,
        temperature_K,
        zeta,
        grad_zeta_m_inv,
        state92,
        direction_n=direction_n,
        temperature_gradient_K_m=temperature_gradient_K_m,
        compute_jacobian=compute_jacobian,
    )


__all__ = [
    "INPUT_COORDINATE_LABELS",
    "N_ACTIVE",
    "N_FIELD",
    "N_SPECTRAL",
    "OUTPUT_COORDINATE_LABELS",
    "SPECTRAL_CHART_SCHEMA",
    "SPECTRAL_EXPORT_SCHEMA",
    "SPECTRAL_STATE_SCHEMA",
    "ContinuousSpectralPointModel",
    "CoordinateBlock",
    "QuantityContract",
    "SpectralActiveState62",
    "SpectralDerivativeOptions",
    "SpectralExportMetadata",
    "SpectralObserverState",
    "SpectralPointDerivatives",
    "SpectralPointExport",
    "spectral_point_export",
]
