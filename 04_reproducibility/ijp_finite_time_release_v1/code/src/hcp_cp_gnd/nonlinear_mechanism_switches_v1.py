"""Registered nonlinear mechanism interventions for the slip-only CP-Ti track."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .cp_ti_material_v1 import simple_shear


@dataclass(frozen=True)
class NonlinearMechanismSwitchV1:
    name: str
    switch_field: str
    physical_role: str
    direct_residual_rows: tuple[str, ...]
    base_history_pathways: tuple[str, ...]
    supported_in_periodic69: bool = True


MECHANISM_SWITCH_MAP_V1: tuple[NonlinearMechanismSwitchV1, ...] = (
    NonlinearMechanismSwitchV1(
        "slip_kinetics",
        "slip",
        "thermally activated signed slip flow",
        ("theta_p", "rho_mobile", "rho_dipole", "gamma_signed", "heat"),
        ("Fp", "density", "temperature", "work_ledgers"),
    ),
    NonlinearMechanismSwitchV1(
        "dislocation_multiplication",
        "multiplication",
        "mobile-dislocation production by mean-free-path-limited glide",
        ("rho_mobile",),
        ("density", "hardening", "temperature_via_future_slip"),
    ),
    NonlinearMechanismSwitchV1(
        "dipole_formation",
        "dipole_formation",
        "mobile-to-dipole transfer controlled by capture distance",
        ("rho_mobile", "rho_dipole"),
        ("density", "hardening", "temperature_via_future_slip"),
    ),
    NonlinearMechanismSwitchV1(
        "glide_recovery",
        "recovery_glide",
        "glide-mediated mobile and dipole annihilation",
        ("rho_mobile", "rho_dipole"),
        ("density", "hardening", "temperature_via_future_slip"),
    ),
    NonlinearMechanismSwitchV1(
        "climb_recovery",
        "recovery_climb",
        "thermally activated dipole annihilation by climb",
        ("rho_dipole",),
        ("density", "hardening", "temperature_via_future_slip"),
    ),
    NonlinearMechanismSwitchV1(
        "thermal_kinetic_feedback",
        "thermal_softening",
        "current temperature in slip and climb activation",
        ("theta_p", "rho_mobile", "rho_dipole", "gamma_signed", "heat"),
        ("Fp", "density", "temperature", "work_ledgers"),
    ),
    NonlinearMechanismSwitchV1(
        "homogeneous_adiabatic_deposition",
        "adiabatic_heating",
        "deposit generated heat into the homogeneous material-point temperature",
        (),
        ("temperature", "temperature_dependent_kinetics"),
    ),
    NonlinearMechanismSwitchV1(
        "deformation_twinning",
        "twinning",
        "polar extension-twin kinetics and reorientation variables",
        (),
        ("Fp", "twin_fraction", "hardening", "temperature"),
        supported_in_periodic69=False,
    ),
)


def mechanism_switch(name: str) -> NonlinearMechanismSwitchV1:
    matches = [item for item in MECHANISM_SWITCH_MAP_V1 if item.name == name]
    if len(matches) != 1:
        raise KeyError(f"unknown nonlinear mechanism switch: {name}")
    return matches[0]


def model_with_mechanism_disabled(base_model: Any, name: str) -> Any:
    """Clone a material point with exactly one registered switch disabled."""

    contract = mechanism_switch(name)
    if not contract.supported_in_periodic69:
        raise ValueError(
            f"{name} requires a state expansion beyond the slip-only 69-state contract"
        )
    switches = replace(base_model.switches, **{contract.switch_field: False})
    arguments = (
        base_model.systems,
        base_model.parameters,
        np.asarray(base_model.orientation, dtype=float).copy(),
        switches,
    )
    if hasattr(base_model, "partition_law"):
        return type(base_model)(*arguments, partition_law=base_model.partition_law)
    return type(base_model)(*arguments)


def evolve_model_to_shears(
    model: Any,
    shears: Any,
    *,
    shear_rate_s_inv: float,
    shear_increment: float = 2.0e-3,
) -> list[Any]:
    """Re-evolve an intervened nonlinear base history to exact shear checkpoints."""

    targets = np.asarray(shears, dtype=float)
    rate = float(shear_rate_s_inv)
    increment = float(shear_increment)
    if (
        targets.ndim != 1
        or targets.size == 0
        or np.any(~np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(np.diff(targets) <= 0.0)
        or not np.isfinite(rate)
        or rate <= 0.0
        or not np.isfinite(increment)
        or increment <= 0.0
    ):
        raise ValueError("invalid nonlinear re-evolution controls")
    state = model.initial_state()
    shear = 0.0
    snapshots: list[Any] = []
    for target in targets:
        while shear < target - 4.0 * np.finfo(float).eps:
            next_shear = min(shear + increment, float(target))
            state = model.advance(
                simple_shear(shear),
                simple_shear(next_shear),
                state,
                (next_shear - shear) / rate,
            ).state
            shear = next_shear
        snapshots.append(state)
    return snapshots


def intervention_effect_decomposition(
    baseline_on: Any,
    fixed_base_off: Any,
    reevolved_base_off: Any,
) -> dict[str, np.ndarray]:
    """Return total = direct + base-mediated intervention effects.

    Effects use the orientation ``baseline_on - mechanism_off``.  ``direct``
    changes the residual law at the baseline state; ``base_mediated`` changes
    only the state/history supplied to the disabled law.
    """

    on = np.asarray(baseline_on)
    fixed = np.asarray(fixed_base_off)
    reevolved = np.asarray(reevolved_base_off)
    if on.shape != fixed.shape or on.shape != reevolved.shape:
        raise ValueError("intervention observables must have identical shapes")
    if not np.all(np.isfinite(on)) or not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(reevolved)):
        raise ValueError("intervention observables must be finite")
    direct = on - fixed
    mediated = fixed - reevolved
    total = on - reevolved
    return {
        "direct": direct,
        "base_mediated": mediated,
        "total": total,
        "closure_residual": total - direct - mediated,
    }


def switch_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "switch_field": item.switch_field,
            "physical_role": item.physical_role,
            "direct_residual_rows": list(item.direct_residual_rows),
            "base_history_pathways": list(item.base_history_pathways),
            "supported_in_periodic69": item.supported_in_periodic69,
        }
        for item in MECHANISM_SWITCH_MAP_V1
    ]


__all__ = [
    "MECHANISM_SWITCH_MAP_V1",
    "NonlinearMechanismSwitchV1",
    "evolve_model_to_shears",
    "intervention_effect_decomposition",
    "mechanism_switch",
    "model_with_mechanism_disabled",
    "switch_manifest",
]
