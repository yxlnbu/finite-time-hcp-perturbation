"""Literature-constrained commercially-pure titanium research card and history.

The module converts the JSON research card into the existing, independently
tested HCP material-point data structure without changing the frozen v0.1
loader or its verification seed.  The card is complete in the computational
sense (every model parameter has a value and unit) but is not a certificate
for a particular plate, heat, chemistry, or EBSD texture.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


CARD_SCHEMA = "CP_TI_GRADE1_DYNAMIC_HCP_MATERIAL_CARD_V1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_card(path: str | Path | None = None) -> dict[str, Any]:
    source = (
        _project_root() / "config/cp_ti_grade1_dynamic_hcp_v1.json"
        if path is None
        else Path(path)
    )
    card = json.loads(source.read_text(encoding="utf-8"))
    if card.get("schema") != CARD_SCHEMA:
        raise ValueError("unexpected CP-Ti material-card schema")
    return card


def heat_capacity_J_kgK(temperature_K: float, card: dict[str, Any] | None = None) -> float:
    """Evaluate the NIST Shomate alpha-Ti heat capacity in J kg-1 K-1."""

    T = float(temperature_K)
    if not np.isfinite(T) or not 298.0 <= T <= 700.0:
        raise ValueError("the frozen alpha-Ti Shomate fit is admitted on 298-700 K")
    raw = (card or load_card())["thermal"]["heat_capacity_model"]
    t = T / 1000.0
    cp_molar_cal = raw["A"] + raw["B"] * t + raw["C"] * t**2 + raw["D"] * t**3 + raw["E"] / t**2
    cp_molar = cp_molar_cal * raw["calorie_to_joule"]
    return float(cp_molar / raw["molar_mass_kg_mol"])


def build_material_objects(card: dict[str, Any] | None = None):
    """Return systems, parameters, and a basal-texture representative point."""

    root = _project_root()
    # The public archive vendors the frozen v0.1 kernel as ``src/hcp_cp`` and
    # the exact verification seed as ``config/verification_seed.yaml``.  The
    # historical sibling checkout remains a development fallback only.
    vendored_seed = root / "config/verification_seed.yaml"
    sibling = root.parent / "HCP_CP_v0.1"
    if vendored_seed.is_file():
        seed_path = vendored_seed
    else:
        if str(sibling / "src") not in sys.path:
            sys.path.insert(0, str(sibling / "src"))
        seed_path = sibling / "config/verification_seed.yaml"
    from hcp_cp.crystal import build_hcp_systems
    from hcp_cp.model import HCPMaterialPoint, MechanismSwitches, orientation_from_bunge
    from hcp_cp.parameters import load_material_parameters

    data = card or load_card()
    systems = build_hcp_systems()
    seed = load_material_parameters(seed_path, systems)
    thermal = data["thermal"]
    load = data["representative_loading"]
    parameters = replace(
        seed,
        status=data["status"],
        mass_density=float(thermal["mass_density_kg_m3"]),
        heat_capacity=heat_capacity_J_kgK(298.15, data),
        taylor_quinney=float(thermal["taylor_quinney"]),
    )
    orientation = orientation_from_bunge(*load["orientation_bunge_deg"])
    model = HCPMaterialPoint(
        systems,
        parameters,
        orientation,
        MechanismSwitches(twinning=False),
    )
    return systems, parameters, model


def simple_shear(amount: float) -> np.ndarray:
    F = np.eye(3)
    F[0, 1] = float(amount)
    return F


def run_representative_history(
    card: dict[str, Any] | None = None,
    *,
    shear_increment: float = 2.0e-3,
) -> tuple[list[dict[str, Any]], Any, Any, Any]:
    """Integrate the homogeneous base history and return registered checkpoints."""

    data = card or load_card()
    systems, parameters, model = build_material_objects(data)
    loading = data["representative_loading"]
    rate = float(loading["macroscopic_shear_rate_s"])
    final = float(loading["final_shear"])
    targets = [float(value) for value in loading["spectral_checkpoint_shears"]]
    if shear_increment <= 0.0 or final <= 0.0 or rate <= 0.0:
        raise ValueError("history controls must be positive")
    state = model.initial_state()
    gamma = 0.0
    output: list[dict[str, Any]] = []
    target_index = 0
    while gamma < final - 4.0 * np.finfo(float).eps:
        next_gamma = min(gamma + shear_increment, final)
        dt = (next_gamma - gamma) / rate
        result = model.advance(simple_shear(gamma), simple_shear(next_gamma), state, dt)
        state = result.state
        gamma = next_gamma
        while target_index < len(targets) and gamma + 1.0e-13 >= targets[target_index]:
            target = targets[target_index]
            response = model.evaluate(simple_shear(target), state)
            output.append(
                {
                    "shear": target,
                    "time_s": target / rate,
                    "temperature_K": float(state.temperature),
                    "cauchy_shear_Pa": float(response.cauchy[0, 1]),
                    "first_piola_shear_Pa": float(response.first_piola[0, 1]),
                    "plastic_work_J_m3": float(state.plastic_work_density),
                    "generated_heat_J_m3": float(state.heat_density),
                    "stored_energy_J_m3": float(state.stored_energy_density),
                    "maximum_slip_rate_s": float(np.max(np.abs(response.slip_rate))),
                    "dominant_slip_index": int(np.argmax(np.abs(response.slip_rate))),
                    "dominant_slip_family": systems.slip_families[int(np.argmax(np.abs(response.slip_rate)))],
                    "accumulated_slip_sum": float(np.sum(state.accumulated_slip)),
                    "substeps_last_increment": int(result.substeps),
                    "energy_balance_relative_error": float(result.energy_balance_relative_error),
                    "work_partition_relative_error": float(result.work_partition_relative_error),
                }
            )
            target_index += 1
    return output, state, model, parameters


def run_representative_checkpoint_states(
    card: dict[str, Any] | None = None,
    *,
    shear_increment: float = 2.0e-3,
) -> tuple[list[dict[str, Any]], list[Any], Any, Any]:
    """Integrate once and retain exact material states at spectral checkpoints.

    Unlike the legacy summary helper, every integration step is clipped to the
    next requested checkpoint.  The returned state and its reported shear are
    therefore coincident even when a checkpoint is not an integer multiple of
    ``shear_increment``.  The state objects are immutable material-point
    snapshots produced by the accepted v0.1 kernel.
    """

    data = card or load_card()
    systems, parameters, model = build_material_objects(data)
    loading = data["representative_loading"]
    rate = float(loading["macroscopic_shear_rate_s"])
    final = float(loading["final_shear"])
    targets = np.asarray(loading["spectral_checkpoint_shears"], dtype=float)
    if (
        not np.isfinite(shear_increment)
        or shear_increment <= 0.0
        or not np.isfinite(rate)
        or rate <= 0.0
        or not np.isfinite(final)
        or final <= 0.0
        or targets.ndim != 1
        or targets.size < 2
        or np.any(~np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(np.diff(targets) <= 0.0)
        or targets[-1] > final + 1.0e-13
    ):
        raise ValueError("checkpoint history controls are invalid")

    state = model.initial_state()
    gamma = 0.0
    records: list[dict[str, Any]] = []
    snapshots: list[Any] = []
    for target in targets:
        result = None
        while gamma < target - 4.0 * np.finfo(float).eps:
            next_gamma = min(gamma + shear_increment, float(target))
            dt = (next_gamma - gamma) / rate
            result = model.advance(simple_shear(gamma), simple_shear(next_gamma), state, dt)
            state = result.state
            gamma = next_gamma
        response = model.evaluate(simple_shear(float(target)), state)
        dominant = int(np.argmax(np.abs(response.slip_rate)))
        records.append(
            {
                "shear": float(target),
                "time_s": float(target / rate),
                "temperature_K": float(state.temperature),
                "cauchy_shear_Pa": float(response.cauchy[0, 1]),
                "first_piola_shear_Pa": float(response.first_piola[0, 1]),
                "plastic_work_J_m3": float(state.plastic_work_density),
                "generated_heat_J_m3": float(state.heat_density),
                "stored_energy_J_m3": float(state.stored_energy_density),
                "maximum_slip_rate_s": float(np.max(np.abs(response.slip_rate))),
                "dominant_slip_index": dominant,
                "dominant_slip_family": systems.slip_families[dominant],
                "accumulated_slip_sum": float(np.sum(state.accumulated_slip)),
                "substeps_last_increment": (
                    0 if result is None else int(result.substeps)
                ),
                "energy_balance_relative_error": (
                    0.0 if result is None else float(result.energy_balance_relative_error)
                ),
                "work_partition_relative_error": (
                    0.0 if result is None else float(result.work_partition_relative_error)
                ),
            }
        )
        snapshots.append(state)
    return records, snapshots, model, parameters


def local_state92_from_material_state(state: Any, model: Any, F_sample: np.ndarray):
    """Map a monotonic material history to the signed v0.2 storage contract."""

    from hcp_cp_gnd.state_contract import LocalState92

    response = model.evaluate(F_sample, state)
    signs = np.sign(response.slip_rate)
    signed = signs * np.asarray(state.accumulated_slip)
    return LocalState92(
        Fp=np.asarray(state.Fp),
        rho_mobile_m2=np.asarray(state.rho_mobile),
        rho_dipole_m2=np.asarray(state.rho_dipole),
        gamma_signed=signed,
        Gamma_absolute=np.asarray(state.accumulated_slip),
        twin_fraction=np.zeros(6),
        temperature_K=float(state.temperature),
        cp_work_density_J_m3=float(state.plastic_work_density),
        generated_heat_density_J_m3=float(state.heat_density),
        stored_energy_density_J_m3=float(state.stored_energy_density),
        time_s=float(state.time),
    )


__all__ = [
    "CARD_SCHEMA",
    "build_material_objects",
    "heat_capacity_J_kgK",
    "load_card",
    "local_state92_from_material_state",
    "run_representative_checkpoint_states",
    "run_representative_history",
    "simple_shear",
]
