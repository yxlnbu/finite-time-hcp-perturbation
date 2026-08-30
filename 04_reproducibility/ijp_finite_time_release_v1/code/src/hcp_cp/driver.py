"""Reproducible material-point driver and neutral history writer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .crystal import build_hcp_systems
from .model import HCPMaterialPoint, MechanismSwitches, orientation_from_bunge
from .parameters import load_material_parameters


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_header(n_slip: int, n_twin: int) -> list[str]:
    scalar = [
        "increment",
        "time_s",
        "dt_s",
        "gamma",
        "temperature_K",
        "sigma12_Pa",
        "P12_Pa",
        "plastic_dissipation_W_per_m3",
        "plastic_work_J_per_m3",
        "heat_J_per_m3",
        "stored_energy_J_per_m3",
        "elastic_energy_J_per_m3",
        "external_work_J_per_m3",
        "mechanical_balance_residual_J_per_m3",
        "det_Fp",
        "total_accumulated_slip",
        "total_twin_fraction",
        "substeps",
        "step_heat_balance_relative_error",
        "step_partition_relative_error",
    ]
    tensor = [f"Fp{i+1}{j+1}" for i in range(3) for j in range(3)]
    slip = []
    for stem in (
        "tau_slip_Pa",
        "slip_rate_per_s",
        "slip_resistance_Pa",
        "rho_mobile_per_m2",
        "rho_forest_derived_per_m2",
        "rho_dipole_per_m2",
        "accumulated_slip",
    ):
        slip.extend(f"{stem}::{index+1}" for index in range(n_slip))
    twin = []
    for stem in ("tau_twin_Pa", "twin_rate_per_s", "twin_fraction"):
        twin.extend(f"{stem}::{index+1}" for index in range(n_twin))
    return scalar + tensor + slip + twin


def run_simple_shear(
    material_path: str | Path,
    output_directory: str | Path,
    *,
    case_id: str,
    increments: int = 200,
    final_shear: float = 0.1,
    shear_rate: float = 1000.0,
    orientation_bunge_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    switches: MechanismSwitches | None = None,
) -> dict[str, Any]:
    if increments <= 0 or final_shear <= 0.0 or shear_rate <= 0.0:
        raise ValueError("increments, final shear and shear rate must be positive")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    material_path = Path(material_path).resolve()

    systems = build_hcp_systems()
    parameters = load_material_parameters(material_path, systems)
    switches = switches or MechanismSwitches()
    orientation = orientation_from_bunge(*orientation_bunge_deg)
    model = HCPMaterialPoint(systems, parameters, orientation, switches)
    state = model.initial_state()
    state.assert_physical(parameters)

    total_time = final_shear / shear_rate
    dt = total_time / increments
    F_previous = np.eye(3)
    response_previous = model.evaluate(F_previous, state)
    initial_elastic_energy = model.elastic_energy_density(F_previous, state)
    external_work = 0.0
    rows: list[dict[str, float | int]] = []
    maximum_substeps = 0
    minimum_density = float("inf")
    maximum_heat_error = 0.0
    maximum_partition_error = 0.0

    for increment in range(1, increments + 1):
        gamma = final_shear * increment / increments
        F_current = np.eye(3)
        F_current[0, 1] = gamma
        result = model.advance(F_previous, F_current, state, dt)
        state = result.state
        response = result.response
        dF = F_current - F_previous
        external_work += 0.5 * float(
            np.sum((response_previous.first_piola + response.first_piola) * dF)
        )
        elastic_energy = model.elastic_energy_density(F_current, state)
        mechanical_residual = (
            external_work
            - (elastic_energy - initial_elastic_energy)
            - state.plastic_work_density
        )
        row: dict[str, float | int] = {
            "increment": increment,
            "time_s": state.time,
            "dt_s": dt,
            "gamma": gamma,
            "temperature_K": state.temperature,
            "sigma12_Pa": response.cauchy[0, 1],
            "P12_Pa": response.first_piola[0, 1],
            "plastic_dissipation_W_per_m3": response.plastic_dissipation,
            "plastic_work_J_per_m3": state.plastic_work_density,
            "heat_J_per_m3": state.heat_density,
            "stored_energy_J_per_m3": state.stored_energy_density,
            "elastic_energy_J_per_m3": elastic_energy,
            "external_work_J_per_m3": external_work,
            "mechanical_balance_residual_J_per_m3": mechanical_residual,
            "det_Fp": float(np.linalg.det(state.Fp)),
            "total_accumulated_slip": float(np.sum(state.accumulated_slip)),
            "total_twin_fraction": float(np.sum(state.twin_fraction)),
            "substeps": result.substeps,
            "step_heat_balance_relative_error": result.energy_balance_relative_error,
            "step_partition_relative_error": result.work_partition_relative_error,
        }
        row.update(
            {f"Fp{i+1}{j+1}": state.Fp[i, j] for i in range(3) for j in range(3)}
        )
        for stem, values in (
            ("tau_slip_Pa", response.resolved_slip),
            ("slip_rate_per_s", response.slip_rate),
            ("slip_resistance_Pa", response.slip_resistance),
            ("rho_mobile_per_m2", state.rho_mobile),
            ("rho_forest_derived_per_m2", response.forest_density),
            ("rho_dipole_per_m2", state.rho_dipole),
            ("accumulated_slip", state.accumulated_slip),
        ):
            row.update({f"{stem}::{index+1}": value for index, value in enumerate(values)})
        for stem, values in (
            ("tau_twin_Pa", response.resolved_twin),
            ("twin_rate_per_s", response.twin_rate),
            ("twin_fraction", state.twin_fraction),
        ):
            row.update({f"{stem}::{index+1}": value for index, value in enumerate(values)})
        rows.append(row)
        maximum_substeps = max(maximum_substeps, result.substeps)
        minimum_density = min(
            minimum_density,
            float(np.min(state.rho_mobile)),
            float(np.min(state.rho_dipole)),
        )
        maximum_heat_error = max(maximum_heat_error, result.energy_balance_relative_error)
        maximum_partition_error = max(maximum_partition_error, result.work_partition_relative_error)
        F_previous = F_current
        response_previous = response

    history_path = output_directory / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_history_header(systems.n_slip, systems.n_twin))
        writer.writeheader()
        writer.writerows(rows)

    final_row = rows[-1]
    mechanical_balance_relative_error = abs(float(final_row["mechanical_balance_residual_J_per_m3"])) / max(
        abs(float(final_row["external_work_J_per_m3"])), 1.0
    )
    summary = {
        "schema": "hcp-cp-simple-shear-result-0.1",
        "case_id": case_id,
        "classification": "NUMERICAL_VERIFICATION_ONLY_NOT_A_TA2_PREDICTION",
        "material_path": str(material_path),
        "material_sha256": _sha256(material_path),
        "increments": increments,
        "final_shear": final_shear,
        "shear_rate_per_s": shear_rate,
        "total_time_s": total_time,
        "orientation_bunge_deg": list(orientation_bunge_deg),
        "switches": asdict(switches),
        "n_slip": systems.n_slip,
        "n_twin": systems.n_twin,
        "final_sigma12_Pa": final_row["sigma12_Pa"],
        "final_temperature_K": final_row["temperature_K"],
        "temperature_rise_K": float(final_row["temperature_K"]) - parameters.T_ref,
        "total_accumulated_slip": final_row["total_accumulated_slip"],
        "total_twin_fraction": final_row["total_twin_fraction"],
        "minimum_density_per_m2": minimum_density,
        "det_Fp_error": abs(float(final_row["det_Fp"]) - 1.0),
        "minimum_plastic_dissipation_W_per_m3": min(
            float(row["plastic_dissipation_W_per_m3"]) for row in rows
        ),
        "cumulative_plastic_work_J_per_m3": final_row["plastic_work_J_per_m3"],
        "maximum_step_heat_balance_relative_error": maximum_heat_error,
        "maximum_step_partition_relative_error": maximum_partition_error,
        "mechanical_balance_relative_error": mechanical_balance_relative_error,
        "maximum_constitutive_substeps": maximum_substeps,
        "history_path": str(history_path),
        "history_sha256": _sha256(history_path),
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", default="single_crystal_monotonic_shear")
    parser.add_argument("--increments", type=int, default=200)
    parser.add_argument("--final-shear", type=float, default=0.1)
    parser.add_argument("--shear-rate", type=float, default=1000.0)
    parser.add_argument("--orientation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--disable-slip", action="store_true")
    parser.add_argument("--disable-twinning", action="store_true")
    parser.add_argument("--disable-multiplication", action="store_true")
    parser.add_argument("--disable-dipole-formation", action="store_true")
    parser.add_argument("--disable-recovery", action="store_true")
    parser.add_argument("--disable-glide-recovery", action="store_true")
    parser.add_argument("--disable-climb-recovery", action="store_true")
    parser.add_argument("--disable-temperature-feedback", action="store_true")
    parser.add_argument("--isothermal", action="store_true")
    args = parser.parse_args()
    switches = MechanismSwitches(
        slip=not args.disable_slip,
        twinning=not args.disable_twinning,
        multiplication=not args.disable_multiplication,
        dipole_formation=not args.disable_dipole_formation,
        recovery_glide=not (args.disable_recovery or args.disable_glide_recovery),
        recovery_climb=not (args.disable_recovery or args.disable_climb_recovery),
        thermal_softening=not args.disable_temperature_feedback,
        adiabatic_heating=not args.isothermal,
    )
    summary = run_simple_shear(
        args.material,
        args.output,
        case_id=args.case_id,
        increments=args.increments,
        final_shear=args.final_shear,
        shear_rate=args.shear_rate,
        orientation_bunge_deg=tuple(args.orientation),
        switches=switches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
