"""Generate the complete material/model/numerical parameter provenance table.

The table distinguishes computational completeness from calibration strength.
Every field used by the active HCP material object is covered, but repository
verification-seed values are never relabelled as specimen-identified data.
"""

from __future__ import annotations

import csv
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V01_SRC = ROOT.parent / "HCP_CP_v0.1/src"
for path in (ROOT, SRC, V01_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hcp_cp.parameters import MaterialParameters  # noqa: E402
from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    build_material_objects,
    heat_capacity_J_kgK,
    load_card,
)


RESULT = ROOT / "05_results/ijp_material_parameter_provenance_v1.json"
CSV_TABLE = ROOT / "05_results/ijp_material_parameter_provenance_v1.csv"
TEX_TABLE = ROOT / "06_manuscript/ijp_spectral_hcp/tables/ft_material_parameter_source_table.tex"


SOURCES = {
    "S1": (
        "Nemat-Nasser, Guo and Cheng, Acta Materialia 47 (1999) 3705--3720, "
        "doi:10.1016/S1359-6454(99)00203-7; grouped alpha-Ti crystal/dynamic-loading constraint."
    ),
    "S2": (
        "NIST-JANAF alpha-Ti table / NIST Chemistry WebBook Shomate heat-capacity "
        "coefficients, admitted over 298--700 K."
    ),
    "S3": (
        "Touloukian/TPRC selected thermal-conductivity value for annealed high-purity "
        "titanium near 300 K, as registered in the material card."
    ),
    "S4": (
        "Dai, L. and Song, W., A strain rate and temperature-dependent crystal "
        "plasticity model for hexagonal close-packed (HCP) materials: Application "
        "to alpha-titanium, International Journal of Plasticity 154 (2022) 103281, "
        "doi:10.1016/j.ijplas.2022.103281 (publisher record accessed 2026-08-29). "
        "The present card adopts the model class only; its numerical entries were "
        "not copied or batch-calibrated parameter by parameter from that paper."
    ),
    "S5": (
        "Public release commit 8aa42e3b5999adc502142f940e42b9222cdb17e6, "
        "config/verification_seed.yaml (frozen kernel version 0.1, SHA-256 "
        "b7ced9294e7d17ba64b5c2eaf843be5eb4defbcfe480bbb04bce1fb8612c124a) "
        "and config/cp_ti_grade1_dynamic_hcp_v1.json (schema V1, SHA-256 "
        "586dd4ed43509aa045f9fe87524b2df51fbd07817853c497ac8c6ad4c082df5d), "
        "accessed 2026-08-29. The seed remains watermarked "
        "VERIFICATION_SEED_NOT_CALIBRATED_FOR_TA2."
    ),
    "S6": (
        "Present-study registered modeling choice in the finite-time spectral runner; "
        "not a measured shear-band width."
    ),
    "S7": "Exact definition, derived quantity, or universal physical constant.",
}


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    raw = float(value)
    if raw == 0.0:
        return "0"
    if abs(raw) >= 1.0e4 or abs(raw) < 1.0e-3:
        return f"{raw:.8g}"
    return f"{raw:.10g}"


def tex_escape(value: str) -> str:
    output = value
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("^", r"\textasciicircum{}"),
        ("<", r"$<$"),
        (">", r"$>$"),
    ):
        output = output.replace(source, target)
    return output


def tex_unit(value: str) -> str:
    if value == "--":
        return "--"
    mapping = {
        "K": r"$\mathrm{K}$",
        "Pa": r"$\mathrm{Pa}$",
        "m": r"$\mathrm{m}$",
        "J": r"$\mathrm{J}$",
        "b": r"$b$",
        "deg": r"degree",
        "kg m^-3": r"$\mathrm{kg\,m^{-3}}$",
        "J kg^-1 K^-1": r"$\mathrm{J\,kg^{-1}\,K^{-1}}$",
        "cal mol^-1 K^-1": r"$\mathrm{cal\,mol^{-1}\,K^{-1}}$",
        "kg mol^-1": r"$\mathrm{kg\,mol^{-1}}$",
        "W m^-1 K^-1": r"$\mathrm{W\,m^{-1}\,K^{-1}}$",
        "m s^-1": r"$\mathrm{m\,s^{-1}}$",
        "m^-2": r"$\mathrm{m^{-2}}$",
        "s^-1": r"$\mathrm{s^{-1}}$",
    }
    if value not in mapping:
        raise ValueError(f"unregistered LaTeX unit: {value}")
    return mapping[value]


def tex_code(value: str) -> str:
    return r"\texttt{\detokenize{" + value + "}}"


def main() -> int:
    card = load_card()
    _, parameters, _ = build_material_objects(card)
    rows: list[dict[str, Any]] = []
    covered_fields: set[str] = {"status"}

    def add(
        group: str,
        symbol: str,
        name: str,
        value: Any,
        unit: str,
        source_key: str,
        evidence_class: str,
        calibration_status: str,
        role: str,
        field: str | None = None,
        active: bool = True,
    ) -> None:
        if field is not None:
            covered_fields.add(field)
        rows.append(
            {
                "group": group,
                "symbol": symbol,
                "parameter": name,
                "value": fmt(value),
                "unit": unit,
                "source_key": source_key,
                "evidence_class": evidence_class,
                "calibration_status": calibration_status,
                "active_in_reported_branch": active,
                "role": role,
            }
        )

    add("crystal", "c/a", "HCP lattice ratio", parameters.c_over_a, "--", "S1", "L2_grouped_literature_constraint", "literature-constrained, not batch identified", "slip geometry", "c_over_a")
    add("crystal", "T_ref", "elastic reference temperature", parameters.T_ref, "K", "S5", "L2_repository_baseline", "verification seed", "reference state", "T_ref")
    elastic_card = card["elastic_Pa"]
    for key, symbol in (("C11", "C_11"), ("C12", "C_12"), ("C13", "C_13"), ("C33", "C_33"), ("C44", "C_44")):
        add("elasticity", symbol, f"HCP stiffness {key}", elastic_card[key], "Pa", "S1/S5", "L2_grouped_literature_constraint", "research baseline, not batch identified", "anisotropic elastic tangent", "elastic_C0")
    add("elasticity", "C_66", "derived HCP stiffness", 0.5 * (elastic_card["C11"] - elastic_card["C12"]), "Pa", "S7", "derived", "exactly derived from C11 and C12", "anisotropic elastic tangent", "elastic_C0")

    thermal = card["thermal"]
    add("thermal", "rho_0", "mass density", parameters.mass_density, "kg m^-3", "S1/S5", "L2_grouped_literature_constraint", "research baseline", "inertia and heat storage", "mass_density")
    add("thermal", "c_p(298.15 K)", "specific heat used at reference state", parameters.heat_capacity, "J kg^-1 K^-1", "S2", "L1_direct_reference_function", "literature function, not fitted", "heat storage", "heat_capacity")
    shomate = thermal["heat_capacity_model"]
    for key, unit in (("A", "cal mol^-1 K^-1"), ("B", "cal mol^-1 K^-1"), ("C", "cal mol^-1 K^-1"), ("D", "cal mol^-1 K^-1"), ("E", "cal mol^-1 K^-1")):
        add("thermal", key, f"Shomate coefficient {key}", shomate[key], unit, "S2", "L1_direct_reference_function", "admitted only on 298--700 K", "temperature-dependent heat capacity")
    add("thermal", "M_Ti", "titanium molar mass", shomate["molar_mass_kg_mol"], "kg mol^-1", "S2", "L1_reference_constant", "reference constant", "molar-to-mass heat-capacity conversion")
    add("thermal", "kappa", "thermal conductivity at 300 K", thermal["conductivity_W_mK_at_300K"], "W m^-1 K^-1", "S3", "L1_selected_reference_value", "literature value, not batch identified", "thermal diffusion")
    add("thermal", "beta", "Taylor--Quinney heat fraction", parameters.taylor_quinney, "--", "S5", "L2_repository_baseline", "not independently identified", "plastic-work heat partition", "taylor_quinney")

    families = card["slip_kinetics_by_family"]["family_order"]
    family_symbols = ("bas", "pri", "pyr")
    family_parameters = (
        ("burgers_m_by_family", card["lattice"]["burgers_m_by_family"], "b", "m", "burgers"),
        ("tau0_Pa", card["slip_kinetics_by_family"]["tau0_Pa"], "tau_0", "Pa", "tau0"),
        ("tau_cut_Pa", card["slip_kinetics_by_family"]["tau_cut_Pa"], "tau_cut", "Pa", "tau_cut"),
        ("activation_energy_J", card["slip_kinetics_by_family"]["activation_energy_J"], "Delta F", "J", "activation_energy"),
        ("reference_velocity_m_s", card["slip_kinetics_by_family"]["reference_velocity_m_s"], "v_0", "m s^-1", "reference_velocity"),
        ("p", card["slip_kinetics_by_family"]["p"], "p", "--", "p"),
        ("q", card["slip_kinetics_by_family"]["q"], "q", "--", "q"),
        ("rho_mobile_0_m2", card["slip_kinetics_by_family"]["rho_mobile_0_m2"], "rho_m0", "m^-2", "rho_mobile_0"),
        ("rho_dipole_0_m2", card["slip_kinetics_by_family"]["rho_dipole_0_m2"], "rho_d0", "m^-2", "rho_dipole_0"),
    )
    for name, values, symbol, unit, field in family_parameters:
        for family, suffix, value in zip(families, family_symbols, values, strict=True):
            add("slip kinetics", f"{symbol}^{suffix}", f"{name}: {family}", value, unit, "S4/S5", "L2_repository_verification_baseline", "not family-by-family calibrated for the reported specimen", "thermally activated slip and initial state", field)

    dislocation = card["dislocation"]
    scalar_dislocation = (
        ("d_g", "grain size", "grain_size_m", "m", "grain_size"),
        ("K_Lambda", "mean-free-path coefficient", "mean_free_path_coefficient", "--", "mean_free_path_coefficient"),
        ("mu_ref", "reference shear modulus", "reference_shear_modulus_Pa", "Pa", "reference_shear_modulus"),
        ("h_self", "forest self interaction", "forest_self", "--", "forest_interaction"),
        ("h_lat", "forest latent interaction", "forest_latent", "--", "forest_interaction"),
        ("a_T", "Taylor coefficient", "taylor_coefficient", "--", "taylor_coefficient"),
        ("c_d", "minimum dipole distance", "dipole_min_distance_burgers", "b", "dipole_min_distance_burgers"),
        ("nu_c", "climb attempt frequency", "climb_frequency_s", "s^-1", "climb_frequency"),
        ("Q_c", "climb activation energy", "climb_activation_J", "J", "climb_activation"),
    )
    for symbol, name, key, unit, field in scalar_dislocation:
        add("dislocation", symbol, name, dislocation[key], unit, "S4/S5", "L2_repository_verification_baseline", "not batch identified", "storage/recovery/forest resistance", field)

    micro = (
        ("H_chi", "micromorphic penalty modulus", parameters.reference_shear_modulus, "Pa"),
        ("ell_N", "Nye/GND energetic length", 1.0e-6, "m"),
        ("ell_chi", "full-rank slip-gradient length", 0.25e-6, "m"),
    )
    for symbol, name, value, unit in micro:
        add("micromorphic/gradient", symbol, name, value, unit, "S6", "L3_registered_model_control", "not identified as a material band width", "algebraic coupling / high-k regularization")

    loading = card["representative_loading"]
    for index, orientation in enumerate(((0.0, 0.0, 0.0), (30.0, 90.0, 30.0), (90.0, 60.0, 30.0)), start=1):
        add("loading", f"g_{index}", f"Bunge orientation case {index}", "/".join(fmt(v) for v in orientation), "deg", "S6", "L3_registered_orientation_case", "theoretical orientation, not an experimental ODF", "cross-orientation transfer")
    add("loading", "T_0", "initial temperature", loading["initial_temperature_K"], "K", "S6", "L3_registered_loading", "prescribed", "base state")
    add("loading", "dot_gamma_0", "macroscopic simple-shear rate", loading["macroscopic_shear_rate_s"], "s^-1", "S1/S6", "L2_dynamic_loading_constraint", "representative, not specimen-matched", "base history")
    add("loading", "gamma_f", "final simple shear", loading["final_shear"], "--", "S6", "L3_registered_loading", "prescribed", "observation window")

    numerics = {
        "density_floor": parameters.density_floor,
        "determinant_tolerance": parameters.determinant_tolerance,
        "maximum_slip_increment": parameters.maximum_slip_increment,
        "maximum_twin_increment": parameters.maximum_twin_increment,
    }
    numeric_meta = (
        ("rho_floor", "density protection floor", "density_floor", "m^-2"),
        ("eps_det", "determinant tolerance", "determinant_tolerance", "--"),
        ("Delta gamma_max", "maximum slip increment", "maximum_slip_increment", "--"),
        ("Delta f_tw_max", "maximum twin increment", "maximum_twin_increment", "--"),
    )
    for symbol, name, field, unit in numeric_meta:
        add("numerical", symbol, name, numerics[field], unit, "S5", "N1_numerical_control", "not material data", "integration/admission control", field)

    inactive = (
        ("gamma_tw", "twin shear override", parameters.twin_shear_override, "--", "twin_shear_override"),
        ("tau_tw", "twin CRSS", parameters.twin_crss, "Pa", "twin_crss"),
        ("dot_f_tw0", "twin reference rate", parameters.twin_reference_rate, "s^-1", "twin_reference_rate"),
        ("m_tw", "twin rate exponent", parameters.twin_rate_exponent, "--", "twin_rate_exponent"),
        ("tau_tw_scale", "twin stress scale", parameters.twin_stress_scale, "Pa", "twin_stress_scale"),
        ("H_tw", "twin latent hardening", parameters.twin_latent_hardening, "Pa", "twin_latent_hardening"),
        ("f_tw_max", "maximum twin fraction", parameters.twin_max_total_fraction, "--", "twin_max_total_fraction"),
    )
    for symbol, name, value, unit, field in inactive:
        add("inactive twinning", symbol, name, value, unit, "S5", "I_inactive_repository_parameter", "inactive in all reported calculations", "retained kernel field", field, active=False)

    required_fields = {field.name for field in fields(MaterialParameters)}
    missing = sorted(required_fields - covered_fields)
    source_keys_used = sorted({token for row in rows for token in row["source_key"].split("/")})
    unknown_sources = sorted(set(source_keys_used) - set(SOURCES))
    gates = {
        "all_material_dataclass_fields_covered": not missing,
        "all_rows_have_units": all(bool(row["unit"]) for row in rows),
        "all_rows_have_source_and_evidence_class": all(row["source_key"] and row["evidence_class"] for row in rows),
        "all_source_keys_defined": not unknown_sources,
        "verification_seed_never_labelled_batch_calibrated": all(
            "batch" not in row["calibration_status"].lower() or "not" in row["calibration_status"].lower()
            for row in rows
            if "S5" in row["source_key"]
        ),
    }

    CSV_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with CSV_TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{landscape}",
        r"\section{Complete material, model and numerical parameter provenance}",
        r"\label{app:parameter_provenance}",
        (
            "Computational completeness does not imply material calibration. "
            "The evidence-class column distinguishes direct reference functions, grouped "
            "literature constraints, repository verification values, registered model controls "
            "and numerical controls. Parameters marked inactive are retained by the material "
            "object but do not enter the reported no-twinning branch."
        ),
        r"\begingroup\tiny",
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.09\linewidth}>{\raggedright\arraybackslash}p{0.09\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.095\linewidth}>{\raggedright\arraybackslash}p{0.095\linewidth}>{\raggedright\arraybackslash}p{0.06\linewidth}>{\raggedright\arraybackslash}p{0.258\linewidth}@{}}",
        r"\toprule",
        r"Group & Symbol & Parameter & Value & Unit & Source & Evidence/calibration status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Group & Symbol & Parameter & Value & Unit & Source & Evidence/calibration status \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        status = f"{row['evidence_class'].replace('_', ' ')}; {row['calibration_status']}"
        cells = [
            tex_escape(str(row["group"]).replace("/", " / ")),
            tex_code(str(row["symbol"])),
            tex_escape(str(row["parameter"]).replace("_", " ")),
            tex_escape(str(row["value"])),
            tex_unit(str(row["unit"])),
            tex_escape(str(row["source_key"])),
            tex_escape(status),
        ]
        tex_lines.append(
            " & ".join(cells) + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", r"\end{landscape}", "", r"\paragraph{Source key.}"])
    for key, value in SOURCES.items():
        tex_lines.append(rf"\textbf{{{key}}}: {tex_escape(value)}\par")
    tex_lines.extend(
        [
            "",
            (
                "The full machine-readable table, including the model role and active/inactive "
                r"flag for every row, is supplied as \path{ijp_material_parameter_provenance_v1.csv}."
            ),
        ]
    )
    TEX_TABLE.parent.mkdir(parents=True, exist_ok=True)
    TEX_TABLE.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    result = {
        "schema": "IJP_MATERIAL_PARAMETER_PROVENANCE_V1",
        "status": "COMPLETE_WITH_EVIDENCE_LEVELS" if all(gates.values()) else "INCOMPLETE_PARAMETER_PROVENANCE",
        "computational_parameter_row_count": len(rows),
        "material_dataclass_fields": sorted(required_fields),
        "covered_material_dataclass_fields": sorted(covered_fields),
        "missing_material_dataclass_fields": missing,
        "unknown_source_keys": unknown_sources,
        "sources": SOURCES,
        "gates": gates,
        "interpretation": {
            "computationally_complete": True,
            "specimen_or_batch_identified": False,
            "gradient_lengths_are_measured_band_widths": False,
            "inactive_twinning_parameters_affect_reported_results": False,
        },
        "configuration_snapshot": {
            "public_release_commit": "8aa42e3b5999adc502142f940e42b9222cdb17e6",
            "frozen_kernel_version": "0.1",
            "verification_seed_path": "config/verification_seed.yaml",
            "verification_seed_sha256": "b7ced9294e7d17ba64b5c2eaf843be5eb4defbcfe480bbb04bce1fb8612c124a",
            "material_card_path": "config/cp_ti_grade1_dynamic_hcp_v1.json",
            "material_card_sha256": "586dd4ed43509aa045f9fe87524b2df51fbd07817853c497ac8c6ad4c082df5d",
            "access_date": "2026-08-29",
        },
        "outputs": {
            "csv": CSV_TABLE.relative_to(ROOT).as_posix(),
            "latex": TEX_TABLE.relative_to(ROOT).as_posix(),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(RESULT), "status": result["status"], "rows": len(rows), "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
