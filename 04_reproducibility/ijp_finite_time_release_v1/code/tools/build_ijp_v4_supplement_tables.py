"""Build human-readable supplementary LaTeX tables from V4 audit receipts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "05_results/ijp_reference_reoptimization_v4.json"
NEAR_SET = ROOT / "05_results/ijp_v4_near_optimal_set_audit_v1.json"
OUTPUT = (
    ROOT
    / "06_manuscript/ijp_spectral_hcp/tables/ft_reference_reoptimization_v4_table.tex"
)


NORM_LABELS = {
    "baseline": "baseline",
    "mechanical_x0.5": r"mechanical $\times0.5$",
    "thermal_x0.5": r"thermal $\times0.5$",
    "dislocation_x2": r"dislocation $\times2$",
    "all_constitutive_x2": r"all constitutive $\times2$",
}
SELECTOR_LABELS = {
    "full_to_full": "full/full",
    "constitutive_to_constitutive": "constitutive/constitutive",
    "full_to_constitutive": r"full$\rightarrow$constitutive",
    "constitutive_to_full": r"constitutive$\rightarrow$full",
    "mechanical_to_mechanical": "mechanical/mechanical",
    "mechanical_to_constitutive": r"mechanical$\rightarrow$constitutive",
    "constitutive_to_temperature": r"constitutive$\rightarrow T$",
}


def main() -> int:
    if not REFERENCE.is_file():
        raise FileNotFoundError("complete V4 reference re-optimization is required")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if reference.get("status") != "REFERENCE_REOPTIMIZATION_PASS":
        raise RuntimeError("V4 reference re-optimization gates are open")

    lines = [
        r"\begin{landscape}",
        r"\begingroup\scriptsize",
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{longtable}{@{}p{1.15cm}p{2.55cm}p{3.60cm}rrrrcc@{}}",
        r"\caption{Complete direct reference-objective re-optimization.  Every row is refined on the 129-state/four-substep propagator; $\Delta\log G$ is relative to the V3 retained coordinate reevaluated on that same reference map.  Class I is a finite-band interior candidate, LW is a lower-boundary long-wave branch and HF is a high-frequency reversible branch; neither boundary class is interpreted as a localization wavelength.}\label{tab:s_reference_reoptimization}\\",
        r"\toprule",
        r"Horizon & Norm & Input--output & $k^*$ ($\mathrm{m^{-1}}$) & $\log G$ & $G$ & $\Delta\log G$ & stat.? & class \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Horizon & Norm & Input--output & $k^*$ ($\mathrm{m^{-1}}$) & $\log G$ & $G$ & $\Delta\log G$ & stat.? & class \\",
        r"\midrule",
        r"\endhead",
    ]
    for key in sorted(reference["records"]):
        record = reference["records"][key]
        classification = {
            "interior_finite_band_candidate": "I",
            "compact_domain_lower_boundary_long_wave_branch_not_a_finite_localization_wavelength": "LW",
            "high_frequency_reversible_wave_branch_not_a_finite_localization_wavelength": "HF",
        }.get(
            record["boundary_classification"],
            record["boundary_classification"].replace("_", " "),
        )
        lines.append(
            " & ".join(
                (
                    record["horizon_id"],
                    NORM_LABELS[record["norm_id"]],
                    SELECTOR_LABELS[record["selector_id"]],
                    f"{record['k_m_inv']:.6g}",
                    f"{record['reference_log_gain']:.6f}",
                    f"{record['reference_gain']:.6g}",
                    f"{record['direct_reference_log_gain_improvement']:+.3e}",
                    "yes" if record["local_stationarity_pass"] else "no",
                    classification,
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{longtable}", r"\endgroup", r"\end{landscape}"))

    if NEAR_SET.is_file():
        near = json.loads(NEAR_SET.read_text(encoding="utf-8"))
        lines.extend(
            (
                "",
                r"\begin{table}[ht]",
                r"\centering",
                r"\caption{Sampled near-optimal sets for the baseline constitutive/constitutive map.  The audit combines an independent scrambled-Sobol challenge with local direction rings and full-domain fixed-direction wavenumber profiles; it is not a continuum global certificate.}",
                r"\begin{tabular}{@{}lrrrr@{}}",
                r"\toprule",
                r"Horizon/$\varepsilon$ & members & $k_{\min}$ ($\mathrm{m^{-1}}$) & $k_{\max}$ ($\mathrm{m^{-1}}$) & max angle (deg) \\",
                r"\midrule",
            )
        )
        for horizon_id, horizon in near["horizons"].items():
            for item in horizon["near_optimal_sets"].values():
                lines.append(
                    f"{horizon_id}/{100*item['epsilon']:.1f}\\% & "
                    f"{item['sampled_member_count']} & "
                    f"{item['k_min_m_inv']:.6g} & {item['k_max_m_inv']:.6g} & "
                    f"{item['maximum_projective_angle_from_winner_deg']:.3f} \\\\"
                )
        lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
