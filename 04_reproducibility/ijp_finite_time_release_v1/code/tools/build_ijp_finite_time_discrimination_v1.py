"""Build the central finite-time versus frozen-spectrum discrimination audit.

The audit compares two quantities for the *same* fixed direction--wavenumber
branch and the same evolving 129-state CP-Ti base history:

1. the observed finite-time singular gain of the time-ordered propagator; and
2. the exponential prediction obtained by integrating the instantaneous
   frozen spectral abscissa.

Their difference is therefore not caused by changing the branch, the state
scaling, or the observation subspace.  It measures the failure of a frozen
eigenvalue accumulation to represent a non-normal, non-autonomous evolution.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.audit_cp_ti_continuous_spectrum_robustness_v1 import (  # noqa: E402
    OBSERVED,
    SpectrumAudit,
)
from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)


ROBUSTNESS = ROOT / "05_results/cp_ti_continuous_spectrum_robustness_v2.json"
RESULT = ROOT / "05_results/cp_ti_finite_time_discrimination_v1.json"
TABLE = ROOT / "05_results/cp_ti_finite_time_discrimination_v1.csv"
FIGURE_DIR = ROOT / "06_manuscript/ijp_spectral_hcp/figures"
FIGURE_STEM = FIGURE_DIR / "fig04_finite_time_discrimination"
REFERENCE_CONTEXT = "factor16"
INTEGRATION_SUBSTEPS = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _branch_records(robustness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selection = robustness["continuous_selection"]
    onset = selection["onset_branches"]
    final = selection["final_branches"]
    final_label = selection["final_winner_label"]
    return {
        "onset_x": onset["sample_x_branch"]["upper"],
        "onset_y": onset["sample_y_branch"]["upper"],
        "terminal_y": final[final_label],
    }


def _cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(times)
    result[1:] = np.cumsum(increments)
    return result


def _scaled_generator(operator: Any, scales: np.ndarray) -> np.ndarray:
    return (operator.generator_A * scales[None, :]) / scales[:, None]


def _commutator_diagnostics(
    operators: list[Any], scales: np.ndarray
) -> dict[str, float]:
    values = []
    for left, right in zip(operators[:-1], operators[1:], strict=True):
        first = _scaled_generator(left, scales)
        second = _scaled_generator(right, scales)
        numerator = float(np.linalg.norm(second @ first - first @ second, ord="fro"))
        denominator = max(
            float(np.linalg.norm(first, ord="fro") * np.linalg.norm(second, ord="fro")),
            np.finfo(float).tiny,
        )
        values.append(numerator / denominator)
    array = np.asarray(values, dtype=float)
    return {
        "maximum_normalized_adjacent_commutator": float(np.max(array)),
        "median_normalized_adjacent_commutator": float(np.median(array)),
        "mean_normalized_adjacent_commutator": float(np.mean(array)),
    }


def _evaluate_branch(
    audit: SpectrumAudit,
    identifier: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    context = audit.contexts[REFERENCE_CONTEXT]
    times = np.asarray(context["times"], dtype=float)
    normal = np.asarray(record["direction_n"], dtype=float)
    normal /= np.linalg.norm(normal)
    wavenumber = float(record["k_m_inv"])
    operators = [
        assemble_dynamic_crystal_operator_v1(
            point,
            wavenumber_m_inv=wavenumber,
            direction_n=normal,
            admission=audit.admission,
        )
        for point in context["points"]
    ]
    history = finite_time_amplification_history(
        operators,
        times,
        coordinate_scales=audit.scales,
        gain_threshold=float(np.e),
        input_indices=OBSERVED,
        output_indices=OBSERVED,
        integration_substeps_per_interval=INTEGRATION_SUBSTEPS,
    )
    spectral_abscissa = []
    frequency = []
    backward_error = []
    for operator in operators:
        roots, _, residuals = operator.admitted_eigenpairs()
        dominant = int(np.argmax(roots.real))
        spectral_abscissa.append(float(roots[dominant].real))
        frequency.append(float(abs(roots[dominant].imag)))
        backward_error.append(float(residuals[dominant]))
    alpha = np.asarray(spectral_abscissa, dtype=float)
    frozen_log_gain = _cumulative_trapezoid(alpha, times)
    exact_log_gain = np.asarray(
        [row["log_gain"] for row in history["prefix"]], dtype=float
    )
    _require(exact_log_gain.shape == times.shape, "prefix/time size mismatch")
    discrepancy = exact_log_gain - frozen_log_gain
    return {
        "branch_id": identifier,
        "direction_n": normal.tolist(),
        "k_m_inv": wavenumber,
        "wavelength_m": float(2.0 * np.pi / wavenumber),
        "critical_time_s": history["critical_time_s"],
        "final_gain": float(history["final_gain"]),
        "final_log_gain": float(history["final_log_gain"]),
        "frozen_integral_final_log_gain": float(frozen_log_gain[-1]),
        "final_log_discrepancy": float(discrepancy[-1]),
        "maximum_abs_log_discrepancy": float(np.max(np.abs(discrepancy))),
        "maximum_modal_backward_error": float(np.max(backward_error)),
        "commutator": _commutator_diagnostics(operators, audit.scales),
        "history": [
            {
                "time_s": float(time),
                "spectral_abscissa_s_inv": float(alpha_value),
                "dominant_frequency_s_inv": float(frequency_value),
                "modal_backward_error": float(error_value),
                "finite_time_gain": float(prefix["maximum_gain"]),
                "finite_time_log_gain": float(prefix["log_gain"]),
                "frozen_integral_log_gain": float(frozen_value),
                "log_gain_discrepancy": float(delta),
            }
            for time, alpha_value, frequency_value, error_value, prefix, frozen_value, delta
            in zip(
                times,
                alpha,
                frequency,
                backward_error,
                history["prefix"],
                frozen_log_gain,
                discrepancy,
                strict=True,
            )
        ],
    }


def _write_table(branches: dict[str, dict[str, Any]]) -> None:
    rows = []
    for branch_id, branch in branches.items():
        for row in branch["history"]:
            rows.append({"branch_id": branch_id, **row})
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot(branches: dict[str, dict[str, Any]]) -> None:
    _configure_style()
    colors = {
        "onset_x": "#2166ac",
        "onset_y": "#67a9cf",
        "terminal_y": "#b2182b",
    }
    labels = {
        "onset_x": r"onset $x$ branch",
        "onset_y": r"onset $y$ branch",
        "terminal_y": r"terminal $y$ branch",
    }
    terminal = branches["terminal_y"]
    history = terminal["history"]
    time_us = np.asarray([row["time_s"] for row in history]) * 1.0e6
    alpha = np.asarray([row["spectral_abscissa_s_inv"] for row in history])
    exact = np.asarray([row["finite_time_log_gain"] for row in history])
    frozen = np.asarray([row["frozen_integral_log_gain"] for row in history])

    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.25))
    ax = axes[0, 0]
    ax.plot(time_us, alpha * 1.0e-6, color="#4d4d4d", lw=1.6)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"frozen abscissa $\alpha$ [$\mu$s$^{-1}$]")
    ax.set_title("(a) Instantaneous frozen spectrum")
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[0, 1]
    ax.plot(time_us, exact, color="#b2182b", lw=2.0, label=r"time-ordered $\log G$")
    ax.plot(time_us, frozen, color="#2166ac", lw=1.7, ls="--", label=r"$\int\alpha\,dt$")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel("logarithmic amplification")
    ax.set_title("(b) Same branch, same history")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for branch_id, branch in branches.items():
        rows = branch["history"]
        branch_time = np.asarray([row["time_s"] for row in rows]) * 1.0e6
        branch_log = np.asarray([row["finite_time_log_gain"] for row in rows])
        ax.plot(
            branch_time,
            branch_log,
            color=colors[branch_id],
            lw=1.8 if branch_id != "terminal_y" else 2.2,
            label=labels[branch_id],
        )
    ax.axhline(1.0, color="0.45", lw=0.9, ls=":", label=r"$G=e$")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"finite-time $\log G$")
    ax.set_title("(c) Horizon-dependent branch ranking")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    for branch_id, branch in branches.items():
        rows = branch["history"]
        branch_time = np.asarray([row["time_s"] for row in rows]) * 1.0e6
        difference = np.asarray([row["log_gain_discrepancy"] for row in rows])
        ax.plot(
            branch_time,
            difference,
            color=colors[branch_id],
            lw=1.8 if branch_id != "terminal_y" else 2.2,
            label=labels[branch_id],
        )
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"$\log G-\int\alpha\,dt$")
    ax.set_title("(d) Frozen-spectrum prediction error")
    ax.grid(True, which="both", alpha=0.22)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    robustness = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    audit = SpectrumAudit(maximum_refinement=16, context_factors=(1, 16))
    source_branches = _branch_records(robustness)
    branches = {
        identifier: _evaluate_branch(audit, identifier, record)
        for identifier, record in source_branches.items()
    }
    final_discrepancies = {
        identifier: branch["final_log_discrepancy"]
        for identifier, branch in branches.items()
    }
    result = {
        "schema": "CP_TI_FINITE_TIME_DISCRIMINATION_V1",
        "status": "COMPLETED_SAME_BRANCH_FINITE_TIME_VERSUS_FROZEN_SPECTRUM_AUDIT",
        "question": (
            "Does an integral of instantaneous frozen spectral abscissae reproduce the "
            "finite-time observable gain on the same evolving crystal branch?"
        ),
        "answer_rule": (
            "A nonzero log-gain discrepancy under the locked branch, norm, base history, "
            "and observation subspace rejects frozen-spectrum accumulation as an equivalent "
            "finite-time selector."
        ),
        "reference_discretization": {
            "base_state_count": len(audit.contexts[REFERENCE_CONTEXT]["times"]),
            "integration_substeps_per_interval": INTEGRATION_SUBSTEPS,
            "observation_indices": OBSERVED.tolist(),
            "coordinate_scale_source": "SpectrumAudit factor1 locked scales",
        },
        "branches": branches,
        "diagnostics": {
            "all_finite": bool(
                all(
                    np.isfinite(value)
                    for branch in branches.values()
                    for row in branch["history"]
                    for value in (
                        row["spectral_abscissa_s_inv"],
                        row["finite_time_log_gain"],
                        row["frozen_integral_log_gain"],
                    )
                )
            ),
            "maximum_modal_backward_error": float(
                max(branch["maximum_modal_backward_error"] for branch in branches.values())
            ),
            "final_log_discrepancies": final_discrepancies,
            "frozen_accumulation_equivalent_to_finite_time_gain": bool(
                all(abs(value) <= 1.0e-2 for value in final_discrepancies.values())
            ),
        },
        "claim_boundary": {
            "supports": (
                "For the registered CP-Ti baseline and tested branches, instantaneous frozen "
                "spectral accumulation is not interchangeable with the non-autonomous finite-time "
                "observable propagator when the reported discrepancy is non-negligible."
            ),
            "does_not_support": (
                "Experimental localization onset, a coordinate-invariant critical time, global "
                "direction--wavenumber uniqueness, or a material-calibrated band width."
            ),
        },
        "sources": [
            ROBUSTNESS.relative_to(ROOT).as_posix(),
            "tools/audit_cp_ti_continuous_spectrum_robustness_v1.py",
            "src/hcp_cp_gnd/dynamic_crystal_perturbation_v1.py",
        ],
    }
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_table(branches)
    _plot(branches)
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "table": str(TABLE),
                "figure_pdf": str(FIGURE_STEM.with_suffix('.pdf')),
                "final_log_discrepancies": final_discrepancies,
                "maximum_modal_backward_error": result["diagnostics"][
                    "maximum_modal_backward_error"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
