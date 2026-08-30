"""Build the reference-objective re-optimization figure from the V4 receipt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "05_results/ijp_reference_reoptimization_v4.json"
FIGURE = (
    ROOT
    / "06_manuscript/ijp_spectral_hcp/figures/fig10_reoptimized_selector_audit"
)
NORM_ORDER = (
    "baseline",
    "mechanical_x0.5",
    "thermal_x0.5",
    "dislocation_x2",
    "all_constitutive_x2",
)
NORM_LABELS = ("base", "mech/2", "thermal/2", "disl.×2", "const.×2")
SELECTOR_ORDER = (
    "full_to_full",
    "constitutive_to_constitutive",
    "full_to_constitutive",
    "constitutive_to_full",
    "mechanical_to_mechanical",
    "mechanical_to_constitutive",
    "constitutive_to_temperature",
)
SELECTOR_LABELS = (
    "full/full",
    "const./const.",
    "full→const.",
    "const.→full",
    "mech./mech.",
    "mech.→const.",
    "const.→T",
)


def main() -> int:
    if not RESULT.is_file():
        raise FileNotFoundError("complete V4 reference re-optimization is required")
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    if report.get("status") != "REFERENCE_REOPTIMIZATION_PASS":
        raise RuntimeError("V4 reference re-optimization gates are open")
    records = report["records"]

    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.55, 6.15), constrained_layout=True)
    matrices: dict[str, np.ndarray] = {}
    all_values = []
    for horizon in ("onset", "terminal"):
        matrix = np.asarray(
            [
                [
                    records[f"{horizon}__{norm}__{selector}"]["reference_log_gain"]
                    for selector in SELECTOR_ORDER
                ]
                for norm in NORM_ORDER
            ]
        )
        matrices[horizon] = matrix
        all_values.extend(matrix.ravel())
    color_limit = max(abs(float(np.min(all_values))), abs(float(np.max(all_values))))

    for ax, horizon, title in (
        (axes[0, 0], "onset", "(a) Onset: direct reference optimization"),
        (axes[0, 1], "terminal", "(b) Terminal: direct reference optimization"),
    ):
        matrix = matrices[horizon]
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        ax.set_xticks(np.arange(len(SELECTOR_LABELS)), SELECTOR_LABELS, rotation=38, ha="right")
        ax.set_yticks(np.arange(len(NORM_LABELS)), NORM_LABELS)
        ax.set_title(title, loc="left")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if abs(value) > 0.55 * color_limit else "black"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.1, color=color)
        fig.colorbar(image, ax=ax, label=r"reference $\log G$", fraction=0.046, pad=0.03)

    ax = axes[1, 0]
    x = np.arange(len(NORM_ORDER))
    for horizon, marker, color in (
        ("onset", "o", "#2166ac"),
        ("terminal", "s", "#b2182b"),
    ):
        values = [
            records[f"{horizon}__{norm}__constitutive_to_constitutive"]["k_m_inv"]
            for norm in NORM_ORDER
        ]
        ax.plot(x, values, marker=marker, color=color, label=horizon)
    ax.set_yscale("log")
    ax.set_xticks(x, NORM_LABELS, rotation=25, ha="right")
    ax.set_ylabel(r"$k^*$ ($\mathrm{m^{-1}}$)")
    ax.set_title("(c) Constitutive/constitutive basin across norms", loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    onset_values = []
    terminal_values = []
    for norm in NORM_ORDER:
        for selector in SELECTOR_ORDER:
            onset_values.append(
                records[f"onset__{norm}__{selector}"][
                    "direct_reference_log_gain_improvement"
                ]
            )
            terminal_values.append(
                records[f"terminal__{norm}__{selector}"][
                    "direct_reference_log_gain_improvement"
                ]
            )
    xx = np.arange(len(onset_values))
    ax.scatter(xx, onset_values, s=14, color="#2166ac", marker="o", label="onset")
    ax.scatter(xx, terminal_values, s=14, color="#b2182b", marker="s", label="terminal")
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0e-3)
    ax.set_xlabel("norm–selector contract index")
    ax.set_ylabel(r"direct-reference $\Delta\log G$")
    ax.set_title("(d) Correction to V3 retained coordinates", loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.12)
    ax.legend(frameon=False, loc="best")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE.with_suffix(".pdf").relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
