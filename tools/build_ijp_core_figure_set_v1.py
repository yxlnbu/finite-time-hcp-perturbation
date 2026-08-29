"""Build the nine-figure IJP evidence architecture around finite-time selection.

Figures 8--10 are produced by their dedicated operator/search/nonlinear audit
scripts.  This script builds the remaining figures and writes one receipt that
records both the positive and falsifying evidence used by the manuscript.
"""

from __future__ import annotations

import json
from math import pi
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcp_cp_gnd.cp_ti_material_v1 import (  # noqa: E402
    load_card,
    local_state92_from_material_state,
    run_representative_checkpoint_states,
    simple_shear,
)


RESULTS = ROOT / "05_results"
FIGURES = ROOT / "06_manuscript/ijp_spectral_hcp/figures"
RECEIPT = RESULTS / "ijp_core_figure_set_v1.json"


def _load(relative: str) -> dict[str, Any]:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.2,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": pdf.relative_to(ROOT).as_posix(),
        "png": png.relative_to(ROOT).as_posix(),
    }


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + height - 0.035, title, weight="bold", va="top", fontsize=8.2)
    ax.text(x + 0.02, y + height - 0.095, body, va="top", fontsize=7.1, linespacing=1.30)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="0.32",
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure01_framework() -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(7.55, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        (0.03, 0.68),
        0.20,
        0.23,
        "Scalar Bai anchor",
        r"Simple shear" "\n" r"$Q,R,P,\rho c,\lambda$" "\nFrozen polynomial",
        face="#f2f2f2",
        edge="#636363",
    )
    _box(
        ax,
        (0.28, 0.68),
        0.20,
        0.23,
        "Evolving HCP path",
        r"$\mathbf{F}^p_0(t),T_0(t)$" "\n" r"$\rho_m^\alpha,\rho_d^\alpha,\gamma^\alpha$" "\nOrientation, multi-slip",
        face="#e5f5f9",
        edge="#2b8cbe",
    )
    _box(
        ax,
        (0.53, 0.68),
        0.20,
        0.23,
        "87-state descriptor",
        r"$[\mathbf{K}+s\mathbf{E}+s^2\mathbf{M}]$" "\n" r"$\hat{\mathbf{x}}=0$" "\n69 finite + 18 algebraic",
        face="#e5f5e0",
        edge="#238b45",
    )
    _box(
        ax,
        (0.78, 0.68),
        0.19,
        0.23,
        "69-state generator",
        r"Exact Schur condensation" "\n" r"$\dot{\mathbf{q}}=\mathbf{A}(t;k,\mathbf{n})\mathbf{q}$" "\nNo fictitious inertia",
        face="#fff7bc",
        edge="#d95f0e",
    )
    _arrow(ax, (0.23, 0.795), (0.28, 0.795))
    _arrow(ax, (0.48, 0.795), (0.53, 0.795))
    _arrow(ax, (0.73, 0.795), (0.78, 0.795))

    _box(
        ax,
        (0.12, 0.35),
        0.27,
        0.20,
        r"Time-ordered $\mathbf{\Phi}$",
        r"$\mathbf{\Phi}=\mathcal{T}\exp\!\int\mathbf{A}(\tau)d\tau$" "\n" r"Path dependent",
        face="#fde0dd",
        edge="#c51b8a",
    )
    _box(
        ax,
        (0.43, 0.35),
        0.22,
        0.20,
        r"Singular gain $G$",
        r"$G=\sigma_{\max}(\mathbf{C}\Phi\mathbf{B})$" "\n" r"Thermal + constitutive" "\nFixed norm",
        face="#fdd49e",
        edge="#d94801",
    )
    _box(
        ax,
        (0.69, 0.35),
        0.24,
        0.20,
        r"Selector $\mathcal{S}_\varepsilon(t)$",
        r"Near-optimal $(\mathbf{n},k)$ set" "\n" r"Optimal state composition" "\nHorizon dependent",
        face="#dadaeb",
        edge="#756bb1",
    )
    _arrow(ax, (0.875, 0.68), (0.30, 0.55))
    _arrow(ax, (0.39, 0.45), (0.43, 0.45))
    _arrow(ax, (0.65, 0.45), (0.69, 0.45))

    ax.add_patch(Rectangle((0.03, 0.06), 0.44, 0.19, facecolor="#f7f7f7", edgecolor="#969696"))
    ax.text(0.05, 0.215, "Classical theory cannot determine", weight="bold", fontsize=8.2, va="top")
    ax.text(
        0.05,
        0.165,
        "which crystal perturbation accumulates the largest gain,\n"
        "or why direction, wavelength, and state composition\n"
        "change with the observation horizon.",
        fontsize=6.9,
        va="top",
    )
    ax.add_patch(Rectangle((0.53, 0.06), 0.44, 0.19, facecolor="#edf8e9", edgecolor="#31a354"))
    ax.text(0.55, 0.215, "Finite-time crystal theory resolves", weight="bold", fontsize=8.2, va="top")
    ax.text(
        0.55,
        0.165,
        "maximum history-conditioned gain and a tested near-optimal\n"
        "direction--wavenumber--state set, subject to explicit\n"
        "norm, search-box, and validation boundaries.",
        fontsize=6.9,
        va="top",
    )
    paths = _save(fig, "fig01_theory_variable_framework")
    return paths, {
        "claim": "Bai scalar anchor to HCP descriptor to finite-time set-valued selector",
        "evidence_kind": "theory and variable map",
    }


def figure02_verification(
    bai: dict[str, Any], qz: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.0))

    ax = axes[0, 0]
    ax.axis("off")
    target = bai["independent_test_reproduction"]["v5_targeted"]
    joint = bai["independent_test_reproduction"][
        "v5_v4_v3_v2_v1_and_related_spectral_regression"
    ]
    items = [
        ("Scalar coefficient/root implementation", True, f"{target['passed']} targeted tests"),
        ("Related regression suite", True, f"{joint['passed']} tests"),
        ("Exact-dyadic sign/zero handling", True, "independent red-team pass"),
        ("Strict HCP → Bai limit", False, "analytical target; still open"),
    ]
    for index, (label, passed, note) in enumerate(items):
        y = 0.88 - 0.22 * index
        color = "#238b45" if passed else "#cb181d"
        marker = "✓" if passed else "!"
        ax.text(0.03, y, marker, color=color, fontsize=16, weight="bold", va="center")
        ax.text(0.12, y + 0.025, label, fontsize=8.8, weight="bold", va="center")
        ax.text(0.12, y - 0.055, note, fontsize=7.8, color="0.35", va="center")
    ax.set_title("(a) Scalar anchor and scope boundary", loc="left")

    ax = axes[0, 1]
    ax.barh(["finite roots", "algebraic roots"], [69, 18], color=["#238b45", "#bdbdbd"])
    ax.set_xlim(0, 75)
    for y, value in enumerate((69, 18)):
        ax.text(value + 1.2, y, str(value), va="center", fontsize=9, weight="bold")
    ax.set_xlabel("root count in the 87-state descriptor")
    ax.set_title("(b) Exact descriptor partition", loc="left")
    ax.grid(True, axis="x", alpha=0.22)

    ax = axes[1, 0]
    metrics = {
        "finite-root match": qz["finite_spectrum_match"]["maximum_relative_error"],
        "mean root match": qz["finite_spectrum_match"]["mean_relative_error"],
        "algebraic identity": qz["strict_equivalence_regularization"][
            "algebraic_identity_max_abs_error"
        ],
        "block decoupling": max(
            qz["strict_equivalence_regularization"]["lower_left_decoupling_max_abs_error"],
            qz["strict_equivalence_regularization"]["upper_right_decoupling_max_abs_error"],
            1.0e-18,
        ),
        "generator equivalence": max(
            qz["strict_equivalence_regularization"]["generator_equivalence_relative_error"],
            1.0e-18,
        ),
    }
    names = list(metrics)
    values = np.asarray([max(metrics[name], 1.0e-18) for name in names])
    y = np.arange(len(names))
    ax.hlines(y, 1.0e-18, values, color="0.65", lw=1.0)
    ax.plot(values, y, "o", color="#2166ac", ms=5)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("reported relative/absolute error")
    ax.set_title("(c) Reduction and reconstruction residuals", loc="left")
    ax.grid(True, axis="x", which="both", alpha=0.22)

    ax = axes[1, 1]
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.56),
            0.92,
            0.31,
            boxstyle="round,pad=0.02",
            facecolor="#e5f5e0",
            edgecolor="#238b45",
        )
    )
    ax.text(0.08, 0.81, "Closed numerical statement", weight="bold", fontsize=9.2)
    ax.text(
        0.08,
        0.72,
        "87-state QZ descriptor = 69 finite generator roots\n"
        "+ 18 structural infinite roots; no fictitious mass.",
        fontsize=8.2,
        va="top",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.12),
            0.92,
            0.30,
            boxstyle="round,pad=0.02",
            facecolor="#fee5d9",
            edgecolor="#cb181d",
        )
    )
    ax.text(0.08, 0.36, "Open analytical statement", weight="bold", fontsize=9.2)
    ax.text(
        0.08,
        0.27,
        "The present receipt does not yet prove controlled\n"
        "HCP slow-root convergence to Bai's scalar spectrum.",
        fontsize=8.2,
        va="top",
    )
    ax.set_title("(d) What is verified—and what is not", loc="left")
    fig.tight_layout()
    paths = _save(fig, "fig02_classical_reduction_descriptor_verification")
    return paths, {
        "qz_status": qz["status"],
        "finite_root_count": qz["descriptor"]["finite_root_count"],
        "structural_infinite_root_count": qz["descriptor"]["structural_infinite_root_count"],
        "maximum_finite_root_relative_error": qz["finite_spectrum_match"][
            "maximum_relative_error"
        ],
        "bai_scalar_reference_status": bai["status"],
        "strict_hcp_to_bai_reduction_closed": False,
    }


def _base_history() -> list[dict[str, float | str]]:
    checkpoints = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1.40]
    card = load_card()
    card["representative_loading"]["spectral_checkpoint_shears"] = checkpoints
    records, states, model, _ = run_representative_checkpoint_states(
        card, shear_increment=2.0e-3
    )
    storages = [
        local_state92_from_material_state(state, model, simple_shear(shear))
        for state, shear in zip(states, checkpoints, strict=True)
    ]
    rows = []
    for raw, storage in zip(records, storages, strict=True):
        rows.append(
            {
                "time_s": float(raw["time_s"]),
                "shear": float(raw["shear"]),
                "first_piola_shear_Pa": float(raw["first_piola_shear_Pa"]),
                "temperature_K": float(raw["temperature_K"]),
                "maximum_slip_rate_s": float(raw["maximum_slip_rate_s"]),
                "accumulated_slip_sum": float(raw["accumulated_slip_sum"]),
                "rho_mobile_sum_m2": float(np.sum(storage.rho_mobile_m2)),
                "rho_dipole_sum_m2": float(np.sum(storage.rho_dipole_m2)),
                "plastic_work_J_m3": float(raw["plastic_work_J_m3"]),
                "generated_heat_J_m3": float(raw["generated_heat_J_m3"]),
                "stored_energy_J_m3": float(raw["stored_energy_J_m3"]),
                "dominant_slip_family": str(raw["dominant_slip_family"]),
            }
        )
    return rows


def figure03_base_history(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    time = np.asarray([row["time_s"] for row in rows]) * 1.0e6
    stress = np.asarray([row["first_piola_shear_Pa"] for row in rows]) * 1.0e-6
    temperature = np.asarray([row["temperature_K"] for row in rows])
    slip_rate = np.asarray([row["maximum_slip_rate_s"] for row in rows])
    slip = np.asarray([row["accumulated_slip_sum"] for row in rows])
    mobile = np.asarray([row["rho_mobile_sum_m2"] for row in rows])
    dipole = np.asarray([row["rho_dipole_sum_m2"] for row in rows])
    work = np.asarray([row["plastic_work_J_m3"] for row in rows]) * 1.0e-6
    heat = np.asarray([row["generated_heat_J_m3"] for row in rows]) * 1.0e-6
    stored = np.asarray([row["stored_energy_J_m3"] for row in rows]) * 1.0e-6

    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.0))
    ax = axes[0, 0]
    line1 = ax.plot(time, stress, "o-", color="#2166ac", lw=1.7, label="shear stress")
    ax2 = ax.twinx()
    line2 = ax2.plot(time, temperature, "s-", color="#b2182b", lw=1.6, label="temperature")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel("first Piola shear stress [MPa]", color="#2166ac")
    ax2.set_ylabel("temperature [K]", color="#b2182b")
    ax.legend(line1 + line2, [line.get_label() for line in line1 + line2], frameon=False)
    ax.set_title("(a) Stress overshoot and thermal rise", loc="left")
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[0, 1]
    ax.loglog(time, np.maximum(slip_rate, 1.0e-30), "o-", color="#7a0177", label="max slip rate")
    ax2 = ax.twinx()
    ax2.semilogx(time, slip, "s-", color="#238b45", label="accumulated slip")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"maximum slip rate [s$^{-1}$]", color="#7a0177")
    ax2.set_ylabel("sum of accumulated slip", color="#238b45")
    ax.set_title("(b) Slip-system activation", loc="left")
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[1, 0]
    ax.loglog(time, mobile, "o-", color="#d95f0e", label="mobile")
    ax.loglog(time, dipole, "s-", color="#756bb1", label="dipole")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"summed dislocation density [m$^{-2}$]")
    ax.set_title("(c) Dislocation-state evolution", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.semilogx(time, work, "o-", color="#252525", label="plastic work")
    ax.semilogx(time, heat, "s-", color="#b2182b", label="generated heat")
    ax.semilogx(time, stored, "^-", color="#2166ac", label="passive nonthermal remainder")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"energy density [MJ m$^{-3}$]")
    ax.set_title("(d) Work--heat--storage partition", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save(fig, "fig03_evolving_base_state")
    peak = int(np.argmax(stress))
    return paths, {
        "checkpoint_count": len(rows),
        "peak_stress_MPa": float(stress[peak]),
        "peak_stress_time_s": float(rows[peak]["time_s"]),
        "terminal_temperature_K": float(temperature[-1]),
        "terminal_accumulated_slip_sum": float(slip[-1]),
        "terminal_mobile_dislocation_sum_m2": float(mobile[-1]),
        "terminal_dipole_dislocation_sum_m2": float(dipole[-1]),
        "base_history": rows,
    }


def _direction_summary(scan: list[dict[str, Any]], time_index: int) -> list[dict[str, Any]]:
    by_direction: dict[tuple[str, str], dict[str, Any]] = {}
    for item in scan:
        key = (item["scan_level"], item["direction_id"])
        gain = float(item["prefix"][time_index]["maximum_gain"])
        if key not in by_direction or gain > by_direction[key]["gain"]:
            by_direction[key] = {
                "direction_id": item["direction_id"],
                "scan_level": item["scan_level"],
                "direction_n": item["direction_n"],
                "gain": gain,
                "k_m_inv": float(item["wavenumber_m_inv"]),
            }
    return list(by_direction.values())


def _sphere_coordinates(direction: Any) -> tuple[float, float]:
    value = np.asarray(direction, dtype=float)
    value /= np.linalg.norm(value)
    return (
        float(np.degrees(np.arctan2(value[1], value[0]))),
        float(np.degrees(np.arcsin(np.clip(value[2], -1.0, 1.0)))),
    )


def _angle(direction: Any, reference: Any) -> float:
    first = np.asarray(direction, dtype=float)
    second = np.asarray(reference, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), 0.0, 1.0))))


def figure05_direction_wavenumber(
    baseline: dict[str, Any], robustness: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    scan = baseline["scan"]
    times = np.asarray(baseline["scan_contract"]["checkpoint_times_s"]) * 1.0e6
    onset_rows = _direction_summary(scan, 2)
    final_rows = _direction_summary(scan, len(times) - 1)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    for ax, rows, title in (
        (axes[0, 0], onset_rows, r"(a) Directional landscape at $2\,\mu$s"),
        (axes[0, 1], final_rows, r"(b) Directional landscape at $140\,\mu$s"),
    ):
        coordinates = np.asarray([_sphere_coordinates(row["direction_n"]) for row in rows])
        color = np.log10(np.maximum([row["gain"] for row in rows], 1.0))
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=color,
            cmap="viridis",
            s=30,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.scatter([0, 90], [0, 0], marker="+", s=90, c=["#2166ac", "#b2182b"], linewidth=1.8)
        ax.text(2, 3, "x", color="#2166ac", fontsize=8)
        ax.text(92, 3, "y", color="#b2182b", fontsize=8)
        ax.set_xlim(-185, 185)
        ax.set_ylim(-5, 95)
        ax.set_xlabel("azimuth [deg]")
        ax.set_ylabel("elevation [deg]")
        ax.set_title(title, loc="left")
        ax.grid(True, alpha=0.2)
        bar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label(r"$\log_{10}G$ after optimizing sampled $k$")

    trajectory = baseline["finite_time_selection"]["selection_trajectory"]
    trajectory_time = np.asarray([row["time_s"] for row in trajectory]) * 1.0e6
    k_star = np.asarray([row["k_star_m_inv"] for row in trajectory])
    wavelength = 2.0 * pi / k_star * 1.0e6
    ax = axes[1, 0]
    ax.loglog(trajectory_time, k_star, "o-", color="#54278f", label=r"$k^*$")
    ax2 = ax.twinx()
    ax2.loglog(trajectory_time, wavelength, "s--", color="#e6550d", label=r"$2\pi/k^*$")
    ax.set_xlabel(r"observation horizon $t$ [$\mu$s]")
    ax.set_ylabel(r"selected $k^*$ [m$^{-1}$]", color="#54278f")
    ax2.set_ylabel(r"selected wavelength [$\mu$m]", color="#e6550d")
    ax.set_title("(c) Wavenumber changes with horizon", loc="left")
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[1, 1]
    angle_x = [_angle(row["direction_n"], [1, 0, 0]) for row in trajectory]
    angle_y = [_angle(row["direction_n"], [0, 1, 0]) for row in trajectory]
    ax.semilogx(trajectory_time, angle_x, "o-", color="#2166ac", label="angle to x")
    ax.semilogx(trajectory_time, angle_y, "s-", color="#b2182b", label="angle to y")
    ax.set_xlabel(r"observation horizon $t$ [$\mu$s]")
    ax.set_ylabel("antipodal direction angle [deg]")
    ax.set_ylim(-2, 94)
    ax.set_title("(d) Directional reorganization", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    ax.text(
        0.03,
        0.06,
        "continuous onset audit:\n{x, y} within 2% at 1.5 µs",
        transform=ax.transAxes,
        fontsize=7.8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.tight_layout()
    paths = _save(fig, "fig05_direction_wavenumber_horizon_map")
    selection = robustness["continuous_selection"]
    return paths, {
        "tested_near_optimal_onset_set": selection["near_optimal_onset_set"],
        "near_optimal_tolerance": selection["near_optimal_tolerance"],
        "global_onset_near_optimal_set_certified": selection[
            "global_onset_near_optimal_set_certified"
        ],
        "final_winner_label": selection["final_winner_label"],
        "final_relative_gap_to_best_tested_competitor": selection[
            "final_relative_gap_to_best_tested_competitor"
        ],
        "trajectory": trajectory,
    }


def _coalition_log_gain(analysis: dict[str, Any], mask: int) -> float:
    match = [row for row in analysis["coalitions"] if int(row["coalition_mask"]) == mask]
    if len(match) != 1:
        raise RuntimeError(f"coalition mask {mask} not uniquely found")
    return float(match[0]["log_gain"])


def figure06_counterfactuals(
    mechanism: dict[str, Any], spectrum: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    scenarios = [
        ("full", 63),
        ("no inertia\ncross-links", 56),
        ("no thermal\ncross-links", 38),
        ("no dislocation\ncross-links", 21),
        ("all cross-links\noff", 0),
    ]
    analyses = [
        ("onset 2 µs", mechanism["analyses"]["onset_2us"]),
        ("terminal 140 µs", mechanism["analyses"]["terminal_140us"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6))
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.36
    intervention_values: dict[str, dict[str, float]] = {}
    for index, (label, analysis) in enumerate(analyses):
        full = _coalition_log_gain(analysis, 63)
        values = np.asarray([_coalition_log_gain(analysis, mask) - full for _, mask in scenarios])
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=("#67a9cf" if index == 0 else "#b2182b"),
        )
        intervention_values[label] = {
            scenario: float(value) for (scenario, _), value in zip(scenarios, values, strict=True)
        }
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xticks(x, [name for name, _ in scenarios], rotation=25, ha="right")
    ax.set_ylabel(r"change in finite-time $\log G$ from full coupling")
    ax.set_title("(a) Fixed-base cross-coupling interventions", loc="left")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    colors = {"LOCAL": "#636363", "SLIP": "#2166ac", "NYE": "#ef8a62", "COMBINED": "#b2182b"}
    for result in spectrum["critical_direction_gradient_ablations"]["results"]:
        k = np.asarray([sample["k_m_inv"] for sample in result["samples"]])
        alpha = np.asarray([sample["dominant_root_s_inv"][0] for sample in result["samples"]])
        ax.semilogx(k, alpha * 1.0e-6, lw=1.6, label=result["mode"].lower(), color=colors[result["mode"]])
    ax.set_xlabel(r"wavenumber $k$ [m$^{-1}$]")
    ax.set_ylabel(r"frozen dominant growth [$\mu$s$^{-1}$]")
    ax.set_title("(b) Independent frozen-QS gradient controls", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    paths = _save(fig, "fig06_counterfactual_mechanism_gradient_controls")
    return paths, {
        "intervention_contract": mechanism["intervention_contract"],
        "intervention_values": intervention_values,
        "mechanism_status": mechanism["status"],
        "gradient_control_scope": spectrum["scope"],
        "cross_panel_quantitative_comparison_authorized": False,
    }


def _seed_case_fields(case: dict[str, Any]) -> tuple[int, int, float, float]:
    match = re.search(r"seed_(\d+)__mode_(\d+)__amplitude_([0-9.]+)", case["case_id"])
    if match is None:
        raise RuntimeError(f"unrecognized seed audit case id {case['case_id']}")
    seed = int(match.group(1))
    mode = int(match.group(2))
    amplitude = float(match.group(3))
    observable = case["observables"]["total_slip_rate"]
    value = observable.get("accepted_fwhm_m")
    width = float("nan") if value is None else float(value) * 1.0e6
    return seed, mode, amplitude, width


def figure07_linear_nonlinear_gate(
    seed_audit: dict[str, Any], q3: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.1))
    ax = axes[0, 0]
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for case in seed_audit["cases"]:
        seed, mode, amplitude, width = _seed_case_fields(case)
        grouped.setdefault((seed, mode), []).append((amplitude, width))
    colors = {4: "#2166ac", 8: "#7a0177", 16: "#d95f0e"}
    styles = {20260821: "-", 20260822: "--", 20260823: ":"}
    for (seed, mode), values in sorted(grouped.items()):
        values.sort()
        ax.plot(
            [row[0] for row in values],
            [row[1] for row in values],
            marker="o",
            ms=3.5,
            lw=1.2,
            color=colors[mode],
            ls=styles[seed],
            label=f"seed {str(seed)[-2:]}, m≤{mode}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("perturbation amplitude")
    ax.set_ylabel(r"accepted slip-rate FWHM [$\mu$m]")
    ax.set_title("(a) Seed/spectrum/amplitude falsification matrix", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=6.2, ncol=3)

    ax = axes[0, 1]
    core = [
        case
        for case in q3["cases"]
        if case["family"] == "single_mode_core"
        and abs(float(case["amplitude_fraction_of_q3_high"]) - 0.5) < 1.0e-12
    ]
    wavelength = np.asarray([case["wavelength_reference_m"] for case in core]) * 1.0e6
    width = np.asarray([case["widths"]["state_intensity"]["width_m"] for case in core]) * 1.0e6
    ax.loglog(wavelength, width, "o", color="#b2182b", ms=6, label="measured")
    domain = np.geomspace(np.min(wavelength) * 0.75, np.max(wavelength) * 1.35, 80)
    ratio = float(np.median(width / wavelength))
    ax.loglog(domain, ratio * domain, "--", color="#2166ac", label=fr"$w={ratio:.3f}\lambda$")
    for case, x_value, y_value in zip(core, wavelength, width, strict=True):
        ax.text(x_value * 1.08, y_value, case["linear_role"], fontsize=7.2, va="center")
    ax.set_xlabel(r"imposed wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"measured state width $w$ [$\mu$m]")
    ax.set_title("(b) Width follows imposed wavelength", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    grids = [case for case in q3["cases"] if case["family"] == "grid_bridge"]
    grids.sort(key=lambda case: int(case["cells"]))
    cells = np.asarray([case["cells"] for case in grids])
    grid_width = np.asarray([case["widths"]["state_intensity"]["width_m"] for case in grids]) * 1.0e6
    ax.plot(cells, grid_width, "o-", color="#238b45", lw=1.7, label="physical width")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells per imposed wavelength")
    ax.set_ylabel(r"winner state width [$\mu$m]")
    ax.set_title("(c) Grid convergence is necessary, not sufficient", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.text(
        0.04,
        0.08,
        "converges to the width of the\nimposed single Fourier mode",
        transform=ax.transAxes,
        fontsize=7.8,
        bbox={"facecolor": "white", "edgecolor": "0.75"},
    )

    ax = axes[1, 1]
    ax.axis("off")
    gate_rows = [
        ("same-window linear gain", True),
        ("amplitude-linear single mode", True),
        ("phase-translation invariance", True),
        ("winner grid convergence", True),
        ("seed/spectrum-independent FWHM", False),
        ("multimode ≥12-cell resolution", False),
        ("emergent material width", False),
    ]
    for index, (label, passed) in enumerate(gate_rows):
        y = 0.91 - 0.12 * index
        color = "#238b45" if passed else "#cb181d"
        ax.add_patch(Rectangle((0.05, y - 0.035), 0.07, 0.07, facecolor=color, edgecolor="none"))
        ax.text(0.085, y, "✓" if passed else "×", color="white", ha="center", va="center", weight="bold")
        ax.text(0.16, y, label, va="center", fontsize=8.2)
    ax.text(
        0.05,
        0.035,
        "Decision: linear transport verified;\nmaterial-scale claim rejected.",
        fontsize=9.0,
        weight="bold",
        color="#cb181d",
    )
    ax.set_title("(d) Acceptance/denial gate", loc="left")
    fig.tight_layout()
    paths = _save(fig, "fig07_same_window_linear_nonlinear_scale_gate")
    return paths, {
        "seed_spectrum_status": seed_audit["status"],
        "seed_spectrum_failed_scientific_tests": seed_audit["scientific_falsification"][
            "failed_scientific_tests"
        ],
        "q3_status": q3["status"],
        "single_mode_width_over_input_wavelength_range": q3["scientific_interpretation"][
            "single_mode_width_over_input_wavelength_range"
        ],
        "multimode_twelve_cell_width_gate": q3["scientific_interpretation"][
            "multimode_twelve_cell_width_gate"
        ],
        "material_scale_identified": False,
    }


def main() -> int:
    _style()
    bai = _load("bai_1982_scalar_regression_reference_v5_independent_redteam_v1.json")
    qz = _load("cp_ti_dynamic_descriptor_qz_v1.json")
    baseline = _load("cp_ti_finite_time_dynamic_perturbation_v2.json")
    robustness = _load("cp_ti_continuous_spectrum_robustness_v2.json")
    mechanism = _load("cp_ti_mechanism_causality_v2.json")
    spectrum = _load("hcp_qs_spectrum_verification_seed_v1.json")
    seed_audit = _load("cp_ti_fwhm_seed_spectrum_v1.json")
    q3 = _load("cp_ti_lsbrp_q3_nonautonomous_bridge_v1.json")
    operator = _load("ijp_operator_consistency_v2.json")
    strengthening = _load("ijp_strengthening_evidence_v1.json")
    nonlinear = _load("ijp_singular_vector_nonlinear_validation_v2.json")

    figures: dict[str, Any] = {}
    paths, evidence = figure01_framework()
    figures["figure_1"] = {"paths": paths, "evidence": evidence}
    paths, evidence = figure02_verification(bai, qz)
    figures["figure_2"] = {"paths": paths, "evidence": evidence}
    base_rows = _base_history()
    paths, evidence = figure03_base_history(base_rows)
    figures["figure_3"] = {"paths": paths, "evidence": evidence}
    figures["figure_4"] = {
        "paths": {
            "pdf": "06_manuscript/ijp_spectral_hcp/figures/fig09_operator_consistency_sensitivity.pdf",
            "png": "06_manuscript/ijp_spectral_hcp/figures/fig09_operator_consistency_sensitivity.png",
        },
        "evidence": {
            "status": operator["status"],
            "gates": operator["gates"],
        },
    }
    figures["figure_5"] = {
        "paths": {
            "pdf": "06_manuscript/ijp_spectral_hcp/figures/fig08_positive_orientation_anchor_audit.pdf",
            "png": "06_manuscript/ijp_spectral_hcp/figures/fig08_positive_orientation_anchor_audit.png",
        },
        "evidence": {
            "status": strengthening["status"],
            "gates": strengthening["gates"],
        },
    }
    paths, evidence = figure05_direction_wavenumber(baseline, robustness)
    figures["figure_6"] = {"paths": paths, "evidence": evidence}
    paths, evidence = figure06_counterfactuals(mechanism, spectrum)
    figures["figure_7"] = {"paths": paths, "evidence": evidence}
    figures["figure_8"] = {
        "paths": {
            "pdf": "06_manuscript/ijp_spectral_hcp/figures/fig10_singular_vector_nonlinear_validation.pdf",
            "png": "06_manuscript/ijp_spectral_hcp/figures/fig10_singular_vector_nonlinear_validation.png",
        },
        "evidence": {"status": nonlinear["status"], "gates": nonlinear["gates"]},
    }
    paths, evidence = figure07_linear_nonlinear_gate(seed_audit, q3)
    figures["figure_9"] = {"paths": paths, "evidence": evidence}

    receipt = {
        "schema": "IJP_CORE_FIGURE_SET_V1",
        "status": "NINE_CORE_FIGURES_BUILT_WITH_EXPLICIT_POSITIVE_AND_FALSIFYING_EVIDENCE",
        "paper_spine": (
            "A crystal-resolved non-autonomous finite-time propagator identifies which "
            "direction--wavenumber--state perturbation accumulates the largest observable "
            "gain over an evolving HCP loading path; the selector changes with horizon, "
            "while the current nonlinear bridge does not identify a material band width."
        ),
        "figures": figures,
        "claim_boundaries": {
            "strict_hcp_to_bai_reduction_closed": False,
            "global_direction_wavenumber_optimum_certified": False,
            "coordinate_invariant_critical_time_claimed": False,
            "experimental_localization_onset_validated": False,
            "material_band_width_identified": False,
            "like_for_like_full_state_finite_time_discrimination_supported": True,
            "singular_vector_small_amplitude_nonlinear_transport_supported": True,
        },
        "source_artifacts": [
            "05_results/bai_1982_scalar_regression_reference_v5_independent_redteam_v1.json",
            "05_results/cp_ti_dynamic_descriptor_qz_v1.json",
            "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json",
            "05_results/cp_ti_continuous_spectrum_robustness_v2.json",
            "05_results/ijp_operator_consistency_v2.json",
            "05_results/ijp_strengthening_evidence_v1.json",
            "05_results/ijp_singular_vector_nonlinear_validation_v2.json",
            "05_results/cp_ti_mechanism_causality_v2.json",
            "05_results/hcp_qs_spectrum_verification_seed_v1.json",
            "05_results/cp_ti_fwhm_seed_spectrum_v1.json",
            "05_results/cp_ti_lsbrp_q3_nonautonomous_bridge_v1.json",
        ],
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "receipt": str(RECEIPT),
                "status": receipt["status"],
                "figures": {key: value["paths"] for key, value in figures.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
