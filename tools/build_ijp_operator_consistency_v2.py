"""Build the like-for-like operator, norm and mechanism audit for the IJP paper.

The primary comparison is deliberately full state:

    log ||D^{-1} Phi D||_2  versus  integral alpha(A(t)) dt.

Both quantities therefore act on the same 69-state space.  Projected
input--output gains are retained as conditioned localization diagnostics, but
are not subtracted from the full-state spectral abscissa.  The script consumes
the released generators and propagators so that every revised manuscript
number remains traceable to the archived V1 evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig, expm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "05_results/ijp_strengthening_evidence_v1_arrays.npz"
RESULT = ROOT / "05_results/ijp_operator_consistency_v2.json"
ARRAYS = ROOT / "05_results/ijp_operator_consistency_v2_arrays.npz"
TABLE_DIR = ROOT / "05_results/ijp_operator_consistency_v2"
FIGURE_STEM = (
    ROOT
    / "06_manuscript/ijp_spectral_hcp/figures/fig09_operator_consistency_sensitivity"
)

BRANCHES = ("onset_x", "onset_y", "terminal_y")
SUBSTEPS = 4
TINY = np.finfo(float).tiny

FULL = np.arange(69, dtype=int)
MECHANICAL = np.arange(0, 6, dtype=int)
CONSTITUTIVE = np.arange(6, 69, dtype=int)
TEMPERATURE = np.asarray([6], dtype=int)
PLASTIC_CHART = np.arange(7, 15, dtype=int)
RHO_MOBILE = np.arange(15, 33, dtype=int)
RHO_DIPOLE = np.arange(33, 51, dtype=int)
SIGNED_SLIP = np.arange(51, 69, dtype=int)
DISLOCATION = np.r_[RHO_MOBILE, RHO_DIPOLE]
PLASTIC_KINEMATIC = np.r_[PLASTIC_CHART, SIGNED_SLIP]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    result[1:] = np.cumsum(
        0.5 * (values[:-1] + values[1:]) * np.diff(times)
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaled_generator(generator: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (generator * scales[None, :]) / scales[:, None]


def rescale_propagator(
    baseline_propagator: np.ndarray,
    baseline_scales: np.ndarray,
    alternative_scales: np.ndarray,
) -> np.ndarray:
    """Apply D_alt^{-1} Phi D_alt without reconstructing dimensional Phi."""

    left = baseline_scales / alternative_scales
    right = alternative_scales / baseline_scales
    return left[:, None] * baseline_propagator * right[None, :]


def log_gain(
    propagator: np.ndarray, output_indices: np.ndarray, input_indices: np.ndarray
) -> float:
    observed = propagator[np.ix_(output_indices, input_indices)]
    value = float(np.linalg.svd(observed, compute_uv=False)[0])
    return float(np.log(max(value, TINY)))


def propagate_history(
    generators: np.ndarray, times: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return all baseline-scaled prefix propagators and their full-state gains."""

    phi = np.eye(69, dtype=np.complex128)
    propagators = [phi.copy()]
    gains = [0.0]
    scaled = [scaled_generator(value, scales) for value in generators]
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index])
        substep_dt = dt / SUBSTEPS
        for substep in range(SUBSTEPS):
            fraction = (substep + 0.5) / SUBSTEPS
            midpoint = (
                (1.0 - fraction) * scaled[index]
                + fraction * scaled[index + 1]
            )
            phi = expm(midpoint * substep_dt) @ phi
        propagators.append(phi.copy())
        gains.append(log_gain(phi, FULL, FULL))
    return np.asarray(propagators), np.asarray(gains)


def projected_control() -> dict[str, Any]:
    """A control showing when a projected frozen comparison is legitimate."""

    times = np.linspace(0.0, 2.0, 401)
    dt = float(times[1] - times[0])
    eigenvalues = np.asarray([1.8, 0.7, -0.4], dtype=float)
    generator = np.diag(eigenvalues)
    phi = np.eye(3)
    for _ in times[:-1]:
        phi = expm(generator * dt) @ phi
    cases = []
    for identifier, indices in (
        ("invariant_projection_including_full_leader", np.asarray([0, 1])),
        ("invariant_projection_excluding_full_leader", np.asarray([1, 2])),
    ):
        projected_log_gain = log_gain(phi, indices, indices)
        restricted_integral = float(2.0 * np.max(eigenvalues[indices]))
        full_integral = float(2.0 * np.max(eigenvalues))
        cases.append(
            {
                "case_id": identifier,
                "indices": indices.tolist(),
                "projected_log_gain": projected_log_gain,
                "restricted_frozen_integral": restricted_integral,
                "full_state_frozen_integral": full_integral,
                "restricted_equality_error": projected_log_gain
                - restricted_integral,
                "full_state_mismatch": projected_log_gain - full_integral,
            }
        )
    return {
        "operator": "stationary diagonal normal generator",
        "interpretation": (
            "A projected frozen comparison is exact only because the selected "
            "subspaces are invariant and the frozen rate is restricted to the "
            "same subspace. Excluding the full-state leader intentionally exposes "
            "the error made by comparing a projected gain with the full-state abscissa."
        ),
        "cases": cases,
        "gate": all(abs(item["restricted_equality_error"]) <= 2.0e-12 for item in cases),
    }


def modal_diagnostics(
    generators: np.ndarray, scales: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    alphas = []
    numerical_abscissae = []
    frequencies = []
    condition_numbers = []
    input_fractions = []
    output_fractions = []
    rotations = [0.0]
    right_vectors: list[np.ndarray] = []
    scaled = [scaled_generator(value, scales) for value in generators]

    for matrix in scaled:
        values, left, right = eig(matrix, left=True, right=True)
        dominant = int(np.argmax(values.real))
        rv = right[:, dominant]
        lv = left[:, dominant]
        rv /= np.linalg.norm(rv)
        lv /= np.linalg.norm(lv)
        overlap = abs(complex(np.vdot(lv, rv)))
        condition_numbers.append(float(1.0 / max(overlap, TINY)))
        input_fractions.append(float(np.linalg.norm(rv[CONSTITUTIVE]) ** 2))
        output_fractions.append(float(np.linalg.norm(lv[CONSTITUTIVE]) ** 2))
        alphas.append(float(values[dominant].real))
        frequencies.append(float(abs(values[dominant].imag)))
        hermitian = 0.5 * (matrix + matrix.conj().T)
        numerical_abscissae.append(float(np.linalg.eigvalsh(hermitian)[-1]))
        if right_vectors:
            previous = right_vectors[-1]
            cosine = np.clip(abs(complex(np.vdot(previous, rv))), 0.0, 1.0)
            rotations.append(float(np.degrees(np.arccos(cosine))))
        right_vectors.append(rv)

    adjacent_commutators = []
    for first, second in zip(scaled[:-1], scaled[1:], strict=True):
        numerator = np.linalg.norm(second @ first - first @ second, ord="fro")
        denominator = max(
            float(np.linalg.norm(first, ord="fro") * np.linalg.norm(second, ord="fro")),
            TINY,
        )
        adjacent_commutators.append(float(numerator / denominator))

    arrays = {
        "spectral_abscissa_s_inv": np.asarray(alphas),
        "numerical_abscissa_s_inv": np.asarray(numerical_abscissae),
        "dominant_frequency_s_inv": np.asarray(frequencies),
        "dominant_mode_condition_number": np.asarray(condition_numbers),
        "dominant_right_constitutive_fraction": np.asarray(input_fractions),
        "dominant_left_constitutive_fraction": np.asarray(output_fractions),
        "adjacent_leader_rotation_deg": np.asarray(rotations),
        "adjacent_commutator": np.r_[0.0, adjacent_commutators],
    }
    summary = {
        "maximum_dominant_mode_condition_number": float(np.max(condition_numbers)),
        "median_dominant_mode_condition_number": float(np.median(condition_numbers)),
        "maximum_adjacent_leader_rotation_deg": float(np.max(rotations)),
        "median_adjacent_leader_rotation_deg": float(np.median(rotations[1:])),
        "maximum_normalized_adjacent_commutator": float(
            max(adjacent_commutators, default=0.0)
        ),
        "median_normalized_adjacent_commutator": float(
            np.median(adjacent_commutators) if adjacent_commutators else 0.0
        ),
        "minimum_dominant_right_constitutive_fraction": float(
            np.min(input_fractions)
        ),
        "minimum_dominant_left_constitutive_fraction": float(
            np.min(output_fractions)
        ),
    }
    return summary, arrays


def magnus_and_ordering(
    generators: np.ndarray, times: np.ndarray, scales: np.ndarray
) -> dict[str, Any]:
    scaled = [scaled_generator(value, scales) for value in generators]
    midpoints = []
    increments = []
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index])
        midpoints.append(0.5 * (scaled[index] + scaled[index + 1]))
        increments.append(dt)

    omega1 = np.zeros((69, 69), dtype=np.complex128)
    omega2 = np.zeros_like(omega1)
    cumulative = np.zeros_like(omega1)
    forward = np.eye(69, dtype=np.complex128)
    steps = []
    for matrix, dt in zip(midpoints, increments, strict=True):
        delta = matrix * dt
        omega2 += 0.5 * (delta @ cumulative - cumulative @ delta)
        cumulative += delta
        omega1 += delta
        step = expm(delta)
        steps.append(step)
        forward = step @ forward
    reverse = np.eye(69, dtype=np.complex128)
    for step in reversed(steps):
        reverse = step @ reverse
    first_magnus = expm(omega1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        second_magnus = expm(omega1 + omega2)
    second_finite = bool(np.all(np.isfinite(second_magnus)))
    return {
        "first_magnus_relative_size": 1.0,
        "second_to_first_magnus_frobenius_ratio": float(
            np.linalg.norm(omega2, ord="fro")
            / max(np.linalg.norm(omega1, ord="fro"), TINY)
        ),
        "forward_one_step_log_gain": log_gain(forward, FULL, FULL),
        "reverse_order_one_step_log_gain": log_gain(reverse, FULL, FULL),
        "first_magnus_log_gain": log_gain(first_magnus, FULL, FULL),
        "second_magnus_finite": second_finite,
        "second_magnus_log_gain": (
            log_gain(second_magnus, FULL, FULL) if second_finite else None
        ),
    }


def norm_variants(scales: np.ndarray) -> dict[str, np.ndarray]:
    variants = {"baseline": scales.copy()}
    groups = {
        "mechanical": MECHANICAL,
        "thermal": TEMPERATURE,
        "dislocation": DISLOCATION,
        "plastic_kinematic": PLASTIC_KINEMATIC,
        "all_constitutive": CONSTITUTIVE,
    }
    for group, indices in groups.items():
        for factor in (0.5, 2.0):
            value = scales.copy()
            value[indices] *= factor
            variants[f"{group}_x{factor:g}"] = value
    return variants


def selectors() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "full_to_full": (FULL, FULL),
        "constitutive_to_constitutive": (CONSTITUTIVE, CONSTITUTIVE),
        "full_to_constitutive": (CONSTITUTIVE, FULL),
        "constitutive_to_full": (FULL, CONSTITUTIVE),
        "mechanical_to_mechanical": (MECHANICAL, MECHANICAL),
        "mechanical_to_constitutive": (CONSTITUTIVE, MECHANICAL),
        "constitutive_to_temperature": (TEMPERATURE, CONSTITUTIVE),
    }


def configure_style() -> None:
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


def make_figure(
    times: np.ndarray,
    branches: dict[str, Any],
    mechanism_arrays: dict[str, np.ndarray],
    sensitivity_rows: list[dict[str, Any]],
) -> None:
    configure_style()
    terminal = branches["terminal_y"]
    history = terminal["history"]
    time_us = times * 1.0e6
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))

    ax = axes[0, 0]
    line_alpha = ax.plot(
        time_us,
        history["spectral_abscissa_s_inv"] * 1.0e-6,
        color="#2166ac",
        label=r"$\alpha$",
    )
    twin = ax.twinx()
    line_mu = twin.plot(
        time_us,
        history["numerical_abscissa_s_inv"] * 1.0e-6,
        color="#d95f02",
        label=r"$\mu_2(\widehat A)$",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel(r"spectral rate $\alpha$ [$\mu$s$^{-1}$]", color="#2166ac")
    twin.set_ylabel(r"logarithmic rate $\mu_2$ [$\mu$s$^{-1}$]", color="#d95f02")
    ax.set_title("(a) Spectrum and same-norm logarithmic rate")
    ax.legend(line_alpha + line_mu, [item.get_label() for item in line_alpha + line_mu], frameon=False)
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[0, 1]
    ax.plot(time_us, history["full_log_gain"], lw=2.0, label=r"full $\log\|\widehat\Phi\|_2$")
    ax.plot(time_us, history["observed_log_gain"], ls=":", label="projected diagnostic")
    ax.plot(time_us, history["frozen_integral"], ls="--", label=r"$\int\alpha\,dt$")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel("log gain")
    ax.set_title("(b) Like-for-like primary comparison")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[1, 0]
    condition = mechanism_arrays["terminal_y_dominant_mode_condition_number"]
    rotation = mechanism_arrays["terminal_y_adjacent_leader_rotation_deg"]
    commutator = mechanism_arrays["terminal_y_adjacent_commutator"]
    ax.semilogy(time_us, condition, label=r"modal $\kappa$")
    ax.semilogy(time_us, np.maximum(rotation, 1.0e-6), label="leader rotation [deg]")
    ax.semilogy(time_us, np.maximum(commutator, 1.0e-12), label="adjacent commutator")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$ [$\mu$s]")
    ax.set_ylabel("diagnostic magnitude")
    ax.set_title("(c) Non-normal/non-autonomous mechanisms")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[1, 1]
    shown_variants = [
        "baseline",
        "mechanical_x0.5",
        "mechanical_x2",
        "thermal_x0.5",
        "thermal_x2",
        "all_constitutive_x0.5",
        "all_constitutive_x2",
    ]
    shown_selectors = [
        "full_to_full",
        "constitutive_to_constitutive",
        "mechanical_to_constitutive",
        "constitutive_to_temperature",
    ]
    lookup = {
        (row["norm_id"], row["selector_id"]): row["log_gain"]
        for row in sensitivity_rows
        if row["branch_id"] == "terminal_y" and row["horizon_id"] == "terminal"
    }
    matrix = np.asarray(
        [[lookup[(norm, selector)] for selector in shown_selectors] for norm in shown_variants]
    )
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(shown_selectors)), ["full/full", "const./const.", "mech.→const.", "const.→T"], rotation=25, ha="right")
    ax.set_yticks(range(len(shown_variants)), [value.replace("_", " ") for value in shown_variants])
    ax.set_title("(d) Terminal norm--selector sensitivity")
    fig.colorbar(image, ax=ax, label="log gain", fraction=0.046, pad=0.04)

    fig.tight_layout()
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    require(SOURCE.is_file(), f"missing released evidence archive: {SOURCE}")
    archive = np.load(SOURCE)
    times = np.asarray(archive["baseline_reference_times_s"], dtype=float)
    baseline_scales = np.asarray(archive["baseline_coordinate_scales"], dtype=float)
    observed = np.asarray(archive["baseline_observed_indices"], dtype=int)
    require(np.array_equal(observed, CONSTITUTIVE), "released observation set changed")

    onset_time = 1.4600e-6
    onset_index = int(np.argmin(np.abs(times - onset_time)))
    horizon_indices = {"near_onset": onset_index, "terminal": len(times) - 1}
    variants = norm_variants(baseline_scales)
    selector_map = selectors()
    branches: dict[str, Any] = {}
    mechanism_arrays: dict[str, np.ndarray] = {}
    prefix_arrays: dict[str, np.ndarray] = {"times_s": times}
    sensitivity_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []

    for branch in BRANCHES:
        generators = np.asarray(archive[f"baseline_{branch}_generators"])
        released_phi = np.asarray(archive[f"baseline_{branch}_propagator"])
        prefix_phi, full_log = propagate_history(generators, times, baseline_scales)
        endpoint_error = float(
            np.linalg.norm(prefix_phi[-1] - released_phi, ord="fro")
            / max(np.linalg.norm(released_phi, ord="fro"), TINY)
        )
        require(endpoint_error <= 2.0e-10, f"{branch} released propagator mismatch")

        modal_summary, modal_arrays = modal_diagnostics(generators, baseline_scales)
        alpha = modal_arrays["spectral_abscissa_s_inv"]
        mu = modal_arrays["numerical_abscissa_s_inv"]
        frozen = cumulative_trapezoid(alpha, times)
        log_norm_bound = cumulative_trapezoid(mu, times)
        observed_log = np.asarray(
            [log_gain(value, CONSTITUTIVE, CONSTITUTIVE) for value in prefix_phi]
        )
        full_discrepancy = full_log - frozen
        observed_diagnostic_difference = observed_log - frozen
        leakage_out = []
        leakage_in = []
        for generator in generators:
            matrix = scaled_generator(generator, baseline_scales)
            oo = np.linalg.norm(matrix[np.ix_(CONSTITUTIVE, CONSTITUTIVE)], ord="fro")
            leakage_out.append(
                float(
                    np.linalg.norm(matrix[np.ix_(MECHANICAL, CONSTITUTIVE)], ord="fro")
                    / max(oo, TINY)
                )
            )
            leakage_in.append(
                float(
                    np.linalg.norm(matrix[np.ix_(CONSTITUTIVE, MECHANICAL)], ord="fro")
                    / max(oo, TINY)
                )
            )
        ordering = magnus_and_ordering(generators, times, baseline_scales)
        history = {
            "spectral_abscissa_s_inv": alpha,
            "numerical_abscissa_s_inv": mu,
            "full_log_gain": full_log,
            "observed_log_gain": observed_log,
            "frozen_integral": frozen,
            "log_norm_integral": log_norm_bound,
            "full_log_discrepancy": full_discrepancy,
            "observed_minus_full_abscissa_diagnostic": observed_diagnostic_difference,
        }
        branches[branch] = {
            "direction_n": np.asarray(archive[f"baseline_{branch}_direction_n"]).tolist(),
            "k_m_inv": float(archive[f"baseline_{branch}_k_m_inv"][0]),
            "released_propagator_relative_error": endpoint_error,
            "primary_contract": {
                "state_space": "complete reduced 69-state generator",
                "norm": "fixed D-scaled Euclidean norm",
                "input_indices": FULL.tolist(),
                "output_indices": FULL.tolist(),
            },
            "projected_contract": {
                "role": "conditioned localization diagnostic; not directly subtracted from full-state alpha",
                "input_indices": CONSTITUTIVE.tolist(),
                "output_indices": CONSTITUTIVE.tolist(),
                "subspace_invariant": False,
                "maximum_mechanical_output_leakage_ratio": float(np.max(leakage_out)),
                "maximum_mechanical_input_coupling_ratio": float(np.max(leakage_in)),
            },
            "final_full_log_gain": float(full_log[-1]),
            "final_projected_log_gain": float(observed_log[-1]),
            "final_frozen_integral": float(frozen[-1]),
            "final_primary_log_discrepancy": float(full_discrepancy[-1]),
            "maximum_abs_primary_log_discrepancy": float(np.max(np.abs(full_discrepancy))),
            "logarithmic_norm_upper_bound_satisfied": bool(
                np.all(full_log <= log_norm_bound + 2.0e-8)
            ),
            "mechanism": modal_summary,
            "ordering_surrogates": ordering,
            "history": history,
        }
        for key, value in modal_arrays.items():
            mechanism_arrays[f"{branch}_{key}"] = value
        for index, time_value in enumerate(times):
            mechanism_rows.append(
                {
                    "branch_id": branch,
                    "time_s": float(time_value),
                    **{key: float(value[index]) for key, value in modal_arrays.items()},
                }
            )
        prefix_arrays[f"{branch}_propagators"] = prefix_phi
        for key, value in history.items():
            prefix_arrays[f"{branch}_{key}"] = value

        for horizon_id, horizon_index in horizon_indices.items():
            base_phi = prefix_phi[horizon_index]
            for norm_id, active_scales in variants.items():
                phi = rescale_propagator(base_phi, baseline_scales, active_scales)
                for selector_id, (output_indices, input_indices) in selector_map.items():
                    sensitivity_rows.append(
                        {
                            "branch_id": branch,
                            "horizon_id": horizon_id,
                            "time_s": float(times[horizon_index]),
                            "norm_id": norm_id,
                            "selector_id": selector_id,
                            "input_dimension": int(len(input_indices)),
                            "output_dimension": int(len(output_indices)),
                            "log_gain": log_gain(phi, output_indices, input_indices),
                        }
                    )

    # Winner stability is reported for each norm/selector/horizon combination.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in sensitivity_rows:
        key = (row["horizon_id"], row["norm_id"], row["selector_id"])
        grouped.setdefault(key, []).append(row)
    sensitivity_summary = []
    for (horizon, norm_id, selector_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: item["log_gain"], reverse=True)
        sensitivity_summary.append(
            {
                "horizon_id": horizon,
                "norm_id": norm_id,
                "selector_id": selector_id,
                "winner_branch": ordered[0]["branch_id"],
                "winner_log_gain": ordered[0]["log_gain"],
                "runner_up_log_gap": ordered[0]["log_gain"] - ordered[1]["log_gain"],
            }
        )

    write_csv(TABLE_DIR / "norm_selector_sensitivity.csv", sensitivity_rows)
    write_csv(TABLE_DIR / "norm_selector_winners.csv", sensitivity_summary)
    write_csv(TABLE_DIR / "mechanism_history.csv", mechanism_rows)
    array_payload = dict(prefix_arrays)
    array_payload.update(mechanism_arrays)
    np.savez_compressed(ARRAYS, **array_payload)

    # Convert arrays to compact JSON summaries only; full histories remain in NPZ/CSV.
    compact_branches = {}
    for branch, record in branches.items():
        compact = {key: value for key, value in record.items() if key != "history"}
        compact_branches[branch] = compact
    control = projected_control()
    gates = {
        "full_state_like_for_like_contract": all(
            item["primary_contract"]["input_indices"] == FULL.tolist()
            and item["primary_contract"]["output_indices"] == FULL.tolist()
            for item in compact_branches.values()
        ),
        "released_propagators_reproduced": all(
            item["released_propagator_relative_error"] <= 2.0e-10
            for item in compact_branches.values()
        ),
        "projected_control_pass": bool(control["gate"]),
        "observed_subspace_not_misrepresented_as_closed": all(
            not item["projected_contract"]["subspace_invariant"]
            for item in compact_branches.values()
        ),
        "logarithmic_norm_bounds_full_gain": all(
            item["logarithmic_norm_upper_bound_satisfied"]
            for item in compact_branches.values()
        ),
        "primary_non_equivalence_on_all_reported_branches": all(
            abs(item["final_primary_log_discrepancy"]) >= 0.10
            for item in compact_branches.values()
        ),
    }
    report = {
        "schema": "IJP_OPERATOR_CONSISTENCY_NORM_MECHANISM_V2",
        "status": "ALL_OPERATOR_CONSISTENCY_GATES_PASS" if all(gates.values()) else "OPEN_OPERATOR_CONSISTENCY_GATES",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "primary_comparison": (
            "full-state D-scaled propagator gain versus the integrated full-state "
            "spectral abscissa on the same 69-state generator"
        ),
        "claim_boundary": (
            "Projected gains quantify declared input-output questions. Because the "
            "constitutive subspace is not invariant, they are not interpreted as a "
            "like-for-like frozen-spectrum discrepancy."
        ),
        "projected_positive_control": control,
        "branches": compact_branches,
        "norm_selector_summary": sensitivity_summary,
        "gates": gates,
        "outputs": {
            "arrays": ARRAYS.relative_to(ROOT).as_posix(),
            "mechanism_table": (TABLE_DIR / "mechanism_history.csv").relative_to(ROOT).as_posix(),
            "sensitivity_table": (TABLE_DIR / "norm_selector_sensitivity.csv").relative_to(ROOT).as_posix(),
            "winner_table": (TABLE_DIR / "norm_selector_winners.csv").relative_to(ROOT).as_posix(),
            "figure_pdf": FIGURE_STEM.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            "figure_png": FIGURE_STEM.with_suffix(".png").relative_to(ROOT).as_posix(),
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figure(times, branches, mechanism_arrays, sensitivity_rows)
    print(json.dumps({"status": report["status"], "gates": gates, "result": str(RESULT)}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
