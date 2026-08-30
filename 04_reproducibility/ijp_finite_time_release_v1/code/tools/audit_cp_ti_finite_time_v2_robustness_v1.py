"""Independent numerical-robustness audit of the CP-Ti finite-time V2 runner.

The original V2 result is retained as an immutable comparison record.  This
audit rebuilds one common 129-state physical history, reuses its exported
spectral points and assembled fixed-(n,k) operators, and separates four
numerical questions:

* base-history density;
* exponential-midpoint substeps;
* factor-two changes of the declared dimensionless coordinate scales; and
* a small nested local direction/wavenumber probe around registered branches.

The local probes are deliberately not advertised as a global nonconvex search.
All reported times and selectors remain fixed-model numerical diagnostics, not
material-level TA2 onset, critical strain, or finite-bandwidth predictions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from math import e, pi
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcp_cp_gnd.dynamic_crystal_perturbation_v1 import (  # noqa: E402
    assemble_dynamic_crystal_operator_v1,
    finite_time_amplification_history,
)
from tools.audit_cp_ti_continuous_spectrum_robustness_v1 import (  # noqa: E402
    SpectrumAudit,
    angle_degrees,
    relative_change,
)
from tools.audit_cp_ti_continuous_spectrum_robustness_v2 import (  # noqa: E402
    tangent_direction,
)


SCHEMA = "CP_TI_FINITE_TIME_V2_ROBUSTNESS_AUDIT_V1"
RESULT = ROOT / "05_results/cp_ti_finite_time_v2_robustness_v1.json"
SUMMARY = ROOT / "05_results/cp_ti_finite_time_v2_robustness_v1.md"
CASES_CSV = ROOT / "05_results/cp_ti_finite_time_v2_robustness_v1.csv"
K_PROFILE_CSV = ROOT / "05_results/cp_ti_finite_time_v2_robustness_v1_k_profiles.csv"
BASELINE = ROOT / "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json"
BRANCH_CACHE = (
    ROOT
    / "05_results/cp_ti_continuous_spectrum_robustness_v2"
    / "continuous_branch_cache_v1.json"
)
MATERIAL_CARD = ROOT / "config/cp_ti_grade1_dynamic_hcp_v1.json"

REFERENCE_CONTEXT = "factor16"
HISTORY_EXTENSION_CONTEXT = "factor32"
REFERENCE_SUBSTEPS = 4
SCALE_AUDIT_SUBSTEPS = 2
SEARCH_SUBSTEPS = 1
GAIN_THRESHOLD = float(e)
ONSET_PROBE_TIME_S = 1.5e-6
ONSET_PROPAGATION_END_S = 2.0e-6
OBSERVED = np.arange(6, 69, dtype=int)

# Predeclared numerical gates.  They are approximation/sensitivity tolerances,
# not material calibration tolerances.
HISTORY_RELATIVE_GATE = 0.02
SUBSTEP_RELATIVE_GATE = 1.0e-3
SCALE_TC_RELATIVE_GATE = 0.05
SCALE_LOG_GAIN_RELATIVE_GATE = 0.10
SEARCH_GAIN_RELATIVE_GATE = 0.02
SEARCH_ANGLE_GATE_DEG = 2.5
SEARCH_K_RELATIVE_GATE = 0.10
NEAR_OPTIMAL_GAIN_GAP = 0.02
K_PROFILE_BOUNDS_M_INV = (3.0e2, 3.0e5)
K_PROFILE_POINT_COUNT = 33


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_time_index(times: np.ndarray, target_s: float) -> int:
    matches = np.flatnonzero(np.isclose(times, target_s, rtol=0.0, atol=2.0e-18))
    require(matches.size == 1, f"time {target_s:.9g} s is not a unique checkpoint")
    return int(matches[0])


def _safe_relative(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return relative_change(float(first), float(second))


def _three_level_extrapolation(
    values: Iterable[float],
    *,
    labels: tuple[str, str, str] = ("65_states", "129_states", "257_states"),
) -> dict[str, Any]:
    first, second, third = (float(value) for value in values)
    coarse_difference = second - first
    fine_difference = third - second
    ratio = (
        None
        if abs(fine_difference) <= np.finfo(float).tiny
        else abs(coarse_difference / fine_difference)
    )
    observed_order = (
        None if ratio is None or ratio <= 0.0 else float(np.log2(ratio))
    )
    denominator = coarse_difference - fine_difference
    extrapolated = (
        None
        if abs(denominator)
        <= 64.0 * np.finfo(float).eps * max(abs(first), abs(second), abs(third), 1.0)
        else float(third + fine_difference**2 / denominator)
    )
    return {
        "labels": list(labels),
        "values": [first, second, third],
        "coarse_difference": coarse_difference,
        "fine_difference": fine_difference,
        "absolute_difference_ratio": ratio,
        "observed_order_diagnostic": observed_order,
        "aitken_extrapolated_limit_diagnostic": extrapolated,
        "interpretation": (
            "Three-level diagnostic only; no asymptotic order is assumed or certified."
        ),
    }


def _unit(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    require(result.shape == (3,), "direction must have three components")
    norm = float(np.linalg.norm(result))
    require(np.isfinite(norm) and norm > 0.0, "direction must have positive finite norm")
    return result / norm


@dataclass(frozen=True)
class Candidate:
    identifier: str
    direction_n: tuple[float, float, float]
    k_m_inv: float
    source: str
    boundary_probe: bool = False

    @classmethod
    def from_record(cls, identifier: str, record: dict[str, Any], source: str) -> "Candidate":
        normal = _unit(record["direction_n"])
        k_value = float(record["k_m_inv"])
        require(np.isfinite(k_value) and k_value > 0.0, "candidate k must be positive")
        return cls(identifier, tuple(float(x) for x in normal), k_value, source)

    def json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.identifier,
            "direction_n": list(self.direction_n),
            "k_m_inv": self.k_m_inv,
            "wavelength_2pi_over_k_m": float(2.0 * pi / self.k_m_inv),
            "source": self.source,
            "boundary_probe": self.boundary_probe,
        }


class FixedContractEvaluator:
    """Reuse actual spectral points and fixed-pair operators across cases."""

    def __init__(self, audit: SpectrumAudit) -> None:
        self.audit = audit
        self.operator_cache: dict[tuple[Any, ...], list[Any]] = {}
        self.history_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.operator_assembly_count = 0
        self.propagation_count = 0

    @staticmethod
    def _candidate_key(candidate: Candidate) -> tuple[Any, ...]:
        return (
            *np.round(np.asarray(candidate.direction_n), 13),
            round(float(np.log(candidate.k_m_inv)), 13),
        )

    def operators(self, context_name: str, candidate: Candidate) -> list[Any]:
        key = (context_name, *self._candidate_key(candidate))
        if key not in self.operator_cache:
            context = self.audit.contexts[context_name]
            normal = np.asarray(candidate.direction_n, dtype=float)
            self.operator_cache[key] = [
                assemble_dynamic_crystal_operator_v1(
                    point,
                    wavenumber_m_inv=candidate.k_m_inv,
                    direction_n=normal,
                    admission=self.audit.admission,
                )
                for point in context["points"]
            ]
            self.operator_assembly_count += len(context["points"])
        return self.operator_cache[key]

    def history(
        self,
        context_name: str,
        prefix_index: int,
        candidate: Candidate,
        *,
        scales: np.ndarray,
        scale_id: str,
        substeps: int,
    ) -> dict[str, Any]:
        key = (
            context_name,
            prefix_index,
            *self._candidate_key(candidate),
            scale_id,
            int(substeps),
        )
        if key in self.history_cache:
            return self.history_cache[key]
        context = self.audit.contexts[context_name]
        raw = finite_time_amplification_history(
            self.operators(context_name, candidate)[: prefix_index + 1],
            context["times"][: prefix_index + 1],
            coordinate_scales=scales,
            gain_threshold=GAIN_THRESHOLD,
            input_indices=OBSERVED,
            output_indices=OBSERVED,
            integration_substeps_per_interval=substeps,
        )
        value = {
            "critical_time_s": raw["critical_time_s"],
            "gain": float(raw["final_gain"]),
            "log_gain": float(raw["final_log_gain"]),
            "prefix": raw["prefix"],
            "dominant_full_state_output_mechanism": raw[
                "full_state_output_mechanism_participation"
            ]["dominant_coarse"],
        }
        self.history_cache[key] = value
        self.propagation_count += 1
        return value


def _gain_at(history: dict[str, Any], target_s: float) -> float:
    for row in history["prefix"]:
        if np.isclose(row["time_s"], target_s, rtol=0.0, atol=2.0e-18):
            return float(row["maximum_gain"])
    raise RuntimeError(f"history has no exact checkpoint at {target_s:.9g} s")


def _near_optimal(values: dict[str, float]) -> list[str]:
    maximum = max(values.values())
    return sorted(
        identifier
        for identifier, value in values.items()
        if (maximum - value) / maximum <= NEAR_OPTIMAL_GAIN_GAP
    )


def _selection_case(
    evaluator: FixedContractEvaluator,
    context_name: str,
    *,
    substeps: int,
    scales: np.ndarray,
    scale_id: str,
    onset_candidates: Iterable[Candidate],
    final_candidates: Iterable[Candidate],
    onset_probe_time_s: float,
) -> dict[str, Any]:
    context = evaluator.audit.contexts[context_name]
    times = np.asarray(context["times"], dtype=float)
    onset_end = _exact_time_index(times, ONSET_PROPAGATION_END_S)
    onset_values: dict[str, tuple[Candidate, dict[str, Any], float]] = {}
    for candidate in onset_candidates:
        history = evaluator.history(
            context_name,
            onset_end,
            candidate,
            scales=scales,
            scale_id=scale_id,
            substeps=substeps,
        )
        probe_gain = _gain_at(history, onset_probe_time_s)
        onset_values[candidate.identifier] = (candidate, history, probe_gain)
    onset_id = max(onset_values, key=lambda key: onset_values[key][2])
    onset_candidate, onset_history, onset_gain = onset_values[onset_id]
    tc_candidates = {
        identifier: value[1]["critical_time_s"]
        for identifier, value in onset_values.items()
        if value[1]["critical_time_s"] is not None
    }
    tc = min(tc_candidates.values()) if tc_candidates else None
    tc_source = min(tc_candidates, key=tc_candidates.get) if tc_candidates else None

    final_index = len(times) - 1
    final_values: dict[str, tuple[Candidate, dict[str, Any]]] = {}
    for candidate in final_candidates:
        final_values[candidate.identifier] = (
            candidate,
            evaluator.history(
                context_name,
                final_index,
                candidate,
                scales=scales,
                scale_id=scale_id,
                substeps=substeps,
            ),
        )
    final_id = max(final_values, key=lambda key: final_values[key][1]["gain"])
    final_candidate, final_history = final_values[final_id]
    onset_probe_gains = {
        identifier: float(value[2]) for identifier, value in onset_values.items()
    }
    onset_common_time_gains = {
        identifier: _gain_at(value[1], ONSET_PROPAGATION_END_S)
        for identifier, value in onset_values.items()
    }
    onset_common_time_winner = max(
        onset_common_time_gains, key=onset_common_time_gains.get
    )
    onset_common_time_envelope_gain = onset_common_time_gains[
        onset_common_time_winner
    ]
    return {
        "context": context_name,
        "base_state_count": len(times),
        "substeps_per_interval": int(substeps),
        "effective_midpoint_steps": int((len(times) - 1) * substeps),
        "scale_id": scale_id,
        "onset_probe_time_s": float(onset_probe_time_s),
        "critical_time_s": tc,
        "critical_time_source_candidate": tc_source,
        "onset_candidate": {
            **onset_candidate.json(),
            "gain": float(onset_gain),
            "log_gain": float(np.log(onset_gain)),
            "critical_time_s": onset_history["critical_time_s"],
            "dominant_full_state_output_mechanism": onset_history[
                "dominant_full_state_output_mechanism"
            ],
        },
        "onset_probe_gains": onset_probe_gains,
        "onset_near_optimal_candidate_set": _near_optimal(onset_probe_gains),
        "onset_common_time_s": ONSET_PROPAGATION_END_S,
        "onset_common_time_winner": onset_common_time_winner,
        "onset_common_time_gains": onset_common_time_gains,
        "onset_common_time_envelope_gain": onset_common_time_envelope_gain,
        "onset_common_time_envelope_log_gain": float(
            np.log(onset_common_time_envelope_gain)
        ),
        "final_candidate": {
            **final_candidate.json(),
            "gain": final_history["gain"],
            "log_gain": final_history["log_gain"],
            "critical_time_s": final_history["critical_time_s"],
            "dominant_full_state_output_mechanism": final_history[
                "dominant_full_state_output_mechanism"
            ],
        },
        "final_candidate_gains": {
            identifier: float(value[1]["gain"])
            for identifier, value in final_values.items()
        },
    }


def _case_changes(case: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "onset_probe_times_same": bool(
            np.isclose(
                case["onset_probe_time_s"],
                reference["onset_probe_time_s"],
                rtol=0.0,
                atol=2.0e-18,
            )
        ),
        "critical_time_relative_change": _safe_relative(
            case["critical_time_s"], reference["critical_time_s"]
        ),
        "onset_log_gain_relative_change": relative_change(
            case["onset_candidate"]["log_gain"],
            reference["onset_candidate"]["log_gain"],
        ),
        "onset_common_time_log_gain_relative_change": relative_change(
            case["onset_common_time_envelope_log_gain"],
            reference["onset_common_time_envelope_log_gain"],
        ),
        "onset_candidate_angle_deg": angle_degrees(
            case["onset_candidate"]["direction_n"],
            reference["onset_candidate"]["direction_n"],
        ),
        "onset_candidate_k_relative_change": relative_change(
            case["onset_candidate"]["k_m_inv"],
            reference["onset_candidate"]["k_m_inv"],
        ),
        "onset_near_optimal_set_same": (
            case["onset_near_optimal_candidate_set"]
            == reference["onset_near_optimal_candidate_set"]
        ),
        "final_log_gain_relative_change": relative_change(
            case["final_candidate"]["log_gain"],
            reference["final_candidate"]["log_gain"],
        ),
        "final_candidate_angle_deg": angle_degrees(
            case["final_candidate"]["direction_n"],
            reference["final_candidate"]["direction_n"],
        ),
        "final_candidate_k_relative_change": relative_change(
            case["final_candidate"]["k_m_inv"],
            reference["final_candidate"]["k_m_inv"],
        ),
    }


def _probe_candidates(
    center: Candidate,
    *,
    angular_radius_deg: float,
    k_factor: float,
    density: str,
) -> list[Candidate]:
    require(density in {"coarse", "fine"}, "unknown probe density")
    normal = np.asarray(center.direction_n, dtype=float)
    records: list[Candidate] = [center]

    def append_direction(radius_deg: float, azimuth_deg: float, label: str) -> None:
        azimuth = np.deg2rad(azimuth_deg)
        radius = np.deg2rad(radius_deg)
        direction = tangent_direction(
            normal,
            radius * float(np.cos(azimuth)),
            radius * float(np.sin(azimuth)),
        )
        records.append(
            Candidate(
                f"{center.identifier}__{label}",
                tuple(float(x) for x in _unit(direction)),
                center.k_m_inv,
                "nested_local_probe",
                boundary_probe=np.isclose(radius_deg, angular_radius_deg),
            )
        )

    for azimuth in (0.0, 90.0, 180.0, 270.0):
        append_direction(angular_radius_deg, azimuth, f"angle_outer_{azimuth:g}")
    for multiplier, label in ((1.0 / k_factor, "k_lower"), (k_factor, "k_upper")):
        records.append(
            Candidate(
                f"{center.identifier}__{label}",
                center.direction_n,
                center.k_m_inv * multiplier,
                "nested_local_probe",
                boundary_probe=True,
            )
        )
    if density == "fine":
        for azimuth in (45.0, 135.0, 225.0, 315.0):
            append_direction(
                0.5 * angular_radius_deg,
                azimuth,
                f"angle_inner_{azimuth:g}",
            )
        root_factor = float(np.sqrt(k_factor))
        for multiplier, label in (
            (1.0 / root_factor, "k_inner_lower"),
            (root_factor, "k_inner_upper"),
        ):
            records.append(
                Candidate(
                    f"{center.identifier}__{label}",
                    center.direction_n,
                    center.k_m_inv * multiplier,
                    "nested_local_probe",
                    boundary_probe=False,
                )
            )
    return records


def _search_branch(
    evaluator: FixedContractEvaluator,
    center: Candidate,
    *,
    window: str,
    angular_radius_deg: float = 5.0,
    k_factor: float = 2.0,
) -> dict[str, Any]:
    require(window in {"onset", "final"}, "unknown search window")
    context = evaluator.audit.contexts[REFERENCE_CONTEXT]
    times = np.asarray(context["times"], dtype=float)
    prefix = (
        _exact_time_index(times, ONSET_PROBE_TIME_S)
        if window == "onset"
        else len(times) - 1
    )

    def evaluate_density(density: str) -> tuple[Candidate, dict[str, Any], int]:
        candidates = _probe_candidates(
            center,
            angular_radius_deg=angular_radius_deg,
            k_factor=k_factor,
            density=density,
        )
        values = [
            (
                candidate,
                evaluator.history(
                    REFERENCE_CONTEXT,
                    prefix,
                    candidate,
                    scales=evaluator.audit.scales,
                    scale_id="baseline",
                    substeps=SEARCH_SUBSTEPS,
                ),
            )
            for candidate in candidates
        ]
        winner, history = max(values, key=lambda value: value[1]["gain"])
        return winner, history, len(candidates)

    coarse_winner, coarse_history, coarse_count = evaluate_density("coarse")
    fine_winner, fine_history, fine_count = evaluate_density("fine")
    verified = evaluator.history(
        REFERENCE_CONTEXT,
        prefix,
        fine_winner,
        scales=evaluator.audit.scales,
        scale_id="baseline",
        substeps=REFERENCE_SUBSTEPS,
    )
    changes = {
        "angle_deg": angle_degrees(
            coarse_winner.direction_n, fine_winner.direction_n
        ),
        "k_relative_change": relative_change(
            coarse_winner.k_m_inv, fine_winner.k_m_inv
        ),
        "log_gain_relative_change": relative_change(
            coarse_history["log_gain"], fine_history["log_gain"]
        ),
    }
    passed = bool(
        changes["angle_deg"] <= SEARCH_ANGLE_GATE_DEG
        and changes["k_relative_change"] <= SEARCH_K_RELATIVE_GATE
        and changes["log_gain_relative_change"] <= SEARCH_GAIN_RELATIVE_GATE
        and not fine_winner.boundary_probe
    )
    return {
        "window": window,
        "branch_id": center.identifier,
        "search_contract": {
            "kind": "axis-separated nested local stencil",
            "angular_radius_deg": angular_radius_deg,
            "k_factor": k_factor,
            "coarse_probe_count": coarse_count,
            "fine_probe_count": fine_count,
            "optimization_substeps_per_interval": SEARCH_SUBSTEPS,
            "verification_substeps_per_interval": REFERENCE_SUBSTEPS,
        },
        "coarse_winner": {
            **coarse_winner.json(),
            "gain": coarse_history["gain"],
            "log_gain": coarse_history["log_gain"],
        },
        "fine_winner": {
            **fine_winner.json(),
            "gain": fine_history["gain"],
            "log_gain": fine_history["log_gain"],
            "verified_gain": verified["gain"],
            "verified_log_gain": verified["log_gain"],
        },
        "coarse_to_fine_changes": changes,
        "passed": passed,
    }


def _expanded_search_if_needed(
    evaluator: FixedContractEvaluator,
    center: Candidate,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    if result["passed"]:
        return None
    expanded = _search_branch(
        evaluator,
        center,
        window=result["window"],
        angular_radius_deg=10.0,
        k_factor=3.0,
    )
    expanded["reason"] = (
        "automatic principled correction after the predeclared local box/density gate failed"
    )
    return expanded


def _superlevel_intervals(
    k_values: np.ndarray,
    gains: np.ndarray,
    *,
    relative_gap: float,
) -> list[dict[str, Any]]:
    require(k_values.ndim == gains.ndim == 1, "profile arrays must be one-dimensional")
    require(k_values.size == gains.size and k_values.size >= 2, "profile arrays mismatch")
    admitted = gains >= (1.0 - relative_gap) * float(np.max(gains))
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate(admitted):
        if value and start is None:
            start = index
        at_end = index == admitted.size - 1
        if start is not None and ((not value) or at_end):
            stop = index if value and at_end else index - 1
            lower = float(k_values[start])
            upper = float(k_values[stop])
            intervals.append(
                {
                    "first_grid_index": start,
                    "last_grid_index": stop,
                    "k_bounds_m_inv": [lower, upper],
                    "wavelength_2pi_over_k_bounds_m": [
                        float(2.0 * pi / upper),
                        float(2.0 * pi / lower),
                    ],
                    "point_count": int(stop - start + 1),
                    "touches_lower_k_box": start == 0,
                    "touches_upper_k_box": stop == admitted.size - 1,
                    "span_decades": float(np.log10(upper / lower)),
                }
            )
            start = None
    return intervals


def _complete_k_profile(
    evaluator: FixedContractEvaluator,
    center: Candidate,
) -> dict[str, Any]:
    context = evaluator.audit.contexts[REFERENCE_CONTEXT]
    times = np.asarray(context["times"], dtype=float)
    prefix = _exact_time_index(times, ONSET_PROPAGATION_END_S)
    k_values = np.geomspace(
        K_PROFILE_BOUNDS_M_INV[0],
        K_PROFILE_BOUNDS_M_INV[1],
        K_PROFILE_POINT_COUNT,
    )
    histories = []
    for index, k_value in enumerate(k_values):
        candidate = Candidate(
            f"{center.identifier}__full_k_{index:02d}",
            center.direction_n,
            float(k_value),
            "complete registered k-box profile",
            boundary_probe=index in (0, K_PROFILE_POINT_COUNT - 1),
        )
        histories.append(
            (
                candidate,
                evaluator.history(
                    REFERENCE_CONTEXT,
                    prefix,
                    candidate,
                    scales=evaluator.audit.scales,
                    scale_id="baseline",
                    substeps=REFERENCE_SUBSTEPS,
                ),
            )
        )

    window_results: dict[str, Any] = {}
    profile_rows: list[dict[str, Any]] = []
    for window, time_s in (
        ("threshold_window", ONSET_PROBE_TIME_S),
        ("post_threshold_window", ONSET_PROPAGATION_END_S),
    ):
        gains = np.asarray([_gain_at(history, time_s) for _, history in histories])
        logs = np.log(gains)
        fine_index = int(np.argmax(gains))
        coarse_indices = np.arange(0, K_PROFILE_POINT_COUNT, 2, dtype=int)
        require(coarse_indices[-1] == K_PROFILE_POINT_COUNT - 1, "coarse k grid must share box ends")
        coarse_local_index = int(np.argmax(gains[coarse_indices]))
        coarse_index = int(coarse_indices[coarse_local_index])
        intervals = _superlevel_intervals(
            k_values,
            gains,
            relative_gap=NEAR_OPTIMAL_GAIN_GAP,
        )
        touches_box = any(
            interval["touches_lower_k_box"] or interval["touches_upper_k_box"]
            for interval in intervals
        )
        fine_candidate = histories[fine_index][0]
        coarse_candidate = histories[coarse_index][0]
        maximum_log_gain_relative_change = relative_change(
            float(logs[coarse_index]), float(logs[fine_index])
        )
        window_results[window] = {
            "time_s": time_s,
            "coarse_point_count": int(coarse_indices.size),
            "fine_point_count": int(K_PROFILE_POINT_COUNT),
            "coarse_winner": {
                **coarse_candidate.json(),
                "gain": float(gains[coarse_index]),
                "log_gain": float(logs[coarse_index]),
            },
            "fine_winner": {
                **fine_candidate.json(),
                "gain": float(gains[fine_index]),
                "log_gain": float(logs[fine_index]),
            },
            "coarse_to_fine": {
                "k_relative_change": relative_change(
                    coarse_candidate.k_m_inv, fine_candidate.k_m_inv
                ),
                "log_gain_relative_change": maximum_log_gain_relative_change,
            },
            "two_percent_superlevel_intervals": intervals,
            "two_percent_superlevel_touches_registered_k_box": touches_box,
            "two_percent_scale_is_box_resolved": not touches_box,
            "profile_density_gain_stable": bool(
                maximum_log_gain_relative_change <= SEARCH_GAIN_RELATIVE_GATE
            ),
            "unique_k_claim_authorized": False,
        }
        for index, (candidate, _) in enumerate(histories):
            profile_rows.append(
                {
                    "branch_id": center.identifier,
                    "window": window,
                    "time_s": time_s,
                    "grid_index": index,
                    "direction_n1": candidate.direction_n[0],
                    "direction_n2": candidate.direction_n[1],
                    "direction_n3": candidate.direction_n[2],
                    "k_m_inv": candidate.k_m_inv,
                    "wavelength_2pi_over_k_m": 2.0 * pi / candidate.k_m_inv,
                    "gain": float(gains[index]),
                    "log_gain": float(logs[index]),
                    "in_two_percent_superlevel_set": bool(
                        gains[index] >= (1.0 - NEAR_OPTIMAL_GAIN_GAP) * np.max(gains)
                    ),
                }
            )
    return {
        "branch_id": center.identifier,
        "direction_n": list(center.direction_n),
        "k_box_m_inv": list(K_PROFILE_BOUNDS_M_INV),
        "fine_point_count": K_PROFILE_POINT_COUNT,
        "coarse_point_count": (K_PROFILE_POINT_COUNT + 1) // 2,
        "windows": window_results,
        "rows": profile_rows,
    }


def _flatten_case(category: str, scenario: str, case: dict[str, Any]) -> dict[str, Any]:
    onset = case["onset_candidate"]
    final = case["final_candidate"]
    return {
        "category": category,
        "scenario": scenario,
        "base_state_count": case["base_state_count"],
        "substeps_per_interval": case["substeps_per_interval"],
        "scale_id": case["scale_id"],
        "critical_time_s": case["critical_time_s"],
        "onset_probe_time_s": case["onset_probe_time_s"],
        "onset_candidate_id": onset["candidate_id"],
        "onset_n1": onset["direction_n"][0],
        "onset_n2": onset["direction_n"][1],
        "onset_n3": onset["direction_n"][2],
        "onset_k_m_inv": onset["k_m_inv"],
        "onset_gain": onset["gain"],
        "onset_near_optimal_set": ";".join(case["onset_near_optimal_candidate_set"]),
        "final_candidate_id": final["candidate_id"],
        "final_n1": final["direction_n"][0],
        "final_n2": final["direction_n"][1],
        "final_n3": final["direction_n"][2],
        "final_k_m_inv": final["k_m_inv"],
        "final_gain": final["gain"],
        "final_log_gain": final["log_gain"],
    }


def _flatten_search(result: dict[str, Any], density: str) -> dict[str, Any]:
    winner = result[f"{density}_winner"]
    return {
        "category": "local_search",
        "scenario": f"{result['window']}__{result['branch_id']}__{density}",
        "base_state_count": 129,
        "substeps_per_interval": SEARCH_SUBSTEPS,
        "scale_id": "baseline",
        "critical_time_s": None,
        "onset_probe_time_s": (
            ONSET_PROBE_TIME_S if result["window"] == "onset" else None
        ),
        "onset_candidate_id": winner["candidate_id"] if result["window"] == "onset" else None,
        "onset_n1": winner["direction_n"][0] if result["window"] == "onset" else None,
        "onset_n2": winner["direction_n"][1] if result["window"] == "onset" else None,
        "onset_n3": winner["direction_n"][2] if result["window"] == "onset" else None,
        "onset_k_m_inv": winner["k_m_inv"] if result["window"] == "onset" else None,
        "onset_gain": winner["gain"] if result["window"] == "onset" else None,
        "onset_near_optimal_set": None,
        "final_candidate_id": winner["candidate_id"] if result["window"] == "final" else None,
        "final_n1": winner["direction_n"][0] if result["window"] == "final" else None,
        "final_n2": winner["direction_n"][1] if result["window"] == "final" else None,
        "final_n3": winner["direction_n"][2] if result["window"] == "final" else None,
        "final_k_m_inv": winner["k_m_inv"] if result["window"] == "final" else None,
        "final_gain": winner["gain"] if result["window"] == "final" else None,
        "final_log_gain": winner["log_gain"] if result["window"] == "final" else None,
    }


def _write_csv(rows: list[dict[str, Any]]) -> None:
    CASES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CASES_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_k_profile_csv(rows: list[dict[str, Any]]) -> None:
    K_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with K_PROFILE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    require(BASELINE.is_file(), f"missing immutable V2 baseline: {BASELINE}")
    require(BRANCH_CACHE.is_file(), f"missing registered branch cache: {BRANCH_CACHE}")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    branch_cache = json.loads(BRANCH_CACHE.read_text(encoding="utf-8"))
    require(
        baseline.get("schema") == "CP_TI_FINITE_TIME_DYNAMIC_PERTURBATION_V2",
        "unexpected V2 baseline schema",
    )
    require(
        branch_cache.get("schema") == "CP_TI_CONTINUOUS_BRANCH_CACHE_V1"
        and branch_cache.get("base_state_count") == 129,
        "registered branch cache is incompatible with the 129-state audit",
    )

    audit = SpectrumAudit(
        maximum_refinement=32,
        context_factors=(1, 4, 8, 16, 32),
    )
    evaluator = FixedContractEvaluator(audit)

    onset_centers = {
        label: Candidate.from_record(
            label,
            branch_cache["onset_branches"][label]["upper"],
            "registered 129-state upper-bracket branch cache",
        )
        for label in ("sample_x_branch", "sample_y_branch")
    }
    final_centers = {
        label: Candidate.from_record(
            label,
            branch_cache["final_branches"][label],
            "registered 129-state final-window branch cache",
        )
        for label in (
            "sample_y_branch",
            "sample_x_branch",
            "oblique_competitor",
        )
    }

    # Run the core history/substep matrices first on the registered branch
    # centers.  Search diagnostics are deliberately downstream so they cannot
    # delay or silently redefine the numerical-convergence evidence.
    resolved_onset = list(onset_centers.values())
    resolved_final_y = final_centers["sample_y_branch"]
    reference_final_candidates = [
        resolved_final_y,
        final_centers["sample_x_branch"],
        final_centers["oblique_competitor"],
    ]

    # Base-history density at one fixed four-substep propagator contract.
    base_history_cases: list[dict[str, Any]] = []
    for factor in (1, 4, 8, 16, 32):
        context_name = f"factor{factor}"
        probe_time = ONSET_PROPAGATION_END_S if factor == 1 else ONSET_PROBE_TIME_S
        final_candidates = reference_final_candidates if factor in (1, 16) else [resolved_final_y]
        base_history_cases.append(
            _selection_case(
                evaluator,
                context_name,
                substeps=REFERENCE_SUBSTEPS,
                scales=audit.scales,
                scale_id="baseline",
                onset_candidates=resolved_onset,
                final_candidates=final_candidates,
                onset_probe_time_s=probe_time,
            )
        )
    base_reference = base_history_cases[-1]
    for case in base_history_cases:
        case["changes_from_257_state_reference"] = _case_changes(case, base_reference)

    # Substep sensitivity on the finest history, plus an original-like 9-state
    # one-substep case used to isolate the source of the old-record discrepancy.
    substep_cases = [
        _selection_case(
            evaluator,
            REFERENCE_CONTEXT,
            substeps=substeps,
            scales=audit.scales,
            scale_id="baseline",
            onset_candidates=resolved_onset,
            final_candidates=(
                reference_final_candidates if substeps in (1, 4) else [resolved_final_y]
            ),
            onset_probe_time_s=ONSET_PROBE_TIME_S,
        )
        for substeps in (1, 2, 4, 8)
    ]
    substep_reference = next(
        case for case in substep_cases if case["substeps_per_interval"] == REFERENCE_SUBSTEPS
    )
    for case in substep_cases:
        case["changes_from_four_substep_reference"] = _case_changes(
            case, substep_reference
        )
    original_like_case = _selection_case(
        evaluator,
        "factor1",
        substeps=1,
        scales=audit.scales,
        scale_id="baseline",
        onset_candidates=resolved_onset,
        final_candidates=reference_final_candidates,
        onset_probe_time_s=ONSET_PROPAGATION_END_S,
    )

    # The physical operator is unchanged here.  Only the declared dimensionless
    # observation coordinates q=D z are perturbed by mechanism group.
    scale_groups = {
        "mechanical": np.arange(0, 6),
        "thermal": np.arange(6, 7),
        "plastic_chart": np.arange(7, 15),
        "dislocation": np.arange(15, 51),
        "signed_slip": np.arange(51, 69),
    }
    scale_scenarios: list[tuple[str, np.ndarray]] = [("baseline", audit.scales.copy())]
    for group, indices in scale_groups.items():
        for factor in (0.5, 2.0):
            scales = audit.scales.copy()
            scales[indices] *= factor
            scale_scenarios.append((f"{group}_x{factor:g}", scales))
    scale_cases = [
        _selection_case(
            evaluator,
            REFERENCE_CONTEXT,
            substeps=SCALE_AUDIT_SUBSTEPS,
            scales=scales,
            scale_id=identifier,
            onset_candidates=resolved_onset,
            final_candidates=[resolved_final_y],
            onset_probe_time_s=ONSET_PROBE_TIME_S,
        )
        for identifier, scales in scale_scenarios
    ]
    scale_reference = scale_cases[0]
    for case in scale_cases:
        case["changes_from_baseline_scale"] = _case_changes(case, scale_reference)

    # Search diagnostics follow the core history/substep matrices.  The local
    # stencil diagnoses directional sensitivity; the complete registered k box
    # diagnoses whether a unique wavelength is numerically identifiable.
    search_results = [
        _search_branch(evaluator, center, window="onset")
        for center in onset_centers.values()
    ]
    search_results.append(
        _search_branch(evaluator, final_centers["sample_y_branch"], window="final")
    )
    expanded_searches: list[dict[str, Any]] = []
    k_profiles = [
        _complete_k_profile(evaluator, center) for center in onset_centers.values()
    ]
    k_profile_rows = [row for profile in k_profiles for row in profile["rows"]]
    _write_k_profile_csv(k_profile_rows)

    base9 = base_history_cases[0]
    base65 = base_history_cases[2]
    base129 = base_history_cases[3]
    base257 = base_history_cases[4]
    step2 = next(case for case in substep_cases if case["substeps_per_interval"] == 2)
    step4 = substep_reference
    step8 = next(case for case in substep_cases if case["substeps_per_interval"] == 8)

    original_record_tc = float(baseline["finite_time_selection"]["critical_time_s"])
    original_record_onset = baseline["finite_time_selection"]["onset_checkpoint_pair"]
    original_record_final = baseline["finite_time_selection"]["final_horizon_pair"]
    decomposition = {
        "immutable_original_v2_record": {
            "base_state_count": len(baseline["base_history"]),
            "substeps_per_interval": 1,
            "critical_time_s": original_record_tc,
            "onset_direction_n": original_record_onset["direction_n"],
            "onset_k_m_inv": original_record_onset["k_star_m_inv"],
            "onset_checkpoint_gain": original_record_onset["maximum_gain"],
            "final_direction_n": original_record_final["direction_n"],
            "final_k_m_inv": original_record_final["wavenumber_m_inv"],
            "final_gain": original_record_final["final_gain"],
            "final_log_gain": original_record_final["final_log_gain"],
        },
        "original_record_to_same_9_state_resolved_candidates": {
            "critical_time_relative_change": relative_change(
                original_record_tc, original_like_case["critical_time_s"]
            ),
            "onset_angle_deg": angle_degrees(
                original_record_onset["direction_n"],
                original_like_case["onset_candidate"]["direction_n"],
            ),
            "onset_k_relative_change": relative_change(
                original_record_onset["k_star_m_inv"],
                original_like_case["onset_candidate"]["k_m_inv"],
            ),
            "final_angle_deg": angle_degrees(
                original_record_final["direction_n"],
                original_like_case["final_candidate"]["direction_n"],
            ),
            "final_k_relative_change": relative_change(
                original_record_final["wavenumber_m_inv"],
                original_like_case["final_candidate"]["k_m_inv"],
            ),
        },
        "same_9_state_one_to_four_substeps": _case_changes(
            original_like_case, base9
        ),
        "same_candidates_9_to129_states_at_four_substeps": _case_changes(
            base9, base129
        ),
        "corrected_65_to129_states_at_four_substeps": _case_changes(
            base65, base129
        ),
        "extended_129_to257_states_at_four_substeps": _case_changes(
            base129, base257
        ),
        "same_candidates_9_to257_states_at_four_substeps": _case_changes(
            base9, base257
        ),
    }

    intermediate_history_change = decomposition[
        "corrected_65_to129_states_at_four_substeps"
    ]
    corrected_history_change = decomposition[
        "extended_129_to257_states_at_four_substeps"
    ]
    original_history_change = decomposition[
        "same_candidates_9_to257_states_at_four_substeps"
    ]
    history_sequence = {
        "critical_time_s": _three_level_extrapolation(
            [
                base65["critical_time_s"],
                base129["critical_time_s"],
                base257["critical_time_s"],
            ]
        ),
        "threshold_window_log_gain": _three_level_extrapolation(
            [
                base65["onset_candidate"]["log_gain"],
                base129["onset_candidate"]["log_gain"],
                base257["onset_candidate"]["log_gain"],
            ]
        ),
        "common_2us_log_gain": _three_level_extrapolation(
            [
                base65["onset_common_time_envelope_log_gain"],
                base129["onset_common_time_envelope_log_gain"],
                base257["onset_common_time_envelope_log_gain"],
            ]
        ),
        "final_window_log_gain": _three_level_extrapolation(
            [
                base65["final_candidate"]["log_gain"],
                base129["final_candidate"]["log_gain"],
                base257["final_candidate"]["log_gain"],
            ]
        ),
    }
    substep2_change = _case_changes(step2, step4)
    substep8_change = _case_changes(step8, step4)
    scale_changes = [case["changes_from_baseline_scale"] for case in scale_cases[1:]]
    history_tc_passed = bool(
        corrected_history_change["critical_time_relative_change"] <= HISTORY_RELATIVE_GATE
    )
    history_onset_log_gain_passed = bool(
        corrected_history_change["onset_log_gain_relative_change"]
        <= HISTORY_RELATIVE_GATE
    )
    history_final_log_gain_passed = bool(
        corrected_history_change["final_log_gain_relative_change"]
        <= HISTORY_RELATIVE_GATE
    )
    history_correction_passed = bool(
        history_tc_passed
        and history_onset_log_gain_passed
        and history_final_log_gain_passed
        and corrected_history_change["onset_near_optimal_set_same"]
    )
    substeps_passed = bool(
        max(
            substep2_change["critical_time_relative_change"],
            substep2_change["onset_log_gain_relative_change"],
            substep2_change["final_log_gain_relative_change"],
            substep8_change["critical_time_relative_change"],
            substep8_change["onset_log_gain_relative_change"],
            substep8_change["final_log_gain_relative_change"],
        )
        <= SUBSTEP_RELATIVE_GATE
    )
    scale_tc_maximum_change = max(
        change["critical_time_relative_change"] for change in scale_changes
    )
    scale_onset_log_gain_maximum_change = max(
        change["onset_log_gain_relative_change"] for change in scale_changes
    )
    scale_final_log_gain_maximum_change = max(
        change["final_log_gain_relative_change"] for change in scale_changes
    )
    scale_tc_passed = bool(scale_tc_maximum_change <= SCALE_TC_RELATIVE_GATE)
    scale_onset_set_preserved = all(
        change["onset_near_optimal_set_same"] for change in scale_changes
    )
    scale_onset_amplitude_passed = bool(
        scale_onset_log_gain_maximum_change <= SCALE_LOG_GAIN_RELATIVE_GATE
    )
    scale_final_amplitude_passed = bool(
        scale_final_log_gain_maximum_change <= SCALE_LOG_GAIN_RELATIVE_GATE
    )
    scale_mechanism_passed = all(
        case["onset_candidate"]["dominant_full_state_output_mechanism"]
        == "plastic_kinematics"
        and case["final_candidate"]["dominant_full_state_output_mechanism"]
        == "plastic_kinematics"
        for case in scale_cases
    )
    onset_local_results = [
        result for result in search_results if result["window"] == "onset"
    ]
    final_local_result = next(
        result for result in search_results if result["window"] == "final"
    )
    local_direction_gain_stable = all(
        result["coarse_to_fine_changes"]["angle_deg"] <= SEARCH_ANGLE_GATE_DEG
        and result["coarse_to_fine_changes"]["log_gain_relative_change"]
        <= SEARCH_GAIN_RELATIVE_GATE
        for result in onset_local_results
    )
    k_profile_density_stable = all(
        window["profile_density_gain_stable"]
        for profile in k_profiles
        for window in profile["windows"].values()
    )
    k_superlevel_box_resolved = all(
        window["two_percent_scale_is_box_resolved"]
        for profile in k_profiles
        for window in profile["windows"].values()
    )
    original_history_passed = bool(
        original_history_change["critical_time_relative_change"]
        <= HISTORY_RELATIVE_GATE
    )
    corrected_reference_passed = bool(
        history_correction_passed
        and substeps_passed
        and local_direction_gain_stable
        and final_local_result["passed"]
        and k_profile_density_stable
    )
    if not history_correction_passed:
        status = "FACTOR32_HISTORY_EXTENSION_NOT_CONVERGED__STOPPED_AT_257_STATES"
    elif corrected_reference_passed and not k_superlevel_box_resolved:
        status = "REFERENCE_NUMERICS_STABLE__ONSET_K_SUPERLEVEL_TOUCHES_REGISTERED_BOX"
    elif corrected_reference_passed:
        status = "REFERENCE_NUMERICS_STABLE__ONSET_K_REPORTED_AS_TWO_PERCENT_SET"
    else:
        status = "CORRECTED_REFERENCE_REMAINS_UNSTABLE"

    gates = {
        "original_9_state_tc_within_2_percent_of_257_state_reference": {
            "passed": original_history_passed,
            "required_for_original_v2_claim": True,
            "observed_critical_time_relative_change": original_history_change[
                "critical_time_relative_change"
            ],
            "limit": HISTORY_RELATIVE_GATE,
        },
        "history_129_to257_tc_within_2_percent": {
            "passed": history_tc_passed,
            "required_for_corrected_reference": True,
            "observed": corrected_history_change["critical_time_relative_change"],
            "limit": HISTORY_RELATIVE_GATE,
        },
        "history_129_to257_threshold_window_log_gain_within_2_percent": {
            "passed": history_onset_log_gain_passed,
            "required_for_corrected_reference": True,
            "observed": corrected_history_change["onset_log_gain_relative_change"],
            "limit": HISTORY_RELATIVE_GATE,
        },
        "history_129_to257_final_window_log_gain_within_2_percent": {
            "passed": history_final_log_gain_passed,
            "required_for_corrected_reference": True,
            "observed": corrected_history_change["final_log_gain_relative_change"],
            "limit": HISTORY_RELATIVE_GATE,
        },
        "factor16_two_four_eight_substeps_within_0p1_percent": {
            "passed": substeps_passed,
            "required_for_corrected_reference": True,
            "two_to_four_changes": substep2_change,
            "eight_to_four_changes": substep8_change,
            "limit": SUBSTEP_RELATIVE_GATE,
        },
        "factor_two_scales_tc_within_5_percent": {
            "passed": scale_tc_passed,
            "required_for_corrected_reference": False,
            "observed_maximum": scale_tc_maximum_change,
            "limit": SCALE_TC_RELATIVE_GATE,
        },
        "factor_two_scales_preserve_xy_near_optimal_set": {
            "passed": scale_onset_set_preserved,
            "required_for_corrected_reference": False,
        },
        "factor_two_scales_threshold_window_log_gain_within_10_percent": {
            "passed": scale_onset_amplitude_passed,
            "required_for_corrected_reference": False,
            "observed_maximum": scale_onset_log_gain_maximum_change,
            "limit": SCALE_LOG_GAIN_RELATIVE_GATE,
        },
        "factor_two_scales_final_window_log_gain_within_10_percent": {
            "passed": scale_final_amplitude_passed,
            "required_for_corrected_reference": False,
            "observed_maximum": scale_final_log_gain_maximum_change,
            "limit": SCALE_LOG_GAIN_RELATIVE_GATE,
        },
        "factor_two_scales_keep_plastic_dominance": {
            "passed": scale_mechanism_passed,
            "required_for_corrected_reference": False,
        },
        "local_direction_and_gain_probe_stable": {
            "passed": local_direction_gain_stable,
            "required_for_corrected_reference": True,
            "angle_limit_deg": SEARCH_ANGLE_GATE_DEG,
            "log_gain_relative_limit": SEARCH_GAIN_RELATIVE_GATE,
        },
        "final_window_local_selector_probe_stable": {
            "passed": final_local_result["passed"],
            "required_for_corrected_reference": True,
        },
        "complete_k_profile_17_to33_point_gain_stable": {
            "passed": k_profile_density_stable,
            "required_for_corrected_reference": True,
            "log_gain_relative_limit": SEARCH_GAIN_RELATIVE_GATE,
        },
        "two_percent_k_superlevel_sets_do_not_touch_registered_box": {
            "passed": k_superlevel_box_resolved,
            "required_for_corrected_reference": False,
            "required_for_finite_k_identification": True,
            "registered_k_box_m_inv": list(K_PROFILE_BOUNDS_M_INV),
        },
    }

    diagnosis = {
        "primary_original_v2_failure_mode": (
            "sparse base history and cross-checkpoint gain interpolation"
            if not original_history_passed
            else "no original history-density failure detected"
        ),
        "evidence": {
            "old_record_vs_same_9_state_resolved_candidate_change": decomposition[
                "original_record_to_same_9_state_resolved_candidates"
            ],
            "same_9_state_one_to_four_substep_change": decomposition[
                "same_9_state_one_to_four_substeps"
            ],
            "same_candidates_9_to257_state_change": original_history_change,
            "corrected_65_to129_state_change": intermediate_history_change,
            "extended_129_to257_state_change": corrected_history_change,
            "three_level_sequence_and_extrapolation": history_sequence,
        },
        "principled_correction": (
            "The history was extended to 257 states at four substeps. If the 129-to-257 "
            "gate remains open, refinement stops here and the three-level error sequence "
            "is reported without asserting convergence."
        ),
        "stability_recovered": corrected_reference_passed,
        "coordinate_scale_interpretation": (
            "Factor-two scale changes audit norm sensitivity. They cannot establish "
            "coordinate-invariant tc; the baseline scale vector remains part of the "
            "computational contract."
        ),
        "candidate_switching_interpretation": (
            "A switch between sample-x and sample-y inside the 2% gain set is reported "
            "as set-valued onset competition, not as a converged unique direction."
        ),
        "wavenumber_interpretation": (
            "Large k movement with nearly unchanged gain is a broad-platform/scale-"
            "identifiability result. It is not labelled optimizer failure; the 2% "
            "superlevel k and wavelength intervals are the primary output."
        ),
    }

    csv_rows: list[dict[str, Any]] = []
    csv_rows.extend(
        _flatten_case("base_history", case["context"], case)
        for case in base_history_cases
    )
    csv_rows.extend(
        _flatten_case(
            "integration_substeps",
            f"substeps_{case['substeps_per_interval']}",
            case,
        )
        for case in substep_cases
    )
    csv_rows.extend(
        _flatten_case("coordinate_scale", case["scale_id"], case)
        for case in scale_cases
    )
    for result in search_results:
        csv_rows.extend(_flatten_search(result, density) for density in ("coarse", "fine"))
    _write_csv(csv_rows)

    report = {
        "schema": SCHEMA,
        "status": status,
        "classification": "NUMERICAL_VERIFICATION_ONLY_NOT_A_TA2_PREDICTION",
        "physical_contract": {
            "material_card": MATERIAL_CARD.relative_to(ROOT).as_posix(),
            "material_card_status": json.loads(MATERIAL_CARD.read_text(encoding="utf-8"))[
                "status"
            ],
            "physical_parameters_changed_across_cases": False,
            "state_dimensions": {
                "dynamic_descriptor": 87,
                "finite_generator": 69,
                "observed_input_output": 63,
            },
            "observed_indices_zero_based": OBSERVED.tolist(),
            "gain_threshold": GAIN_THRESHOLD,
            "gain_norm": "q=Dz; Euclidean norm on thermal plus 62 constitutive coordinates",
            "reference_coordinate_scale_vector": audit.scales.tolist(),
            "base_shear_checkpoints": audit.contexts["factor1"]["shears"].tolist(),
            "reference_base_state_count": len(
                audit.contexts[REFERENCE_CONTEXT]["times"]
            ),
            "history_extension_state_count": len(
                audit.contexts[HISTORY_EXTENSION_CONTEXT]["times"]
            ),
            "reference_substeps_per_interval": REFERENCE_SUBSTEPS,
            "onset_probe_time_s": ONSET_PROBE_TIME_S,
            "onset_propagation_end_s": ONSET_PROPAGATION_END_S,
            "candidate_seed_source": BRANCH_CACHE.relative_to(ROOT).as_posix(),
        },
        "audit_design": {
            "base_history_state_counts": [
                len(audit.contexts[f"factor{factor}"]["times"])
                for factor in (1, 4, 8, 16, 32)
            ],
            "substeps_per_interval": [1, 2, 4, 8],
            "coordinate_scale_scenarios": [identifier for identifier, _ in scale_scenarios],
            "local_probe": {
                "branches": [result["branch_id"] for result in search_results],
                "windows": [result["window"] for result in search_results],
                "coarse_probe_count_per_branch": 7,
                "fine_probe_count_per_branch": 13,
                "box_angular_radius_deg": 5.0,
                "box_k_factor": 2.0,
                "axis_separated_not_full_tensor_grid": True,
            },
            "complete_k_profiles": {
                "context": REFERENCE_CONTEXT,
                "substeps_per_interval": REFERENCE_SUBSTEPS,
                "branches": [profile["branch_id"] for profile in k_profiles],
                "times_s": [ONSET_PROBE_TIME_S, ONSET_PROPAGATION_END_S],
                "registered_bounds_m_inv": list(K_PROFILE_BOUNDS_M_INV),
                "coarse_point_count": (K_PROFILE_POINT_COUNT + 1) // 2,
                "fine_point_count": K_PROFILE_POINT_COUNT,
                "profile_csv": K_PROFILE_CSV.relative_to(ROOT).as_posix(),
                "profile_csv_row_count": len(k_profile_rows),
            },
            "csv_case_count": len(csv_rows),
        },
        "registered_seed_candidates": {
            "onset": {key: value.json() for key, value in onset_centers.items()},
            "final": {key: value.json() for key, value in final_centers.items()},
        },
        "local_search": {
            "results": search_results,
            "complete_registered_k_profiles": [
                {key: value for key, value in profile.items() if key != "rows"}
                for profile in k_profiles
            ],
            "old_local_k_box_failures_reinterpreted_as_platform_diagnostics": True,
            "global_continuum_optimum_proved": False,
        },
        "base_history_density": {
            "cases": base_history_cases,
            "history_extension_context": HISTORY_EXTENSION_CONTEXT,
            "three_level_sequence_and_extrapolation": history_sequence,
            "stopped_at_257_states": True,
        },
        "integration_substeps": {
            "cases": substep_cases,
            "reference_substeps_per_interval": REFERENCE_SUBSTEPS,
        },
        "coordinate_scale_sensitivity": {
            "mechanism_groups_zero_based": {
                group: indices.tolist() for group, indices in scale_groups.items()
            },
            "cases": scale_cases,
            "factors": [0.5, 2.0],
            "substeps_per_interval": SCALE_AUDIT_SUBSTEPS,
            "coordinate_invariance_claimed": False,
            "separate_diagnostics": {
                "maximum_critical_time_relative_change": scale_tc_maximum_change,
                "xy_near_optimal_set_preserved": scale_onset_set_preserved,
                "maximum_threshold_window_log_gain_relative_change": scale_onset_log_gain_maximum_change,
                "maximum_final_window_log_gain_relative_change": scale_final_log_gain_maximum_change,
                "plastic_dominance_preserved": scale_mechanism_passed,
            },
        },
        "original_v2_gap_decomposition": decomposition,
        "gates": gates,
        "failed_original_gates": [
            name
            for name, value in gates.items()
            if value.get("required_for_original_v2_claim") and not value["passed"]
        ],
        "failed_corrected_reference_gates": [
            name
            for name, value in gates.items()
            if value.get("required_for_corrected_reference") and not value["passed"]
        ],
        "failed_coordinate_scale_diagnostics": [
            name
            for name, value in gates.items()
            if name.startswith("factor_two_scales") and not value["passed"]
        ],
        "failed_finite_k_identification_gates": [
            name
            for name, value in gates.items()
            if value.get("required_for_finite_k_identification") and not value["passed"]
        ],
        "failure_diagnosis_and_correction": diagnosis,
        "execution": {
            "operator_assembly_count": evaluator.operator_assembly_count,
            "propagation_count": evaluator.propagation_count,
            "in_memory_operator_reuse": True,
        },
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "script": Path(__file__).relative_to(ROOT).as_posix(),
            "source_sha256": {
                BASELINE.relative_to(ROOT).as_posix(): _sha256(BASELINE),
                BRANCH_CACHE.relative_to(ROOT).as_posix(): _sha256(BRANCH_CACHE),
                MATERIAL_CARD.relative_to(ROOT).as_posix(): _sha256(MATERIAL_CARD),
            },
        },
        "claim_boundary": {
            "evidence_kind": "fixed-model structure_and_code_level_numerical_robustness",
            "single_crystal_literature_constrained_verification_seed": True,
            "material_validation": False,
            "ta2_onset_time_prediction": False,
            "ta2_critical_strain_prediction": False,
            "ta2_finite_bandwidth_prediction": False,
            "global_direction_wavenumber_optimum_proved": False,
            "coordinate_invariant_critical_time_proved": False,
            "tested_local_candidate_set_only": True,
            "unique_onset_wavenumber_claim_authorized": False,
        },
        "cannot_prove": [
            "a global nonconvex direction-wavenumber maximum",
            "coordinate-invariant gain-threshold crossing time",
            "continuum-uniform high-k boundedness",
            "material-level TA2 onset, critical strain, or finite bandwidth",
            "batch-specific parameter validity or omitted twinning/DRX/damage mechanisms",
        ],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# CP-Ti finite-time V2 robustness audit V1",
        "",
        f"Status: **{status}**.",
        "",
        "This is numerical verification only and is not a TA2 material prediction.",
        "",
        "## Main diagnosis",
        "",
        f"- Original V2 tc: {original_record_tc * 1.0e6:.9g} us.",
        f"- Extended 257-state/four-substep tc: {base257['critical_time_s'] * 1.0e6:.9g} us.",
        f"- Same-candidate 9-to-257-state tc change: {100.0 * original_history_change['critical_time_relative_change']:.6g}%.",
        f"- 65-to-129-state tc change: {100.0 * intermediate_history_change['critical_time_relative_change']:.6g}%.",
        f"- 129-to-257-state tc change: {100.0 * corrected_history_change['critical_time_relative_change']:.6g}%.",
        f"- 129-to-257 threshold-window log-gain change: {100.0 * corrected_history_change['onset_log_gain_relative_change']:.6g}%.",
        f"- 129-to-257 final-window log-gain change: {100.0 * corrected_history_change['final_log_gain_relative_change']:.6g}%.",
        f"- Primary failure mode: {diagnosis['primary_original_v2_failure_mode']}.",
        f"- Stability recovered under the corrected reference: {corrected_reference_passed}.",
        "",
        "## Corrected reference selectors",
        "",
        f"- Onset probe time: {ONSET_PROBE_TIME_S * 1.0e6:.9g} us; tested near-optimal set: {base257['onset_near_optimal_candidate_set']}.",
        f"- Numerical onset winner: n={base257['onset_candidate']['direction_n']}, k={base257['onset_candidate']['k_m_inv']:.9g} 1/m, gain={base257['onset_candidate']['gain']:.9g}.",
        f"- Final-window winner: n={base257['final_candidate']['direction_n']}, k={base257['final_candidate']['k_m_inv']:.9g} 1/m, gain={base257['final_candidate']['gain']:.9g}.",
        "",
        "## Coordinate-scale diagnostics",
        "",
        f"- Maximum tc change: {100.0 * scale_tc_maximum_change:.6g}%.",
        f"- x/y near-optimal set preserved: {scale_onset_set_preserved}.",
        f"- Maximum threshold-window log-gain change: {100.0 * scale_onset_log_gain_maximum_change:.6g}%.",
        f"- Maximum final-window log-gain change: {100.0 * scale_final_log_gain_maximum_change:.6g}%.",
        f"- Plastic dominance preserved: {scale_mechanism_passed}.",
        "",
        "## Complete registered k-box profiles",
        "",
    ]
    for profile in k_profiles:
        for window_name, window in profile["windows"].items():
            intervals = [
                {
                    "k_m_inv": interval["k_bounds_m_inv"],
                    "wavelength_m": interval["wavelength_2pi_over_k_bounds_m"],
                    "touches_lower": interval["touches_lower_k_box"],
                    "touches_upper": interval["touches_upper_k_box"],
                }
                for interval in window["two_percent_superlevel_intervals"]
            ]
            lines.append(
                f"- {profile['branch_id']} / {window_name} at {window['time_s'] * 1.0e6:.6g} us: "
                f"fine-grid winner k={window['fine_winner']['k_m_inv']:.9g} 1/m; "
                f"2% intervals={json.dumps(intervals, sort_keys=True)}."
            )
    lines.extend(["", "## Gates", "", "| gate | passed |", "|---|---:|"])
    lines.extend(f"| {name} | {value['passed']} |" for name, value in gates.items())
    lines.extend(
        [
            "",
            f"Failed original gates: {report['failed_original_gates']}.",
            f"Failed corrected-reference gates: {report['failed_corrected_reference_gates']}.",
            f"Failed coordinate-scale diagnostics: {report['failed_coordinate_scale_diagnostics']}.",
            f"Failed finite-k-identification gates: {report['failed_finite_k_identification_gates']}.",
            "",
            "## Scope",
            "",
            "The local search is a nested, axis-separated probe around registered branches; it is not a global optimizer certificate. Factor-two coordinate-scale cases measure norm sensitivity and do not make tc coordinate invariant. The x/y onset branches are retained as a tested near-optimal set when their gains lie within 2%.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "result": str(RESULT),
                "summary": str(SUMMARY),
                "cases_csv": str(CASES_CSV),
                "k_profile_csv": str(K_PROFILE_CSV),
                "status": status,
                "csv_case_count": len(csv_rows),
                "original_tc_s": original_record_tc,
                "extended_257_state_tc_s": base257["critical_time_s"],
                "failed_original_gates": report["failed_original_gates"],
                "failed_corrected_reference_gates": report[
                    "failed_corrected_reference_gates"
                ],
                "operator_assembly_count": evaluator.operator_assembly_count,
                "propagation_count": evaluator.propagation_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
