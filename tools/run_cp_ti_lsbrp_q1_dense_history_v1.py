"""LSBRP-1 Q1 dense-history gate on the locked CP-Ti base cache.

This first Q1 stage deliberately evaluates only the preregistered sample-x and
sample-y onset branches through 2 microseconds.  It answers the expensive
257/513-state and 4/8-substep question before any 65-point wavenumber scan is
allowed to redefine the branch.  A failed 257-to-513 threshold-window gate is
the sole trigger for a 1025-state cache.
"""

from __future__ import annotations

import hashlib
import json
from math import e
from pathlib import Path
import pickle
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.audit_cp_ti_finite_time_v2_robustness_v1 import (  # noqa: E402
    Candidate,
    FixedContractEvaluator,
)


SCHEMA = "CP_TI_LSBRP_Q1_DENSE_HISTORY_V1"
CACHE = ROOT / "05_results/cp_ti_lsbrp_base_cache_v1_ref64.pkl"
MANIFEST = CACHE.with_suffix(".json")
BRANCH_CACHE = (
    ROOT
    / "05_results/cp_ti_continuous_spectrum_robustness_v2"
    / "continuous_branch_cache_v1.json"
)
RESULT = ROOT / "05_results/cp_ti_lsbrp_q1_dense_history_v1.json"
SUMMARY = ROOT / "05_results/cp_ti_lsbrp_q1_dense_history_v1.md"

ONSET_PROBE_TIME_S = 1.5e-6
END_TIME_S = 2.0e-6
TC_GATE = 0.01
THRESHOLD_LOG_GAIN_GATE = 0.02
SUBSTEP_LOG_GAIN_GATE = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    scale = max(abs(float(second)), np.finfo(float).tiny)
    return float(abs(float(first) - float(second)) / scale)


def _exact_index(times: Any, target_s: float) -> int:
    values = np.asarray(times, dtype=float)
    matches = np.flatnonzero(np.isclose(values, target_s, rtol=0.0, atol=2.0e-18))
    if matches.size != 1:
        raise RuntimeError(f"time {target_s:.9g} s is not a unique cache checkpoint")
    return int(matches[0])


def load_verified_cache() -> dict[str, Any]:
    if not CACHE.is_file() or not MANIFEST.is_file():
        raise RuntimeError("locked ref64 cache and manifest must be built first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed = _sha256(CACHE)
    if observed != manifest.get("cache_sha256"):
        raise RuntimeError("base-cache SHA-256 does not match its manifest")
    with CACHE.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - verified local generated cache
    if payload.get("schema") != "CP_TI_LSBRP_BASE_CACHE_V1":
        raise RuntimeError("unexpected base-cache schema")
    if tuple(payload.get("context_factors", ())) != (1, 32, 64):
        raise RuntimeError("Q1 requires the locked factor 1/32/64 cache")
    return payload


def _gain_at(prefix: list[dict[str, Any]], target_s: float) -> float:
    rows = [
        row
        for row in prefix
        if np.isclose(float(row["time_s"]), target_s, rtol=0.0, atol=2.0e-18)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"propagator prefix does not contain {target_s:.9g} s")
    return float(rows[0]["maximum_gain"])


def _case(
    evaluator: FixedContractEvaluator,
    context_name: str,
    candidates: dict[str, Candidate],
    substeps: int,
) -> dict[str, Any]:
    context = evaluator.audit.contexts[context_name]
    end_index = _exact_index(context["times"], END_TIME_S)
    branches: dict[str, Any] = {}
    for label, candidate in candidates.items():
        history = evaluator.history(
            context_name,
            end_index,
            candidate,
            scales=evaluator.audit.scales,
            scale_id="locked_empirical_D",
            substeps=substeps,
        )
        threshold_gain = _gain_at(history["prefix"], ONSET_PROBE_TIME_S)
        branches[label] = {
            **candidate.json(),
            "critical_time_s": history["critical_time_s"],
            "threshold_gain": threshold_gain,
            "threshold_log_gain": float(np.log(threshold_gain)),
            "two_microsecond_gain": history["gain"],
            "two_microsecond_log_gain": history["log_gain"],
        }
    threshold_winner = max(branches, key=lambda key: branches[key]["threshold_log_gain"])
    end_winner = max(branches, key=lambda key: branches[key]["two_microsecond_log_gain"])
    finite_tc = {
        key: value["critical_time_s"]
        for key, value in branches.items()
        if value["critical_time_s"] is not None
    }
    return {
        "context": context_name,
        "global_base_state_count": len(context["times"]),
        "early_window_state_count": end_index + 1,
        "integration_substeps_per_interval": substeps,
        "gain_threshold": e,
        "critical_time_s": min(finite_tc.values()) if finite_tc else None,
        "critical_time_branch": min(finite_tc, key=finite_tc.get) if finite_tc else None,
        "threshold_winner": threshold_winner,
        "threshold_log_gain": branches[threshold_winner]["threshold_log_gain"],
        "two_microsecond_winner": end_winner,
        "two_microsecond_log_gain": branches[end_winner]["two_microsecond_log_gain"],
        "branches": branches,
    }


def main() -> int:
    started = time.perf_counter()
    payload = load_verified_cache()
    branch_cache = json.loads(BRANCH_CACHE.read_text(encoding="utf-8"))
    if branch_cache.get("schema") != "CP_TI_CONTINUOUS_BRANCH_CACHE_V1":
        raise RuntimeError("unexpected registered-branch schema")
    audit = SimpleNamespace(
        contexts=payload["contexts"],
        scales=np.asarray(payload["coordinate_scales"], dtype=float),
        admission=payload["admission"],
    )
    evaluator = FixedContractEvaluator(audit)
    candidates = {
        label: Candidate.from_record(
            label,
            branch_cache["onset_branches"][label]["upper"],
            "registered 129-state upper-bracket branch cache",
        )
        for label in ("sample_x_branch", "sample_y_branch")
    }

    case257 = _case(evaluator, "factor32", candidates, 4)
    case513_4 = _case(evaluator, "factor64", candidates, 4)
    case513_8 = _case(evaluator, "factor64", candidates, 8)
    history_changes = {
        "critical_time_relative_change": _relative(
            case257["critical_time_s"], case513_4["critical_time_s"]
        ),
        "threshold_log_gain_relative_change": _relative(
            case257["threshold_log_gain"], case513_4["threshold_log_gain"]
        ),
        "two_microsecond_log_gain_relative_change": _relative(
            case257["two_microsecond_log_gain"], case513_4["two_microsecond_log_gain"]
        ),
        "threshold_winner_preserved": case257["threshold_winner"]
        == case513_4["threshold_winner"],
        "two_microsecond_winner_preserved": case257["two_microsecond_winner"]
        == case513_4["two_microsecond_winner"],
    }
    substep_changes = {
        "critical_time_relative_change": _relative(
            case513_4["critical_time_s"], case513_8["critical_time_s"]
        ),
        "threshold_log_gain_relative_change": _relative(
            case513_4["threshold_log_gain"], case513_8["threshold_log_gain"]
        ),
        "two_microsecond_log_gain_relative_change": _relative(
            case513_4["two_microsecond_log_gain"], case513_8["two_microsecond_log_gain"]
        ),
        "threshold_winner_preserved": case513_4["threshold_winner"]
        == case513_8["threshold_winner"],
        "two_microsecond_winner_preserved": case513_4["two_microsecond_winner"]
        == case513_8["two_microsecond_winner"],
    }
    gates = {
        "257_to_513_tc_below_1_percent": {
            "passed": history_changes["critical_time_relative_change"] < TC_GATE,
            "observed": history_changes["critical_time_relative_change"],
            "limit": TC_GATE,
        },
        "257_to_513_threshold_log_gain_below_2_percent": {
            "passed": history_changes["threshold_log_gain_relative_change"]
            < THRESHOLD_LOG_GAIN_GATE,
            "observed": history_changes["threshold_log_gain_relative_change"],
            "limit": THRESHOLD_LOG_GAIN_GATE,
        },
        "513_state_four_to_eight_substeps_log_gains_below_0p5_percent": {
            "passed": max(
                substep_changes["threshold_log_gain_relative_change"],
                substep_changes["two_microsecond_log_gain_relative_change"],
            )
            < SUBSTEP_LOG_GAIN_GATE,
            "observed_maximum": max(
                substep_changes["threshold_log_gain_relative_change"],
                substep_changes["two_microsecond_log_gain_relative_change"],
            ),
            "limit": SUBSTEP_LOG_GAIN_GATE,
        },
    }
    requires_1025 = not gates[
        "257_to_513_threshold_log_gain_below_2_percent"
    ]["passed"]
    if requires_1025:
        status = "DENSE_HISTORY_GATE_OPEN__BUILD_1025_STATE_CACHE"
    elif not all(value["passed"] for value in gates.values()):
        status = "DENSE_HISTORY_OR_SUBSTEP_GATE_FAILED"
    else:
        status = "DENSE_HISTORY_GATE_PASSED__PROCEED_TO_K_SET"

    report = {
        "schema": SCHEMA,
        "status": status,
        "classification": "FIXED_MODEL_NUMERICAL_VERIFICATION_NOT_TA2_PREDICTION",
        "contract": {
            "window_s": [0.5e-6, END_TIME_S],
            "threshold_probe_time_s": ONSET_PROBE_TIME_S,
            "candidate_set": list(candidates),
            "coordinate_metric": "locked empirical D only; W_E/W_O are separate unbuilt contracts",
            "physical_parameters_changed": False,
        },
        "cases": {
            "states257_substeps4": case257,
            "states513_substeps4": case513_4,
            "states513_substeps8": case513_8,
        },
        "history_changes": history_changes,
        "substep_changes": substep_changes,
        "gates": gates,
        "requires_1025_state_cache": requires_1025,
        "next_action": (
            "build ref128 cache and repeat the fixed-branch gate"
            if requires_1025
            else "run adaptive 65-point k-set interior/Hausdorff audit"
        ),
        "provenance": {
            "base_cache": CACHE.relative_to(ROOT).as_posix(),
            "base_cache_sha256": _sha256(CACHE),
            "base_cache_manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "branch_cache": BRANCH_CACHE.relative_to(ROOT).as_posix(),
            "branch_cache_sha256": _sha256(BRANCH_CACHE),
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": _sha256(Path(__file__)),
        },
        "runtime": {
            "elapsed_s": time.perf_counter() - started,
            "operator_assembly_count": evaluator.operator_assembly_count,
            "propagation_count": evaluator.propagation_count,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LSBRP-1 Q1 dense-history gate V1",
        "",
        f"- Status: `{status}`.",
        f"- 257-state / 4-substep tc: {case257['critical_time_s']*1e6:.9g} us.",
        f"- 513-state / 4-substep tc: {case513_4['critical_time_s']*1e6:.9g} us.",
        f"- 513-state / 8-substep tc: {case513_8['critical_time_s']*1e6:.9g} us.",
        f"- 257->513 tc relative change: {history_changes['critical_time_relative_change']:.6%}.",
        f"- 257->513 threshold-window log-gain relative change: {history_changes['threshold_log_gain_relative_change']:.6%}.",
        f"- 4->8 substep maximum registered log-gain relative change: {gates['513_state_four_to_eight_substeps_log_gains_below_0p5_percent']['observed_maximum']:.6%}.",
        f"- 1025-state cache required by preregistered rule: {requires_1025}.",
        "",
        "This stage does not claim a finite wavelength: the adaptive 65-point k-set and W_E/W_O metric gates remain separate.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "result": str(RESULT), "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
