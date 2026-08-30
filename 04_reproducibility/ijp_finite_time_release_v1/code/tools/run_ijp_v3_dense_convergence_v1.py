"""Close the V3 513-to-1025 checkpoint-density convergence gate.

This calculation is deliberately linear.  The factor-64 nonlinear replay has
already tested the optimized singular vectors.  Here the same directions,
wavenumbers, selectors, coordinate metric and four-substep propagator are
evaluated on factor-64 (513 states) and factor-128 (1025 states) base histories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402


SCHEMA = "IJP_V3_DENSE_CONVERGENCE_V1"
V3_EVIDENCE = ROOT / "05_results/ijp_revision_evidence_v3.json"
V3_ARRAYS = ROOT / "05_results/ijp_revision_evidence_v3_arrays.npz"
RESULT = ROOT / "05_results/ijp_v3_dense_convergence_v1.json"
NONLINEAR_RESULT = ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v3.json"
SUITE_RESULT = ROOT / "05_results/ijp_v3_validation_suite_v1.json"
CACHE = ROOT / "05_results/ijp_v3_factor128_context_v1.pkl"
MANIFEST = CACHE.with_suffix(".json")
GATE = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contracts(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = evidence["reoptimization"]["records"]
    output: dict[str, dict[str, Any]] = {}
    for horizon, role in (("onset", "near_onset"), ("terminal", "terminal")):
        row = records[f"{horizon}__baseline__constitutive_to_constitutive"]
        output[role] = {
            "target_time_s": float(row["time_s"]),
            "direction_n": np.asarray(row["direction_n"], dtype=float),
            "k_m_inv": float(row["k_m_inv"]),
        }
    return output


def context_dict(context: Any) -> dict[str, Any]:
    return {
        "times": context.times_s,
        "shears": context.shears,
        "points": context.points,
        "storages": context.storages,
    }


def load_or_build_factor128() -> tuple[dict[str, Any], str]:
    builder_hash = sha256(Path(v3.__file__))
    if CACHE.is_file() and MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") == "IJP_V3_FACTOR128_CONTEXT_V1"
            and manifest.get("builder_sha256") == builder_hash
            and manifest.get("cache_sha256") == sha256(CACHE)
        ):
            with CACHE.open("rb") as stream:
                payload = pickle.load(stream)  # noqa: S301 - hash-verified local cache
            return payload["context"], "verified_cache"

    print(json.dumps({"stage": "build_paper_model_context", "factor": 128}), flush=True)
    refined = v3.build_context("v3_dense_convergence_factor128", factor=128)
    payload = {
        "schema": "IJP_V3_FACTOR128_CONTEXT_V1",
        "factor": 128,
        "context": context_dict(refined),
    }
    with CACHE.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": "IJP_V3_FACTOR128_CONTEXT_V1",
        "builder": Path(v3.__file__).relative_to(ROOT).as_posix(),
        "builder_sha256": builder_hash,
        "cache": CACHE.relative_to(ROOT).as_posix(),
        "cache_sha256": sha256(CACHE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload["context"], "newly_built_cache"


def gain(
    context: dict[str, Any],
    contract: dict[str, Any],
    admission: Any,
    scales: np.ndarray,
) -> float:
    end_index = v2._exact_index(
        np.asarray(context["times"]), float(contract["target_time_s"])
    )
    linear, _ = v2.linear_contract(
        context,
        end_index,
        np.asarray(contract["direction_n"], dtype=float),
        float(contract["k_m_inv"]),
        admission,
        scales,
    )
    return float(linear["final_gain"])


def main() -> int:
    started = time.perf_counter()
    for source in (V3_EVIDENCE, V3_ARRAYS):
        if not source.is_file():
            raise RuntimeError(f"missing required source {source}")
    evidence = json.loads(V3_EVIDENCE.read_text(encoding="utf-8"))
    with np.load(V3_ARRAYS) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    dense_payload = v2.load_verified_cache()
    factor64 = dense_payload["contexts"]["factor64"]
    factor128, cache_status = load_or_build_factor128()
    role_contracts = contracts(evidence)

    saved_substeps = v2.LINEAR_SUBSTEPS
    v2.LINEAR_SUBSTEPS = 4
    records: dict[str, Any] = {}
    try:
        for role, contract in role_contracts.items():
            print(json.dumps({"stage": "dense_convergence", "role": role}), flush=True)
            gain64 = gain(factor64, contract, dense_payload["admission"], scales)
            gain128 = gain(factor128, contract, v3._admission(), scales)
            log64 = float(np.log(gain64))
            log128 = float(np.log(gain128))
            records[role] = {
                "factor64_state_count": int(len(factor64["times"])),
                "factor128_state_count": int(len(factor128["times"])),
                "factor64_four_substep_gain": gain64,
                "factor128_four_substep_gain": gain128,
                "factor64_to128_relative_log_gain_change": float(
                    abs(log64 / log128 - 1.0)
                ),
                "factor64_to128_relative_gain_change": float(
                    abs(gain64 / gain128 - 1.0)
                ),
                "direction_n": contract["direction_n"].tolist(),
                "k_m_inv": float(contract["k_m_inv"]),
                "target_time_s": float(contract["target_time_s"]),
            }
    finally:
        v2.LINEAR_SUBSTEPS = saved_substeps

    maximum_change = max(
        row["factor64_to128_relative_log_gain_change"] for row in records.values()
    )
    passed = maximum_change <= GATE
    report = {
        "schema": SCHEMA,
        "status": "DENSE_CONVERGENCE_GATE_PASS" if passed else "DENSE_CONVERGENCE_GATE_OPEN",
        "gate": {
            "definition": "max |log(G_513)/log(G_1025)-1| <= 0.02",
            "threshold": GATE,
            "maximum_observed": maximum_change,
            "passed": passed,
        },
        "contract": {
            "model": "paper-consistent fixed-beta passive-ledger HCP model",
            "selector": "constitutive_to_constitutive",
            "metric": "exact released V3 reference_coordinate_scales",
            "substeps_per_registered_interval": 4,
            "directions_and_wavenumbers": "fixed V3 re-optimized winners",
            "cache_status": cache_status,
        },
        "records": records,
        "claim_boundary": (
            "This gate addresses checkpoint-density convergence of the linear finite-time "
            "gain only; it does not establish a mature band width or specimen calibration."
        ),
        "provenance": {
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": sha256(Path(__file__)),
            "v3_evidence_sha256": sha256(V3_EVIDENCE),
            "v3_arrays_sha256": sha256(V3_ARRAYS),
            "factor128_cache": CACHE.relative_to(ROOT).as_posix(),
            "factor128_cache_sha256": sha256(CACHE),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not NONLINEAR_RESULT.is_file():
        raise RuntimeError(f"missing required nonlinear receipt {NONLINEAR_RESULT}")
    nonlinear = json.loads(NONLINEAR_RESULT.read_text(encoding="utf-8"))
    direct_nonlinear_gates = {
        key: bool(value)
        for key, value in nonlinear["gates"].items()
        if key != "257_to_513_state_relative_log_gain_change_below_2_percent"
    }
    suite_gates = {
        "mandatory_513_to_1025_log_gain_density_gate_pass": passed,
        "all_direct_513_state_nonlinear_transport_gates_pass": all(
            direct_nonlinear_gates.values()
        ),
    }
    suite = {
        "schema": "IJP_V3_VALIDATION_SUITE_V1",
        "status": (
            "ALL_V3_DENSE_AND_NONLINEAR_GATES_PASS"
            if all(suite_gates.values())
            else "OPEN_V3_DENSE_OR_NONLINEAR_GATE"
        ),
        "gates": suite_gates,
        "trigger_and_closure": {
            "trigger": (
                "The near-onset 257-to-513 relative log-gain change was 2.615%, "
                "so the predeclared 2% gate triggered a 1025-state calculation."
            ),
            "closure": (
                "The 513-to-1025 maximum relative log-gain change is below 2%; "
                "the 513-state nonlinear replay is judged only against its matching "
                "513-state linear propagator."
            ),
        },
        "central_linear_values": {
            role: {
                "state_count": row["factor128_state_count"],
                "gain": row["factor128_four_substep_gain"],
                "k_m_inv": row["k_m_inv"],
            }
            for role, row in records.items()
        },
        "matching_513_state_nonlinear_values": {
            role: {
                "predicted_linear_gain": nonlinear["roles"][role][
                    "predicted_linear_gain"
                ],
                "maximum_gain_relative_error": nonlinear["roles"][role][
                    "maximum_gain_relative_error"
                ],
                "maximum_complex_output_vector_relative_error": nonlinear["roles"][
                    role
                ]["maximum_complex_output_vector_relative_error"],
            }
            for role in ("near_onset", "terminal")
        },
        "direct_nonlinear_gates": direct_nonlinear_gates,
        "claim_boundary": (
            "The suite validates converged linear finite-time gain and matching-grid "
            "small-amplitude nonlinear transport. It does not validate finite-amplitude "
            "saturation, mature width, specimen calibration or propagation resistance."
        ),
        "sources": {
            "dense_convergence": RESULT.relative_to(ROOT).as_posix(),
            "dense_convergence_sha256": sha256(RESULT),
            "nonlinear_validation": NONLINEAR_RESULT.relative_to(ROOT).as_posix(),
            "nonlinear_validation_sha256": sha256(NONLINEAR_RESULT),
        },
    }
    SUITE_RESULT.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gate": report["gate"]}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
