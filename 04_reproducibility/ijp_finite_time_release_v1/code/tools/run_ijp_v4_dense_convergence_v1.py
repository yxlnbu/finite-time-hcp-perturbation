"""Recheck 513-to-1025 density convergence at the V4 reference optima."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402
import tools.run_ijp_v3_dense_convergence_v1 as dense_v3  # noqa: E402


SCHEMA = "IJP_V4_DENSE_CONVERGENCE_V1"
V4_EVIDENCE = ROOT / "05_results/ijp_reference_reoptimization_v4.json"
V4_ARRAYS = ROOT / "05_results/ijp_reference_reoptimization_v4_arrays.npz"
RESULT = ROOT / "05_results/ijp_v4_dense_convergence_v1.json"
GATE = 0.02


def contracts(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = evidence["records"]
    output: dict[str, dict[str, Any]] = {}
    for horizon, role in (("onset", "near_onset"), ("terminal", "terminal")):
        row = records[f"{horizon}__baseline__constitutive_to_constitutive"]
        output[role] = {
            "target_time_s": float(row["time_s"]),
            "direction_n": np.asarray(row["direction_n"], dtype=float),
            "k_m_inv": float(row["k_m_inv"]),
        }
    return output


def main() -> int:
    started = time.perf_counter()
    for source in (V4_EVIDENCE, V4_ARRAYS):
        if not source.is_file():
            raise RuntimeError(f"missing required source {source}")
    evidence = json.loads(V4_EVIDENCE.read_text(encoding="utf-8"))
    if evidence.get("status") != "REFERENCE_REOPTIMIZATION_PASS":
        raise RuntimeError("V4 reference re-optimization gates are open")
    with np.load(V4_ARRAYS) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    dense_payload = v2.load_verified_cache()
    factor64 = dense_payload["contexts"]["factor64"]
    factor128, cache_status = dense_v3.load_or_build_factor128()
    role_contracts = contracts(evidence)

    saved_substeps = v2.LINEAR_SUBSTEPS
    v2.LINEAR_SUBSTEPS = 4
    records: dict[str, Any] = {}
    try:
        for role, contract in role_contracts.items():
            print(json.dumps({"stage": "v4_dense_convergence", "role": role}), flush=True)
            gain64 = dense_v3.gain(
                factor64, contract, dense_payload["admission"], scales
            )
            gain128 = dense_v3.gain(
                factor128, contract, v3._admission(), scales
            )
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
        "status": "V4_DENSE_CONVERGENCE_GATE_PASS" if passed else "V4_DENSE_CONVERGENCE_GATE_OPEN",
        "gate": {
            "definition": "max |log(G_513)/log(G_1025)-1| <= 0.02",
            "threshold": GATE,
            "maximum_observed": maximum_change,
            "passed": passed,
        },
        "contract": {
            "selector": "constitutive_to_constitutive",
            "metric": "V4 reference coordinate scales",
            "substeps_per_registered_interval": 4,
            "directions_and_wavenumbers": "fixed direct-reference V4 winners",
            "cache_status": cache_status,
        },
        "records": records,
        "claim_boundary": (
            "This gate addresses checkpoint-density convergence of the linear finite-time "
            "gain at the V4 optima; it does not establish nonlinear closure, mature width "
            "or specimen calibration."
        ),
        "provenance": {
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": dense_v3.sha256(Path(__file__)),
            "v4_evidence_sha256": dense_v3.sha256(V4_EVIDENCE),
            "v4_arrays_sha256": dense_v3.sha256(V4_ARRAYS),
            "factor128_cache": dense_v3.CACHE.relative_to(ROOT).as_posix(),
            "factor128_cache_sha256": dense_v3.sha256(dense_v3.CACHE),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "gate": report["gate"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
