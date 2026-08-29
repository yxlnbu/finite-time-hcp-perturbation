"""Audit grid-resolved nonlinear transport with every 16-cell Fourier mode.

The four-mode V3 replay is a Galerkin transport check.  This audit removes
that spectral truncation, retains all nonnegative modes 0..8 and records
whether the same onset and terminal singular inputs remain finite for one and
four nonlinear midpoint substeps per factor-64 history interval.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import tools.build_ijp_revision_evidence_v3 as v3  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v2 as v2  # noqa: E402
import tools.run_ijp_singular_vector_nonlinear_validation_v3 as v3nl  # noqa: E402


SCHEMA = "IJP_V3_FULL_FOURIER_NONLINEAR_AUDIT_V1"
RESULT = ROOT / "05_results/ijp_v3_full_fourier_nonlinear_audit_v1.json"


def main() -> int:
    started = time.perf_counter()
    evidence = json.loads(v3nl.V3_EVIDENCE.read_text(encoding="utf-8"))
    with np.load(v3nl.V3_ARRAYS) as released:
        scales = np.asarray(released["reference_coordinate_scales"], dtype=float)
    dense_payload = v2.load_verified_cache()
    context = dense_payload["contexts"]["factor64"]
    _, _, _, spectral = v3.build_case_models("v3_full_fourier_audit")
    admission = v3._admission()
    role_contracts = v3nl.contracts(evidence)

    v2.LINEAR_SUBSTEPS = 1
    v2.RETAINED_MODES = tuple(range(v2.CELLS // 2 + 1))
    trials = []
    for nonlinear_substeps in (1, 4):
        v2.NONLINEAR_SUBSTEPS = nonlinear_substeps
        for role in ("near_onset", "terminal"):
            trial_started = time.perf_counter()
            record = {
                "role": role,
                "nonlinear_substeps_per_interval": nonlinear_substeps,
                "retained_nonnegative_fourier_modes": list(v2.RETAINED_MODES),
                "cells": v2.CELLS,
            }
            caught = []
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    summary, cases, _ = v2.execute_role(
                        role,
                        role_contracts[role],
                        context,
                        spectral,
                        admission,
                        scales,
                    )
                    caught = [str(item.message) for item in observed]
                record.update(
                    {
                        "completed": True,
                        "summary": summary,
                        "cases": cases,
                        "maximum_discarded_rhs_fourier_energy_fraction": max(
                            case["integration"][
                                "maximum_discarded_rhs_fourier_energy_fraction"
                            ]
                            for case in cases
                        ),
                    }
                )
            except (FloatingPointError, OverflowError, ValueError) as error:
                record.update(
                    {
                        "completed": False,
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
            record["warnings"] = caught
            record["runtime_s"] = float(time.perf_counter() - trial_started)
            trials.append(record)
            if not record["completed"]:
                break

    passed = all(
        trial["completed"]
        and trial["maximum_discarded_rhs_fourier_energy_fraction"] <= 0.01
        for trial in trials
    ) and len(trials) == 4
    report = {
        "schema": SCHEMA,
        "status": (
            "FULL_FOURIER_NONLINEAR_GATE_PASS"
            if passed
            else "FULL_FOURIER_NONLINEAR_GATE_OPEN"
        ),
        "gate": {
            "definition": (
                "Both optimized horizons remain finite with all 16-cell Fourier modes "
                "for one and four nonlinear substeps, and discarded RHS energy <=1%."
            ),
            "passed": passed,
        },
        "contract": {
            "base_history": "factor64 / 513 states",
            "selector": "constitutive_to_constitutive",
            "directions_and_wavenumbers": "fixed V3 re-optimized winners",
            "coordinate_metric": "exact released V3 metric",
            "cells_per_fundamental_wavelength": v2.CELLS,
            "retained_nonnegative_fourier_modes": list(v2.RETAINED_MODES),
        },
        "trials": trials,
        "interpretation": (
            "Failure means the four-mode replay cannot be promoted to complete "
            "grid-resolved nonlinear validation. It is evidence of an unresolved "
            "high-wavenumber/nonlinear closure problem, not evidence for mature width."
        ),
        "provenance": {
            "tool": Path(__file__).relative_to(ROOT).as_posix(),
            "tool_sha256": v2.sha256(Path(__file__)),
            "v3_evidence_sha256": v2.sha256(v3nl.V3_EVIDENCE),
            "v3_arrays_sha256": v2.sha256(v3nl.V3_ARRAYS),
            "base_context_cache": v2.CACHE.relative_to(ROOT).as_posix(),
            "base_context_cache_sha256": v2.sha256(v2.CACHE),
        },
        "runtime": {
            "wall_time_s": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": report["gate"]}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
