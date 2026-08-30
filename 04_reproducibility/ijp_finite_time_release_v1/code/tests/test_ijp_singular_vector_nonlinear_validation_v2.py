from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_onset_and_terminal_singular_vectors_pass_nonlinear_gate() -> None:
    report = json.loads(
        (
            ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "BOTH_SINGULAR_VECTOR_NONLINEAR_GATES_PASS"
    assert all(report["gates"].values())

    by_role: dict[str, list[dict]] = {}
    for case in report["cases"]:
        by_role.setdefault(case["role"], []).append(case)
        assert case["gain_relative_error"] < 0.02
        assert case["complex_output_vector_relative_error"] < 0.03
        assert case["nonretained_fourier_energy_fraction"] < 0.01
        assert case["ledger_partition_maximum_relative_residual"] < 1.0e-10
        assert case["integration"]["completed_requested_history"]

    assert set(by_role) == {"near_onset", "terminal"}
    assert all(len(cases) == 2 for cases in by_role.values())
    assert max(case["gain_relative_error"] for case in by_role["near_onset"]) < 0.013
    assert max(case["gain_relative_error"] for case in by_role["terminal"]) < 0.003

    for role in report["roles"].values():
        assert role["cells_per_wavelength"] >= 16
        assert role["factor_two_gain_collapse"] < 0.02
