from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_full_state_operator_contract_and_primary_differences() -> None:
    report = load("05_results/ijp_operator_consistency_v2.json")
    assert report["status"] == "ALL_OPERATOR_CONSISTENCY_GATES_PASS"
    assert all(report["gates"].values())

    expected = {
        "onset_x": 4.784123,
        "onset_y": 7.217005,
        "terminal_y": 6.402256,
    }
    for branch, target in expected.items():
        record = report["branches"][branch]
        contract = record["primary_contract"]
        assert contract["state_space"] == "complete reduced 69-state generator"
        assert contract["input_indices"] == list(range(69))
        assert contract["output_indices"] == list(range(69))
        assert abs(record["final_primary_log_discrepancy"] - target) < 2.0e-6
        assert record["logarithmic_norm_upper_bound_satisfied"]

    onset_x = report["branches"]["onset_x"]
    assert onset_x["final_full_log_gain"] - onset_x["final_projected_log_gain"] > 5.0


def test_norm_selector_winners_are_stable_within_each_declared_question() -> None:
    report = load("05_results/ijp_operator_consistency_v2.json")
    rows = report["norm_selector_summary"]
    assert len(rows) == 11 * 7 * 2

    winners: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        winners.setdefault((row["horizon_id"], row["selector_id"]), set()).add(
            row["winner_branch"]
        )
    assert all(len(value) == 1 for value in winners.values())

    assert winners[("near_onset", "full_to_full")] == {"onset_y"}
    assert winners[("near_onset", "constitutive_to_temperature")] == {"terminal_y"}
    assert winners[("terminal", "full_to_full")] == {"terminal_y"}
    assert winners[("terminal", "mechanical_to_mechanical")] == {"onset_x"}


def test_orientation_and_search_claim_boundaries() -> None:
    report = load("05_results/ijp_strengthening_evidence_v1.json")
    assert report["status"] == "ALL_REQUIRED_STRENGTHENING_GATES_PASS"
    assert report["global_search"]["status"] == "ANCHOR_ASSISTED_SEARCH_AUDIT_PASS"
    assert "numerical_global_search_certificate_pass" not in report["gates"]

    for record in report["orientation_transfer"]["orientations"].values():
        contract = record["comparison_contract"]
        assert contract["state_space"] == "complete reduced 69-state generator"
        assert contract["input_selector"] == "identity"
        assert contract["output_selector"] == "identity"
        assert record["released_propagator_relative_error"] < 2.0e-10
        assert record["non_equivalence_gate"]
