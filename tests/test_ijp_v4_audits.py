from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_full_grid_convergence_is_separated_from_rhs_closure() -> None:
    report = load("05_results/ijp_v4_full_fourier_convergence_audit_v1.json")
    audit = report["audit"]

    assert report["status"] == "NUMERICAL_CONVERGENCE_PASS_NONLINEAR_RHS_CLOSURE_OPEN"
    assert audit["numerical_convergence_passed"]
    assert not audit["full_nonlinear_rhs_closure_passed"]
    assert audit["gates"]["two_thirds_grid_ladder_completes"]
    assert audit["gates"]["n32_to_n64_gain_change_below_1_percent"]
    assert audit["gates"]["time_density_crosscheck_below_1_percent"]
    assert audit["gates"]["integration_format_change_below_2_percent"]
    assert not audit["gates"]["primary_dealiased_discarded_rhs_energy_below_1_percent"]
    assert audit["maximum_primary_dealiased_discarded_rhs_fourier_energy_fraction"] > 0.8


def test_raw_failure_is_localized_to_the_nyquist_coordinate() -> None:
    report = load("05_results/ijp_v4_full_fourier_convergence_audit_v1.json")
    failure = report["audit"]["first_failure_localization"]

    assert report["audit"]["gates"]["raw_negative_control_fails_at_nyquist"]
    assert report["audit"]["gates"]["matched_no_nyquist_control_completes_both_horizons"]
    assert failure["cells"] == 16
    assert failure["conjugacy_defect_nonnegative_mode"] == 8
    assert failure["conjugacy_defect_state_group"] == "signed_slip"
    assert failure["dominant_state_group"] == "velocity"
    assert failure["stage"] == "midpoint_stage"


def test_start_branch_and_metric_claim_boundaries_are_explicit() -> None:
    report = load("05_results/ijp_v4_start_branch_metric_audit_v1.json")

    assert report["status"] == "START_AND_BRANCH_AUDIT_PASS__EXPERIMENTAL_METRIC_OPEN"
    assert report["gates"]["branch_convention_gain_change_below_1_percent"]
    assert report["gates"]["energy_metrics_constructed_without_diagonal_regularization"]
    assert not report["gates"]["experimental_observation_covariance_available"]

    onset = report["roles"]["near_onset"]
    terminal = report["roles"]["terminal"]
    assert onset["branch_transition_count"] == 2
    assert terminal["branch_transition_count"] == 4
    assert terminal["physical_metric_audit"]["initial_energy_metric"]["rank"] == 28
    assert terminal["physical_metric_audit"]["initial_energy_metric"]["nullity"] == 41
    assert terminal["start_time_sensitivity"][-1]["relative_to_0p5us_gain"] < 0.33
    assert not terminal["physical_metric_audit"]["experimental_covariance_available"]
