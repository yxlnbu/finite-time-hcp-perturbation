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
    assert report["contract"]["released_inputs"].startswith("V4 direct-reference")
    assert report["provenance"]["optimization_evidence"].endswith(
        "ijp_reference_reoptimization_v4.json"
    )


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

    assert report["status"] == (
        "START_AND_BRANCH_AUDIT_PASS__POSITIVE_ENERGY_SUBSPACE_ONLY__"
        "EXPERIMENTAL_METRIC_OPEN"
    )
    assert report["gates"]["branch_convention_gain_change_below_1_percent"]
    assert report["gates"]["threshold_rate_jump_below_1e_minus_12_of_imposed_rate"]
    assert report["gates"]["all_registered_yield_events_transverse_in_secant_audit"]
    assert report["gates"]["energy_metrics_constructed_without_diagonal_regularization"]
    assert not report["gates"]["energy_metric_quotient_invariance_available"]
    assert not report["gates"]["experimental_observation_covariance_available"]

    onset = report["roles"]["near_onset"]
    terminal = report["roles"]["terminal"]
    assert onset["branch_transition_count"] == 2
    assert terminal["branch_transition_count"] == 4
    assert terminal["physical_metric_audit"]["initial_energy_metric"]["rank"] == 28
    assert terminal["physical_metric_audit"]["initial_energy_metric"]["nullity"] == 41
    assert terminal["maximum_threshold_rate_jump_to_imposed_rate"] < 2.0e-17
    assert terminal["physical_metric_audit"][
        "energy_to_energy_nullspace_invariance"
    ]["relative_nullspace_leakage"] > 0.99
    assert not terminal["physical_metric_audit"][
        "energy_to_energy_nullspace_invariance"
    ]["quotient_map_well_defined_within_tolerance"]
    assert terminal["start_time_sensitivity"][-1]["relative_to_0p5us_gain"] < 0.33
    assert not terminal["physical_metric_audit"]["experimental_covariance_available"]


def test_all_reference_contracts_are_directly_optimized_and_stationary() -> None:
    report = load("05_results/ijp_reference_reoptimization_v4.json")

    assert report["status"] == "REFERENCE_REOPTIMIZATION_PASS"
    assert len(report["records"]) == 70
    assert all(report["gates"].values())
    assert all(item["local_stationarity_pass"] for item in report["records"].values())
    classifications = {
        item["boundary_classification"] for item in report["records"].values()
    }
    assert "interior_finite_band_candidate" in classifications
    assert any("long_wave" in item for item in classifications)
    assert any("high_frequency" in item for item in classifications)


def test_near_optimal_sets_distinguish_flat_onset_from_terminal_basin() -> None:
    report = load("05_results/ijp_v4_near_optimal_set_audit_v1.json")

    assert report["status"] == "NEAR_OPTIMAL_SET_AUDIT_PASS"
    assert all(report["gates"].values())
    onset = report["horizons"]["onset"]["near_optimal_sets"]["epsilon_0.001"]
    terminal = report["horizons"]["terminal"]["near_optimal_sets"]["epsilon_0.001"]
    assert onset["sampled_member_count"] > 400
    assert onset["maximum_projective_angle_from_winner_deg"] > 80.0
    assert terminal["sampled_member_count"] <= 3
    assert terminal["maximum_projective_angle_from_winner_deg"] < 0.1


def test_v4_dense_and_galerkin_transport_gates_are_layered() -> None:
    dense = load("05_results/ijp_v4_dense_convergence_v1.json")
    replay = load("05_results/ijp_singular_vector_nonlinear_validation_v4.json")

    assert dense["status"] == "V4_DENSE_CONVERGENCE_GATE_PASS"
    assert dense["gate"]["passed"]
    assert replay["status"] == "GALERKIN_TRANSPORT_PASS_NONLINEAR_RHS_CLOSURE_OPEN"
    assert replay["transport_passed"]
    assert all(replay["transport_gates"].values())
    assert not replay["complete_nonlinear_rhs_closure_passed"]
    assert not replay["closure_gates"]["discarded_rhs_fourier_energy_below_1_percent"]
