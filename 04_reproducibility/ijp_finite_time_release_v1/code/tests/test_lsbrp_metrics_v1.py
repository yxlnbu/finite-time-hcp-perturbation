import numpy as np

from hcp_cp_gnd.lsbrp_metrics_v1 import (
    QuadraticMetricV1,
    construct_observation_metric,
    metric_nullspace_invariance_audit,
    weighted_propagator_gain,
)


def test_psd_metric_exposes_nullspace_without_regularization():
    metric = QuadraticMetricV1(
        np.diag([4.0, 0.0, 9.0]), "test", "unit-test exact PSD"
    )
    assert metric.rank == 2
    assert metric.nullity == 1
    B = metric.state_from_unit_coordinates()
    C = metric.unit_coordinates_from_state()
    assert np.allclose(C @ B, np.eye(2), atol=2.0e-14)
    assert np.allclose(B.conj().T @ metric.matrix @ B, np.eye(2), atol=2.0e-14)


def test_observation_metric_is_H_star_R_inverse_H():
    H = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0]])
    R = np.diag([4.0, 9.0])
    metric = construct_observation_metric(
        H, R, state_dimension=3, provenance="unit-test measured covariance"
    )
    assert metric.rank == 2
    assert np.allclose(metric.matrix, H.T @ np.linalg.solve(R, H))


def test_weighted_gain_matches_direct_spd_formula():
    Phi = np.array([[2.0, 1.0], [0.0, 0.5]], dtype=complex)
    Win = QuadraticMetricV1(np.diag([4.0, 1.0]), "in", "unit-test")
    Wout = QuadraticMetricV1(np.diag([1.0, 9.0]), "out", "unit-test")
    result = weighted_propagator_gain(
        Phi, input_metric=Win, output_metric=Wout
    )
    expected = np.linalg.svd(
        np.diag([1.0, 3.0]) @ Phi @ np.diag([0.5, 1.0]),
        compute_uv=False,
    )[0]
    assert np.isclose(result["gain"], expected, rtol=2.0e-14)
    assert np.isclose(Win.quadratic_value(result["input_state"]), 1.0)


def test_indefinite_metric_is_rejected_not_clipped():
    try:
        QuadraticMetricV1(np.diag([1.0, -1.0e-3]), "bad", "unit-test")
    except ValueError as error:
        assert "indefinite" in str(error)
    else:
        raise AssertionError("an indefinite metric was silently accepted")


def test_psd_quotient_invariance_is_verified_when_nullspace_stays_null():
    metric = QuadraticMetricV1(np.diag([1.0, 0.0]), "psd", "unit-test")
    audit = metric_nullspace_invariance_audit(
        np.diag([2.0, 3.0]), input_metric=metric, output_metric=metric
    )
    assert audit["quotient_map_well_defined_within_tolerance"]
    assert audit["relative_nullspace_leakage"] == 0.0


def test_psd_quotient_invariance_rejects_null_to_positive_coupling():
    metric = QuadraticMetricV1(np.diag([1.0, 0.0]), "psd", "unit-test")
    propagator = np.array([[1.0, 0.5], [0.0, 1.0]])
    audit = metric_nullspace_invariance_audit(
        propagator, input_metric=metric, output_metric=metric
    )
    assert not audit["quotient_map_well_defined_within_tolerance"]
    assert audit["relative_nullspace_leakage"] > 0.0
