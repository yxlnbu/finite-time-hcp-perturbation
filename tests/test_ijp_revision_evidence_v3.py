from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tools.build_ijp_revision_evidence_v3 as v3

from tools.build_ijp_revision_evidence_v3 import (
    DEGENERACY_RELATIVE_GAP,
    refined_shears,
    singular_diagnostics,
)


def test_factor16_history_has_129_strictly_increasing_states() -> None:
    values = refined_shears(16)
    assert values.shape == (129,)
    assert np.all(np.diff(values) > 0.0)
    assert values[0] == 0.005
    assert values[-1] == 1.4


def test_factor128_history_has_1025_strictly_increasing_states() -> None:
    values = refined_shears(128)
    assert values.shape == (1025,)
    assert np.all(np.diff(values) > 0.0)
    assert values[0] == 0.005
    assert values[-1] == 1.4


def test_formal_singular_gap_gate_and_rank_one_contract() -> None:
    nearly_repeated = np.diag([1.0, 0.96, 0.2]).astype(complex)
    diagnostic, _ = singular_diagnostics(
        nearly_repeated, np.asarray([0, 1]), np.asarray([0, 1])
    )
    assert np.isclose(diagnostic["relative_singular_gap"], 0.04)
    assert diagnostic["relative_singular_gap"] < DEGENERACY_RELATIVE_GAP
    assert diagnostic["near_degenerate"] is True
    assert diagnostic["individual_vector_interpretation_authorized"] is False

    separated = np.diag([1.0, 0.5, 0.2]).astype(complex)
    diagnostic, _ = singular_diagnostics(
        separated, np.asarray([0, 1]), np.asarray([0, 1])
    )
    assert diagnostic["near_degenerate"] is False
    assert diagnostic["individual_vector_interpretation_authorized"] is True

    rank_one, _ = singular_diagnostics(
        separated, np.asarray([0]), np.asarray([0, 1])
    )
    assert rank_one["sigma_2"] is None
    assert rank_one["sigma_1_over_sigma_2"] is None
    assert rank_one["rank_one_observable"] is True


def test_v3_uses_the_paper_fixed_beta_passive_ledger_contract() -> None:
    _, parameters, model, spectral = v3.build_case_models("contract_test")
    assert spectral.power_partition_law is None
    assert model.parameters is parameters
    assert parameters.taylor_quinney == 0.9


def test_v3_receipts_separate_linear_galerkin_and_full_fourier_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    dense = json.loads(
        (root / "05_results/ijp_v3_dense_convergence_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert dense["status"] == "DENSE_CONVERGENCE_GATE_PASS"
    assert dense["records"]["terminal"]["factor128_state_count"] == 1025
    assert dense["gate"]["maximum_observed"] <= 0.02

    galerkin = json.loads(
        (
            root
            / "05_results/ijp_singular_vector_nonlinear_validation_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert galerkin["classification"].startswith("FOUR_MODE_GALERKIN")
    assert galerkin["gates"]["discarded_rhs_fourier_energy_below_1_percent"] is False

    full = json.loads(
        (root / "05_results/ijp_v3_full_fourier_nonlinear_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert full["status"] == "FULL_FOURIER_NONLINEAR_GATE_OPEN"
    assert full["gate"]["passed"] is False
    assert {trial["nonlinear_substeps_per_interval"] for trial in full["trials"]} == {
        1,
        4,
    }
