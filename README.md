# Finite-time perturbation selection in evolving HCP crystal plasticity

Public reproducibility materials for the manuscript:

> *Finite-time perturbation selection in evolving HCP crystal plasticity: path-dependent direction, wavelength and mechanism*

The repository contains the audited manuscript, machine-readable numerical evidence, the 69-state generator histories, optimal singular vectors, search trajectories, Galerkin/full-Fourier gate receipts, parameter provenance and the scripts needed to reassess the principal claims.

## Main findings represented by this release

- Propagator and frozen-spectrum quantities are compared on the same complete 69-state space, fixed scaling and identity selectors.
- The three audited full-state log-gain discrepancies are `+4.784`, `+7.217` and `+6.402`.
- All seven input--output selectors are independently re-optimized over direction and wavenumber under five norms at onset and terminal horizons. Fifteen unrestricted/mechanical cases hit the upper-wavenumber boundary; constitutive-response selectors retain interior maxima.
- A strictly matched piecewise-frozen baseline uses the same input/output projectors and metric and converges toward the time-ordered propagator under interval-preserving refinement.
- Commutators, modal condition numbers, eigenvector rotation and chronological-order controls diagnose the frozen/finite-time difference.
- Three independently selected crystal orientations retain the non-equivalence.
- The direction--wavenumber evidence is classified as an **anchor-assisted basin-retention audit**, not an independent global-optimum certificate.
- Three added rate/temperature paths, four local parameter sensitivities and a bounded Grade-II CP-Ti comparison expose transfer and calibration limits.
- The mandatory 513-to-1025 history-density audit passes in the declared log-gain objective; the 1025-state terminal constitutive gain is `6.2906e5` at `k = 6.8369e3 m^-1`.
- A four-mode Galerkin replay reproduces the matching 513-state gains within `0.730%` and `0.223%`, but its terminal discarded-RHS energy gate fails. Retaining every Fourier mode resolvable on the 16-cell grid makes near-onset continuation non-finite for both one and four nonlinear substeps.
- The present theory therefore closes finite-time finite-band **linear selection**, but not complete nonlinear harmonic transfer, mature material width or propagation resistance.

## Repository layout

- `src/hcp_cp/` -- vendored, frozen v0.1 constitutive kernel used by the release.
- `src/hcp_cp_gnd/` -- HCP crystal-plasticity and perturbation kernels used by the released audits.
- `config/verification_seed.yaml` -- exact watermarked v0.1 verification configuration; no sibling checkout is required.
- `tools/` -- operator, search, nonlinear-validation, provenance and packaging scripts.
- `05_results/` -- JSON/CSV/NPZ evidence and complete numerical histories.
- `04_reproducibility/literature_digitization/guo2019_prl/` -- the external CP-Ti landmarks and extraction record used by the bounded comparison.
- `06_manuscript/ijp_spectral_hcp/` -- LaTeX source, figures, evidence ledger and audited PDF.
- `environment/requirements-lock.txt` -- minimal public Python runtime.
- `05_results/ijp_reproducibility_release_v1.json` -- release checksum and array-contract receipt.

The latest audited manuscript is [`06_manuscript/ijp_spectral_hcp/main_finite_time.pdf`](06_manuscript/ijp_spectral_hcp/main_finite_time.pdf).

## Quick verification

Python 3.11 or newer is required.

```bash
python -m venv .venv
python -m pip install -r environment/requirements-lock.txt
python -m pip install -e .
python -m pytest tests/test_ijp_operator_consistency_v2.py tests/test_ijp_singular_vector_nonlinear_validation_v2.py tests/test_ijp_revision_evidence_v3.py -q
```

Rebuild the like-for-like operator, norm and mechanism evidence with:

```bash
python tools/build_ijp_operator_consistency_v2.py
```

The V3 re-optimization, density ladder and nonlinear audits are intentionally more expensive than the receipt-based tests:

```bash
python tools/build_ijp_revision_evidence_v3.py
python tools/run_ijp_singular_vector_nonlinear_validation_v3.py
python tools/run_ijp_v3_dense_convergence_v1.py
python tools/audit_ijp_v3_full_fourier_nonlinear_v1.py
```

The factor-128 context cache is intentionally not versioned because it exceeds GitHub's per-file limit; `run_ijp_v3_dense_convergence_v1.py` rebuilds it and verifies its SHA-256 manifest locally.

## Evidence and claim boundaries

The machine-readable claim ledger is available at [`06_manuscript/ijp_spectral_hcp/metadata/finite_time_claim_evidence_v2.csv`](06_manuscript/ijp_spectral_hcp/metadata/finite_time_claim_evidence_v2.csv).

The release does **not** claim:

- specimen-calibrated onset or experimental validation;
- an independent global direction--wavenumber optimum;
- a seed-independent mature shear-band width;
- complete grid-resolved nonlinear harmonic closure;
- an identified dislocation Helmholtz free energy;
- areal dissipation or propagation resistance;
- a strict analytical HCP-to-Bai degeneration.

## Integrity and privacy

The public environment record contains no local user paths, credentials or access tokens. Numerical arrays have explicit shape and finite-value contracts in the release receipt, while Git commit hashes provide repository integrity.

## License status

No software or data license has yet been selected by the authors. Public visibility should not be interpreted as a grant of reuse rights beyond applicable law.
