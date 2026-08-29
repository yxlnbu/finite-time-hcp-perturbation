# Finite-time perturbation selection in evolving HCP crystal plasticity

Public reproducibility materials for the manuscript:

> *Finite-time perturbation selection in evolving HCP crystal plasticity: path-dependent direction, wavelength and mechanism*

The repository contains the audited manuscript, machine-readable numerical evidence, the 69-state generator histories, optimal singular vectors, search trajectories, nonlinear validation fields, parameter provenance and the scripts needed to reassess the principal claims.

## Main findings represented by this release

- Propagator and frozen-spectrum quantities are compared on the same complete 69-state space, fixed scaling and identity selectors.
- The three audited full-state log-gain discrepancies are `+4.784`, `+7.217` and `+6.402`.
- Eleven norm definitions and seven input--output selector pairs expose both robust winners and physically meaningful selector dependence.
- Commutators, modal condition numbers, eigenvector rotation and chronological-order controls diagnose the frozen/finite-time difference.
- Three independently selected crystal orientations retain the non-equivalence.
- The direction--wavenumber evidence is classified as an **anchor-assisted basin-retention audit**, not an independent global-optimum certificate.
- Complete nonlinear periodic calculations reproduce the optimized near-onset and terminal singular-vector gains within `1.24%` and `0.223%`.
- The present theory closes finite-time finite-band selection and small-amplitude transport, but does not identify a mature material band width or propagation resistance.

## Repository layout

- `src/hcp_cp_gnd/` -- HCP crystal-plasticity and perturbation kernels used by the released audits.
- `tools/` -- operator, search, nonlinear-validation, provenance and packaging scripts.
- `05_results/` -- JSON/CSV/NPZ evidence and complete numerical histories.
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
python -m pytest tests/test_ijp_operator_consistency_v2.py tests/test_ijp_singular_vector_nonlinear_validation_v2.py -q
```

Rebuild the like-for-like operator, norm and mechanism evidence with:

```bash
python tools/build_ijp_operator_consistency_v2.py
```

The terminal nonlinear validation is intentionally more expensive than the receipt-based tests:

```bash
python tools/run_ijp_singular_vector_nonlinear_validation_v2.py
```

## Evidence and claim boundaries

The machine-readable claim ledger is available at [`06_manuscript/ijp_spectral_hcp/metadata/finite_time_claim_evidence_v2.csv`](06_manuscript/ijp_spectral_hcp/metadata/finite_time_claim_evidence_v2.csv).

The release does **not** claim:

- specimen-calibrated onset or experimental validation;
- an independent global direction--wavenumber optimum;
- a seed-independent mature shear-band width;
- an identified dislocation Helmholtz free energy;
- areal dissipation or propagation resistance;
- a strict analytical HCP-to-Bai degeneration.

## Integrity and privacy

The public environment record contains no local user paths, credentials or access tokens. Numerical arrays have explicit shape and finite-value contracts in the release receipt, while Git commit hashes provide repository integrity.

## License status

No software or data license has yet been selected by the authors. Public visibility should not be interpreted as a grant of reuse rights beyond applicable law.
