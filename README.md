# Finite-time direction--wavenumber selection in evolving HCP crystal plasticity

Public reproducibility materials for the manuscript:

> *Finite-time direction--wavenumber selection in evolving HCP crystal plasticity: exact descriptor reduction and path-dependent amplification*

The repository contains the manuscript source, machine-readable numerical evidence, the 69-state generator histories, optimal singular vectors, search trajectories, Galerkin/dealiased full-grid gate receipts, parameter provenance and the scripts needed to reassess the principal claims.

## Main findings represented by this release

- Propagator and frozen-spectrum quantities are compared on the same complete 69-state space, fixed scaling and identity selectors.
- The three audited full-state log-gain discrepancies are `+4.784`, `+7.217` and `+6.402`.
- All 70 horizon--norm--selector contracts are directly optimized on the 129-state/four-substep objective. Fifty are finite-band interior candidates, five onset temperature maps are lower-boundary long-wave branches and 15 reversible/mechanical maps are high-frequency branches; all pass the retained six-neighbour stationarity and boundary-extension checks.
- A strictly matched piecewise-frozen baseline uses the same input/output projectors and metric and converges toward the time-ordered propagator under interval-preserving refinement.
- Commutators, modal condition numbers, eigenvector rotation and chronological-order controls diagnose the frozen/finite-time difference.
- Three independently selected crystal orientations retain the non-equivalence.
- An independent sampled challenge and local-ring/wavenumber-profile audit quantifies the near-optimal sets. The onset landscape is nearly flat and does not define a unique length or direction; the terminal constitutive basin is narrow. This remains a sampled audit, not a continuum global-optimum proof.
- Three added rate/temperature paths, four local parameter sensitivities and a bounded Grade-II CP-Ti comparison expose transfer and calibration limits.
- The mandatory 513-to-1025 history-density audit passes in the declared relative log-gain objective; the 1025-state terminal constitutive gain is `6.3378e5` at the V4 basin `k = 7.7240e3 m^-1`. The 3.46% raw-gain change is reported separately.
- A raw 16-cell all-mode negative control first fails at `1.3203125 microseconds`, interval 84, midpoint substep 0, at the even-grid Nyquist mode (`k = 4.3640e5 m^-1`): velocity coordinate 4 dominates and signed-slip coordinate 55 first violates conjugacy. The matched no-Nyquist control completes at both horizons. Two-thirds-dealiased 16/32/64-cell replays converge in gain, but the primary terminal audit places `84.58%` of instantaneous nonlinear RHS energy above the retained cutoff.
- Four registered active-set transition intervals have been audited with left-, right- and secant-split propagation conventions; terminal gain varies by at most `0.0622%`. The rank-28 recoverable-energy seminorm fails nullspace invariance (`0.9996` terminal leakage), so only positive-energy-subspace restricted gains are reported; no quotient norm is claimed.
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

The latest audited source is [`06_manuscript/ijp_spectral_hcp/main_finite_time.tex`](06_manuscript/ijp_spectral_hcp/main_finite_time.tex); the matching compiled manuscript and supplementary PDFs are committed in the same directory. Both builds have no undefined citations/references or overfull boxes.

## Quick verification

Python 3.11 or newer is required.

```bash
python -m venv .venv
python -m pip install -r environment/requirements-lock.txt
python -m pip install -e .
python -m pytest -q
```

Rebuild the like-for-like operator, norm and mechanism evidence with:

```bash
python tools/build_ijp_operator_consistency_v2.py
```

The direct V4 re-optimization, near-optimal-set challenge, density ladder and nonlinear audits are intentionally more expensive than the receipt-based tests:

```bash
python tools/build_ijp_reference_reoptimization_v4.py
python tools/audit_ijp_v4_near_optimal_set.py
python tools/run_ijp_v4_dense_convergence_v1.py
python tools/run_ijp_singular_vector_nonlinear_validation_v3.py
python tools/audit_ijp_v4_full_fourier_convergence_v1.py
python tools/audit_ijp_v4_start_branch_metric_v1.py
```

The factor-128 context cache is intentionally not versioned because it exceeds GitHub's per-file limit; `run_ijp_v4_dense_convergence_v1.py` rebuilds the required 1025-state receipt locally.

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
