# Finite-time HCP perturbation reproducibility release V1

This archive supports the finite-time HCP crystal-plasticity manuscript.

## Scope

- `code/src/hcp_cp/` contains the frozen, watermarked v0.1 constitutive kernel; no sibling checkout is required.
- `data/05_results/ijp_strengthening_evidence_v1_arrays.npz` contains every 69 x 69 complex generator at all 129 reference checkpoints for the three reported baseline branches, their full complex leading input/output singular vectors, complete observed singular spectra, propagators, three-orientation generators, nested Sobol samples, anchor-assisted audit data, and positive controls.
- `ijp_operator_consistency_v2.*` records the like-for-like full-state audit, norm--selector matrix, commutators, modal conditioning and eigenvector rotation.
- `ijp_revision_evidence_v3.*` records selector-specific re-optimization, matched piecewise freezing, loading transfer and local sensitivity; it is retained as reconnaissance and historical provenance.
- `ijp_reference_reoptimization_v4.*` records direct 129-state/four-substep optimization, local-stationarity gates, boundary extensions, complete V4 singular vectors and optimizer traces for all 70 contracts.
- `ijp_v4_near_optimal_set_audit_v1.*` and `ijp_v4_dense_convergence_v1.json` record the independent sampled challenge, set-valued basin audit and 513-to-1025 state recheck at the V4 coordinates.
- `ijp_v4_full_fourier_convergence_audit_v1.*` records the Nyquist negative control and dealiased space--time--integration convergence ladder.
- `ijp_v4_start_branch_metric_audit_v1.*` records start-time, active-set switching and positive-energy-subspace/nullspace-invariance audits.
- `reference_optimizer_trace.csv` contains every retained objective evaluation for the direct-reference local-search starts.
- `ijp_material_parameter_provenance_v1.csv` distinguishes literature constraints, repository verification values, model controls, and numerical controls.
- `environment/` freezes the runtime and Python package set.
- `manuscript/` contains the audited PDF, its edited source sections, the generated parameter and 70-contract tables, the active figures, and the comprehensive V4 revision audit.
- `MANIFEST.sha256` authenticates every packaged file.

## Claim boundary

The package supports like-for-like full-state finite-time/frozen discrimination, norm--selector robustness, direct reference-objective direction--wavenumber optimization, an anchor-assisted near-optimal-set audit and resolved dealiased finite-band transport. It does not claim specimen calibration, nonlinear spectral closure, material band width, propagation resistance, an analytic global optimum proof, or a strict HCP-to-Bai analytical degeneration.

## Public repository

https://github.com/yxlnbu/finite-time-hcp-perturbation
