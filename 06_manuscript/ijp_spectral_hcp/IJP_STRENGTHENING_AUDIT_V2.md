# IJP finite-time crystal-perturbation strengthening audit V2

## Decision

The requested strengthening actions have been implemented at the level needed for the manuscript's **like-for-like full-state finite-time versus frozen discrimination claim**. The evidence does not close, and the manuscript does not claim, specimen calibration, an independent global optimum, nonlinear band width or propagation resistance, or a strict analytical HCP-to-Bai degeneration.

## Acceptance matrix

| No. | Requested strengthening | Implemented evidence | Acceptance result | Claim boundary |
|---:|---|---|---|---|
| 1 | Add a stationary or commuting normal positive control | A stationary normal generator and a time-varying, pairwise-commuting normal family with one fixed leading common mode give maximum frozen/propagator log-gain discrepancies of (9.33\times10^{-15}) and (1.78\times10^{-15}), respectively. A commuting-normal family with a switching pointwise leader gives a discrepancy of (1.1459), furnishing the necessary negative boundary case. | **Pass** | Commutation and normality are not alone sufficient; equality additionally requires one common mode to remain pointwise leading almost everywhere. |
| 2 | Add at least three orientations or loading paths | Three independently selected Bunge orientations, ((0,0,0)^\circ), ((30,90,30)^\circ), and ((90,60,30)^\circ), are evaluated under the registered simple-shear loading path. Their selection-horizon frozen/finite-time log-gain discrepancies are 6.234, 5.810, and 5.532; terminal discrepancies are 6.737, 4.461, and 3.307. | **Pass for orientation transfer** | This demonstrates that non-equivalence is not a single-orientation artifact. It is not yet a full multi-loading-path campaign. |
| 3 | Strengthen sphere--wavenumber search and convergence audit | Nested scrambled Sobol levels of 64, 256, and 1024 points cover one projective hemisphere and (k\in[300,3\times10^5],\mathrm{m}^{-1}), followed by eight independent local starts per horizon and a 129-state/four-substep reassessment. Sphere fill angle improves from (18.52^\circ) to (4.45^\circ); the maximum log-(k) gap improves from 0.1994 to 0.01160. Winner-basin recurrence is 4/8 for onset and 5/8 for terminal. Relative high-resolution deficits are 0 and 0.5155%. | **Pass as an anchor-assisted basin-retention audit** | The blind Sobol cloud misses the narrow terminal basin unless exact/prior-branch anchors are retained. This is convergence and basin-retention evidence for the registered protocol, not independent global discovery. |
| 4 | Complete the material parameter--source table | The generated provenance card contains 74 rows and covers every field in the registered `MaterialParameters` dataclass. Each entry records value, unit, code location, source key, evidence class, activation status, and interpretation. | **Pass for computational provenance** | Literature constraints, repository verification values, model controls, numerical controls, and inactive twinning parameters are deliberately distinguished. The card is not represented as a single-batch Ti calibration. |
| 5 | Release generators, singular vectors, optimizer traces, and environment | The release contains all three reported baseline generator histories at 129 checkpoints ((129\times69\times69), complex128), three orientation histories, leading input/output singular vectors, all observed singular values, propagators and responses, all retained optimizer evaluations, deterministic seed, package lock, BLAS/LAPACK configuration, repository-state probe, and SHA-256 manifest. The environment record explicitly reports that the supplied workspace has no Git metadata instead of inventing a revision. | **Pass as a publication-ready local archive** | The ZIP is ready for Zenodo/institutional-repository deposition but is not itself a public URL or DOI until uploaded. |
| 6 | Keep strict HCP-to-Bai analytical reduction open and non-essential | The abstract, introduction, discussion, conclusion, reproducibility appendix, and release README state that Bai is an independently verified scalar classical anchor; strict analytical HCP-to-Bai degeneration is neither closed nor required for the finite-time discrimination claim. | **Pass** | No equivalence claim is made beyond the implemented comparison contract. |

## Central scientific claim now supported

On the same complete 69-state space, fixed scaling, identity selectors, base history, direction and wavenumber, replacing the non-autonomous propagator by the time integral of a frozen instantaneous growth rate underpredicts the three reported terminal log gains by 4.784, 7.217 and 6.402. The historical negative onset-$x$ difference was a projected/full-state mismatch and is not retained as evidence. Positive controls recover equality when the sufficient normal/commuting/fixed-leader conditions hold; commutators, modal condition numbers, eigenvector rotations and ordering controls explain the HCP discrepancy. The localization-relevant winners persist across 11 norm definitions, seven selector pairs and three independently selected crystal orientations.

## What remains outside this paper

1. A strict asymptotic or algebraic reduction of the complete HCP crystal-plasticity perturbation system to Bai's scalar criterion.
2. An analytic proof of the global direction--wavenumber maximizer.
3. Batch-calibrated prediction of experimental onset strain, finite nonlinear band width, dissipated energy, propagation resistance, and structural failure.
4. Public DOI deposition of the local release archive.

## Registered artifacts

- Manuscript: `06_manuscript/ijp_spectral_hcp/main_finite_time.pdf`
- Operator/norm/mechanism ledger: `05_results/ijp_operator_consistency_v2.json`
- Search/orientation ledger: `05_results/ijp_strengthening_evidence_v1.json`
- Full numerical arrays: `05_results/ijp_strengthening_evidence_v1_arrays.npz`
- Singular-vector nonlinear validation: `05_results/ijp_singular_vector_nonlinear_validation_v2.json`
- Parameter provenance: `05_results/ijp_material_parameter_provenance_v1.csv`
- Reproducibility receipt: `05_results/ijp_reproducibility_release_v1.json`
- Reproducibility archive: `04_reproducibility/ijp_finite_time_release_v1.zip`
