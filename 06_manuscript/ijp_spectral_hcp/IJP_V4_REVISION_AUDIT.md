# IJP V4 comprehensive revision audit

## Editorial decision

The manuscript is now consistently framed as a **theory--method paper**. Its strongest defensible contribution is the exact descriptor reduction and selector-dependent finite-time direction--wavenumber selection on an evolving HCP path. It is not framed as a calibrated titanium prediction, a mature shear-band-width theory, or a completed cross-scale propagation-resistance model.

## Reviewer-action ledger

| Requested action | Disposition | Main evidence |
|---|---|---|
| Same state space, norm and input--output contract for frozen and propagated quantities | Closed | Full 69-state identity-selector comparison; invariant-projection positive control |
| Seven selectors and adverse norms re-optimized over direction and wavenumber | Closed | 70 horizon--norm--selector optimizations; boundary wave branches classified |
| Stationary/commuting-normal positive controls | Closed | Equality recovered under normal, commuting, fixed-leader conditions |
| Difference explained by commutators, eigenvector rotation and modal conditioning | Closed | Same-branch operator diagnostics and order reversal |
| Same-selector piecewise-frozen baseline | Closed | Interval-preserving refinement with identical selectors and metric |
| Three orientations and added loading paths | Closed within theory-method scope | Three Bunge orientations; three added rate/temperature paths |
| Local sensitivity of dislocation and gradient parameters | Closed as branch-local sensitivity | Symmetric 20% perturbations of four parameters |
| Titanium literature comparison | Closed as a negative external check | Stress-before-temperature ordering only; timing/magnitude fail |
| Formal singular-value degeneracy threshold and ratios | Closed | 5% relative-gap threshold; onset 1.862, terminal 3.73e5 |
| Full 11-by-7 matrix in supplementary material | Closed | Supplementary Fig. S1 |
| Orientation ordering | Closed | Minimum--median--maximum in text and figure source |
| Parameter provenance | Closed for computational provenance | 74-row table; exact NIST, IAEA, paper, commit, config hashes and dates |
| Abstract and conclusions condensed | Closed | 184-word abstract; five conclusions |
| Spatial, temporal and integration convergence of nonlinear replay | Closed for resolved quantities | Dealiased 16/32/64 cells; 257/513 states; 1/2/4 substeps; Lawson--Euler control |
| First failed equation, variable and wavenumber | Closed | Raw failure at 1.3515625 microseconds; Nyquist; velocity/signed-slip coordinates |
| Complete nonlinear harmonic closure | Open by evidence, not omitted | Primary terminal discarded RHS energy 0.8466 exceeds the 0.01 gate |
| Physical norm | Partially closed | Recoverable-energy quotient metric added at fixed direction/wavenumber; no experimental covariance or metric-specific re-optimization |
| Branch switching | Numerically bounded, theorem open | Four transition intervals; terminal convention spread below 0.0617%; no semismooth theorem |
| Independent global optimum | Open and relabelled | Anchor-assisted basin-retention audit only |
| Strict HCP-to-Bai analytical degeneration | Open and non-essential | Bai retained as an independently verified classical anchor |

## Remaining pre-submission blockers

1. Author names, affiliations, funding and CRediT roles remain placeholders and must be supplied by the authors.
2. A reuse license and persistent repository DOI remain unset.
3. The revised LaTeX source has not been compiled in this revision pass; the committed PDF is therefore an earlier build.
4. Complete nonlinear spectral-flux closure, experimental observation covariance, energy-metric direction--wavenumber re-optimization, semismooth switching theory, specimen calibration, material width and propagation resistance remain outside the supported claims.

## Verification status

- Public test suite: 12 tests passed.
- V4 receipt status: `NUMERICAL_CONVERGENCE_PASS_NONLINEAR_RHS_CLOSURE_OPEN`.
- Start/branch/metric receipt status: `START_AND_BRANCH_AUDIT_PASS__EXPERIMENTAL_METRIC_OPEN`.
- Active-manuscript citation and cross-reference static audit: passed.
