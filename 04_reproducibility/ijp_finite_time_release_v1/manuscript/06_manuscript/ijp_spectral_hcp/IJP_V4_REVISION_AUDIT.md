# IJP V4 comprehensive revision audit

## Editorial decision

The manuscript is now consistently framed as a **theory--method paper**. Its strongest defensible contribution is the exact descriptor reduction and selector-dependent finite-time direction--wavenumber selection on an evolving HCP path. It is not framed as a calibrated titanium prediction, a mature shear-band-width theory, or a completed cross-scale propagation-resistance model.

## Reviewer-action ledger

| Requested action | Disposition | Main evidence |
|---|---|---|
| Same state space, norm and input--output contract for frozen and propagated quantities | Closed | Full 69-state identity-selector comparison; invariant-projection positive control |
| Seven selectors and adverse norms re-optimized over direction and wavenumber | Closed | All 70 horizon--norm--selector contracts directly optimized on the 129-state/four-substep objective; six-neighbour stationarity, boundary extensions and retained trajectories supplied |
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
| Abstract and conclusions condensed | Closed | 193-word abstract (18.6% shorter than the reviewed 237-word version); five conclusions |
| Spatial, temporal and integration convergence of nonlinear replay | Closed for resolved quantities | Dealiased 16/32/64 cells; 257/513/1025-state linear density ladder; independent time-density ladder; Lawson--Euler control |
| First failed equation, variable and wavenumber | Closed | Raw failure at 1.3203125 microseconds in interval 84, midpoint substep 0; Nyquist mode 8 at $k=4.3640\times10^5$ m$^{-1}$; velocity coordinate dominates and signed-slip coordinate 55 first violates conjugacy |
| Complete nonlinear harmonic closure | Open by evidence, not omitted | Primary terminal discarded RHS energy 0.8458 exceeds the 0.01 gate |
| Physical norm | Negative audit made explicit | The rank-28 energy seminorm fails nullspace invariance (terminal relative leakage 0.9996); only a positive-energy-subspace restricted gain is reported, with no experimental covariance or metric-specific re-optimization |
| Branch switching | Numerically bounded, theorem open | Four transition intervals; terminal convention spread below 0.0622%; no semismooth theorem |
| Independent global optimum | Open and relabelled | Direct optimization, stationarity and sampled near-optimal-set challenges close the reported finite search; no continuum global proof is claimed |
| Strict HCP-to-Bai analytical degeneration | Open and non-essential | Bai retained as an independently verified classical anchor |

## Remaining pre-submission blockers

1. Author names, affiliations, funding and CRediT roles remain placeholders and must be supplied by the authors.
2. A reuse license and persistent repository DOI remain unset.
3. Complete nonlinear spectral-flux closure, positive density-storage curvature, experimental observation covariance, energy-metric direction--wavenumber re-optimization, specimen calibration, material width and propagation resistance remain outside the supported claims.

## Verification status

- Submission-scope public test suite: 21 tests passed in the synchronized standalone repository.  The larger private ASBs workspace contains unrelated UEL/capability and one-shot absence fixtures and is not represented as globally green.
- V4 receipt status: `NUMERICAL_CONVERGENCE_PASS_NONLINEAR_RHS_CLOSURE_OPEN`.
- Start/branch/metric receipt status: `START_AND_BRANCH_AUDIT_PASS__POSITIVE_ENERGY_SUBSPACE_ONLY__EXPERIMENTAL_METRIC_OPEN`.
- Main manuscript and supplementary PDF compilation: passed; no undefined citations/references or overfull boxes.
- Rendered-page audit: passed for the title/abstract, key V4 figures, nonlinear convergence table, complete parameter provenance and all pages of the 70-row supplementary table.
