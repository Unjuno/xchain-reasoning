# Stage 52–54

Observation-only reliability diagnostics for exterior relations.

- Stage 52 primary: **FAIL**. Per-request LOO reliability improved 50%-corrupted relations but damaged intact relations beyond the frozen noninferiority margin.
- Stage 53: **conditional PASS** with 32 additional unlabeled same-regime fields on independent seeds/noise. Calibration cost and extra data are explicit.
- Stage 54: exact observation-equivalent counterexample after removing the trusted query-edge assumption. Observation-only diagnostics cannot distinguish paired worlds with opposite query truth.

Runnable source is in `experiments/stage52/`; detailed report: `docs/STAGE52_54.md`.
