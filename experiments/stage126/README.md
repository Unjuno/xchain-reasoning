# Stage 126: accuracy versus actual execution time

This is a prospective, implementation-controlled study, not a result announcement.
The frozen protocol and source precede the first evaluation run.

## Question

Can the early decision-reversal policy retain its accuracy advantage after both
adaptive and fixed-depth BP receive a compiled implementation, when compared at
measured rather than nominal cost?

The mathematical update, clipping, graph distribution and sign decision remain
unchanged. A single-thread C++ kernel keeps each field's two-step message state
and continues only selected fields. Uncertainty and random controls have the
same selected count. Every fixed-depth control receives the same kernel; the
original NumPy controls remain in the comparator set as well.

Run: `python experiments/stage126/run.py --out runs/stage126`.
Smoke test: add `--quick` (not a significance test).
Requires g++ and the existing pinned Python dependencies. The compiler is invoked
with no fast-math and floating-point contraction disabled. Compilation is a
separately recorded setup expense, not an inference speedup.

## Decision rule and units

| Symbol | Meaning | SI unit | Definition / domain | Type |
|---|---|---|---|---|
| b1, b2 | query log-fields after one and two updates | 1 | finite reals; zero predicts positive | scalars |
| f | fields selected to continue | 1 | fraction whose predicted signs differ, [0,1] | scalar |
| d | mean nominal depth | 1 | 2 + 62 f, [2,64] | scalar |
| t | measured batch time | s | packing, allocation, selection, update and output | scalar |
| A | expected true sign accuracy | 1 | exact conditional risk averaged over observed fields, [0,1] | probability |

Depth and accuracy are dimensionless; neither is equated with seconds.
The primary comparator is the hindsight upper concave accuracy/time frontier of
compiled and NumPy fixed depths 2/4/8/12/16/24/32/64, evaluated separately in each
condition. It allows fixed-policy mixtures and is a stronger benchmark than a
single fixed depth. This oracle envelope is a diagnostic, not a deployed policy.

Truth and exact-posterior tables enter scoring only, never policy selection.
The query observation is masked. All test seeds and endpoints are fixed in
[protocol.json](protocol.json). Ten seeds are the statistical units; repetitions
and individual fields do not inflate the sample size. Failed comparisons stay.

## Scope

Local execution was unavailable when this study was prepared, so the bounded
experiment runs in an isolated GitHub Actions branch. Runner CPU, compiler,
Python versions and compilation time are recorded. These timings must not be
compared directly with earlier container timings. No LLM, compression, energy,
new inference-principle or universal wall-clock claim is made.

The artifact contains per-condition rows, individual timing repeats, inputs,
predictions, protocol, environment, numerical audit and SHA-256 manifest. A pass
of the workflow alone is not a scientific PASS: read the frozen primary test.
