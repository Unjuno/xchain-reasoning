# Stage 52–54: label-free exterior reliability and its limits

## Result, before interpretation

Stage 52's **primary hypothesis failed**. Per-request leave-one-observed-node-out
(LOO) predictive density improved sign accuracy when half of the exterior edge
signs were wrong, but damaged intact graphs beyond the fixed noninferiority
margin. A separate Stage 53 followup with **32 independent unlabeled fields**
passed its conditional criterion under new noise levels. This needs additional
same-regime data; it is not free single-request inference. Neither study proves
a speed advantage, general correctness guarantee, LLM benefit, or new inference
principle.

| Experiment and comparison | Sign-accuracy difference, percentage points (95% seed-t CI) |
|---|---:|
| 52: per-request LOO vs full messages, 50% flipped exterior signs | +1.4765 [0.9865, 1.9664] |
| 52: same policy, intact relations | -0.5194 [-0.5423, -0.4966] |
| 53: independent batch-32 LOO vs full messages, 50% flips | +0.9122 [0.7101, 1.1142] |
| 53: same policy, intact relations | -0.0240 [-0.0782, 0.0303] |
| 53: batch-32 vs **no exterior messages**, 50% flips | **-0.3273 [-0.5754, -0.0791]** |

Both primary tests require a positive lower CI bound in the 50%-flip condition
and a lower bound above -0.5 pp on intact relations. Stage 52 is **FAIL** and
remains so. Stage 53 is **conditional PASS**. The last row is an important
negative control: attenuation avoids some damage but does not recover the
correct exterior relations. In Stage 53 the true normalized MSE under 50% flips
changes from 0.667980 (full messages) to 0.675950 (batch-32): classification and
squared-error objectives must not be conflated.

[Key means](../results/stage52_54/means.csv) ·
[Primary and strong-control summaries](../results/stage52_54/summary.json)

## Scope and protocol

The frozen [Stage 52 protocol](../experiments/stage52/protocol.json) uses the
existing 49-node Gaussian generator: one unobserved query, 48 observed nodes,
chain/cross/tree/grid/degree-preserving rewired grid, regularization 0.05,
noise ratios 1 and 4, and 10 fresh seeds (62001–62010). Four conditions are
intact, 15% exterior sign flips, 50% exterior sign flips, and a wrong but
cycle-balanced vertex-gauge perturbation. Query-incident edges remain correct.
Each condition has 1,024 test fields. There are 400 conditions and 5,600 result
rows, not 5,600 independent experiments. Random inputs are paired across noise
and corruption conditions. The 48 observations within a field are dependent.

A gate in {0, 0.125, 0.25, 0.5, 0.75, 1} scales only exterior-to-exterior
messages; query-incident messages and precision diagonal stay fixed, matching
Stage 51. This also changes the effective Gaussian prior, not just a runtime
scheduler. Output inference uses 16 updates. The diagnostic uses exact Gaussian
conditional predictions and does not pretend they are a cheap 16-step probe.

For each gate, LOO predicts each **observed** node using the other observations.
It scores the held-out *noisy observed value*, not the latent query answer.
Gate selection minimizes mean negative log predictive density. Query truth,
true covariance, corruption rate and graph-family labels are not passed to the
selection API. Observation-noise variances, edge magnitudes, prior family and
correct query-incident relations **are supplied assumptions**. This is not
unknown-noise or arbitrary-model reliability estimation.

Stage 53 was specified after Stage 52 and uses new seeds 63001–63010, noise
ratios 0.25 and 8 (not used in Stage 52), the same five graph families and four
corruption conditions: another 400 conditions and 4,400 rows. It selects a gate
from an independent library with the same graph, noise and relation quality as
the test field. Sizes 1/4/16/32/128 are recorded, but **32 was fixed as primary**
before test generation; other sizes are exploratory. No selector was trained
on a graph family, so this is not leave-one-family-out learned generalization.

The primary accuracy estimate integrates the latent query conditional on each
test observation vector (Rao–Blackwellization), after policy selection. It is
still Monte Carlo over the observations, not an exact population risk. Sampled
query-label accuracies are reported separately and preserve the main direction.
Within each seed, family and noise cells are averaged before paired Student-t
intervals. The 95% multiplier is 2.262157, with 10 seed blocks. Main and followup
are separate hypotheses; later success does not revise earlier failure. These
are local frozen protocols, not external preregistration.

## Three distinctions

1. Observation prediction is a **proxy**, not query correctness. LOO loss need
   not rank gates in the same way as query sign accuracy. More calibration data
   reduces selection noise but cannot fix an unsuitable model family/objective.
2. Cycle consistency is not truth. The `coherent50` control remains balanced on
   every cycle, yet has wrong exterior relations. LOO detects some discrepancy
   using observed data; cycle balance alone has no such information.
3. Suppressing messages is not repairing relations. On the severe-corruption
   strata, simply disabling exterior messages still beats the calibrated gate
   in sign accuracy. That policy loses useful information on intact graphs.

## Cost and implementation

Each graph/noise/candidate set builds six models, each with a 49-by-49 prior
and 48-by-48 observed-covariance Cholesky factorization (12 total). Diagnostics
store six dense observed-precision matrices and coefficient vectors: 115,200
bytes of persistent float64 arrays in the reference representation. Per-field
LOO scoring is quadratic in 48 observations for each of six candidates; it is
not free. Candidate coefficient construction, calibration data acquisition,
32-field calibration scoring and final prediction are additional work. Exact
conditional inference is already available after this preprocessing; no claim
of beating direct inference is supported. Timings in generated `costs.csv` are
single execution logs, not comparative latency benchmarks.

Measured environment: Intel Xeon Platinum 8573C, Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0, float64, BLAS/OpenMP 1 thread, test batch 1,024. No quantization;
CPU clock not fixed or measured. No neural training or hidden external data.
Raw arrays and full result rows are generated locally, not stored in Git.

## Formulas and variable table

All data are standardized, dimensionless real quantities. Logs are natural.
The superscript T denotes transpose; expectation/probability are under the
specified Gaussian world, not unknown real-world data.

| Symbol | Meaning / definition | SI unit | Domain / assumption | Type |
|---|---|---|---|---|
| m, i, j | Number/index of observed nodes | 1 | m=48; 0<=i<m | integers |
| g | Exterior attenuation gate | 1 | Six values in [0,1] | scalar |
| O, o | Observation vector and its realization | 1 | m real components | random/vector |
| C_g | Candidate observation covariance | 1 | Symmetric positive definite | matrix |
| H_g | C_g inverse | 1 | Symmetric positive definite | matrix |
| mu_gi, s_gi^2 | Conditional held-out mean and variance | 1 | Condition on all observations except i | scalar/positive scalar |
| L_g | Average LOO negative log predictive density | 1 | Normalized, constant omitted | scalar |
| Y | Latent query answer | 1 | Continuous, P(Y=0)=0 | random scalar |
| mu, v | True conditional mean/variance of Y given O | 1 | Used **only by scoring** | scalar/positive scalar |
| f | Policy's query prediction | 1 | Function of observations and independent calibration | scalar |
| Phi | Standard normal CDF | 1 | Real argument, range [0,1] | function |
| Sigma, k | Joint latent covariance and query/observation cross-covariance | 1 | Positive definite joint law | matrix/vector |
| S | Sign matrix reversing only the unobserved query | 1 | Diagonal, entries +/-1 | matrix |
| delta | Any binary observation-only decision | 1 | May include independent randomization | function |

Gaussian conditioning, obtained by completing the square in the exponent of
the joint density, gives

$$\mu_{g i}=o_i-\frac{(H_g o)_i}{(H_g)_{ii}},\qquad s_{g i}^2=\frac{1}{(H_g)_{ii}}.$$

To see the exclusion explicitly, the terms involving the held-out value in
$o^T H_g o$ are $(H_g)_{ii}o_i^2+2o_i\sum_{j\ne i}(H_g)_{ij}o_j$.
Completing that square gives mean $-\sum_{j\ne i}(H_g)_{ij}o_j/(H_g)_{ii}$;
it has no dependence on the held-out value. Its variance is the reciprocal
diagonal. Substitution into the Gaussian log density yields, up to a shared
constant,

$$L_g(o)=\frac{1}{2m}\sum_i\left[\frac{((H_g o)_i)^2}{(H_g)_{ii}}-\log (H_g)_{ii}\right].$$

Selection uses only this score. After selection, conditional accuracy is
$\Phi(\operatorname{sign}(f)\mu/\sqrt v)$ and conditional MSE is
$(f-\mu)^2+v$. These evaluator quantities never enter the selector.
The variance terms, squared residuals and Gaussian standardization have
consistent dimensionless units; this is not information measured in bits.

## Stage 54: an exact observability limit

This deliberately **removes the trusted-query-edge assumption** of Stages 52–53.
It must not be presented as an outer-only counterexample to those conditions.
Construct two Gaussian worlds with covariances $\Sigma$ and $S\Sigma S$.
Only the unobserved query changes sign. Their observed covariance matrices are
identical, while query/observation cross-covariances are $k$ and $-k$.

**Proof.** Couple the worlds by keeping every observation and every independent
algorithmic random draw identical, and replacing $Y$ by $-Y$. Any decision
$\delta(O)$ is identical in both worlds. Since $Y$ is nonzero with probability
one, exactly one of these opposite labels agrees with that decision. Averaging
the two correctness indicators and then taking expectation gives accuracy
exactly 1/2. This remains true with arbitrarily many unlabeled observations:
their distribution is the same in both worlds. Knowing which world is true
requires an additional anchor/assumption, not more computation on those same
observations. Proof complete.

A new 100-case diagnostic (10 seeds, five graphs, two noise levels) confirms:
observed-covariance difference 0; diagnostic difference 0; chosen-gate mismatch
0; opposite-prediction error 0; paired-world accuracy exactly 0.5.
[Counterexample summary](../results/stage52_54/summary.json)

## Audit and reproducibility

- The local baseline Git tree exactly matched the public main tree before edits.
- Stage 52 and 53 were each rerun completely: 10,000 rows, numerical difference
  0, categorical matches. Stage 54's 100 rows also match exactly.
- Frozen source/protocol hashes match the executed files.
- 23 tests pass, including explicit-deletion versus fast LOO, own-observation
  exclusion, query-edge preservation, cycle-balanced corruption, and the
  observational-equivalence construction.
- A smoke-run JSON serialization issue was corrected before main evaluation;
  thresholds were not tuned from the test results.
- This is same-environment computational reproduction, not an independent
  laboratory replication or rigorous directed-rounding numerical proof.

[Audit](../provenance/stage52_54/verification.json) ·
[Environment](../provenance/stage52_54/environment.json)

```bash
python -m unittest discover -s tests -v
python experiments/stage52/run.py --out runs/stage52
python experiments/stage52/followup.py --out runs/stage53
python experiments/stage52/blind_spot.py --out runs/stage54
```

Each command also accepts `--quick` for smoke tests; those are not full-study
confidence intervals. Existing output directories are rejected. No network,
credentials, latent-query labels for selection, or unsafe deserialization is
required.

## Relation to prior work and next constraint

LOO predictive scoring and marginal likelihood are established model-selection
methods, not XChain-specific innovations. Their different targets and the
choice of model family matter. See [Petit et al., arXiv:2107.06006](https://arxiv.org/abs/2107.06006)
and the dependent-observation caveat in [Liu and Rue, arXiv:2210.04482](https://arxiv.org/abs/2210.04482).
The present LOO calculation is exact conditional Gaussian interpolation; it is
not a theorem that interpolating observed nodes optimizes a different query.

The next useful test is whether a small, trusted relation anchor plus
observation-only repair can outperform simply disabling exterior messages,
while accounting for independent calibration data and diagnostic work. No such
repair experiment was executed in this batch.
