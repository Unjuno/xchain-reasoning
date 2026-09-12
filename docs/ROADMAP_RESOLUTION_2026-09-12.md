# XChain roadmap v1 resolution — 2026-09-12

> **Scope.** Synthetic, interpretable graph models only. This is not evidence of LLM gains.
> PASS, FAIL/UNCERTAIN, identifiability limits, and cost failures are preserved together.

## Status

The public roadmap items defined after Stage 60 have now all received a falsifiable experimental or theoretical disposition. This document distinguishes PASS, FAIL/UNCERTAIN, and identifiability limits. It does **not** claim LLM generalization.

## Roadmap verdicts

| Item | Evidence | Verdict |
|---|---|---|
| Unknown support / magnitudes | Stages 62–72; sparse precision reconstruction, active query-relation acquisition, Schur-complement residual | Conditional PASS; complete unknown query support has an identifiability limit |
| Single-request / low calibration | Stages 93–105, 119 | One independent unlabeled field can suffice in dense regimes; same-request-only reliability selection failed equal-work random; history reuse works at regime level |
| Non-gauge-balanced relations | Stages 73–99 | PASS for exterior-value mechanism in Ising trees/loops; structural frustration-only gate FAILs on intrinsically frustrated correct models |
| Learned relation model | Stage 122 | Error components separated: model, relation, shallow/deep inference |
| Cost accounting | Stages 115, 124, 125 | Accuracy-per-nominal-work can improve while wall-clock efficiency fails; repair setup is amortizable only across repeated requests |

## Stage 119 — instance-level revision trigger vs uncertainty

New 10-seed test, seven graph families, same selected deep fraction.

- sign-flip trigger mean accuracy: 0.790536
- uncertainty trigger: 0.783508
- condition-matched random: 0.777843
- mean nominal depth: 11.431
- sign-flip - uncertainty: **+0.7029 pp [0.6381, 0.7676]**
- sign-flip - random: **+1.2693 pp [1.1892, 1.3494]**

Exception: on a simple cycle, sign-flip was **-0.1432 pp [-0.2189, -0.0675]** below uncertainty. The trigger is not topology-universal.

## Stage 120 — naive history × instance hierarchy

A regime-history gate was stacked on top of the sign-flip trigger under relation drift.

- policy - global random: **+0.2507 pp [0.1559, 0.3454]**
- policy - segment-matched random: **-0.0521 pp [-0.1378, 0.0336]**
- policy - matched uncertainty: **-0.0081 pp [-0.0180, 0.0018]**

Verdict: **FAIL/UNCERTAIN for additive hierarchical value**. The history gate mostly encoded which relation regime deserved compute and suppressed the instance signal too aggressively.

## Stage 121 — stale-history reset under drift

Independent 10-seed test with unseen beta/noise/drift periods.

- reset policy accuracy: 0.796196
- no-reset: 0.795024
- global equal-allocation random: 0.793156
- reset - no-reset: **+0.1172 pp [0.0498, 0.1846]**
- reset - global random: **+0.3040 pp [0.2039, 0.4041]**
- mean depth reset: 1.3721; no-reset: 1.4171

The reset policy improved overall accuracy while using slightly less nominal depth. The first-request-after-change effect alone remained uncertain.

## Stage 122 — learned relation model error decomposition

An interpretable learned Ising relation model was fit from complete latent training fields using edge correlations, then inference was evaluated with a hidden query and noisy observations.

At K=512 training fields:

- true-model exact Bayes accuracy: 0.806763
- learned-model exact accuracy: 0.804598
- learned model depth-64 BP: 0.802799
- learned model depth-2 BP: 0.793034
- 20% exterior relation corruption, exact inference: 0.769784

Paired 10-seed mean gaps:

- model error: **0.2165 pp [0.1186, 0.3143]**
- residual depth-64 inference error: **0.1799 pp [0.0895, 0.2702]**
- depth-2 -> depth-64 recoverable gap: **0.9766 pp [0.8374, 1.1158]**
- relation-corruption error: **3.4814 pp [2.3185, 4.6443]**

Thus relation error dominated the remaining model/inference errors in this test family.

## Stage 124 — nominal work vs actual CPU time

Seven topology conditions per seed; 10 seeds; B=2048; same selected deep count for sign-flip, uncertainty, and random.

Means:

- sign-flip accuracy: 0.774282
- uncertainty: 0.763735
- random: 0.761663
- fixed depth-12: 0.747635
- sign-flip mean nominal depth: 12.023

Paired accuracy differences:

- sign-flip - random: **+1.2619 pp [0.5576, 1.9661]**
- sign-flip - uncertainty: **+1.0547 pp [0.4219, 1.6874]**
- sign-flip - fixed depth-12: **+2.6646 pp [1.7180, 3.6112]**

But the recorded NumPy CPU implementation was not wall-clock efficient:

- sign-flip total median-per-condition sum: 229.49 ms/seed block
- fixed depth-12: 83.45 ms/seed block
- mean sign-flip/fixed runtime ratio: **2.77x**

The cause is implementation-level: subset continuation loses dense batch vectorization. Therefore `accuracy / nominal message updates` improved, while current wall-clock efficiency did not.

## Stage 125 — repair setup cost

Stage-57-style K=512 spectral relation repair on 49 nodes, single CPU thread.

Per-problem medians:

- covariance/eigendecomposition/repair: **0.332 ms**
- repaired coefficient construction: **0.263 ms**
- one coefficient application: **0.911 us**
- exact-Bayes coefficient application: **0.942 us**
- K=512 calibration array: **196,608 bytes (192 KiB)**
- covariance: 18,432 bytes
- adjacency: 19,208 bytes
- output coefficient: 384 bytes

Setup/application ratio is about **653 single-request coefficient applications**, before charging acquisition of the 512 unlabeled calibration fields. Therefore the repair branch is an accuracy method whose setup can only be amortized over repeated requests in a stable relation regime; it is not currently an online efficiency result.

## Variables for the final adaptive-compute claim

| Symbol | Meaning | SI unit | Definition / domain | Type |
|---|---|---|---|---|
| `b1,b2` | query beliefs after depths 1 and 2 | 1 | real log-fields | scalars |
| `I_flip` | early decision-reversal indicator | 1 | `1[sign(b1) != sign(b2)]` | binary |
| `f` | fraction selected for deep compute | 1 | [0,1] | scalar |
| `D` | nominal mean inference depth | 1 | `2 + 62 f` in Stage 119 | scalar |
| `A` | true query sign accuracy | 1 | [0,1] | probability |
| `T_cpu` | measured online inference time | s | wall-clock process time, single-thread CPU | scalar |
| `K` | number of independent calibration fields | 1 | nonnegative integer | integer |

Dimensional check: `D`, `f`, and accuracy are dimensionless. They cannot be directly equated with seconds; Stage 124 explicitly demonstrates why nominal depth and wall-clock time must be reported separately.

## Final falsifiable statement

Within the tested synthetic graph families:

1. Exterior compute can increase true output accuracy when reliable target-relevant information remains outside current reach.
2. Early internal decision reversal is a stronger instance-level allocation signal than uncertainty on average, but has topology-specific failures.
3. History primarily identifies relation regimes; naively stacking history and instance triggers does not automatically add value.
4. Stale history should be invalidated under relation drift; the tested reset improved overall accuracy and did not increase nominal depth.
5. Learned relation error can dominate both model-fit and inference-depth errors.
6. Better nominal compute allocation does **not** imply better wall-clock runtime; implementation and calibration setup costs can erase the advantage.

## ERROR CHECK

- Stage 118 incomplete legacy run was not filled by guessing; Stage 119 is a fresh 10-seed replacement with saved code.
- Stage 120 negative result is retained.
- Confidence intervals use the 10 independent seed blocks, not rows/requests as independent replicates.
- CPU clock was not pinned; Stage 124/125 runtime conclusions are specific to the recorded single-thread NumPy/SciPy implementation.
- No LLM, semantic-compression, or universal graph-inference claim is made.
