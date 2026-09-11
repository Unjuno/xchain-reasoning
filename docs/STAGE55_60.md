# XChain Stage 55–60: unlabeled relation repair, trusted anchors, and repaired exterior value

## 0. Scope and conclusion

These experiments continue the Stage 52–54 question: can wrong exterior relations be **repaired**, rather than merely suppressed, without reading query truth?

The answer is conditional.

- **Stage 55 — FAIL.** Replacing every observed-observed edge sign by the sign of a 32-field sample covariance improves 50%-corrupted relations but damages intact graphs too much.
- **Stage 56 — PASS under its frozen criteria.** A conservative repair rule, tuned only on Stage 55 and then tested on new seeds/noise, repairs only high-confidence contradictions. On 50% corruption it beats both exterior-off and batch-gating controls while reducing intact accuracy by only about 0.14 percentage points.
- **Stage 57 — PASS.** If all edge signs are unreliable but graph support/magnitudes are known, 512 unlabeled fields plus the exact sign of **one query-incident edge** can reconstruct enough relation structure to beat an oracle that knows all true query-incident relations but disables exterior-exterior messages.
- **Stage 58 — descriptive sample-efficiency curve.** The one-anchor method needs substantial unlabeled calibration in this model. It is below the oracle-local baseline at K<=32, statistically unresolved at K=64/128, and clearly above it at K=256/512/1024.
- **Stage 59 — FAIL under its primary criterion.** Redundant noisy query anchors are much more robust than one noisy anchor, but at 10% anchor error they do not significantly beat an impossible oracle-local control that knows the true query-edge signs.
- **Stage 60 — PASS.** Holding the *same noisy anchor-derived query orientation fixed*, enabling the repaired exterior improves accuracy by about 2.69 percentage points at 10% anchor error. This isolates positive incremental value in the repaired exterior itself.

These are synthetic balanced-sign Gaussian models. Calibration fields, trusted anchors, graph support and magnitudes are additional information. No LLM, semantic-compression, or runtime-superiority claim follows.

## 1. Experimental model

The 49-node Stage 49 family is reused: one unobserved query node and 48 observed exterior nodes. The true signed adjacency has the gauge-balanced form

\[
A = G W G,
\]

where `G=diag(g_i)`, `g_i in {-1,+1}`, and `W` is symmetric, nonnegative and has zero diagonal. The Gaussian prior precision is

\[
Q = \operatorname{diag}\!\left(\tau + \sum_j W_{ij}\right)-A.
\]

The observations are all non-query latent coordinates plus independent diagonal Gaussian noise. Inference uses the same 16 synchronous local updates as Stage 49–54 unless noted.

### Variables

| Symbol | Meaning | SI unit | Definition / assumptions | Type |
|---|---|---:|---|---|
| `A` | signed graph relation matrix | 1 | `A=GWG` in the true model | real matrix |
| `G` | node-gauge matrix | 1 | diagonal entries `g_i` are ±1 | diagonal matrix |
| `W` | unsigned relation magnitudes | 1 | symmetric, nonnegative | real matrix |
| `Q` | prior precision | 1 | expression above, `tau>0` | SPD matrix |
| `Sigma` | prior covariance | 1 | `Q^{-1}` | SPD matrix |
| `q` | unobserved query node | 1 | one node index | integer |
| `Y` | unlabeled calibration observations | 1 | K independent observed fields | matrix |
| `K` | calibration-field count | 1 | 4–1024 depending on stage | integer |
| `g_hat` | estimated gauge | 1 | signs inferred from calibration covariance | ±1 vector |
| `s_qr` | trusted query-anchor relation | 1 | true sign on one edge `(q,r)` | ±1 scalar |

All model values are normalized and dimensionless. Accuracy is a dimensionless probability; NMSE is MSE divided by query prior variance and is dimensionless.

## 2. Why unlabeled covariance contains relation signs

Let

\[
L = GQG = \operatorname{diag}\!\left(\tau + \sum_j W_{ij}\right)-W.
\]

`L` is symmetric, strictly diagonally dominant by `tau>0`, with non-positive off-diagonal entries. It is therefore a nonsingular M-matrix. For a connected graph, `L^{-1}` is entrywise strictly positive. Since

\[
\Sigma = Q^{-1} = G L^{-1} G,
\]

we obtain for every distinct pair

\[
\operatorname{sign}(\Sigma_{ij}) = g_i g_j.
\]

Independent observation noise changes only the covariance diagonal, so the off-diagonal observed covariance has the same sign. Thus, with enough unlabeled fields, observed-observed relation signs are statistically identifiable in this model.

**Unit check:** covariance entries, relation magnitudes and latent values are all dimensionless under this normalized model.

**Limitation:** finite K can reverse a sample-covariance sign. Stage 55 is the direct empirical demonstration that blindly trusting those finite-sample signs damages intact graphs.

## 3. Stage 55 — unconditional pairwise repair fails intact noninferiority

The candidate graph keeps correct support/magnitudes and correct query-incident signs, but exterior signs may be corrupted. For K=32, every observed-observed edge sign is replaced by its sample-covariance sign.

Primary paired-seed accuracy differences:

| Comparison | Mean | 95% CI |
|---|---:|---:|
| flip50 repair − exterior off | +1.657 pp | +1.255 to +2.059 pp |
| flip50 repair − batch gate | +1.987 pp | +1.516 to +2.458 pp |
| intact repair − full correct relations | **−2.195 pp** | **−2.557 to −1.832 pp** |

The intact noninferiority requirement was a lower bound above −0.5 pp. **FAIL.**

## 4. Stage 56 — conservative repair on independent seeds

Stage 55 was used only as tuning data. Stage 56 uses new seeds and noise ratios `[0.25,1,4,8]`. Calibration size is fixed at K=128. A candidate edge is flipped only if:

1. candidate sign and sample-correlation sign disagree, and
2. absolute sample correlation is at least 0.10.

Query-incident edges remain trusted and unchanged.

| Comparison | Mean | 95% CI |
|---|---:|---:|
| flip50 repair − exterior off | +1.620 pp | +1.411 to +1.829 pp |
| flip50 repair − batch-128 gate | +1.840 pp | +1.610 to +2.070 pp |
| intact repair − full correct relations | −0.139 pp | −0.224 to −0.054 pp |

All frozen criteria pass. Mean exterior edge-sign accuracy is 89.55% at flip50 and 99.53% on intact graphs.

This is **not** a proof that threshold 0.10 is universal. It was selected using the preceding Stage 55 dataset.

## 5. Why one trusted query edge matters

The observed covariance does **not** determine the query gauge. Flip only `g_q` while leaving all observed gauges unchanged. Then every observed-observed covariance remains unchanged, while every query-observed cross-covariance changes sign. This is the Stage 54 ambiguity.

Suppose a single query-incident edge `(q,r)` is trusted. Its sign satisfies

\[
s_{qr}=g_q g_r.
\]

If the observed gauge is known up to global sign, then

\[
\widehat g_q = s_{qr}\widehat g_r
\]

fixes the query orientation. A global simultaneous sign flip of all estimated gauges leaves every repaired edge product unchanged, so the arbitrary eigenvector orientation is harmless.

At population covariance, `G_obs C G_obs` is entrywise positive. Perron–Frobenius therefore gives a strictly positive leading eigenvector, so the sign pattern of the leading eigenvector of `C` recovers the observed gauges up to one global sign. Stage 57 uses the finite-sample analogue.

## 6. Stage 57 — one trusted relation plus unlabeled spectral reconstruction

All edge signs, including query edges, are first corrupted at 50%. Support and magnitudes remain correct. The policy ignores candidate signs, estimates observed gauges from the leading eigenvector of the centered K=512 calibration covariance, and uses exactly one trusted query-edge sign to orient the query gauge.

Mean accuracies:

| Policy | Accuracy | NMSE |
|---|---:|---:|
| corrupted full messages | 50.856% | 1.0972 |
| oracle local-only: every true query edge known, exterior disabled | 70.352% | 0.7237 |
| **spectral repair + one trusted query edge** | **72.505%** | **0.5802** |
| oracle true relations, 16 updates | 74.002% | 0.5488 |
| exact Bayes, true model | 74.123% | 0.5262 |

Primary paired-seed accuracy differences:

- versus corrupted full: **+21.649 pp [19.455, 23.843]**;
- versus oracle local-only: **+2.153 pp [1.682, 2.624]**.

The repaired relation signs are 95.37% correct overall and 96.75% correct on query-incident edges on average.

## 7. Stage 58 — how much unlabeled calibration is needed?

New seeds were used. One exact query edge is retained while K varies.

| K | edge-sign accuracy | output accuracy | accuracy gain vs oracle local-only | 95% CI of gain |
|---:|---:|---:|---:|---:|
| 8 | 69.54% | 63.02% | −7.354 pp | −8.857 to −5.850 |
| 16 | 74.53% | 64.65% | −5.720 pp | −7.586 to −3.855 |
| 32 | 80.00% | 67.56% | −2.810 pp | −4.257 to −1.364 |
| 64 | 85.82% | 69.10% | −1.273 pp | −2.740 to +0.195 |
| 128 | 89.45% | 70.56% | +0.183 pp | −0.404 to +0.770 |
| 256 | 93.23% | 71.71% | **+1.341 pp** | **+0.450 to +2.232** |
| 512 | 95.88% | 72.91% | **+2.539 pp** | **+2.191 to +2.887** |
| 1024 | 97.34% | 72.64% | **+2.270 pp** | **+1.570 to +2.971** |

The 1024-versus-512 accuracy difference is not statistically resolved; its paired-seed 95% interval crosses zero. More correctly recovered edges do not guarantee monotone output accuracy because the **location** of remaining mistakes matters.

## 8. Stage 59 — imperfect anchors and redundancy

Each query-edge anchor is independently wrong with probability 0, 5%, 10%, or 20%. `one_anchor` uses one relation. `all_query_anchors` uses an edge-magnitude-weighted vote of every query-incident anchor.

At 10% anchor error, redundant anchors beat one anchor by:

**+2.115 pp [1.921, 2.309].**

But they are still −0.363 pp from the impossible `oracle_local_only` control whose query edges are all exactly correct; its interval is −0.963 to +0.236 pp. Therefore the frozen Stage 59 primary criterion is **FAIL**.

At 5% anchor error, redundant anchors are +0.917 pp above that oracle-local baseline [0.195,1.638], but this was not the prespecified primary anchor-error level.

## 9. Stage 60 — incremental value of the repaired exterior under the same anchor uncertainty

Stage 59's oracle-local comparison mixes two effects: anchor error and exterior value. Stage 60 holds the inferred query orientation **identical** between two policies:

- `reconstructed_local`: reconstructed query relations, exterior-exterior messages disabled;
- `repaired_full`: the same query orientation and reconstructed graph, exterior enabled.

At 10% anchor error:

\[
\boxed{\Delta\mathrm{accuracy}=+2.685\text{ pp}}
\]

with paired-seed 95% interval **+2.452 to +2.919 pp**. The frozen criterion passes.

The incremental exterior gain remains positive across the tested anchor error rates:

| anchor error | repaired full − reconstructed local |
|---:|---:|
| 0% | +2.998 pp [2.686,3.309] |
| 5% | +2.841 pp [2.543,3.139] |
| 10% | +2.685 pp [2.452,2.919] |
| 20% | +2.182 pp [2.000,2.363] |

This is the cleanest result in this batch for the original XChain question: **conditional on the same imperfect local/query orientation, repaired exterior relations still add useful information.**

## 10. What is and is not established

### Supported in this synthetic family

1. Unlabeled repeated fields can reveal relation signs because off-diagonal covariance carries the gauge products.
2. Finite-sample relation repair must be conservative; unconditional repair damages intact graphs.
3. One trusted query-incident relation resolves the query-orientation ambiguity left by observation-only diagnostics.
4. After controlling for the same noisy anchor estimate, repaired exterior computation improves query accuracy.

### Not established

- relation repair from a **single** request;
- unknown graph support or relation magnitudes;
- non-Gaussian or non-gauge-balanced relation families;
- learned relation discovery in neural models;
- runtime/energy superiority after charging calibration acquisition and eigendecomposition;
- LLM reasoning, semantic compression, factuality or safety.

## 11. Error checks

The public scripts expose all protocol constants. Full-output deterministic reruns and selected numerical identities are checked before publication. Calibration/evaluation seed blocks are disjoint across Stages 55–60. Stage 56's threshold/K were explicitly tuned on Stage 55 and therefore are tested only on Stage 56's independent block.
