# Stage 49–51: True output accuracy and relation corruption

**Evidence level:** Fully re-executed during this publication.

This is the actively supported public experiment and was fully rerun during
publication. The other historical pages summarize supplied records only.

Forty-nine latent Gaussian values are related by a signed weighted graph. One
query is unobserved; 48 exterior observations remain fixed. Iteration preserves
explicit numerical workspace state. Five topologies and ten graph/data seed
blocks define the main test; each distribution has 2,048 Monte Carlo samples.

| Method | Exact sign accuracy | Normalized MSE |
|---|---:|---:|
| Two synchronous updates | 68.6726% | 0.765215 |
| Sixteen synchronous updates | 70.4028% | 0.634800 |
| Optimal neighbor-only inference | 68.6916% | 0.678551 |
| All-observation Bayes inference | 70.4262% | 0.629650 |

The 2→16 gain is +1.7302 percentage points, with a 95% seed interval of
[+1.6979,+1.7624]. The NMSE reduction is 17.0538%; against optimal neighbor-only
inference it is 6.4534%. Eight two-step restarts do not propagate farther and
return the two-step prediction despite the same nominal 784 node updates.
Gauss–Seidel and the direct Bayes solution are explicit strong controls.

Stage 50 leaves true data and all query-adjacent relations unchanged and corrupts
only exterior-edge signs. At 50% corruption, exact sign accuracy falls from
70.2817% at two updates to 68.4471% at sixteen. The iteration still contracts.
MSE and classification can move in different directions.

Stage 51 selects a single exterior-message gate using five separate calibration
seeds with 50% corrupted relations. The classification objective selects zero;
mean normalized-MSE selects one. In ten held-out seeds, classification gating
gains 2.1975 percentage points at 50% corruption but loses 3.9759 points on intact
relations. This avoids bad messages; it does not discover which relation is
reliable without labels.

**What passed:** a conditional mechanism test of true accuracy improvement.
**What did not follow:** X-shape necessity, universal monotonic improvement,
solver novelty, superiority to full Bayes inference, or LLM/semantic-compression
validity. Topology changes also change the prior, precluding a topology-only
causal interpretation.

See [theory](../THEORY.md), [protocol 49](../../reference/stage49_51/protocol49.json),
[protocol 50](../../reference/stage49_51/protocol50.json),
[protocol 51](../../reference/stage49_51/protocol51.json),
[results](../../reference/stage49_51/KEY_RESULTS.csv), and
[re-execution audit](../../provenance/publication_reproduction.json).

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
