# Stage 55–60

Unlabeled relation repair and trusted-anchor experiments in the balanced-sign Gaussian family.

- Stage 55: **FAIL**. Unconditional K=32 pairwise sign repair helps corrupted graphs but damages intact graphs.
- Stage 56: **PASS under frozen criteria**. A confidence-thresholded K=128 repair rule, tuned only on Stage 55, generalizes to new seeds/noise and preserves intact accuracy within the predeclared margin.
- Stage 57: **PASS**. With all signs unreliable, 512 unlabeled fields plus one trusted query-incident relation reconstruct enough structure to beat an oracle local-only baseline that knows every true query-edge sign but disables exterior-exterior messages.
- Stage 58: prespecified calibration-size curve; exterior value becomes clearly positive versus oracle local-only at K>=256 in this test.
- Stage 59: **FAIL** primary. Redundant noisy anchors beat one-anchor dependence, but at 10% anchor error do not significantly beat the impossible oracle-local control.
- Stage 60: **PASS**. Holding the same noisy query orientation fixed, repaired exterior computation adds +2.685 pp at 10% anchor error, 95% paired-seed interval +2.452 to +2.919 pp.

These stages assume known graph support and magnitudes, repeated unlabeled calibration fields, and the gauge-balanced Gaussian relation family. They do not establish single-request relation discovery or LLM factuality.
