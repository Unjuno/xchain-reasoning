# Stage 09: Independent-headroom reconstruction

**Evidence level:** Historical report; not re-executed during this publication.

Stage 9 rebuilt 30 toy DDPM models and evaluated 960 conditions; it did not
resume the previously reported 13/30 jobs. The old weights and required modules
could not be confirmed in that handoff.

Using separate samples for headroom and target gain, the product of headroom and
decision-flip rate improved gain-prediction MAE by 35.26% in one distribution
transfer direction and 19.19% in the reverse. The fixed two-way 20% criterion
was not met. Adding flip rate beyond headroom alone improved MAE by 2.15% and
3.43%, with intervals crossing zero.

Sharing the same noisy R1 estimate between predictor and outcome changed the
reverse-direction improvement from 19.19% to 28.63%. A null simulation with no
true gain produced a correlation near 0.705 under shared error and near zero
under independent measurement. This is a measurement-design correction, not a
claim that every earlier effect was spurious.

**Status:** primary threshold FAIL; incremental flip-rate value UNCERTAIN.
The remaining empirical observation was that reachable evidence can benefit
from more computation. R1 was a 24-evaluation trajectory, not one neural call.

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
