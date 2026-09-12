# Research roadmap

## Roadmap v1 status: resolved with mixed outcomes

The five items defined after Stage 60 now have falsifiable experimental or theoretical dispositions. See [the resolution report](ROADMAP_RESOLUTION_2026-09-12.md).

| Item | Disposition |
|---|---|
| Unknown support / magnitudes | Conditional PASS for exterior reconstruction; fully unknown query support has an identifiability limit. |
| Single-request / low calibration | Independent unlabeled fields and reusable history work at regime level; same-request-only reliability selection is insufficient in the tested setting. |
| Non-gauge-balanced relations | Exterior-compute mechanism reproduced in signed Ising trees and loopy graphs; structural-frustration-only gating FAILs on intrinsically frustrated correct models. |
| Learned relation model | Model error, relation error and finite-depth inference error were separated in Stage 122. |
| Cost accounting | Nominal-work allocation can improve accuracy while current wall-clock implementations are slower; repair setup is amortizable only across repeated requests. |

### Final adaptive-compute findings

- Stage 119: shallow **decision reversal** is a stronger instance-level trigger than uncertainty on average, but fails on a simple cycle.
- Stage 120: naively stacking history and instance triggers does **not** add value over a segment-matched random control.
- Stage 121: stale-history reset gives a small Pareto improvement under relation drift.
- Stage 124: current subset-continuation NumPy implementation is **2.77x slower** than fixed depth 12 despite higher accuracy per nominal depth.
- Stage 125: Stage-57-style relation repair has setup cost equivalent to roughly 653 single coefficient applications, before data-acquisition cost.

## Next roadmap (v2)

The original roadmap is closed. New work should be treated as a new roadmap rather than retroactively changing v1:

1. Replace synthetic explicit relations with learned latent relations while retaining auditability.
2. Design a vectorization-friendly adaptive continuation implementation and retest wall-clock efficiency.
3. Test whether the regime/instance decomposition survives larger state spaces and learned representations.
4. Keep exact/strong solvers as controls; do not treat extra iteration as a contribution by itself.

Exploratory runs belong under ignored `runs/` or `local_work/`. Promote a finding only after a frozen protocol, held-out evaluation, numerical audit, privacy review and a result-specific public change.
