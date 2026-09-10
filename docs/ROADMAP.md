# Research roadmap (not completed experiments)

The active question is whether an exterior workspace improves **true output
accuracy**, and under what information and relation-reliability assumptions.

The next research branch should test label-free reliability signals for exterior
relations, with calibration distributions and evaluation seeds separated before
looking at outcomes. Candidate diagnostics include held-out observation
prediction and relation-consistency checks. Compare them against direct Bayesian
inference, local optimal inference, shallow restarts, unweighted iteration and a
fixed gate. Include wrong-but-consistent relations and distribution shift.

A separate experiment should hold the generating distribution fixed while
changing only the computational routing graph. The present cross-family table
cannot identify a topology-only causal effect. Count preprocessing, reliability
probes, memory, total work and sequential depth independently.

Local exploratory runs belong under ignored `runs/` or `local_work/`. Promote a
finding only after a frozen protocol, held-out evaluation, numerical audit,
privacy review and a result-specific public change. This document is not a
promise of unattended or background execution.
