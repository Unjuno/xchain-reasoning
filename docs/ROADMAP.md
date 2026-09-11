# Research roadmap (not completed experiments)

Stages 52–60 established a conditional path from **relation suppression** to **relation repair** in the balanced-sign Gaussian family. The next branch should deliberately remove the assumptions that made that repair possible.

Priority experiments:

1. **Unknown support/magnitudes.** Infer whether an edge exists and its strength, not only its sign. Hold the true generating law fixed while corrupting the computational graph.
2. **Single-request / low-calibration regime.** Stage 58 shows that repeated unlabeled fields are expensive; test active acquisition and confidence bounds that choose which relations need more evidence.
3. **Non-gauge-balanced relations.** Use models where pairwise covariance signs do not directly encode a globally consistent node gauge, and compare repair with suppression and direct Bayesian inference.
4. **Learned relation model.** Train a small interpretable recurrent/graph model and separate model error, relation error and inference error.
5. **Cost accounting.** Charge calibration acquisition, eigendecomposition, repair, memory and sequential depth before making an efficiency claim.

Local exploratory runs belong under ignored `runs/` or `local_work/`. Promote a finding only after a frozen protocol, held-out evaluation, numerical audit, privacy review and a result-specific public change. This document is not a promise of unattended or background execution.
