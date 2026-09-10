# XChain Reasoning

Experimental research on **graph-structured external workspaces for iterative inference**.
The central question is whether computation outside a visible output node can improve
that output's **true task accuracy**, rather than merely reproduce an existing model faster.

[日本語](README.ja.md) · [Research history](docs/RESEARCH_HISTORY.md) ·
[Theory](docs/THEORY.md) · [Reproduction](docs/REPRODUCIBILITY.md) ·
[Limitations](docs/LIMITATIONS.md)

> Research prototype, not a validated LLM method. A cross is one graph topology,
> not a necessary shape. “External workspace” means explicit numerical states;
> it does not mean publishing private conversational reasoning or transcripts.

## Current evidence: Stage 49–51

A 49-node synthetic Gaussian model has one unobserved output node and 48 noisy
observed exterior nodes. Inputs remain fixed while exterior states are updated.
There are five graph families and ten paired graph/data seeds per main experiment.

| Method | Exact sign accuracy | Normalized MSE |
|---|---:|---:|
| Two synchronous updates | 68.67% | 0.7652 |
| Two updates restarted eight times | 68.67% | 0.7652 |
| Sixteen state-preserving updates | **70.40%** | **0.6348** |
| All-observation Bayes solution | 70.43% | 0.6297 |

The 2→16 update gain is **+1.730 percentage points** (95% seed interval
+1.698 to +1.762), with **17.05% normalized-MSE reduction**. Against the stronger
optimal *neighbor-only* estimator, the MSE reduction is **6.45%**, not 17%.
Exact expectations and Monte Carlo checks are kept separate.

**Negative result:** flipping 50% of non-query-adjacent relation signs, while
keeping the true data and query-adjacent relations fixed, changes 2→16-step
accuracy by **−1.835 percentage points**. Convergence is not correctness.
A scalar gate fitted on corrupted calibration data helps that regime but harms
intact relations; it is not a general trust detector.

Sources: [reference tables](reference/stage49_51/KEY_RESULTS.csv),
[protocols](reference/stage49_51/protocol49.json),
[public study report](docs/history/stage49_51.md), and
[publication re-execution audit](provenance/publication_reproduction.json).
The public export was rerun on all **11,600 result rows**; numeric CSV values
matched the supplied results in the recorded environment. This is a reproduction,
not an independent new study.

## Run locally

Python 3.11 or newer is recommended. No API keys, model downloads, or network
access are needed by the experiment after installing dependencies.

```bash
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/reproduce.py --quick --out runs/smoke
python tools/reproduce.py --out runs/full
python tools/compare_reference.py runs/full
```

The quick command is an execution check, **not** a reproduction of confidence
intervals or gate selection. Full reproduction uses the original seeds and
2,048 samples per distribution. Add `--save-inputs` to save the synthetic arrays;
large run outputs are intentionally ignored by Git. Existing results are not
silently overwritten.

## What is public here

- Portable, tested Stage 49–51 source and deterministic synthetic-input generation.
- Main reference summaries, protocols, mathematical definitions and proofs.
- A curated Stage 1–51 history, including corrections, FAIL and UNCERTAIN results.
- Input-package hashes, reproduction scope, publication policy and review checks.

Earlier stages are **historical records**, not claimed to have been rerun during
this publication. Their complete legacy execution environments, bulk arrays and
checkpoints are not silently represented as a supported release. See
[publication scope](docs/PUBLICATION.md) and the
[historical evidence ledger](docs/RESEARCH_HISTORY.md).

## Non-claims

No demonstrated superiority to an exact Bayesian solver, Gauss–Seidel, all graph
inference algorithms, or an LLM benchmark. No lossless semantic compression claim.
Changing graph topology also changes the generating distribution, so the tables
are not a topology-only causal ranking. Nominal local-update counts are **not**
wall-clock, energy, or a claim that direct inference is free.

## License and citation

[Apache-2.0](LICENSE). Dependencies retain their own licenses; none are vendored.
See [CITATION.cff](CITATION.cff), [NOTICE](NOTICE),
[references](docs/REFERENCES.md), and [contributing](CONTRIBUTING.md).
