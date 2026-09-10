# Reproduction and evidence levels

## Supported study

The maintained public runner covers Stage 49–51. It uses Python, NumPy, SciPy and
pandas, with float64 inference and one BLAS/OpenMP thread. The recorded full run
used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and pandas 2.2.3. Dependencies are
pinned for reproduction, not described as the latest versions.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/reproduce.py --quick --out runs/smoke
python tools/reproduce.py --out runs/full
python tools/compare_reference.py runs/full
```

Use `--stage 49`, `--stage 50` or `--stage 51` to run one stage. Stage 51 includes
both calibration and held-out evaluation; inference never receives the latent
query label. `--save-inputs` saves all generated arrays. A complete input export
adds roughly 256 MiB. The runner refuses to overwrite an existing stage CSV.

The full protocol creates 5,400 Stage 49 rows, 3,500 Stage 50 rows and 2,700 Stage
51 rows (300 calibration + 2,400 test). Stage 49 has 300 problem conditions, not
5,400 independent experiments. Covariance parameters share base random draws.
The statistical unit is ten graph/data seed blocks; Stage 51 calibration uses
five different seeds. Monte Carlo samples per distribution are 2,048.

## What each check means

- Unit tests cover dimensions, positive definiteness, contraction, unobserved
  query inputs, causal reach, risk decomposition, direct-state/coefficient
  equivalence, reset controls, corruption locality and node permutation.
- The quick run uses one seed per phase and 64 samples. It tests execution, not
  published confidence intervals or the full gate-calibration conclusion.
- `compare_reference.py` compares grouped outputs against the supplied reference
  summaries. The publication audit additionally compared all 11,600 row values.
- A full rerun in this environment matched the supplied numeric CSV values.
  This is a reproducibility check, not an independent held-out confirmation.
- Historical Stage 1–48 findings were curated from saved reports. They were not
  all rerun during this publication and are not covered by the smoke CI.

## Accounting and limitations

Sixteen full synchronous sweeps and eight fresh two-sweep restarts each imply
784 nominal node updates for 49 nodes. The coefficient implementation exploits
linearity; its wall-clock runtime is not that nominal algorithmic work. Direct
Bayes baselines have nonzero setup/solve cost, even where legacy CSVs used zero
as an unaccounted-cost marker. Do not use those zeros in latency ratios.

The all-observation Bayes estimator has more information than a neighbor-only
Bayes estimator. That comparison diagnoses unused information, not equal-input
solver superiority. Cross-family graph changes also change the prior.

Bitwise agreement across hardware/library versions is not promised. Floating-
point tolerances are numerical audit thresholds, not interval-arithmetic proofs.
Confidence intervals measure graph/data-seed variation within this protocol,
not every source of uncertainty and not historical multiple-comparison control.

See [provenance](../provenance/publication_reproduction.json) and
[limitations](LIMITATIONS.md).
