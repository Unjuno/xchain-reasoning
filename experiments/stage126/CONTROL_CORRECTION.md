# Fixed-control fairness correction

During source inspection, before reading the first full-run outcomes, the
original runner was found to compute intermediate query beliefs and reversal
features even for fixed-depth controls. Fixed policies do not need these
features. The initial run is therefore not used for the primary runtime claim.

`strict_controls.py` substitutes a dedicated fixed C++ entry point and a plain
fixed NumPy loop. Both avoid these unused features. The candidate algorithm,
all seeds, batch sizes, repetitions, measurements and primary criteria are
unchanged. The complete study is rerun under this correction; original source
and the initial run remain available as an audit trail.

Canonical execution is now:

```
python experiments/stage126/strict_controls.py --out runs/stage126
```

The wrapper records its own source hashes and refreshes the artifact manifest.
It does not use labels or observed results to choose a comparator or threshold.
