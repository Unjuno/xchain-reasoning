# SPDX-License-Identifier: Apache-2.0
"""Compare a completed full run with the public grouped reference tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ['stage49_means.csv', 'stage49_families.csv', 'stage50_means.csv', 'stage51_means.csv']


def compare(run: Path, atol: float = 1e-12) -> dict:
    record = json.loads((run / 'run.json').read_text())
    if record.get('mode') != 'full' or record.get('status') != 'completed' or record.get('stages') != ['49', '50', '51']:
        raise ValueError('Need a completed full all-stage run; smoke results are not the reference protocol.')
    results = {}
    for name in TABLES:
        expected = pd.read_csv(ROOT / 'reference' / 'stage49_51' / name)
        actual = pd.read_csv(run / 'results' / name)
        if expected.shape != actual.shape or list(expected.columns) != list(actual.columns):
            raise AssertionError(f'Shape/columns differ: {name}')
        numeric = expected.select_dtypes(include=np.number).columns
        other = [c for c in expected if c not in numeric]
        if not expected[other].equals(actual[other]):
            raise AssertionError(f'Label/order mismatch: {name}')
        np.testing.assert_allclose(actual[numeric], expected[numeric], atol=atol, rtol=0, equal_nan=True, err_msg=name)
        results[name] = {'rows': len(actual), 'max_abs_difference': float(np.nanmax(np.abs(actual[numeric].to_numpy() - expected[numeric].to_numpy())))}
    expected = json.loads((ROOT / 'reference' / 'stage49_51' / 'stage51_selected.json').read_text())
    actual = json.loads((run / 'results' / 'stage51_selected.json').read_text())
    if expected != actual:
        raise AssertionError('Frozen calibration selection differs.')
    return {'status': 'passed', 'atol': atol, 'tables': results, 'selection_matches': True}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    print(json.dumps(compare(args.run), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
