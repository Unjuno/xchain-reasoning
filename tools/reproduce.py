# SPDX-License-Identifier: Apache-2.0
"""Run the portable Stage 49-51 experiments without network access."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['49', '50', '51', 'all'], default='all')
    parser.add_argument('--out', type=Path, default=Path('runs/stage49_51'))
    parser.add_argument('--quick', action='store_true', help='One seed per phase, 64 samples; not the full study.')
    parser.add_argument('--save-inputs', action='store_true', help='Save synthetic arrays; full run adds about 256 MiB.')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    stages = ['49', '50', '51'] if args.stage == 'all' else [args.stage]
    if any((out / 'results' / f'stage{s}_rows.csv').exists() for s in stages):
        parser.error('Requested output already exists. Use a different --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               XCHAIN_OUTPUT_DIR=str(out), XCHAIN_SAMPLES='64' if args.quick else '2048',
               XCHAIN_SEED_LIMIT='1' if args.quick else '10',
               XCHAIN_SAVE_INPUTS='1' if args.save_inputs else '0')
    run_record = {'mode': 'smoke' if args.quick else 'full', 'stages': stages,
                  'samples_per_distribution': 64 if args.quick else 2048,
                  'seed_limit': 1 if args.quick else 10, 'python': platform.python_version(),
                  'thread_limits': {k: env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                  'save_inputs': args.save_inputs, 'status': 'running'}
    record = out / 'run.json'
    record.write_text(json.dumps(run_record, indent=2) + '\n', encoding='utf-8')
    try:
        for stage in stages:
            subprocess.run([sys.executable, str(ROOT / 'experiments' / 'stage49_51' / f'run{stage}.py')],
                           env=env, check=True)
        if args.stage == 'all' and not args.quick:
            subprocess.run([sys.executable, str(ROOT / 'experiments' / 'stage49_51' / 'analyze.py')],
                           env=env, check=True)
    except subprocess.CalledProcessError as exc:
        run_record['status'] = 'failed'
        run_record['exit_code'] = exc.returncode
        record.write_text(json.dumps(run_record, indent=2) + '\n', encoding='utf-8')
        return exc.returncode
    run_record['status'] = 'completed'
    record.write_text(json.dumps(run_record, indent=2) + '\n', encoding='utf-8')
    print('Completed', run_record['mode'], 'run. Results:', out)
    if args.quick:
        print('Smoke tests do not reproduce full-study confidence intervals or calibration selection.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
