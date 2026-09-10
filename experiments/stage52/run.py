# SPDX-License-Identifier: Apache-2.0
"""Reproducible label-free reliability experiment. Outputs stay outside source."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set these before NumPy/SciPy imports; no runtime credentials or network used.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'stage49_51'))
from core import graph, make_problem, subset_coef, coefficient_path
from reliability import build_candidates, diagnostic_scores, select_predictions, corrupt_relations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_ci(values):
    x = np.asarray(values, dtype=float)
    u = float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
    k = float(student_t.ppf(.975, len(x)-1)) if len(x)>1 else 0.
    m = float(x.mean())
    return {'mean': m, 'low': m-k*u, 'high': m+k*u, 'standard_uncertainty': u,
            'coverage_factor': k, 'seed_blocks': len(x)}


def summarize(frame):
    means = frame.groupby(['corruption','policy'])[['acc_rb','acc_mc','nmse_rb','nmse_mc','mean_gate']].mean().reset_index()
    result = {'means': means.astype(object).where(means.notna(), None).to_dict('records')}
    pairs = []
    for corruption in frame.corruption.unique():
        block = frame[frame.corruption == corruption].groupby(['seed','policy']).mean(numeric_only=True)
        for policy in ['loo_nlpd','evidence','loo_mse','loo_batch32','evidence_batch32']:
            for baseline in ['fixed_1','fixed_0','shallow2']:
                a = block.xs(policy, level='policy'); b = block.xs(baseline, level='policy')
                pairs.append({'corruption':corruption,'policy':policy,'baseline':baseline,
                    'acc_gain':paired_ci(a.acc_rb-b.acc_rb),
                    'acc_gain_mc':paired_ci(a.acc_mc-b.acc_mc),
                    'nmse_decrease':paired_ci(b.nmse_rb-a.nmse_rb)})
    result['paired_comparisons'] = pairs
    required = [p for p in pairs if p['policy']=='loo_nlpd' and p['baseline']=='fixed_1']
    corrupt = next(p for p in required if p['corruption']=='flip50')
    intact = next(p for p in required if p['corruption']=='intact')
    result['primary'] = {'corrupt_gain':corrupt['acc_gain'], 'intact_gain':intact['acc_gain'],
        'criteria':'Corrupt sign-gain CI lower > 0 AND intact CI lower > -0.005',
        'status': 'PASS' if corrupt['acc_gain']['low']>0 and intact['acc_gain']['low']>-.005 else 'FAIL',
        'scope':'Known Gaussian family and noise; no computational-efficiency claim.'}
    return result


def run(out, quick=False):
    if out.exists():
        raise ValueError('Output directory already exists; choose a new path.')
    protocol = json.loads((HERE/'protocol.json').read_text())
    out.mkdir(parents=True)
    (out/'inputs').mkdir()
    hashes = {p.name:digest(p) for p in [HERE/'protocol.json',HERE/'run.py',HERE/'reliability.py']}
    hashes['stage49_51/core.py'] = digest(HERE.parent/'stage49_51/core.py')
    (out/'frozen.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    count = 64 if quick else protocol['samples_per_condition']
    seeds = [62991] if quick else protocol['seeds']
    families = protocol['families'][:2] if quick else protocol['families']
    rows, diagnostics, timings = [], [], []
    saved_bytes = 0
    for seed in seeds:
        for fi, family in enumerate(families):
            adjacency, query, edges, gauge, distance = graph(family, seed)
            rng = np.random.default_rng(seed*107+fi)
            z = rng.standard_normal((count+32,49)); noise = rng.standard_normal((count+32,48))
            path = out/'inputs'/f'{seed}_{family}.npz'
            np.savez_compressed(path, adjacency=adjacency,query=query,z=z,noise=noise)
            saved_bytes += path.stat().st_size
            for nr in protocol['noise_ratios']:
                problem = make_problem(adjacency,query,protocol['tau'],nr)
                latent = z @ np.linalg.cholesky(problem['Sigma']).T
                observations = latent[:,problem['obs']] + noise*np.sqrt(problem['noisevar'])
                y = observations[:count]; calibration = observations[count:]
                truth = latent[:count,query]
                # Ground truth quantities below are used only by the evaluator.
                posterior_mean = y @ problem['bayes_coef']
                posterior_var = problem['bayes_var']
                true_local = y @ subset_coef(problem, np.flatnonzero(distance[query]<=1))
                shallow = y @ coefficient_path(problem,[2])[2][query]
                for ki,kind in enumerate(protocol['corruptions']):
                    bad = corrupt_relations(adjacency,query,kind,np.random.default_rng(seed*701+fi*10+ki))
                    assert np.array_equal(bad[query],adjacency[query])
                    start = time.perf_counter()
                    candidates = build_candidates(bad,query,protocol['tau'],problem['noisevar'],protocol['gates'],protocol['steps'])
                    build_seconds = time.perf_counter()-start
                    start = time.perf_counter(); scores = diagnostic_scores(y,candidates)
                    score_seconds = time.perf_counter()-start
                    start = time.perf_counter(); cal_scores = diagnostic_scores(calibration,candidates)
                    calibration_seconds = time.perf_counter()-start
                    predictions = {'shallow2':(shallow,None),'bayes_all':(posterior_mean,None),'bayes_local':(true_local,None)}
                    for name, score in scores.items():
                        predictions[name] = select_predictions(y,candidates,score)
                    for name in ['loo_nlpd','evidence']:
                        index = int(np.argmin(cal_scores[name].mean(axis=0)))
                        predictions[('loo' if name=='loo_nlpd' else name)+'_batch32'] = (
                            y @ candidates[index].output_coef, np.full(count,index,dtype=int))
                    for gi,candidate in enumerate(candidates):
                        predictions[f'fixed_{candidate.gate:g}']=(y@candidate.output_coef,np.full(count,gi,dtype=int))
                    for policy,(pred,index) in predictions.items():
                        signed = np.where(pred>=0,1.,-1.)
                        # Rao-Blackwellized accuracy integrates latent query noise;
                        # the adaptive policy depends only on observations, not truth.
                        row = dict(seed=seed,family=family,noise_ratio=nr,corruption=kind,policy=policy,
                            acc_rb=float(ndtr(signed*posterior_mean/np.sqrt(posterior_var)).mean()),
                            acc_mc=float(np.mean((pred>=0)==(truth>=0))),
                            nmse_rb=float(((pred-posterior_mean)**2+posterior_var).mean()/problem['prior']),
                            nmse_mc=float(((pred-truth)**2).mean()/problem['prior']),
                            mean_gate=float(np.asarray(protocol['gates'])[index].mean()) if index is not None else np.nan)
                        rows.append(row)
                    for name,score in scores.items():
                        selected = np.argmin(score,axis=1)
                        for gi,gate in enumerate(protocol['gates']):
                            diagnostics.append(dict(seed=seed,family=family,noise_ratio=nr,corruption=kind,
                                diagnostic=name,gate=gate,selection_fraction=float(np.mean(selected==gi))))
                    timings.append(dict(seed=seed,family=family,noise_ratio=nr,corruption=kind,
                        build_seconds=build_seconds,score_seconds=score_seconds,
                        calibration_seconds=calibration_seconds,samples=count,candidates=len(candidates),
                        cached_float64_bytes=sum(c.observed_precision.nbytes+c.output_coef.nbytes+c.exact_output_coef.nbytes for c in candidates)))
        pd.DataFrame(rows).to_csv(out/'rows.csv',index=False)
        print('Completed seed',seed,'rows',len(rows),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'rows.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'selection.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'costs.csv',index=False)
    summary=summarize(frame)
    summary.update(rows=len(rows),conditions=len(timings),samples_per_condition=count,
        independent_seed_blocks=len(seeds),mode='smoke' if quick else 'full',
        raw_inputs_bytes=saved_bytes,scoring='Conditional expected accuracy/MSE integrated over query truth; Monte Carlo over observation vectors, not exact population risks.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if 'model name' in line),'unknown')
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cpu=cpu,
        dtype='float64',threads=1,batch=count,clock='not fixed or measured',timing='single diagnostic logs, NOT speed benchmarks'),indent=2)+'\n')
    assert all(digest(HERE/name)==value for name,value in hashes.items() if '/' not in name)
    print(json.dumps(summary['primary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    run(args.out,args.quick)
