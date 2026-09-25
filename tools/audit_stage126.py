# SPDX-License-Identifier: Apache-2.0
"""Independently rescore Stage 126's saved synthetic inputs and predictions.

This audit is not another independent scientific replication. It checks the
artifact's hashes, input regeneration, exact scoring and time-budget envelope.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage126'))
import run as stage


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def interval(x):
    x=np.asarray(x,float);se=float(x.std(ddof=1)/np.sqrt(len(x)))
    k=float(t.ppf(.975,len(x)-1));m=float(x.mean())
    return dict(mean=m,low=m-k*se,high=m+k*se,standard_error=se,coverage_factor=k,n=len(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    data=args.data;args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'SHA256.json').read_text())
    for name,h in manifest.items():
        assert Path(name).name==name,'Unsafe manifest path'
        assert digest(data/name)==h,f'Artifact mismatch: {name}'
    env=json.loads((data/'environment.json').read_text())
    for name,h in {**env['source_sha256'],**env['control_source_sha256']}.items():
        assert digest(ROOT/'experiments'/'stage126'/name)==h,f'Source changed: {name}'
    cfg=json.loads((data/'protocol.json').read_text())
    df=pd.read_csv(data/'rows.csv');tm=pd.read_csv(data/'timings.csv')
    assert len(df)==len(cfg['seeds'])*len(cfg['families'])*len(cfg['batch_sizes'])*20
    assert len(tm)==len(df)*cfg['timing_repetitions']
    max_score=0.;max_mu=0.;max_hull=0.;inputs=0;conditions=0;lp_cases=0
    for (seed,family,batch),z in df.groupby(['seed','family','batch']):
        seed=int(seed);batch=int(batch);gi=cfg['families'].index(family)
        a=np.load(data/f'input_{seed}_{family}_{batch}.npz',allow_pickle=False)
        edges,q,J,bad,obs,truth=stage.generate(seed,gi,family,batch,cfg)
        for name,value in [('edges',np.array(edges)),('query',q),('true_couplings',J),
                           ('inference_couplings',bad),('observations',obs),('truth',truth)]:
            np.testing.assert_array_equal(a[name],value)
        assert np.all(obs[:,q]==0)
        mu=stage.conditional_mean(edges,J,q,obs,cfg['observation_flip_probability'])
        max_mu=max(max_mu,float(np.max(abs(mu-a['conditional_mean']))))
        np.testing.assert_allclose(mu,a['conditional_mean'],rtol=1e-12,atol=1e-12)
        for r in z.itertuples():
            pred=a[r.policy]
            expected=float(np.mean(.5*(1+pred*mu)))
            empirical=float(np.mean(pred==truth))
            max_score=max(max_score,abs(expected-r.accuracy_expected),abs(empirical-r.accuracy_sampled))
            assert abs(expected-r.accuracy_expected)<1e-12
            assert abs(empirical-r.accuracy_sampled)<1e-12
            observed=tm[(tm.seed==seed)&(tm.family==family)&(tm.batch==batch)&(tm.policy==r.policy)].seconds
            assert len(observed)==cfg['timing_repetitions']
            assert abs(float(np.median(observed))-r.median_seconds)<1e-12
        fixed=z[z.policy.str.contains('fixed')]
        for r in z[~z.policy.str.contains('fixed')].itertuples():
            # Normalize time by budget, avoiding absolute small-second LP tolerances.
            answer=linprog(-fixed.accuracy_expected.to_numpy(),
                A_ub=(fixed.median_seconds.to_numpy()/r.median_seconds)[None,:],b_ub=[1.],
                A_eq=np.ones((1,len(fixed))),b_eq=[1.],bounds=(0,None),method='highs')
            if answer.success:
                assert np.isfinite(r.frontier_accuracy)
                error=abs(float(-answer.fun)-r.frontier_accuracy)
                max_hull=max(max_hull,error);assert error<1e-9
            else:
                assert np.isnan(r.frontier_accuracy) and answer.status==2
            lp_cases+=1
        inputs+=batch;conditions+=1
    table=[]
    for (batch,policy),z in df.groupby(['batch','policy']):
        table.append(dict(batch=int(batch),policy=policy,
            expected_accuracy=float(z.accuracy_expected.mean()),sampled_accuracy=float(z.accuracy_sampled.mean()),
            median_batch_ms=float(1000*z.median_seconds.median()),
            q1_batch_ms=float(1000*z.median_seconds.quantile(.25)),
            q3_batch_ms=float(1000*z.median_seconds.quantile(.75)),
            mean_nominal_depth=float(z.nominal_depth.mean())))
    pd.DataFrame(table).to_csv(args.out/'timing_table.csv',index=False)
    result={'status':'passed','artifact_files_verified':len(manifest),'conditions':conditions,
        'inputs_evaluated':inputs,'policy_rows':len(df),'timing_rows':len(tm),'independent_lp_checks':lp_cases,
        'max_score_difference':max_score,'max_conditional_mean_difference':max_mu,
        'max_frontier_difference_from_linprog':max_hull,
        'scope':'All saved inputs regenerated, every policy row rescored, every fixed-time frontier checked by independent LP. Timing is not rerun here.',
        'batches':{}}
    for batch in cfg['batch_sizes']:
        z=df[df.batch==batch];f=z[z.policy=='cpp_flip'];g=f.groupby('seed').apply(
            lambda x:100*(x.accuracy_expected-x.frontier_accuracy).mean(),include_groups=False)
        wide=z.pivot(index=['seed','family'],columns='policy',values='median_seconds')
        sums=wide.groupby('seed').sum()
        ratios={}
        for comparator in ('cpp_fixed12','numpy_fixed12','numpy_flip'):
            c=interval(np.log(sums.cpp_flip/sums[comparator]))
            ratios[comparator]={k:float(np.exp(c[k])) for k in ('mean','low','high')}
        result['batches'][str(batch)]={'frontier_gain_pp':interval(g),
            'ratio_of_family_summed_time':ratios,
            'selected_table':[r for r in table if r['batch']==batch and r['policy'] in
                 ('cpp_flip','cpp_uncertainty','cpp_random','cpp_fixed2','cpp_fixed12','cpp_fixed64',
                  'numpy_flip','numpy_fixed2','numpy_fixed12','numpy_fixed64')]}
    (args.out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('STAGE126_INDEPENDENT_AUDIT_BEGIN')
    print(json.dumps(result,indent=2))
    print('STAGE126_INDEPENDENT_AUDIT_END')

if __name__=='__main__':main()
