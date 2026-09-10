# SPDX-License-Identifier: Apache-2.0
"""Stage54 diagnostic: observation-equivalent worlds with opposite query truth.

This deliberately relaxes Stage52's trusted-query-edge assumption. It is not
an outer-only corruption test, nor a refutation under known correct query edges.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from run import digest
from core import FAMILIES,graph,make_problem
from reliability import build_candidates,diagnostic_scores,select_predictions
HERE=Path(__file__).resolve().parent


def run(out,quick=False):
    if out.exists():raise ValueError('Output already exists.')
    out.mkdir(parents=True)
    (out/'frozen.json').write_text(json.dumps({n:digest(HERE/n) for n in ['blind_spot.py','reliability.py']},indent=2)+'\n')
    rows=[]
    for seed in ([64991] if quick else range(64001,64011)):
        for family in (FAMILIES[:2] if quick else FAMILIES):
            a,q,_,_,_=graph(family,seed)
            signs=np.ones(49);signs[q]=-1
            b=a*signs[:,None]*signs[None,:]
            for nr in [1.,4.]:
                p=make_problem(a,q,.05,nr);other=make_problem(b,q,.05,nr)
                rng=np.random.default_rng(seed*109+(0 if nr==1. else 1))
                theta=rng.standard_normal((128 if quick else 1024,49))@np.linalg.cholesky(p['Sigma']).T
                y=theta[:,p['obs']]+rng.normal(size=(len(theta),48))*np.sqrt(p['noisevar'])
                ca=build_candidates(a,q,.05,p['noisevar'],[0.,.125,.25,.5,.75,1.])
                cb=build_candidates(b,q,.05,p['noisevar'],[0.,.125,.25,.5,.75,1.])
                sa=diagnostic_scores(y,ca)['loo_nlpd'];sb=diagnostic_scores(y,cb)['loo_nlpd']
                pred,index=select_predictions(y,ca,sa)
                pred_b,index_b=select_predictions(y,cb,sb)
                acc_plus=float(np.mean((pred>=0)==(theta[:,q]>=0)))
                acc_minus=float(np.mean((pred>=0)==(-theta[:,q]>=0)))
                rows.append(dict(seed=seed,family=family,noise_ratio=nr,
                    observation_covariance_difference=float(np.max(np.abs(p['C']-other['C']))),
                    query_cross_covariance_sum=float(np.max(np.abs(p['k']+other['k']))),
                    diagnostic_difference=float(np.max(np.abs(sa-sb))),
                    selected_gate_difference=int(np.count_nonzero(index!=index_b)),
                    opposite_prediction_error=float(np.max(np.abs(pred+pred_b))),
                    accuracy_world_plus=acc_plus,accuracy_world_minus=acc_minus,
                    paired_world_accuracy=(acc_plus+acc_minus)/2))
    df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False)
    summary=dict(cases=len(df),max_observation_covariance_difference=float(df.observation_covariance_difference.max()),
        max_diagnostic_difference=float(df.diagnostic_difference.max()),
        gate_mismatch_count=int(df.selected_gate_difference.sum()),
        max_opposite_prediction_error=float(df.opposite_prediction_error.max()),
        paired_world_accuracy=float(df.paired_world_accuracy.mean()),
        scope='Trusted query-incident relations REMOVED. Same observable law, opposite query labels. Not an outer-only corruption test.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--quick',action='store_true')
    args=p.parse_args();run(args.out,args.quick)
