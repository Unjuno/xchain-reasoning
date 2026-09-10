# SPDX-License-Identifier: Apache-2.0
"""Stage53: new-seed, new-noise independent unlabeled calibration followup."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.special import ndtr
from run import paired_ci,digest
from core import graph,make_problem
from reliability import build_candidates,diagnostic_scores,corrupt_relations
HERE=Path(__file__).resolve().parent


def run_followup(out,quick=False):
    if out.exists():raise ValueError('Output already exists.')
    cfg=json.loads((HERE/'followup_protocol.json').read_text())
    out.mkdir(parents=True);(out/'inputs').mkdir()
    hashes={n:digest(HERE/n) for n in ['followup_protocol.json','followup.py','run.py','reliability.py']}
    (out/'frozen.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    count=64 if quick else cfg['samples_per_condition']
    seeds=[63991] if quick else cfg['seeds']
    families=cfg['families'][:2] if quick else cfg['families']
    rows=[]
    for seed in seeds:
        for fi,family in enumerate(families):
            a,q,_,_,_=graph(family,seed)
            rng=np.random.default_rng(seed*119+fi)
            z=rng.standard_normal((count+128,49));e=rng.standard_normal((count+128,48))
            np.savez_compressed(out/'inputs'/f'{seed}_{family}.npz',adjacency=a,query=q,z=z,noise=e)
            for nr in cfg['noise_ratios']:
                p=make_problem(a,q,cfg['tau'],nr)
                theta=z@np.linalg.cholesky(p['Sigma']).T
                y=theta[:,p['obs']]+e*np.sqrt(p['noisevar'])
                test=y[:count];cal=y[count:];truth=theta[:count,q]
                mu=test@p['bayes_coef'];var=p['bayes_var']
                for ki,kind in enumerate(cfg['corruptions']):
                    bad=corrupt_relations(a,q,kind,np.random.default_rng(seed*709+fi*10+ki))
                    candidates=build_candidates(bad,q,cfg['tau'],p['noisevar'],cfg['gates'],cfg['steps'])
                    scores=diagnostic_scores(cal,candidates)['loo_nlpd']
                    policies=[(f'fixed_{c.gate:g}',i) for i,c in enumerate(candidates)]
                    for size in cfg['unlabeled_calibration_sizes']:
                        index=int(np.argmin(scores[:size].mean(0)))
                        policies.append((f'loo_batch{size}',index))
                    for name,index in policies:
                        pred=test@candidates[index].output_coef
                        signed=np.where(pred>=0,1.,-1.)
                        rows.append(dict(seed=seed,family=family,noise_ratio=nr,corruption=kind,policy=name,
                            selected_gate=candidates[index].gate,
                            acc_rb=float(ndtr(signed*mu/np.sqrt(var)).mean()),
                            acc_mc=float(np.mean((pred>=0)==(truth>=0))),
                            nmse_rb=float(((pred-mu)**2+var).mean()/p['prior'])))
        print('Stage53 completed seed',seed,'rows',len(rows),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False)
    comparisons=[]
    for kind in cfg['corruptions']:
        block=df[df.corruption==kind].groupby(['seed','policy']).mean(numeric_only=True)
        for size in cfg['unlabeled_calibration_sizes']:
            for baseline in ['fixed_0','fixed_1']:
                aa=block.xs(f'loo_batch{size}',level='policy');bb=block.xs(baseline,level='policy')
                comparisons.append(dict(corruption=kind,calibration_size=size,baseline=baseline,
                    acc_gain=paired_ci(aa.acc_rb-bb.acc_rb),acc_gain_mc=paired_ci(aa.acc_mc-bb.acc_mc),
                    nmse_decrease=paired_ci(bb.nmse_rb-aa.nmse_rb)))
    corr=next(c for c in comparisons if c['corruption']=='flip50' and c['calibration_size']==32 and c['baseline']=='fixed_1')
    intact=next(c for c in comparisons if c['corruption']=='intact' and c['calibration_size']==32 and c['baseline']=='fixed_1')
    summary=dict(rows=len(rows),conditions=len(df)//11,samples_per_condition=count,seed_blocks=len(seeds),
        primary=dict(corrupt_gain=corr['acc_gain'],intact_gain=intact['acc_gain'],
            status='PASS' if corr['acc_gain']['low']>0 and intact['acc_gain']['low']>-.005 else 'FAIL'),
        comparisons=comparisons,means=df.groupby(['corruption','policy']).mean(numeric_only=True).reset_index().to_dict('records'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert all(digest(HERE/n)==h for n,h in hashes.items())
    print(json.dumps(summary['primary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true')
    a=p.parse_args();run_followup(a.out,a.quick)
