# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'))
sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
from core import graph, make_problem, risks
from reliability import build_candidates, diagnostic_scores, corrupt_relations

def ci(x):
    x=np.asarray(x,float); n=len(x); m=float(x.mean())
    if n<2: return dict(mean=m,low=m,high=m,n=n)
    h=float(student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
    return dict(mean=m,low=m-h,high=m+h,n=n)

def corrupt30(adj,q,rng):
    a=np.asarray(adj).copy(); edges=np.argwhere(np.triu(a!=0,1)); eligible=edges[np.all(edges!=q,axis=1)]
    chosen=rng.permutation(len(eligible))[:round(.30*len(eligible))]
    for i,j in eligible[chosen]: a[i,j]*=-1; a[j,i]*=-1
    return a

def repair_pairwise(candidate,q,obs_y,k):
    a=np.asarray(candidate,float).copy(); n=len(a); obs=np.delete(np.arange(n),q); pos={int(v):i for i,v in enumerate(obs)}
    y=np.asarray(obs_y[:k],float); cov=y.T@y/k
    edges=np.argwhere(np.triu(a!=0,1))
    for i,j in edges:
        if i==q or j==q: continue
        s=1.0 if cov[pos[int(i)],pos[int(j)]]>=0 else -1.0
        mag=abs(a[i,j]); a[i,j]=a[j,i]=mag*s
    return a

def run(out:Path, quick=False):
    if out.exists(): raise ValueError('output exists')
    cfg=json.loads((HERE/'protocol55.json').read_text()); out.mkdir(parents=True)
    seeds=cfg['seeds'][:1] if quick else cfg['seeds']; fams=cfg['families'][:2] if quick else cfg['families']
    ks=[8,32] if quick else cfg['calibration_sizes']; rows=[]
    for seed in seeds:
      for fi,fam in enumerate(fams):
        true_a,q,_,_,_=graph(fam,seed)
        for ni,nr in enumerate(cfg['noise_ratios']):
          p=make_problem(true_a,q,cfg['tau'],nr)
          rng=np.random.default_rng(seed*1009+fi*37+ni)
          z=rng.standard_normal((max(ks),49)); e=rng.standard_normal((max(ks),48))
          th=z@np.linalg.cholesky(p['Sigma']).T; cal=th[:,p['obs']]+e*np.sqrt(p['noisevar'])
          for ki,kind in enumerate(cfg['corruptions']):
            crng=np.random.default_rng(seed*7001+fi*101+ni*17+ki)
            bad=corrupt30(true_a,q,crng) if kind=='flip30' else corrupt_relations(true_a,q,kind,crng)
            base=build_candidates(bad,q,cfg['tau'],p['noisevar'],cfg['gate_candidates'],cfg['steps'])
            score=diagnostic_scores(cal,base)['loo_nlpd']
            policies=[('gate0',base[0].output_coef),('bad_full',base[-1].output_coef)]
            oracle=build_candidates(true_a,q,cfg['tau'],p['noisevar'],[1.0],cfg['steps'])[0].output_coef
            policies.append(('oracle_true_relations',oracle))
            for K in ks:
              gi=int(np.argmin(score[:K].mean(0))); policies.append((f'gate_batch{K}',base[gi].output_coef))
              repaired=repair_pairwise(bad,q,cal,K)
              rw=build_candidates(repaired,q,cfg['tau'],p['noisevar'],[1.0],cfg['steps'])[0].output_coef
              policies.append((f'repair_pairwise{K}',rw))
            for pol,w in policies:
              r=risks(w,p)
              rows.append(dict(seed=seed,family=fam,noise_ratio=nr,corruption=kind,policy=pol,
                               acc=r['acc_exact'],nmse=r['nmse_exact']))
      print('done seed',seed,'rows',len(rows),flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'rows.csv',index=False)
    primaryK=32 if not quick else 32
    def diff(kind,a,b,metric='acc'):
      z=df[df.corruption==kind].pivot_table(index=['seed','family','noise_ratio'],columns='policy',values=metric)
      d=(z[a]-z[b]).groupby('seed').mean(); return ci(d.values)
    summary={'rows':len(df),'seed_blocks':len(seeds),'primary':{
      'flip50_repair_vs_gate0':diff('flip50',f'repair_pairwise{primaryK}','gate0'),
      'flip50_repair_vs_gatebatch':diff('flip50',f'repair_pairwise{primaryK}',f'gate_batch{primaryK}'),
      'intact_repair_vs_full':diff('intact',f'repair_pairwise{primaryK}','bad_full')},
      'by_policy':df.groupby(['corruption','policy']).mean(numeric_only=True).reset_index().to_dict('records')}
    a=summary['primary']; summary['primary']['status']='PASS' if (a['flip50_repair_vs_gate0']['low']>0 and a['flip50_repair_vs_gatebatch']['low']>0 and a['intact_repair_vs_full']['low']>-.005) else 'FAIL'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(summary['primary'],indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
