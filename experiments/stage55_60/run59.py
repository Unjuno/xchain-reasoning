# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'));sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
from core import graph,make_problem,risks
from reliability import build_candidates

def ci(x):
 x=np.asarray(x,float);m=float(x.mean());n=len(x);h=0 if n<2 else float(student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n));return dict(mean=m,low=m-h,high=m+h,n=n)
def observed_gauges(A,q,Y):
 obs=np.delete(np.arange(len(A)),q);X=Y-Y.mean(0);S=X.T@X/len(X);_,V=np.linalg.eigh(S);return obs,np.where(V[:,-1]>=0,1.,-1.)
def build_from_gauge(A,q,obs,gobs,gq):
 g=np.empty(len(A));g[obs]=gobs;g[q]=gq;R=np.abs(A)*g[:,None]*g[None,:];np.fill_diagonal(R,0);return R
def run(out,quick=False):
 if out.exists():raise ValueError('output exists')
 cfg=json.loads((HERE/'protocol59.json').read_text());out.mkdir(parents=True);seeds=cfg['seeds'][:1] if quick else cfg['seeds'];fams=cfg['families'][:2] if quick else cfg['families'];R=8 if quick else cfg['anchor_replicates'];rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed);neigh=np.flatnonzero(A[q]!=0);first=int(sorted(neigh,key=lambda j:(-abs(A[q,j]),int(j)))[0])
   for ni,nr in enumerate(cfg['noise_ratios']):
    p=make_problem(A,q,cfg['tau'],nr);rng=np.random.default_rng(seed*7211+fi*113+ni);K=cfg['calibration_size'];z=rng.standard_normal((K,49));e=rng.standard_normal((K,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar']);obs,gobs=observed_gauges(A,q,Y);pos={int(v):i for i,v in enumerate(obs)}
    local=risks(build_candidates(A,q,cfg['tau'],p['noisevar'],[0.],cfg['steps'])[0].output_coef,p)['acc_exact']
    for er in cfg['anchor_error_rates']:
     for rep in range(R):
      rr=np.random.default_rng(seed*99991+fi*1009+ni*101+int(er*1000)*17+rep)
      noisy={int(j):float(np.sign(A[q,j]))*(-1.0 if rr.random()<er else 1.0) for j in neigh}
      gq1=noisy[first]*gobs[pos[first]]
      votes=np.array([noisy[int(j)]*gobs[pos[int(j)]] for j in neigh]);weights=np.abs(A[q,neigh]);score=float(weights@votes)
      if score==0: gqa=gq1
      else:gqa=1.0 if score>0 else -1.0
      if rep==0 and er==cfg['anchor_error_rates'][0]:
       orient={}
       for gg in (-1.0,1.0):
        B=build_from_gauge(A,q,obs,gobs,gg);w=build_candidates(B,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef;orient[gg]=risks(w,p)
      for pol,gq in [('one_anchor',gq1),('all_query_anchors',gqa)]:
       r=orient[float(gq)];rows.append(dict(seed=seed,family=fam,noise_ratio=nr,anchor_error=er,rep=rep,policy=pol,acc=r['acc_exact'],nmse=r['nmse_exact'],oracle_local_acc=local))
  print('done',seed,len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False)
 sub=df[df.anchor_error==.10].groupby(['seed','family','noise_ratio','policy']).mean(numeric_only=True).reset_index();piv=sub.pivot_table(index=['seed','family','noise_ratio'],columns='policy',values='acc');d=(piv.all_query_anchors-piv.one_anchor).groupby('seed').mean();allm=sub[sub.policy=='all_query_anchors'].groupby('seed').apply(lambda x:(x.acc-x.oracle_local_acc).mean(),include_groups=False)
 primary={'all_vs_one_at_10pct':ci(d.values),'all_vs_oracle_local_at_10pct':ci(allm.values)};primary['status']='PASS' if primary['all_vs_one_at_10pct']['low']>0 and primary['all_vs_oracle_local_at_10pct']['low']>0 else 'FAIL'
 means=df.groupby(['anchor_error','policy']).mean(numeric_only=True).reset_index();summary={'rows':len(df),'seed_blocks':len(seeds),'primary':primary,'means':means.to_dict('records')};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');means.to_csv(out/'means.csv',index=False);print(json.dumps(primary,indent=2));print(means[['anchor_error','policy','acc','oracle_local_acc']].to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
