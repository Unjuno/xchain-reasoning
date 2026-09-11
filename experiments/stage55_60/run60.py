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
def obs_gauge(A,q,Y):
 obs=np.delete(np.arange(len(A)),q);X=Y-Y.mean(0);S=X.T@X/len(X);_,V=np.linalg.eigh(S);return obs,np.where(V[:,-1]>=0,1.,-1.)
def graph_from(A,q,obs,gobs,gq):
 g=np.empty(len(A));g[obs]=gobs;g[q]=gq;B=np.abs(A)*g[:,None]*g[None,:];np.fill_diagonal(B,0);return B

def run(out,quick=False):
 if out.exists():raise ValueError('output exists')
 cfg=json.loads((HERE/'protocol60.json').read_text());out.mkdir(parents=True);seeds=cfg['seeds'][:1] if quick else cfg['seeds'];fams=cfg['families'][:2] if quick else cfg['families'];R=8 if quick else cfg['anchor_replicates'];rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed);neigh=np.flatnonzero(A[q]!=0)
   for ni,nr in enumerate(cfg['noise_ratios']):
    p=make_problem(A,q,cfg['tau'],nr);rng=np.random.default_rng(seed*8123+fi*131+ni);K=cfg['calibration_size'];z=rng.standard_normal((K,49));e=rng.standard_normal((K,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar']);obs,gobs=obs_gauge(A,q,Y);pos={int(v):i for i,v in enumerate(obs)}
    orient={}
    for gg in (-1.,1.):
     B=graph_from(A,q,obs,gobs,gg);wf=build_candidates(B,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef;wl=build_candidates(B,q,cfg['tau'],p['noisevar'],[0.],cfg['steps'])[0].output_coef;orient[gg]=(risks(wf,p),risks(wl,p))
    for er in cfg['anchor_error_rates']:
     for rep in range(R):
      rr=np.random.default_rng(seed*100003+fi*1009+ni*103+int(er*1000)*19+rep);votes=[];weights=[]
      for j in neigh:
       s=float(np.sign(A[q,j]))*(-1 if rr.random()<er else 1);votes.append(s*gobs[pos[int(j)]]);weights.append(abs(A[q,j]))
      score=float(np.dot(weights,votes));gq=(1. if score>0 else -1.) if score!=0 else float(votes[0]);rf,rl=orient[gq]
      for pol,r in [('repaired_full',rf),('reconstructed_local',rl)]: rows.append(dict(seed=seed,family=fam,noise_ratio=nr,anchor_error=er,rep=rep,policy=pol,acc=r['acc_exact'],nmse=r['nmse_exact']))
  print('done',seed,len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False);means=df.groupby(['anchor_error','policy']).mean(numeric_only=True).reset_index();means.to_csv(out/'means.csv',index=False)
 sub=df[df.anchor_error==.10].groupby(['seed','family','noise_ratio','policy']).mean(numeric_only=True).reset_index();piv=sub.pivot_table(index=['seed','family','noise_ratio'],columns='policy',values='acc');d=(piv.repaired_full-piv.reconstructed_local).groupby('seed').mean();primary={'full_vs_local_at_10pct':ci(d.values)};primary['status']='PASS' if primary['full_vs_local_at_10pct']['low']>0 else 'FAIL';(out/'summary.json').write_text(json.dumps({'rows':len(df),'seed_blocks':len(seeds),'primary':primary,'means':means.to_dict('records')},indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(primary,indent=2));print(means[['anchor_error','policy','acc','nmse']].to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
