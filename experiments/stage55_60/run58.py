# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'));sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
from core import graph,make_problem,risks
from reliability import build_candidates

def reconstruct(A,q,Y):
 obs=np.delete(np.arange(len(A)),q);X=Y-Y.mean(0);S=X.T@X/len(X);_,V=np.linalg.eigh(S);gobs=np.where(V[:,-1]>=0,1.,-1.);pos={int(v):i for i,v in enumerate(obs)}
 neigh=np.flatnonzero(A[q]!=0);anchor=int(sorted(neigh,key=lambda j:(-abs(A[q,j]),int(j)))[0]);gq=float(np.sign(A[q,anchor]))*gobs[pos[anchor]]
 g=np.empty(len(A));g[obs]=gobs;g[q]=gq;R=np.abs(A)*g[:,None]*g[None,:];np.fill_diagonal(R,0)
 ed=np.argwhere(np.triu(A!=0,1));acc=np.mean([np.sign(R[i,j])==np.sign(A[i,j]) for i,j in ed])
 return R,acc

def run(out,quick=False):
 if out.exists():raise ValueError('output exists')
 cfg=json.loads((HERE/'protocol58.json').read_text());out.mkdir(parents=True);seeds=cfg['seeds'][:1] if quick else cfg['seeds'];fams=cfg['families'][:2] if quick else cfg['families'];ks=cfg['calibration_sizes'][:3] if quick else cfg['calibration_sizes'];rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed)
   for ni,nr in enumerate(cfg['noise_ratios']):
    p=make_problem(A,q,cfg['tau'],nr);rng=np.random.default_rng(seed*8191+fi*97+ni);N=max(ks);z=rng.standard_normal((N,49));e=rng.standard_normal((N,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar'])
    local=risks(build_candidates(A,q,cfg['tau'],p['noisevar'],[0.],cfg['steps'])[0].output_coef,p);truth16=risks(build_candidates(A,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef,p)
    for K in ks:
     R,signacc=reconstruct(A,q,Y[:K]);w=build_candidates(R,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef;r=risks(w,p)
     rows.append(dict(seed=seed,family=fam,noise_ratio=nr,K=K,acc=r['acc_exact'],nmse=r['nmse_exact'],edge_sign_accuracy=signacc,oracle_local_acc=local['acc_exact'],oracle_true16_acc=truth16['acc_exact'],acc_gain_vs_local=r['acc_exact']-local['acc_exact'],gap_to_true16=truth16['acc_exact']-r['acc_exact']))
  print('done',seed,len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False);means=df.groupby('K').mean(numeric_only=True).reset_index();means.to_csv(out/'means.csv',index=False)
 (out/'summary.json').write_text(json.dumps({'rows':len(df),'seed_blocks':len(seeds),'means':means.to_dict('records')},indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');print(means[['K','acc','edge_sign_accuracy','acc_gain_vs_local','gap_to_true16']].to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
