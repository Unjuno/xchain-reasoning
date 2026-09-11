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
def corrupt_all50(a,rng):
 b=a.copy();ed=np.argwhere(np.triu(b!=0,1));ch=rng.permutation(len(ed))[:round(.5*len(ed))]
 for i,j in ed[ch]:b[i,j]*=-1;b[j,i]*=-1
 return b
def spectral_repair(candidate,true_a,q,Y):
 obs=np.delete(np.arange(len(candidate)),q);X=Y-Y.mean(0);S=X.T@X/len(X);vals,vecs=np.linalg.eigh(S);v=vecs[:,-1];gobs=np.where(v>=0,1.,-1.);pos={int(v):i for i,v in enumerate(obs)}
 neigh=np.flatnonzero(true_a[q]!=0);anchor=int(sorted(neigh,key=lambda j:(-abs(true_a[q,j]),int(j)))[0]);anchor_sign=float(np.sign(true_a[q,anchor]));gq=anchor_sign*gobs[pos[anchor]]
 g=np.empty(len(candidate));g[obs]=gobs;g[q]=gq
 repaired=np.abs(candidate)*g[:,None]*g[None,:];np.fill_diagonal(repaired,0)
 ed=np.argwhere(np.triu(true_a!=0,1));allacc=np.mean([np.sign(repaired[i,j])==np.sign(true_a[i,j]) for i,j in ed])
 qedges=ed[np.any(ed==q,axis=1)];qacc=np.mean([np.sign(repaired[i,j])==np.sign(true_a[i,j]) for i,j in qedges])
 return repaired,anchor,allacc,qacc

def run(out,quick=False):
 if out.exists():raise ValueError('output exists')
 cfg=json.loads((HERE/'protocol57.json').read_text());out.mkdir(parents=True);seeds=cfg['seeds'][:1] if quick else cfg['seeds'];fams=cfg['families'][:2] if quick else cfg['families'];K=cfg['calibration_size'];rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed)
   for ni,nr in enumerate(cfg['noise_ratios']):
    p=make_problem(A,q,cfg['tau'],nr);rng=np.random.default_rng(seed*6121+fi*101+ni);z=rng.standard_normal((K,49));e=rng.standard_normal((K,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar'])
    bad=corrupt_all50(A,np.random.default_rng(seed*9311+fi*37+ni))
    repaired,anchor,allacc,qacc=spectral_repair(bad,A,q,Y)
    w_bad=build_candidates(bad,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef
    w_repair=build_candidates(repaired,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef
    w_local_oracle=build_candidates(A,q,cfg['tau'],p['noisevar'],[0.],cfg['steps'])[0].output_coef
    w_true=build_candidates(A,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef
    for pol,w in [('corrupted_full',w_bad),('oracle_local_only',w_local_oracle),('spectral_anchor1',w_repair),('oracle_true_relations',w_true),('exact_bayes',p['bayes_coef'])]:
     r=risks(w,p);rows.append(dict(seed=seed,family=fam,noise_ratio=nr,policy=pol,acc=r['acc_exact'],nmse=r['nmse_exact'],anchor=anchor,edge_sign_accuracy=allacc if pol=='spectral_anchor1' else np.nan,query_edge_sign_accuracy=qacc if pol=='spectral_anchor1' else np.nan))
  print('done',seed,len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False)
 def diff(a,b):
  z=df.pivot_table(index=['seed','family','noise_ratio'],columns='policy',values='acc');d=(z[a]-z[b]).groupby('seed').mean();return ci(d.values)
 primary={'repair_vs_corrupted_full':diff('spectral_anchor1','corrupted_full'),'repair_vs_oracle_local_only':diff('spectral_anchor1','oracle_local_only')};primary['status']='PASS' if primary['repair_vs_corrupted_full']['low']>0 and primary['repair_vs_oracle_local_only']['low']>0 else 'FAIL'
 summary={'rows':len(df),'seed_blocks':len(seeds),'primary':primary,'means':df.groupby('policy').mean(numeric_only=True).reset_index().to_dict('records')}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(primary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
