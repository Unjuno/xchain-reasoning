# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'stage49_51'))
from core import graph,make_problem

def build_candidate(a,q,tau,noise,steps=16):
 n=len(a);obs=np.delete(np.arange(n),q);diag=tau+np.abs(a).sum(1);prec=np.zeros(n);prec[obs]=1/noise;pd=diag+prec;F=np.zeros((n,n-1));F[obs,np.arange(n-1)]=prec[obs]/pd[obs];prior=np.diag(diag)-a;sigma=cho_solve(cho_factor(prior,lower=True),np.eye(n));cov=sigma[np.ix_(obs,obs)]+np.diag(noise);transition=a/pd[:,None];coef=np.zeros_like(F)
 for _ in range(steps):coef=transition@coef+F
 return coef[q].copy()
def repair(candidate,true_a,q,Y):
 obs=np.delete(np.arange(len(candidate)),q);X=Y-Y.mean(0);S=X.T@X/len(X);_,V=np.linalg.eigh(S);gobs=np.where(V[:,-1]>=0,1.,-1.);pos={int(v):i for i,v in enumerate(obs)};neigh=np.flatnonzero(true_a[q]!=0);anchor=int(sorted(neigh,key=lambda j:(-abs(true_a[q,j]),int(j)))[0]);gq=np.sign(true_a[q,anchor])*gobs[pos[anchor]];g=np.empty(len(candidate));g[obs]=gobs;g[q]=gq;R=np.abs(candidate)*g[:,None]*g[None,:];np.fill_diagonal(R,0);return R,S
def corrupt(a,rng):
 b=a.copy();ed=np.argwhere(np.triu(b!=0,1));ch=rng.permutation(len(ed))[:round(.5*len(ed))]
 for i,j in ed[ch]:b[i,j]*=-1;b[j,i]*=-1
 return b
def medtime(fn,n):
 x=[];r=None
 for _ in range(n):t=time.perf_counter_ns();r=fn();x.append((time.perf_counter_ns()-t)/1e6)
 return float(np.median(x)),r
def run(out,quick=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[135001] if quick else list(range(135001,135011));fams=['grid'] if quick else ['chain','cross','tree','grid','grid_rewired'];rep=3 if quick else 11;rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed);p=make_problem(A,q,.05,1.);K=512;rng=np.random.default_rng(seed*17+fi);z=rng.standard_normal((K,49));e=rng.standard_normal((K,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar']);bad=corrupt(A,np.random.default_rng(seed*31+fi));tr,(R,S)=medtime(lambda:repair(bad,A,q,Y),rep);tc,w=medtime(lambda:build_candidate(R,q,.05,p['noisevar']),rep);ta,_=medtime(lambda:float(Y[0]@w),max(11,rep));rows.append(dict(seed=seed,family=fam,repair_ms=tr,coef_build_ms=tc,online_apply_us=ta*1000,calibration_fields=K,calibration_bytes=Y.nbytes,covariance_bytes=S.nbytes,adjacency_bytes=R.nbytes,coef_bytes=w.nbytes))
 D=pd.DataFrame(rows);D.to_csv(out/'rows.csv',index=False);med=D.median(numeric_only=True).to_dict();med['setup_to_apply_ratio']=float((med['repair_ms']+med['coef_build_ms'])*1000/med['online_apply_us']);(out/'summary.json').write_text(json.dumps(med,indent=2)+'\n');print(json.dumps(med,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
