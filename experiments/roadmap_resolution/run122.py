# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from ising_core import N,STATES,make_edge_params,sample_prior,beliefs
FAMS=('cycle','cactus','grid');BETAS=(.6,.9);POBS=(.15,.30);KTRAIN=(64,512)

def stage122_graph(fam):
 if fam=='cycle': return [(i,(i+1)%N) for i in range(N)],0
 if fam=='cactus':
  e=[]
  for a in (0,3,6,9): e += [(a,a+1),(a+1,a+2),(a+2,a)]
  e += [(2,3),(5,6),(8,9)];return e,5
 if fam=='grid':
  e=[];R,C=3,4
  for r in range(R):
   for c in range(C):
    i=r*C+c
    if r+1<R:e.append((i,(r+1)*C+c))
    if c+1<C:e.append((i,r*C+c+1))
  return e,5
 raise ValueError(fam)
def corrupt_stage122(edges,J,q,seed,gi,beta,pobs,K):
 rng=np.random.default_rng(seed*10000000+gi*1000000+int(beta*100000)+int(pobs*10000)+K);out=J.copy();elig=[i for i,(u,v) in enumerate(edges) if u!=q and v!=q];fl=rng.random(len(elig))<.2
 for f,idx in zip(fl,elig):
  if f:out[idx]*=-1
 return out
def ci(x):
 x=np.asarray(x,float);m=float(x.mean());h=0 if len(x)<2 else float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}
def learn(edges,s):return np.array([np.arctanh(np.clip(np.mean(s[:,u]*s[:,v]),-.95,.95)) for u,v in edges])
def obs_index(obs,q):
 c=[i for i in range(N) if i!=q];return (obs[:,c]>0).astype(np.int64)@(1<<np.arange(len(c),dtype=np.int64))
def exact_map(edges,J,q,p):
 c=[i for i in range(N) if i!=q];m=len(c);n=1<<m;sidx=(STATES[:,c]>0).astype(np.int64)@(1<<np.arange(m,dtype=np.int64));en=np.zeros(len(STATES))
 for (u,v),j in zip(edges,J):en+=j*STATES[:,u]*STATES[:,v]
 en-=en.max();w=np.exp(en);num=np.bincount(sidx,weights=w*STATES[:,q],minlength=n);ids=np.arange(n,dtype=np.uint16);pc=np.array([int(x).bit_count() for x in range(n)],dtype=np.int16);d=pc[ids[:,None]^ids[None,:]];L=((1-p)**(m-d))*(p**d);return np.where(L@num>=0,1,-1).astype(np.int8)
def run_seed(seed,n_eval=1024,kset=KTRAIN):
 rows=[]
 for gi,fam in enumerate(FAMS):
  edges,q=stage122_graph(fam)
  for beta in BETAS:
   J=make_edge_params(edges,beta,np.random.default_rng(seed*10000+gi*1000+int(beta*100)));train=sample_prior(edges,J,np.random.default_rng(seed*100000+gi*10000+int(beta*1000)),max(kset))
   for pobs in POBS:
    er=np.random.default_rng(seed*1000000+gi*100000+int(beta*10000)+int(pobs*1000));sp=sample_prior(edges,J,er,n_eval);obs=sp*np.where(er.random((n_eval,N))<pobs,-1,1).astype(np.int8);oi=obs_index(obs,q);truth=sp[:,q];oracle=float(np.mean(exact_map(edges,J,q,pobs)[oi]==truth))
    for K in kset:
     Jh=learn(edges,train[:K]);lex=float(np.mean(exact_map(edges,Jh,q,pobs)[oi]==truth));b=beliefs(edges,Jh,q,obs,pobs,(2,64));a2=float(np.mean(np.where(b[2]>=0,1,-1)==truth));a64=float(np.mean(np.where(b[64]>=0,1,-1)==truth));Jc=corrupt_stage122(edges,Jh,q,seed,gi,beta,pobs,K);ce=float(np.mean(exact_map(edges,Jc,q,pobs)[oi]==truth));rows.append(dict(seed=seed,family=fam,beta=beta,pobs=pobs,K=K,oracle_true=oracle,learned_exact=lex,learned_bp2=a2,learned_bp64=a64,corrupted_learned_exact=ce,model_gap_pp=100*(oracle-lex),inference_gap64_pp=100*(lex-a64),shallow_gap_pp=100*(a64-a2),relation_error_pp=100*(lex-ce),edge_mae=float(np.mean(np.abs(Jh-J))),edge_sign_acc=float(np.mean(np.sign(Jh)==np.sign(J)))))
 return pd.DataFrame(rows)
def run(out,quick=False,seed=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[seed] if seed is not None else ([132001] if quick else list(range(132001,132011)));ne=256 if quick else 1024;ks=(64,) if quick else KTRAIN;D=pd.concat([run_seed(s,ne,ks) for s in seeds],ignore_index=True);D.to_csv(out/'rows.csv',index=False);summary={'rows':len(D),'by_K':{}}
 for K,z in D.groupby('K'):
  g=z.groupby('seed').mean(numeric_only=True);summary['by_K'][str(K)]={'means':g.mean().to_dict(),**{c:ci(g[c].values) for c in ['model_gap_pp','inference_gap64_pp','shallow_gap_pp','relation_error_pp']}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args();run(a.out,a.quick,a.seed)
