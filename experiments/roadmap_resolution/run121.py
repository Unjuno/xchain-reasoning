# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,sys
from collections import deque
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from ising_core import N,graph,make_edge_params,sample_prior,beliefs,mismatch_scores,corrupt_exterior
FAMS=('barbell','tri_ladder','torus3x4','grid');POBS=(.14,.26);TRUE_FLIPS=(.1,.35);DRIFTS=(7,23);BETA=.82;CORR_LEVELS=(0.,.2,.4,.55);HIST_MAX=16;GOOD=.35;RESET=.45

def ci(x):
 x=np.asarray(x,float);m=float(x.mean());h=0 if len(x)<2 else float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}
def actions(mis,pred,do_reset):
 hist=deque(maxlen=HIST_MAX);a=np.ones(len(mis),int);resets=np.zeros(len(mis),bool)
 for i in range(len(mis)):
  if do_reset and mis[i]>RESET:
   hist.clear();resets[i]=True
  good=len(hist)>0 and float(np.mean(hist))<=GOOD;a[i]=64 if good and pred[1][i]!=pred[2][i] else (2 if good else 1);hist.append(float(mis[i]))
 return a,resets
def acc(a,pred,truth):
 out=np.empty(len(truth),np.int8)
 for d in np.unique(a):out[a==d]=pred[int(d)][a==d]
 return float(np.mean(out==truth))
def run_seed(seed,T=184):
 rows=[]
 for gi,fam in enumerate(FAMS):
  edges,q=graph(fam);rng=np.random.default_rng(seed*10000+gi*1000+82);Jmag=make_edge_params(edges,BETA,rng)
  for tf in TRUE_FLIPS:
   tr=np.random.default_rng(seed*1000000+gi*100000+int(tf*1000)+82);Jtrue=Jmag*np.where(tr.random(len(edges))<tf,-1.,1.)
   for pobs in POBS:
    for drift in DRIFTS:
     rr=np.random.default_rng(seed*10000000+gi*1000000+int(tf*10000)+int(pobs*1000)+drift);sp=sample_prior(edges,Jtrue,rr,T);obs=sp*np.where(rr.random((T,N))<pobs,-1,1).astype(np.int8);nseg=(T+drift-1)//drift;bb={d:np.zeros(T) for d in (1,2,64)};mis=np.zeros(T)
     for sj in range(nseg):
      corr=CORR_LEVELS[(2*sj+seed+gi)%4];lo=sj*drift;hi=min(T,(sj+1)*drift);cr=np.random.default_rng(seed*100000000+gi*10000000+int(tf*100000)+int(pobs*10000)+drift*100+sj);Jinf=corrupt_exterior(edges,Jtrue,q,corr,cr);b=beliefs(edges,Jinf,q,obs[lo:hi],pobs,(1,2,64))
      for d in bb:bb[d][lo:hi]=b[d]
      mis[lo:hi]=mismatch_scores(edges,Jinf,q,obs[lo:hi])
     pred={d:np.where(x>=0,1,-1) for d,x in bb.items()};truth=sp[:,q];ar,rs=actions(mis,pred,True);an,rn=actions(mis,pred,False);rg=ar.copy();np.random.default_rng(seed*91919+gi*991+drift*17+int(pobs*1000)+int(tf*10000)).shuffle(rg)
     post=np.zeros(T,bool);post[np.arange(drift,T,drift)]=True
     def subacc(a,mask):
      if not mask.any():return np.nan
      out=np.empty(mask.sum(),np.int8);inds=np.where(mask)[0]
      for d in np.unique(a[mask]):out[a[mask]==d]=pred[int(d)][inds[a[mask]==d]]
      return float(np.mean(out==truth[mask]))
     rows.append(dict(seed=seed,family=fam,beta=BETA,pobs=pobs,true_flip=tf,drift=drift,reset_acc=acc(ar,pred,truth),noreset_acc=acc(an,pred,truth),global_random=acc(rg,pred,truth),reset_depth=float(ar.mean()),noreset_depth=float(an.mean()),reset_rate=float(rs.mean()),post_reset_acc=subacc(ar,post),post_noreset_acc=subacc(an,post),post_d1_acc=float(np.mean(pred[1][post]==truth[post])) if post.any() else np.nan))
 return pd.DataFrame(rows)
def run(out,quick=False,seed=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[seed] if seed is not None else ([131001] if quick else list(range(131001,131011)));T=64 if quick else 184;D=pd.concat([run_seed(s,T) for s in seeds],ignore_index=True);D.to_csv(out/'rows.csv',index=False);g=D.groupby('seed').mean(numeric_only=True);summary={'rows':len(D),'means':g.mean().to_dict(),'reset_minus_noreset_pp':ci(100*(g.reset_acc-g.noreset_acc)),'reset_minus_random_pp':ci(100*(g.reset_acc-g.global_random))};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args();run(a.out,a.quick,a.seed)
