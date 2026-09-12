# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,sys
from collections import deque
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from ising_core import N,graph,make_edge_params,sample_prior,beliefs,mismatch_scores,corrupt_exterior
FAMS=('barbell','tri_ladder','torus3x4','grid');POBS=(.18,.32);TRUE_FLIPS=(0.,.3);DRIFTS=(11,37);BETA=.85;CORR_LEVELS=(0.,.15,.35,.50);HIST_MAX=16;HIST_GOOD=.35;RESET=.45

def ci(x):
 x=np.asarray(x,float);m=float(x.mean());h=0 if len(x)<2 else float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}
def acc(actions,preds,truth):
 out=np.empty(len(truth),np.int8)
 for d in np.unique(actions):out[actions==d]=preds[int(d)][actions==d]
 return float(np.mean(out==truth))
def shuffle_actions(a,rng,groups=None):
 out=np.empty_like(a)
 if groups is None:out=a.copy();rng.shuffle(out);return out
 for g in np.unique(groups):
  ix=np.where(groups==g)[0];v=a[ix].copy();rng.shuffle(v);out[ix]=v
 return out

def run_seed(seed,T=192):
 rows=[]
 for gi,fam in enumerate(FAMS):
  edges,q=graph(fam);rng=np.random.default_rng(seed*10000+gi*1000+85);Jmag=make_edge_params(edges,BETA,rng)
  for tf in TRUE_FLIPS:
   tr=np.random.default_rng(seed*1000000+gi*100000+int(tf*1000)+85);Jtrue=Jmag*np.where(tr.random(len(edges))<tf,-1.,1.)
   for pobs in POBS:
    for drift in DRIFTS:
     rr=np.random.default_rng(seed*10000000+gi*1000000+int(tf*10000)+int(pobs*1000)+drift);sp=sample_prior(edges,Jtrue,rr,T);obs=sp*np.where(rr.random((T,N))<pobs,-1,1).astype(np.int8);nseg=(T+drift-1)//drift;corrseq=[CORR_LEVELS[(j+seed+gi)%4] for j in range(nseg)];bb={d:np.zeros(T) for d in (1,2,3,4,64)};mis=np.zeros(T);seg=np.zeros(T,int)
     for sj,corr in enumerate(corrseq):
      lo=sj*drift;hi=min(T,(sj+1)*drift);cr=np.random.default_rng(seed*100000000+gi*10000000+int(tf*100000)+int(pobs*10000)+drift*100+sj);Jinf=corrupt_exterior(edges,Jtrue,q,corr,cr);b=beliefs(edges,Jinf,q,obs[lo:hi],pobs,(1,2,3,4,64))
      for d in bb:bb[d][lo:hi]=b[d]
      mis[lo:hi]=mismatch_scores(edges,Jinf,q,obs[lo:hi]);seg[lo:hi]=sj
     pred={d:np.where(x>=0,1,-1) for d,x in bb.items()};truth=sp[:,q];hist=deque(maxlen=HIST_MAX);actions=np.ones(T,int);good_arr=np.zeros(T,bool);reset_arr=np.zeros(T,bool)
     for i in range(T):
      if mis[i]>RESET:
       hist.clear();reset_arr[i]=True
      good=len(hist)>0 and float(np.mean(hist))<=HIST_GOOD;good_arr[i]=good
      actions[i]=64 if good and pred[1][i]!=pred[2][i] else (2 if good else 1);hist.append(float(mis[i]))
     gr=shuffle_actions(actions,np.random.default_rng(seed*99991+gi*1009+drift*17+int(pobs*100)+int(tf*1000)));sr=shuffle_actions(actions,np.random.default_rng(seed*999983+gi*10007+drift*19+int(pobs*100)+int(tf*1000)),seg);unc=np.where(actions==1,1,2);cand=np.where(actions!=1)[0];k=int(np.sum(actions==64))
     if k:unc[cand[np.argsort(np.abs(bb[2][cand]))[:k]]]=64
     post=np.zeros(T,bool);post[np.arange(drift,T,drift)]=True
     chosen=np.array([pred[int(a)][i] for i,a in enumerate(actions)],dtype=np.int8)
     fixed={d:float(np.mean(pred[d]==truth)) for d in (1,2,3,4,64)}
     rows.append(dict(seed=seed,family=fam,beta=BETA,pobs=pobs,true_flip=tf,drift=drift,policy_acc=acc(actions,pred,truth),global_random=acc(gr,pred,truth),segment_random=acc(sr,pred,truth),uncertainty_matched=acc(unc,pred,truth),avg_depth=float(actions.mean()),frac_d1=float(np.mean(actions==1)),frac_d2=float(np.mean(actions==2)),frac_d64=float(np.mean(actions==64)),reset_rate=float(np.mean(reset_arr)),good_rate=float(np.mean(good_arr)),postchange_policy=float(np.mean(chosen[post]==truth[post])) if post.any() else np.nan,postchange_d1=float(np.mean(pred[1][post]==truth[post])) if post.any() else np.nan,**{f'acc{d}':fixed[d] for d in fixed}))
 return pd.DataFrame(rows)
def run(out,quick=False,seed=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[seed] if seed is not None else ([130001] if quick else list(range(130001,130011)));T=64 if quick else 192;D=pd.concat([run_seed(s,T) for s in seeds],ignore_index=True);D.to_csv(out/'rows.csv',index=False);g=D.groupby('seed').mean(numeric_only=True);summary={'rows':len(D),'means':g.mean().to_dict(),'policy_minus_global_random_pp':ci(100*(g.policy_acc-g.global_random)),'policy_minus_segment_random_pp':ci(100*(g.policy_acc-g.segment_random)),'policy_minus_uncertainty_pp':ci(100*(g.policy_acc-g.uncertainty_matched))};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args();run(a.out,a.quick,a.seed)
