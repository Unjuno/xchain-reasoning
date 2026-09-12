# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from ising_core import N,graph,make_edge_params,sample_prior,beliefs,corrupt_exterior
FAMILIES=('barbell','tri_ladder','torus3x4','cycle','cactus','ladder','grid');BETAS=(.75,.95);POBS=(.12,.28);TRUE_FLIPS=(0.,.2,.4);CORR=(0.,.3,.5)

def ci(x):
 x=np.asarray(x,float);m=float(x.mean());h=0 if len(x)<2 else float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}

def run_seed(seed,n_samples=1024):
 rows=[]
 for gi,fam in enumerate(FAMILIES):
  edges,q=graph(fam)
  for beta in BETAS:
   rng=np.random.default_rng(seed*10000+gi*1000+int(beta*100));Jmag=make_edge_params(edges,beta,rng)
   for tf in TRUE_FLIPS:
    srng=np.random.default_rng(seed*1000000+gi*100000+int(beta*10000)+int(tf*1000));J=Jmag*np.where(srng.random(len(edges))<tf,-1.,1.)
    for pobs in POBS:
     er=np.random.default_rng(seed*100000+gi*10000+int(beta*1000)+int(tf*100)+int(pobs*10000));sp=sample_prior(edges,J,er,n_samples);obs=sp*np.where(er.random((n_samples,N))<pobs,-1,1).astype(np.int8)
     for corr in CORR:
      qr=np.random.default_rng(seed*100000000+gi*10000000+int(beta*100000)+int(tf*10000)+int(corr*1000)+int(pobs*100));Jinf=corrupt_exterior(edges,J,q,corr,qr);b=beliefs(edges,Jinf,q,obs,pobs,(1,2,64));pred={d:np.where(x>=0,1,-1) for d,x in b.items()};truth=sp[:,q];deep=pred[1]!=pred[2];k=int(deep.sum());frac=k/n_samples;flip_pred=np.where(deep,pred[64],pred[2]);unc=pred[2].copy()
      if k:
       ix=np.argsort(np.abs(b[2]))[:k];unc[ix]=pred[64][ix]
      acc2=float(np.mean(pred[2]==truth));acc64=float(np.mean(pred[64]==truth));rows.append(dict(seed=seed,family=fam,beta=beta,pobs=pobs,true_flip=tf,corruption=corr,deep_fraction=frac,avg_depth=2+62*frac,flip_acc=float(np.mean(flip_pred==truth)),uncertainty_acc=float(np.mean(unc==truth)),condition_random=(1-frac)*acc2+frac*acc64,acc2=acc2,acc64=acc64))
 return pd.DataFrame(rows)

def run(out,quick=False,seed=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[seed] if seed is not None else ([129001] if quick else list(range(129001,129011)));ns=256 if quick else 1024;D=pd.concat([run_seed(s,ns) for s in seeds],ignore_index=True);D.to_csv(out/'rows.csv',index=False);g=D.groupby('seed').mean(numeric_only=True);summary={'rows':len(D),'seed_blocks':len(g),'means':g.mean().to_dict(),'flip_minus_uncertainty_pp':ci(100*(g.flip_acc-g.uncertainty_acc)),'flip_minus_random_pp':ci(100*(g.flip_acc-g.condition_random))};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args();run(a.out,a.quick,a.seed)
