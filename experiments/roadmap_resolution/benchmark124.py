# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from ising_core import N,graph,make_edge_params,sample_prior,corrupt_exterior,init_bp,sweep,belief
FAMS=('barbell','tri_ladder','torus3x4','cycle','cactus','ladder','grid')
def ci(x):
 x=np.asarray(x,float);m=float(x.mean());h=0 if len(x)<2 else float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}
def setup(seed,fam,B):
 edges,q=graph(fam);gi=FAMS.index(fam);J=make_edge_params(edges,.85,np.random.default_rng(seed*10000+gi*1000+85));tr=np.random.default_rng(seed*1000000+gi*100000+285);J=J*np.where(tr.random(len(edges))<.2,-1.,1.);er=np.random.default_rng(seed*100000+gi*10000+2850);sp=sample_prior(edges,J,er,B);obs=sp*np.where(er.random((B,N))<.2,-1,1).astype(np.int8);Jinf=corrupt_exterior(edges,J,q,.3,np.random.default_rng(seed*100000000+gi*10000000+87320));return edges,q,Jinf,obs,.2,sp[:,q]
def adaptive(edges,J,q,obs,pobs,truth,mode,rng=None):
 src,dst,jc,rev,h,inc,qin=init_bp(edges,J,q,obs,pobs);msg=np.zeros((len(src),len(obs)));msg=sweep(msg,h,src,dst,jc,rev,inc);b1=belief(msg,h,q,qin);msg=sweep(msg,h,src,dst,jc,rev,inc);b2=belief(msg,h,q,qin);p1=np.where(b1>=0,1,-1);p2=np.where(b2>=0,1,-1);k=int(np.sum(p1!=p2))
 if mode=='flip':sel=np.flatnonzero(p1!=p2)
 elif mode=='uncertainty':sel=np.argsort(np.abs(b2))[:k]
 else:sel=np.sort(rng.choice(len(obs),k,replace=False)) if k else np.array([],int)
 out=p2.copy()
 if k:
  hs=h[sel];ms=msg[:,sel].copy()
  for _ in range(62):ms=sweep(ms,hs,src,dst,jc,rev,inc)
  out[sel]=np.where(belief(ms,hs,q,qin)>=0,1,-1)
 return float(np.mean(out==truth)),k/len(obs),2+62*k/len(obs)
def fixed(edges,J,q,obs,pobs,truth,d=12):
 src,dst,jc,rev,h,inc,qin=init_bp(edges,J,q,obs,pobs);msg=np.zeros((len(src),len(obs)))
 for _ in range(d):msg=sweep(msg,h,src,dst,jc,rev,inc)
 return float(np.mean(np.where(belief(msg,h,q,qin)>=0,1,-1)==truth))
def run(out,quick=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[134001] if quick else list(range(134001,134011));fams=FAMS[:2] if quick else FAMS;B=512 if quick else 2048;reps=1 if quick else 3;rows=[]
 for seed in seeds:
  for fam in fams:
   e,q,J,o,p,t=setup(seed,fam,B);af,fr,av=adaptive(e,J,q,o,p,t,'flip');au,_,_=adaptive(e,J,q,o,p,t,'uncertainty');ar,_,_=adaptive(e,J,q,o,p,t,'random',np.random.default_rng(seed));a12=fixed(e,J,q,o,p,t);times={m:[] for m in ('flip','uncertainty','random','fixed12')}
   for rep in range(reps):
    order=list(times);random.Random(seed+rep).shuffle(order)
    for m in order:
     ts=time.perf_counter_ns();fixed(e,J,q,o,p,t) if m=='fixed12' else adaptive(e,J,q,o,p,t,m,np.random.default_rng(seed+rep) if m=='random' else None);times[m].append((time.perf_counter_ns()-ts)/1e6)
   row=dict(seed=seed,family=fam,deep_fraction=fr,avg_depth=av,flip_acc=af,uncertainty_acc=au,random_acc=ar,fixed12_acc=a12);row.update({m+'_ms':float(np.median(v)) for m,v in times.items()});rows.append(row)
 D=pd.DataFrame(rows);D.to_csv(out/'rows.csv',index=False);g=D.groupby('seed').agg({**{c:'mean' for c in ['deep_fraction','avg_depth','flip_acc','uncertainty_acc','random_acc','fixed12_acc']},**{c:'sum' for c in ['flip_ms','uncertainty_ms','random_ms','fixed12_ms']}});summary={'means':g.mean().to_dict(),'flip_minus_random_pp':ci(100*(g.flip_acc-g.random_acc)),'flip_minus_uncertainty_pp':ci(100*(g.flip_acc-g.uncertainty_acc)),'flip_minus_fixed12_pp':ci(100*(g.flip_acc-g.fixed12_acc)),'runtime_ratio_flip_fixed_mean':float(np.mean(g.flip_ms/g.fixed12_ms))};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
