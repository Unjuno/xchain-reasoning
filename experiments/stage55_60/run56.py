# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'));sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
from core import graph,make_problem,risks
from reliability import build_candidates,diagnostic_scores,corrupt_relations

def ci(x):
 x=np.asarray(x,float);m=float(x.mean());n=len(x)
 h=0 if n<2 else float(student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
 return dict(mean=m,low=m-h,high=m+h,n=n)
def corrupt30(adj,q,rng):
 a=adj.copy();ed=np.argwhere(np.triu(a!=0,1));el=ed[np.all(ed!=q,axis=1)];ch=rng.permutation(len(el))[:round(.3*len(el))]
 for i,j in el[ch]:a[i,j]*=-1;a[j,i]*=-1
 return a
def conservative_repair(candidate,true_a,q,Y,thr):
 a=np.asarray(candidate,float).copy();obs=np.delete(np.arange(len(a)),q);pos={int(v):i for i,v in enumerate(obs)}
 Y=Y-Y.mean(0);ss=np.sqrt(np.sum(Y*Y,0));corr=(Y.T@Y)/np.maximum(ss[:,None]*ss[None,:],1e-300)
 edges=np.argwhere(np.triu(a!=0,1));changed=0;eligible=0;correct=0
 for i,j in edges:
  if i==q or j==q:continue
  eligible+=1;c=float(corr[pos[int(i)],pos[int(j)]]);cand=np.sign(a[i,j])
  if c*cand<0 and abs(c)>=thr:
   a[i,j]*=-1;a[j,i]*=-1;changed+=1
  if np.sign(a[i,j])==np.sign(true_a[i,j]): correct+=1
 return a,changed,eligible,correct/eligible if eligible else 1.0

def run(out,quick=False):
 if out.exists():raise ValueError('output exists')
 cfg=json.loads((HERE/'protocol56.json').read_text());out.mkdir(parents=True)
 seeds=cfg['seeds'][:1] if quick else cfg['seeds'];fams=cfg['families'][:2] if quick else cfg['families'];K=cfg['calibration_size'];rows=[]
 for seed in seeds:
  for fi,fam in enumerate(fams):
   A,q,*_=graph(fam,seed)
   for ni,nr in enumerate(cfg['noise_ratios']):
    p=make_problem(A,q,cfg['tau'],nr);rng=np.random.default_rng(seed*4111+fi*71+ni);z=rng.standard_normal((K,49));e=rng.standard_normal((K,48));theta=z@np.linalg.cholesky(p['Sigma']).T;Y=theta[:,p['obs']]+e*np.sqrt(p['noisevar'])
    for ki,kind in enumerate(cfg['corruptions']):
     crng=np.random.default_rng(seed*8111+fi*101+ni*19+ki);bad=corrupt30(A,q,crng) if kind=='flip30' else corrupt_relations(A,q,kind,crng)
     cand=build_candidates(bad,q,cfg['tau'],p['noisevar'],cfg['gate_candidates'],cfg['steps']);scores=diagnostic_scores(Y,cand)['loo_nlpd'];gi=int(np.argmin(scores.mean(0)))
     repaired,changed,eligible,signacc=conservative_repair(bad,A,q,Y,cfg['absolute_sample_correlation_threshold'])
     rw=build_candidates(repaired,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef;ow=build_candidates(A,q,cfg['tau'],p['noisevar'],[1.],cfg['steps'])[0].output_coef
     for pol,w in [('gate0',cand[0].output_coef),('bad_full',cand[-1].output_coef),('gate_batch128',cand[gi].output_coef),('repair_conservative128',rw),('oracle_true_relations',ow)]:
      r=risks(w,p);rows.append(dict(seed=seed,family=fam,noise_ratio=nr,corruption=kind,policy=pol,acc=r['acc_exact'],nmse=r['nmse_exact'],changed_edges=changed if pol=='repair_conservative128' else np.nan,edge_sign_accuracy=signacc if pol=='repair_conservative128' else np.nan))
  print('done',seed,len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'rows.csv',index=False)
 def diff(kind,a,b,metric='acc'):
  z=df[df.corruption==kind].pivot_table(index=['seed','family','noise_ratio'],columns='policy',values=metric);d=(z[a]-z[b]).groupby('seed').mean();return ci(d.values)
 pr={'flip50_repair_vs_gate0':diff('flip50','repair_conservative128','gate0'),'flip50_repair_vs_gatebatch':diff('flip50','repair_conservative128','gate_batch128'),'intact_repair_vs_full':diff('intact','repair_conservative128','bad_full')}
 pr['status']='PASS' if pr['flip50_repair_vs_gate0']['low']>0 and pr['flip50_repair_vs_gatebatch']['low']>0 and pr['intact_repair_vs_full']['low']>-.005 else 'FAIL'
 summary={'rows':len(df),'seed_blocks':len(seeds),'primary':pr,'means':df.groupby(['corruption','policy']).mean(numeric_only=True).reset_index().to_dict('records')}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(pr,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.out,a.quick)
