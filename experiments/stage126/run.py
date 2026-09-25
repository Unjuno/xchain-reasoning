# SPDX-License-Identifier: Apache-2.0
"""Frozen Stage 126 runner. Truth never enters a prediction function.

Run from the repository root: python experiments/stage126/run.py --out runs/stage126
Requires g++ and the repository's pinned Python requirements. No network in this script.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'roadmap_resolution'))
import ising_core as ref


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign(x):
    return np.where(np.asarray(x) >= 0, 1, -1).astype(np.int8)


def ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0
    k = float(student_t.ppf(.975,len(x)-1)) if len(x)>1 else 0.0
    return dict(mean=mean, low=mean-k*se, high=mean+k*se, standard_error=se,
                coverage_factor=k, seed_blocks=len(x))


class Compiled:
    def __init__(self, out):
        self.library_path = out/'kernel.so'
        self.command = ['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off',
                        '-fPIC','-shared',str(HERE/'kernel.cpp'),'-o',str(self.library_path)]
        start = time.perf_counter()
        subprocess.run(self.command, check=True, capture_output=True, text=True)
        self.compile_seconds = time.perf_counter()-start
        self.lib = ctypes.CDLL(str(self.library_path.resolve()))
        self.fn = self.lib.xchain_bp
        i1 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
        d1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        d2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        self.fn.argtypes = [ctypes.c_int]*4+[i1]*3+[d1,d2]+[ctypes.c_int]*2+[ctypes.c_ulonglong]+[d1]*3+[i1]
        self.fn.restype = ctypes.c_int

    def predict(self, edges, J, query, observations, pobs, depth, mode=0, random_seed=0):
        obs = np.asarray(observations)
        if obs.ndim != 2 or obs.shape[1] != ref.N or len(obs)==0:
            raise ValueError('Invalid observation shape')
        if not 0 < pobs < .5 or not 0 <= query < ref.N or not 2 <= depth <= 256:
            raise ValueError('Invalid inference parameters')
        src,dst,jc,rev = ref.directed_graph(edges,J)
        src,dst,rev = [np.ascontiguousarray(a,dtype=np.int32) for a in (src,dst,rev)]
        jc = np.ascontiguousarray(jc,dtype=np.float64)
        fields = np.ascontiguousarray(.5*math.log((1-pobs)/pobs)*obs,dtype=np.float64)
        fields[:,query] = 0
        if not np.isfinite(fields).all(): raise ValueError('Nonfinite input')
        b = len(obs)
        out,b1,b2 = [np.empty(b,dtype=np.float64) for _ in range(3)]
        mask = np.empty(b,dtype=np.int32)
        status = self.fn(ref.N,len(src),b,query,src,dst,rev,jc,fields,int(depth),int(mode),
                         int(random_seed),out,b1,b2,mask)
        if status: raise RuntimeError(f'Compiled BP returned {status}')
        return out,b1,b2,mask.astype(bool)


def numpy_predict(edges,J,q,obs,pobs,depth,adaptive=False):
    src,dst,jc,rev,h,inc,qin = ref.init_bp(edges,J,q,obs,pobs)
    msg = np.zeros((len(src),len(obs)))
    msg = ref.sweep(msg,h,src,dst,jc,rev,inc)
    b1 = ref.belief(msg,h,q,qin)
    msg = ref.sweep(msg,h,src,dst,jc,rev,inc)
    b2 = ref.belief(msg,h,q,qin)
    mask = sign(b1)!=sign(b2)
    if adaptive:
        out = b2.copy()
        if mask.any():
            hs = h[mask]; ms = msg[:,mask].copy()
            for _ in range(2,depth): ms=ref.sweep(ms,hs,src,dst,jc,rev,inc)
            out[mask]=ref.belief(ms,hs,q,qin)
    else:
        for _ in range(2,depth): msg=ref.sweep(msg,h,src,dst,jc,rev,inc)
        out=ref.belief(msg,h,q,qin)
    return out,b1,b2,mask


def conditional_mean(edges,J,q,obs,pobs):
    """Exact truth scoring by all 4096 spins and a product BSC convolution.
    Scoring only: this output is never passed to a policy.
    """
    cols=np.delete(np.arange(ref.N),q); m=len(cols); states=ref.STATES
    index=(states[:,cols]>0).astype(np.int64)@(1<<np.arange(m,dtype=np.int64))
    energy=np.zeros(len(states))
    for (u,v),j in zip(edges,J): energy+=j*states[:,u]*states[:,v]
    weights=np.exp(energy-energy.max());weights/=weights.sum()
    den=np.bincount(index,weights=weights,minlength=1<<m)
    num=np.bincount(index,weights=weights*states[:,q],minlength=1<<m)
    def channel(a):
        a=a.copy()
        for k in range(m):
            block=a.reshape(-1,2,1<<k)
            left=block[:,0,:].copy();right=block[:,1,:].copy()
            block[:,0,:]=(1-pobs)*left+pobs*right
            block[:,1,:]=pobs*left+(1-pobs)*right
        return a
    den=channel(den);num=channel(num)
    assert np.all(den>0) and abs(den.sum()-1)<1e-12
    oi=(obs[:,cols]>0).astype(np.int64)@(1<<np.arange(m,dtype=np.int64))
    return np.clip(num[oi]/den[oi],-1,1)


def fixed_frontier(times,accuracy,budget):
    """Hindsight upper envelope allowing ANY mixture of <=2 fixed policies.
    Budget is a measured per-batch time. Spending all of it is not required.
    """
    times=np.asarray(times,float);accuracy=np.asarray(accuracy,float)
    best=-np.inf
    for i,ti in enumerate(times):
        if ti<=budget: best=max(best,float(accuracy[i]))
        for j,tj in enumerate(times):
            if ti<budget<tj:
                weight=(budget-ti)/(tj-ti)
                best=max(best,float((1-weight)*accuracy[i]+weight*accuracy[j]))
    return float(best) if np.isfinite(best) else None


def generate(seed,gi,family,batch,cfg):
    edges,q=ref.graph(family)
    streams=[np.random.default_rng(np.random.SeedSequence([seed,gi,j])) for j in range(4)]
    J=ref.make_edge_params(edges,cfg['beta'],streams[0])
    J*=np.where(streams[1].random(len(edges))<cfg['intrinsic_negative_edge_probability'],-1.,1.)
    spins=ref.sample_prior(edges,J,streams[2],batch)
    obs=spins*np.where(streams[2].random((batch,ref.N))<cfg['observation_flip_probability'],-1,1).astype(np.int8)
    obs[:,q]=0
    bad=ref.corrupt_exterior(edges,J,q,cfg['exterior_corruption_probability'],streams[3])
    return edges,q,J,bad,obs,spins[:,q]


def self_test(engine):
    rng=np.random.default_rng(20260925)
    edges,q=ref.graph('grid');J=rng.normal(0,.3,len(edges))
    obs=rng.choice([-1,1],size=(19,ref.N)).astype(np.int8);obs[:,q]=0
    maximum=0.0
    for d in (2,4,12,64):
        a=engine.predict(edges,J,q,obs,.2,d);b=numpy_predict(edges,J,q,obs,.2,d)
        maximum=max(maximum,float(np.max(abs(a[0]-b[0]))))
        np.testing.assert_allclose(a[0],b[0],atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(sign(a[0]),sign(b[0]))
    deep=engine.predict(edges,J,q,obs,.2,64)[0]
    shallow=engine.predict(edges,J,q,obs,.2,2)[0]
    for mode in (1,2,3):
        a=engine.predict(edges,J,q,obs,.2,64,mode,42)
        np.testing.assert_allclose(a[0],np.where(a[3],deep,shallow),atol=1e-10,rtol=1e-10)
        assert int(a[3].sum())==int((sign(a[1])!=sign(a[2])).sum())
        changed=obs.copy();changed[:,q]=1
        z=engine.predict(edges,J,q,changed,.2,64,mode,42)
        np.testing.assert_array_equal(a[0],z[0])
    assert abs(fixed_frontier([1,3],[.5,.9],2)-.7)<1e-12
    assert fixed_frontier([1,2],[.9,.6],3)==.9
    # Independent direct enumeration check of the convolution scorer.
    mu=conditional_mean(edges,J,q,obs,.2)
    energy=np.array([sum(j*s[u]*s[v] for (u,v),j in zip(edges,J)) for s in ref.STATES])
    prior=np.exp(energy-energy.max());prior/=prior.sum()
    cols=np.delete(np.arange(ref.N),q)
    for z,y in zip(mu,obs):
        d=np.sum(ref.STATES[:,cols]!=y[cols],axis=1)
        w=prior*(.2**d)*(.8**(len(cols)-d))
        assert abs(z-np.dot(w,ref.STATES[:,q])/w.sum())<1e-12
    return dict(status='passed',max_fixed_belief_difference=maximum,
                checks=['fixed depth equivalence','same-count adaptive continuation',
                        'query observation masked','frontier interpolation','exact scorer enumeration'])


def summarize(rows,cfg,audit):
    df=pd.DataFrame(rows);result={'batches':{},'audit':audit}
    for batch,z in df.groupby('batch'):
        def contrast(a,b):
            wide=z.pivot(index=['seed','family'],columns='policy',values='accuracy_expected')
            return ci((100*(wide[a]-wide[b])).groupby('seed').mean().values)
        f=z[z.policy=='cpp_flip'].copy()
        available=f.frontier_accuracy.notna().all()
        front=ci((100*(f.accuracy_expected-f.frontier_accuracy)).groupby(f.seed).mean().values) if available else None
        wide=z.pivot(index=['seed','family'],columns='policy',values='median_seconds')
        ratio=(wide['cpp_flip']/wide['numpy_flip']).groupby('seed').apply(lambda a:float(np.log(a).mean()))
        r=ci(ratio.values)
        item={'frontier_gain_pp':front,'all_frontiers_feasible':bool(available),
              'flip_minus_uncertainty_pp':contrast('cpp_flip','cpp_uncertainty'),
              'flip_minus_random_pp':contrast('cpp_flip','cpp_random'),
              'compiled_to_numpy_flip_time_ratio':{k:math.exp(r[k]) for k in ('mean','low','high')},
              'means':z.groupby('policy')[['accuracy_expected','accuracy_sampled','median_seconds','nominal_depth']].mean().to_dict('index'),
              'family_frontier_gain_pp':f.groupby('family').apply(lambda a:float(100*(a.accuracy_expected-a.frontier_accuracy).mean()),include_groups=False).to_dict()}
        item['status']='PASS' if available and front['mean']>=.5 and front['low']>0 and audit['decision_mismatches']==0 and audit['selection_mismatches']==0 else 'FAIL_OR_UNCERTAIN'
        result['batches'][str(batch)]=item
    result['primary_batch']=cfg['primary_batch']
    return result


def run(out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text())
    seeds=cfg['seeds'][:1] if quick else cfg['seeds']
    families=cfg['families'][:2] if quick else cfg['families']
    batches=[32] if quick else cfg['batch_sizes']
    repetitions=1 if quick else cfg['timing_repetitions']
    frozen={p.name:sha(p) for p in (HERE/'protocol.json',HERE/'kernel.cpp',HERE/'run.py')}
    engine=Compiled(out)
    audit=self_test(engine)
    audit.update(decision_mismatches=0,selection_mismatches=0,max_belief_difference=0.,conditions=0)
    cpu='unknown'
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu=next((s.split(':',1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith('model name')),'unknown')
    environment={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                 'platform':platform.platform(),'cpu':cpu,'threads':1,'clock_pinned':False,
                 'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
                 'compile_flags':engine.command[1:7],'compile_seconds':engine.compile_seconds,
                 'source_sha256':frozen,'base_core_sha256':sha(HERE.parent/'roadmap_resolution'/'ising_core.py')}
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    rows=[];timing_rows=[]
    for seed in seeds:
        for gi,family in enumerate(families):
            for batch in batches:
                edges,q,J,bad,obs,truth=generate(seed,gi,family,batch,cfg)
                p=cfg['observation_flip_probability'];mu=conditional_mean(edges,J,q,obs,p)
                calls={}
                for d in cfg['fixed_depths']:
                    calls[f'cpp_fixed{d}']=lambda d=d:engine.predict(edges,bad,q,obs,p,d)
                    calls[f'numpy_fixed{d}']=lambda d=d:numpy_predict(edges,bad,q,obs,p,d)
                for label,mode in [('flip',1),('uncertainty',2),('random',3)]:
                    calls['cpp_'+label]=lambda mode=mode:engine.predict(edges,bad,q,obs,p,64,mode,seed+gi)
                calls['numpy_flip']=lambda:numpy_predict(edges,bad,q,obs,p,64,True)
                predictions={name:fn() for name,fn in calls.items()}
                for d in cfg['fixed_depths']:
                    a=predictions[f'cpp_fixed{d}'];b=predictions[f'numpy_fixed{d}']
                    audit['max_belief_difference']=max(audit['max_belief_difference'],float(np.max(abs(a[0]-b[0]))))
                    audit['decision_mismatches']+=int(np.count_nonzero(sign(a[0])!=sign(b[0])))
                a=predictions['cpp_flip'];b=predictions['numpy_flip']
                audit['selection_mismatches']+=int(np.count_nonzero(a[3]!=b[3]))
                audit['decision_mismatches']+=int(np.count_nonzero(sign(a[0])!=sign(b[0])))
                for label in ('cpp_flip','cpp_uncertainty','cpp_random'):
                    a=predictions[label]
                    assert int(a[3].sum())==int(predictions['cpp_flip'][3].sum())
                    np.testing.assert_allclose(a[0],np.where(a[3],predictions['cpp_fixed64'][0],predictions['cpp_fixed2'][0]),atol=1e-9,rtol=1e-9)
                times={name:[] for name in calls}
                order_rng=np.random.default_rng(np.random.SeedSequence([seed,gi,batch,126]))
                for rep in range(repetitions):
                    for name in order_rng.permutation(list(calls)):
                        start=time.perf_counter_ns();answer=calls[name]();elapsed=(time.perf_counter_ns()-start)*1e-9
                        times[name].append(elapsed)
                        timing_rows.append(dict(seed=seed,family=family,batch=batch,policy=name,repeat=rep,seconds=elapsed))
                        np.testing.assert_array_equal(sign(answer[0]),sign(predictions[name][0]))
                condition=[]
                for name,a in predictions.items():
                    adaptive='fixed' not in name
                    depth=2+62*float(a[3].mean()) if adaptive else int(name.split('fixed')[1])
                    condition.append(dict(seed=seed,family=family,batch=batch,policy=name,
                        accuracy_expected=float(np.mean(.5*(1+sign(a[0])*mu))),
                        accuracy_sampled=float(np.mean(sign(a[0])==truth)),
                        bayes_accuracy_expected=float(np.mean(.5*(1+abs(mu)))),
                        median_seconds=float(np.median(times[name])),
                        q1_seconds=float(np.quantile(times[name],.25)),q3_seconds=float(np.quantile(times[name],.75)),
                        nominal_depth=depth,message_updates=int(round(depth*batch*2*len(edges))),
                        frontier_accuracy=None))
                fixed=[r for r in condition if 'fixed' in r['policy']]
                for r in condition:
                    if 'fixed' not in r['policy']:
                        r['frontier_accuracy']=fixed_frontier([v['median_seconds'] for v in fixed],
                                                [v['accuracy_expected'] for v in fixed],r['median_seconds'])
                rows.extend(condition);audit['conditions']+=1
                np.savez_compressed(out/f'input_{seed}_{family}_{batch}.npz',edges=np.array(edges),query=q,
                   true_couplings=J,inference_couplings=bad,observations=obs,truth=truth,conditional_mean=mu,
                   **{name:sign(a[0]) for name,a in predictions.items()})
        pd.DataFrame(rows).to_csv(out/'rows.csv',index=False)
        pd.DataFrame(timing_rows).to_csv(out/'timings.csv',index=False)
        print(json.dumps({'completed_seed':seed,'conditions':audit['conditions']}),flush=True)
    audit['frozen_sources_unchanged']=all(sha(HERE/name)==value for name,value in frozen.items())
    assert audit['frozen_sources_unchanged']
    summary=summarize(rows,cfg,audit);summary['quick']=quick
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix!='.so'}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('STAGE126_SUMMARY_BEGIN',flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print('STAGE126_SUMMARY_END',flush=True)
    print('STAGE126_ENVIRONMENT '+json.dumps(environment),flush=True)
    if not quick and (audit['decision_mismatches'] or audit['selection_mismatches']):
        print('WARNING: numerical equivalence failed; no primary PASS is permitted.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    run(args.out,args.quick)
