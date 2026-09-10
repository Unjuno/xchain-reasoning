# SPDX-License-Identifier: Apache-2.0
# Public export adapted from the supplied Stage 49-51 research code.
"""Analytic XChain accuracy test. Standard units are dimensionless.
No fitted models, hidden truth is never input to the inference policies.
"""
from __future__ import annotations
import math, time, json, os, platform, hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import shortest_path

# Public export: output paths and smoke-test sizes are configurable.
ROOT=Path(os.environ.get("XCHAIN_OUTPUT_DIR", "runs/stage49_51")).expanduser().resolve()
SAMPLES=int(os.environ.get("XCHAIN_SAMPLES", "2048"))
SEED_LIMIT=int(os.environ.get("XCHAIN_SEED_LIMIT", "10"))
SAVE_INPUTS=os.environ.get("XCHAIN_SAVE_INPUTS", "0")=="1"
if SAMPLES < 2 or not 1 <= SEED_LIMIT <= 10:
    raise ValueError("Require samples >= 2 and seed limit in [1, 10].")

def seed_range(start, count=10):
    return range(start, start + min(count, SEED_LIMIT))

def save_input(name, **arrays):
    if SAVE_INPUTS:
        (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ROOT / "inputs" / name, **arrays)

FAMILIES=['chain','cross','tree','grid','grid_rewired']
STEPS=[1,2,4,8,16,32,64]


def connected(edges,n):
    adj=[[] for _ in range(n)]
    for i,j in edges: adj[i].append(j); adj[j].append(i)
    seen={0}; stack=[0]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen: seen.add(j); stack.append(j)
    return len(seen)==n


def graph(family:str, seed:int, n:int=49):
    if n!=49: raise ValueError('This protocol uses exactly 49 nodes.')
    rng=np.random.default_rng(seed)
    if family=='chain': edges={(i,i+1) for i in range(n-1)}; q=24
    elif family=='cross':
        edges=set(); q=0
        for arm in range(4):
            prev=0
            for p in range(12):
                cur=1+arm*12+p; edges.add(tuple(sorted((prev,cur)))); prev=cur
    elif family=='tree': edges={((i-1)//2,i) for i in range(1,n)}; q=0
    elif family in ('grid','grid_rewired'):
        edges=set(); q=24
        for r in range(7):
            for c in range(7):
                i=r*7+c
                if r<6: edges.add((i,i+7))
                if c<6: edges.add((i,i+1))
        if family=='grid_rewired':
            # Exactly eight accepted degree-preserving double-edge swaps.
            swaps=0
            for _ in range(10000):
                ed=sorted(edges); k=rng.choice(len(ed),2,replace=False)
                a,b=ed[k[0]]; c,d=ed[k[1]]
                if len({a,b,c,d})<4: continue
                if rng.integers(2): c,d=d,c
                e1=tuple(sorted((a,c))); e2=tuple(sorted((b,d)))
                if e1 in edges or e2 in edges: continue
                new=edges-{ed[k[0]],ed[k[1]]}|{e1,e2}
                if not connected(new,n): continue
                edges=new; swaps+=1
                if swaps==8: break
            if swaps!=8: raise RuntimeError('Could not complete graph swaps')
    else: raise ValueError(family)
    ed=np.asarray(sorted(edges),dtype=np.int64)
    # Gauge signs make naive averaging invalid. Latents themselves are not copies.
    signs=rng.choice(np.array([-1.,1.]),n)
    weights=rng.uniform(.6,1.4,len(ed))
    adj=np.zeros((n,n))
    for (i,j),w in zip(ed,weights):
        adj[i,j]=adj[j,i]=w*signs[i]*signs[j]
    dist=shortest_path((np.abs(adj)>0).astype(float),directed=False,unweighted=True)
    return adj,q,ed,signs,dist


def make_problem(adj, q, tau, noise_ratio):
    adj=np.asarray(adj, dtype=np.float64)
    if (adj.ndim != 2 or adj.shape[0] != adj.shape[1] or
            not np.isfinite(adj).all() or not np.allclose(adj,adj.T) or
            np.any(np.diag(adj) != 0)):
        raise ValueError("adj must be finite, symmetric, square, and zero diagonal")
    n=len(adj)
    if not isinstance(q,(int,np.integer)) or not 0 <= q < n:
        raise ValueError("query index outside graph")
    if not np.isfinite(tau) or not np.isfinite(noise_ratio) or tau <= 0 or noise_ratio <= 0:
        raise ValueError("tau and noise_ratio must be finite and positive")
    Q=np.diag(tau+np.abs(adj).sum(1))-adj
    Sigma=cho_solve(cho_factor(Q,lower=True),np.eye(n))
    obs=np.array([i for i in range(n) if i!=q])
    noisevar=noise_ratio*np.diag(Sigma)[obs]
    prec=np.zeros(n);prec[obs]=1/noisevar
    M=Q+np.diag(prec)
    D=np.diag(M)
    T=adj/D[:,None]
    U=np.zeros((n,n-1));U[obs,np.arange(n-1)]=prec[obs]/D[obs]
    C=Sigma[np.ix_(obs,obs)]+np.diag(noisevar)
    k=Sigma[obs,q]
    post=cho_solve(cho_factor(M,lower=True),np.eye(n))
    allcoef=cho_solve(cho_factor(M,lower=True),np.diag(prec)[:,obs])
    assert np.max(np.abs(T).sum(1))<1
    return dict(Q=Q,Sigma=Sigma,obs=obs,noisevar=noisevar,prec=prec,M=M,D=D,T=T,U=U,C=C,k=k,
                prior=float(Sigma[q,q]),bayes_var=float(post[q,q]),allcoef=allcoef,
                bayes_coef=allcoef[q],q=q)


def risks(w, p, addvar=0.0):
    predvar=float(w@p['C']@w)+float(addvar)
    covariance=float(w@p['k'])
    risk=float(p['prior']-2*covariance+predvar)
    if predvar<=1e-28: corr=0.; accuracy=.5
    else:
        corr=float(np.clip(covariance/np.sqrt(p['prior']*predvar),-1,1))
        accuracy=.5+np.arcsin(corr)/np.pi
    excess=float((w-p['bayes_coef'])@p['C']@(w-p['bayes_coef'])+addvar)
    return {'mse_exact':risk,'nmse_exact':risk/p['prior'],'acc_exact':float(accuracy),
            'excess_exact':excess,'excess_identity_error':abs(risk-p['bayes_var']-excess),
            'predvar':predvar,'covariance':covariance,'corr':corr}


def subset_coef(p, nodes):
    allowed=np.isin(p['obs'],nodes);ix=np.flatnonzero(allowed)
    w=np.zeros(len(p['obs']))
    if len(ix): w[ix]=cho_solve(cho_factor(p['C'][np.ix_(ix,ix)],lower=True),p['k'][ix])
    return w


def coefficient_path(p, steps=STEPS):
    A=np.zeros_like(p['U']);ret={}
    for t in range(1,max(steps)+1):
        A=p['T']@A+p['U']
        if t in steps: ret[t]=A.copy()
    return ret


def gs_coefficient(p,adj,steps):
    A=np.zeros_like(p['U']);ret={}
    for t in range(1,max(steps)+1):
        for i in range(len(A)):
            A[i]=p['U'][i]+p['T'][i]@A
        if t in steps:ret[t]=A.copy()
    return ret


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import scipy
    try:
        import torch
        tv=torch.__version__
    except ImportError:tv=None
    info={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'torch':tv,'platform':platform.platform(),'dtype':'float64','batch_samples':SAMPLES,
          'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
          'clock':'not fixed or measured','training_runs':0,'quantization':None,
          'primary_metric':'true latent NMSE and sign accuracy, not runtime/model fidelity'}
    if Path('/proc/cpuinfo').exists():
        info['cpu']=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if 'model name' in s),'unknown')
    return info
