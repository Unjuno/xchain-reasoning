# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import math
import numpy as np
N=12

def graph(f):
 e=set(); add=lambda a,b:e.add(tuple(sorted((a,b)))) if a!=b else None
 if f=='barbell':
  for group in (range(0,5),range(7,12)):
   g=list(group)
   for ii,i in enumerate(g):
    for j in g[ii+1:]: add(i,j)
  add(4,5);add(5,6);add(6,7);q=1
 elif f=='tri_ladder':
  for c in range(5): add(c,c+1);add(6+c,6+c+1);add(c,6+c);add(c,7+c)
  add(5,11);q=2
 elif f=='torus3x4':
  R,C=3,4
  for r in range(R):
   for c in range(C):
    i=r*C+c;add(i,r*C+(c+1)%C);add(i,((r+1)%R)*C+c)
  q=5
 elif f=='cycle':
  for i in range(N): add(i,(i+1)%N)
  q=0
 elif f=='cactus':
  for a in (0,3,6,9): add(a,a+1);add(a+1,a+2);add(a+2,a)
  add(2,3);add(5,6);add(8,9);q=5
 elif f=='ladder':
  for r in range(2):
   for c in range(5): add(r*6+c,r*6+c+1)
  for c in range(6): add(c,6+c)
  q=2
 elif f=='grid':
  R,C=3,4
  for r in range(R):
   for c in range(C):
    i=r*C+c
    if r+1<R:add(i,(r+1)*C+c)
    if c+1<C:add(i,r*C+c+1)
  q=5
 else: raise ValueError(f)
 return sorted(e),q

def make_edge_params(edges,beta,rng): return beta*rng.uniform(.85,1.15,size=len(edges))

def enumerate_states(n=N):
 vals=np.arange(1<<n,dtype=np.uint16)[:,None];bits=((vals>>np.arange(n,dtype=np.uint16))&1).astype(np.int8);return (2*bits-1).astype(np.int8)
STATES=enumerate_states()

def sample_prior(edges,J,rng,n):
 energy=np.zeros(len(STATES),float)
 for (u,v),j in zip(edges,J):energy+=j*STATES[:,u]*STATES[:,v]
 energy-=energy.max();p=np.exp(energy);p/=p.sum();idx=rng.choice(len(STATES),size=n,p=p);return STATES[idx].astype(np.int8)

def directed_graph(edges,J):
 src=[];dst=[];jc=[];lookup={}
 for (u,v),j in zip(edges,J):
  a=len(src);src.extend([u,v]);dst.extend([v,u]);jc.extend([j,j]);lookup[(u,v)]=a;lookup[(v,u)]=a+1
 rev=[lookup[(d,s)] for s,d in zip(src,dst)]
 return np.array(src),np.array(dst),np.array(jc,float),np.array(rev)

def init_bp(edges,J,q,obs,pobs):
 src,dst,jc,rev=directed_graph(edges,J);lam=.5*math.log((1-pobs)/pobs);h=lam*obs.astype(float);h[:,q]=0.;incoming=[np.where(dst==i)[0] for i in range(N)];qin=np.where(dst==q)[0];return src,dst,jc,rev,h,incoming,qin

def sweep(msg,h,src,dst,jc,rev,incoming):
 total=np.zeros((N,h.shape[0]),float)
 for i in range(N):
  if len(incoming[i]): total[i]=msg[incoming[i]].sum(axis=0)
 new=np.empty_like(msg)
 for a in range(len(src)):
  cav=h[:,src[a]]+total[src[a]]-msg[rev[a]];x=np.clip(np.tanh(jc[a])*np.tanh(cav),-1+1e-14,1-1e-14);new[a]=np.arctanh(x)
 return new

def belief(msg,h,q,qin):
 b=h[:,q].copy()
 if len(qin):b+=msg[qin].sum(axis=0)
 return b

def beliefs(edges,J,q,obs,pobs,depths=(1,2,64)):
 src,dst,jc,rev,h,inc,qin=init_bp(edges,J,q,obs,pobs);msg=np.zeros((len(src),len(obs)));out={}
 for t in range(1,max(depths)+1):
  msg=sweep(msg,h,src,dst,jc,rev,inc)
  if t in depths:out[t]=belief(msg,h,q,qin).copy()
 return out

def mismatch_scores(edges,J,q,obs):
 elig=[(ei,u,v) for ei,(u,v) in enumerate(edges) if u!=q and v!=q];bad=np.zeros(len(obs),float)
 for ei,u,v in elig:bad+=(obs[:,u]*obs[:,v]*np.sign(J[ei])<0)
 return bad/max(1,len(elig))

def corrupt_exterior(edges,J,q,corr,rng):
 out=J.copy();elig=[i for i,(u,v) in enumerate(edges) if u!=q and v!=q];fl=rng.random(len(elig))<corr
 for f,idx in zip(fl,elig):
  if f:out[idx]*=-1
 return out
