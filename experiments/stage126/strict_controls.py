# SPDX-License-Identifier: Apache-2.0
"""Fair-control correction fixed before reading the first full-run outcomes.

The original runner records two shallow beliefs even for fixed policies.
This wrapper removes that unused work from BOTH fixed backends, preserving
all test seeds, inputs, criteria and candidate-policy implementations.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import math
from pathlib import Path
import subprocess
import time
import run as study
import numpy as np

BaseCompiled=study.Compiled
base_numpy=study.numpy_predict
HERE=Path(__file__).resolve().parent

class StrictCompiled(BaseCompiled):
    def __init__(self,out):
        super().__init__(out)
        path=out/'fixed_kernel.so'
        command=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off',
                 '-fPIC','-shared',str(HERE/'fixed_kernel.cpp'),'-o',str(path)]
        start=time.perf_counter()
        subprocess.run(command,check=True,capture_output=True,text=True)
        self.compile_seconds+=time.perf_counter()-start
        self.fixed_library=ctypes.CDLL(str(path.resolve()))
        self.fixed_fn=self.fixed_library.xchain_fixed
        i1=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
        d1=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
        d2=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
        self.fixed_fn.argtypes=[ctypes.c_int]*4+[i1]*3+[d1,d2,ctypes.c_int,d1]
        self.fixed_fn.restype=ctypes.c_int

    def predict(self,edges,J,query,observations,pobs,depth,mode=0,random_seed=0):
        if mode:
            return super().predict(edges,J,query,observations,pobs,depth,mode,random_seed)
        obs=np.asarray(observations)
        if obs.ndim!=2 or obs.shape[1]!=study.ref.N or len(obs)==0:
            raise ValueError('Invalid observation shape')
        if not 0<pobs<.5 or not 0<=query<study.ref.N or not 2<=depth<=256:
            raise ValueError('Invalid parameters')
        src,dst,jc,rev=study.ref.directed_graph(edges,J)
        src,dst,rev=[np.ascontiguousarray(x,dtype=np.int32) for x in (src,dst,rev)]
        jc=np.ascontiguousarray(jc,dtype=np.float64)
        fields=np.ascontiguousarray(.5*math.log((1-pobs)/pobs)*obs,dtype=np.float64)
        fields[:,query]=0
        if not np.isfinite(fields).all():raise ValueError('Nonfinite input')
        out=np.empty(len(obs),dtype=np.float64)
        status=self.fixed_fn(study.ref.N,len(src),len(obs),query,src,dst,rev,jc,fields,int(depth),out)
        if status:raise RuntimeError(f'Fixed kernel returned {status}')
        # Aliases preserve the evaluator interface without computing early features.
        return out,out,out,np.zeros(len(obs),dtype=bool)


def strict_numpy(edges,J,q,obs,pobs,depth,adaptive=False):
    if adaptive:return base_numpy(edges,J,q,obs,pobs,depth,True)
    src,dst,jc,rev,h,inc,qin=study.ref.init_bp(edges,J,q,obs,pobs)
    msg=np.zeros((len(src),len(obs)))
    for _ in range(depth):msg=study.ref.sweep(msg,h,src,dst,jc,rev,inc)
    out=study.ref.belief(msg,h,q,qin)
    return out,out,out,np.zeros(len(obs),dtype=bool)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    frozen={p.name:study.sha(p) for p in (HERE/'strict_controls.py',HERE/'fixed_kernel.cpp')}
    study.Compiled=StrictCompiled;study.numpy_predict=strict_numpy
    study.run(args.out,args.quick)
    out=Path(args.out)
    assert all(study.sha(HERE/name)==h for name,h in frozen.items())
    correction={'reason':'Remove unnecessary early-feature work from fixed controls before interpreting results.',
       'first_run_used_for_primary':False,'criterion_seeds_and_candidate_unchanged':True,
       'control_source_sha256':frozen,'all_control_sources_unchanged':True}
    (out/'strict_control_audit.json').write_text(json.dumps(correction,indent=2)+'\n')
    env=json.loads((out/'environment.json').read_text());env.update(correction)
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    manifest={p.name:study.sha(p) for p in out.iterdir() if p.is_file() and p.suffix!='.so' and p.name!='SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('STAGE126_CONTROL_CORRECTION '+json.dumps(correction),flush=True)

if __name__=='__main__':main()
