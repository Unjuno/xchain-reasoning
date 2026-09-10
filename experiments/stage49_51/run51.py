# SPDX-License-Identifier: Apache-2.0
# Public export adapted from the supplied Stage 49-51 research code.
from core import *
import pandas as pd
GATES=[0.,.125,.25,.5,.75,1.]


def evaluate_block(seed,family,fi,fraction,phase,z,eps,adj,q,edges,order,nr):
    p=make_problem(adj,q,.05,nr)
    truth=z@np.linalg.cholesky(p['Sigma']).T
    y=truth[:,p['obs']]+eps*np.sqrt(p['noisevar'])
    bad=adj.copy(); count=int(round(fraction*len(order)))
    for e in order[:count]:
        i,j=edges[e];bad[i,j]*=-1;bad[j,i]*=-1
    outer=bad.copy();outer[q,:]=0;outer[:,q]=0
    near=bad-outer
    rows=[]
    for gate in GATES:
        T=(near+gate*outer)/p['D'][:,None]
        A=np.zeros_like(p['U'])
        for _ in range(16):A=T@A+p['U']
        w=A[q];pred=y@w;rr=risks(w,p)
        rows.append(dict(phase=phase,seed=seed,family=family,noise_ratio=nr,fraction=fraction,gate=gate,
              nmse_mc=float(np.mean((pred-truth[:,q])**2)/p['prior']),
              acc_mc=float(np.mean((pred>=0)==(truth[:,q]>=0))),**rr))
    return rows


def run():
    (ROOT/"results").mkdir(parents=True, exist_ok=True)
    rows=[]
    for phase,seeds,fractions in [('calibration',seed_range(51001,5),[.5]),('test',seed_range(52001,10),[0.,.15,.30,.50])]:
        if phase=='test':
            tr=pd.DataFrame(rows)
            agg=tr.groupby('gate')[['acc_mc','nmse_mc']].mean()
            selected={'classification_gate':float(agg.acc_mc.idxmax()),'mse_gate':float(agg.nmse_mc.idxmin()),
                      'selection':'Only calibration labels, no exact population risks used in selection',
                      'train_seeds':list(seed_range(51001,5)),'calibration_corruption':.5}
            (ROOT/'results/stage51_selected.json').write_text(json.dumps(selected,indent=2))
            print('FROZEN TRAIN SELECTION',selected,flush=True)
        for seed in seeds:
            for fi,family in enumerate(FAMILIES):
                adj,q,edges,gauge,dist=graph(family,seed)
                rng=np.random.default_rng(seed*113+fi)
                z=rng.standard_normal((SAMPLES,49));eps=rng.standard_normal((SAMPLES,48))
                eligible=np.flatnonzero(np.all(edges!=q,axis=1));order=rng.permutation(eligible)
                save_input(f'branch51_{seed}_{family}.npz',adj=adj,q=q,edges=edges,z=z,eps=eps,flip_order=order)
                for nr in (1.,4.):
                    for fraction in fractions:
                        rows.extend(evaluate_block(seed,family,fi,fraction,phase,z,eps,adj,q,edges,order,nr))
            pd.DataFrame(rows).to_csv(ROOT/'results/stage51_rows.csv',index=False)
            print('stage51',phase,seed,'complete',len(rows),flush=True)
    (ROOT/'results/stage51_done.json').write_text(json.dumps({'rows':len(rows),'training_type':'one scalar grid calibration; not neural training'},indent=2))

if __name__=='__main__':run()
