# SPDX-License-Identifier: Apache-2.0
# Public export adapted from the supplied Stage 49-51 research code.
from core import *
import pandas as pd


def run():
    (ROOT/"results").mkdir(parents=True, exist_ok=True)
    rows=[]
    for seed in seed_range(50001,10):
        for fi,family in enumerate(FAMILIES):
            adj,q,edges,gauge,dist=graph(family,seed)
            rng=np.random.default_rng(seed*109+fi)
            z=rng.standard_normal((SAMPLES,49));eps=rng.standard_normal((SAMPLES,48))
            eligible=np.flatnonzero(np.all(edges!=q,axis=1));order=rng.permutation(eligible)
            save_input(f'branch50_{seed}_{family}.npz',adj=adj,q=q,edges=edges,gauge=gauge,dist=dist,z=z,eps=eps,flip_order=order)
            for nr in (1.,4.):
                p=make_problem(adj,q,.05,nr)
                theta=z@np.linalg.cholesky(p['Sigma']).T
                y=theta[:,p['obs']]+eps*np.sqrt(p['noisevar'])
                for fraction in (0.,.05,.15,.30,.50):
                    bad=adj.copy();count=int(round(fraction*len(order)))
                    for e in order[:count]:
                        i,j=edges[e];bad[i,j]*=-1;bad[j,i]*=-1
                    # Keep all query-adjacent relations correct, and keep diagonal,
                    # evidence precision, sample, parameter count and edge count fixed.
                    Tbad=bad/p['D'][:,None]
                    A=np.zeros_like(p['U'])
                    for t in range(1,65):
                        A=Tbad@A+p['U']
                        if t not in STEPS:continue
                        w=A[q];pred=y@w;rr=risks(w,p)
                        rows.append(dict(seed=seed,family=family,tau=.05,noise_ratio=nr,
                            fraction=fraction,flipped_edges=count,eligible_edges=len(order),
                            t=t,nmse_mc=float(np.mean((pred-theta[:,q])**2)/p['prior']),
                            acc_mc=float(np.mean((pred>=0)==(theta[:,q]>=0))),**rr))
        pd.DataFrame(rows).to_csv(ROOT/'results/stage50_rows.csv',index=False)
        print('stage50 seed',seed,'complete',len(rows),flush=True)
    (ROOT/'results/stage50_done.json').write_text(json.dumps({'rows':len(rows),'true_data_configs':len(rows)//35,'inference_variants':len(rows)//7},indent=2))

if __name__=='__main__':run()
