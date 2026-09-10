# SPDX-License-Identifier: Apache-2.0
# Public export adapted from the supplied Stage 49-51 research code.
from core import *
import pandas as pd


def run():
    (ROOT/"results").mkdir(parents=True, exist_ok=True)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];configs=[];maxid=0
    for seed in seed_range(49001,10):
        for fi,family in enumerate(FAMILIES):
            # Topologies use different data RNG streams and different priors.
            # This is not a matched topology-only causal contrast.
            adj,q,edges,gauge,dist=graph(family,seed)
            data_rng=np.random.default_rng(seed*101+fi)
            z=data_rng.standard_normal((SAMPLES,49));eps=data_rng.standard_normal((SAMPLES,48))
            save_input(f'base_{seed}_{family}.npz',adj=adj,q=q,edges=edges,
                                gauge=gauge,dist=dist,z=z,eps=eps)
            for tau in (.05,.5):
                for nr in (.25,1.,4.):
                    p=make_problem(adj,q,tau,nr)
                    truth=z@np.linalg.cholesky(p['Sigma']).T
                    y=truth[:,p['obs']]+eps*np.sqrt(p['noisevar'])
                    target=truth[:,q]
                    paths=coefficient_path(p); gs=gs_coefficient(p,adj,STEPS)
                    methods={f'outer_t{t}':(a[q],t) for t,a in paths.items()}
                    methods.update({f'gs_t{t}':(a[q],t) for t,a in gs.items() if t in (2,4,8,16)})
                    methods['frozen_outer_t16']=(paths[2][q],16)
                    methods['reset_2x8']=(paths[2][q],16)
                    methods['output_only']=(np.zeros(48),16)
                    methods['local_bayes_r1']=(subset_coef(p,np.flatnonzero(dist[q]<=1)),0)
                    methods['local_bayes_r2']=(subset_coef(p,np.flatnonzero(dist[q]<=2)),0)
                    methods['bayes_all']=(p['bayes_coef'],0)
                    # Global sign-aligned mean, optimally rescaled using true covariance.
                    u=gauge[q]*gauge[p['obs']]/48
                    scale=float(u@p['k'])/float(u@p['C']@u)
                    methods['aligned_mean_calibrated']=(scale*u,0)
                    for name,(w,t) in methods.items():
                        rr=risks(w,p);pred=y@w
                        actual=(pred>=0)==(target>=0)
                        row=dict(seed=seed,family=family,tau=tau,noise_ratio=nr,n=49,edges=len(edges),query=q,
                                 method=name,sweeps=t,node_updates=49*t,edge_products=2*len(edges)*t,
                                 spectral_radius=float(np.max(np.abs(np.linalg.eigvals(p['T'])))),
                                 inf_contraction=float(np.max(np.abs(p['T']).sum(1))),
                                 prior_var=p['prior'],bayes_var=p['bayes_var'],samples=len(target),
                                 nmse_mc=float(np.mean((pred-target)**2)/p['prior']),acc_mc=float(actual.mean()),**rr)
                        rows.append(row);maxid=max(maxid,rr['excess_identity_error'])
                    configs.append(dict(seed=seed,family=family,tau=tau,noise_ratio=nr))
        pd.DataFrame(rows).to_csv(out/'stage49_rows.csv',index=False)
        print('stage49 seed',seed,'complete;',len(rows),'rows',flush=True)
    (out/'stage49_done.json').write_text(json.dumps({'configs':len(configs),'rows':len(rows),'max_risk_identity_error':maxid},indent=2))

if __name__=='__main__':run()
