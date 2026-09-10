# SPDX-License-Identifier: Apache-2.0
# Public export adapted from the supplied Stage 49-51 research code.
from core import *
import pandas as pd
from scipy.stats import t as tdist


def ci(values):
    a=np.asarray(values,dtype=float);m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    k=float(tdist.ppf(.975,len(a)-1))
    return {'mean':m,'ci95':[m-k*se,m+k*se],'standard_uncertainty':se,'coverage_factor':k,'n_seeds':len(a)}


def paired(df,a,b,metrics=('nmse_exact','acc_exact'),condition=None):
    d=df if condition is None else df.query(condition)
    g=d.groupby(['seed','method'])[list(metrics)].mean()
    aa=g.xs(a,level='method');bb=g.xs(b,level='method')
    return {'accuracy_gain_pp':ci(100*(aa.acc_exact-bb.acc_exact)),
            'nmse_reduction':ci(bb.nmse_exact-aa.nmse_exact),
            'relative_nmse_reduction_pct':ci(100*(1-aa.nmse_exact/bb.nmse_exact))}


def run():
    a=pd.read_csv(ROOT/'results/stage49_rows.csv')
    b=pd.read_csv(ROOT/'results/stage50_rows.csv')
    c=pd.read_csv(ROOT/'results/stage51_rows.csv')
    means=a.groupby('method')[['nmse_exact','acc_exact','nmse_mc','acc_mc']].mean()
    fam=a[a.method.isin(['outer_t2','outer_t16','bayes_all','local_bayes_r1'])].groupby(['family','method'])[['nmse_exact','acc_exact']].mean()
    effect=paired(a,'outer_t16','frozen_outer_t16')
    byfam={f:paired(a,'outer_t16','frozen_outer_t16',condition=f"family == '{f}'") for f in FAMILIES}
    s49={'conditions':300,'rows':len(a),'inference_prediction_evaluations':int(a.samples.sum()),
         'primary':effect,'vs_optimal_local_bayes':paired(a,'outer_t16','local_bayes_r1'),
         'gauss_seidel16_vs_jacobi16':paired(a,'gs_t16','outer_t16'),
         'by_family':byfam,'means':means.reset_index().to_dict(orient='records'),
         'finite32_to_bayes_excess_nmse':float((a[a.method=='outer_t32'].nmse_exact.to_numpy()-a[a.method=='bayes_all'].nmse_exact.to_numpy()).mean()),
         'primary_pass':bool(effect['nmse_reduction']['ci95'][0]>0 and effect['accuracy_gain_pp']['ci95'][0]>0)}
    t50=b[b.t.isin([2,4,8,16,64])].groupby(['fraction','t'])[['nmse_exact','acc_exact','nmse_mc','acc_mc']].mean()
    effects50={}
    for frac in sorted(b.fraction.unique()):
        d=b[b.fraction==frac].groupby(['seed','t'])[['nmse_exact','acc_exact']].mean()
        r={}
        for target,base in [(16,2),(64,16),(64,4)]:
            aa=d.xs(target,level='t');bb=d.xs(base,level='t')
            r[f't{target}_vs_t{base}']={'acc_gain_pp':ci(100*(aa.acc_exact-bb.acc_exact)),
                                     'nmse_reduction':ci(bb.nmse_exact-aa.nmse_exact)}
        effects50[str(frac)]=r
    s50={'true_conditions':100,'rows':len(b),'means':t50.reset_index().to_dict(orient='records'),'effects':effects50}
    selected=json.loads((ROOT/'results/stage51_selected.json').read_text())
    test=c[c.phase=='test'];ef51={}
    for frac in sorted(test.fraction.unique()):
        d=test[test.fraction==frac].groupby(['seed','gate'])[['nmse_exact','acc_exact']].mean()
        aa=d.xs(selected['classification_gate'],level='gate');bb=d.xs(1.,level='gate')
        ef51[str(frac)]={'acc_gain_pp':ci(100*(aa.acc_exact-bb.acc_exact)),
                        'nmse_reduction':ci(bb.nmse_exact-aa.nmse_exact)}
    t51=test.groupby(['fraction','gate'])[['nmse_exact','acc_exact','nmse_mc','acc_mc']].mean()
    s51={'selected':selected,'rows':len(c),'effects':ef51,'means':t51.reset_index().to_dict(orient='records')}
    s={'stage49':s49,'stage50':s50,'stage51':s51,'statistical_note':'Exact Gaussian expectation; t intervals vary graphs, not Monte Carlo labels. Exploratory branches separately frozen. No multiplicity-wide historical claim.'}
    (ROOT/'SUMMARY.json').write_text(json.dumps(s,indent=2))
    means.to_csv(ROOT/'results/stage49_means.csv');fam.to_csv(ROOT/'results/stage49_families.csv')
    t50.to_csv(ROOT/'results/stage50_means.csv');t51.to_csv(ROOT/'results/stage51_means.csv')
    key=[]
    for method in ['output_only','outer_t2','outer_t4','outer_t8','outer_t16','outer_t32','outer_t64','local_bayes_r1','bayes_all','aligned_mean_calibrated','gs_t16']:
        row=means.loc[method].to_dict();row.update(stage=49,method=method);key.append(row)
    for frac in [0.,.15,.3,.5]:
        for t in [2,16,64]:
            row=t50.loc[(frac,t)].to_dict();row.update(stage=50,method=f'corrupt{frac}_t{t}');key.append(row)
    for frac in [0.,.5]:
        for gate in [0.,1.]:
            row=t51.loc[(frac,gate)].to_dict();row.update(stage=51,method=f'corrupt{frac}_gate{gate}');key.append(row)
    pd.DataFrame(key).to_csv(ROOT/'KEY_RESULTS.csv',index=False)
    print('STAGE49 PRIMARY',json.dumps(effect,indent=2));print('VERSUS OPTIMAL LOCAL',s49['vs_optimal_local_bayes'])
    print('STAGE50 FRAC .5',effects50['0.5']);print('STAGE51',ef51)

if __name__=='__main__':run()
