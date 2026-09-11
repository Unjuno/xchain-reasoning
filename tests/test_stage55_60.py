# SPDX-License-Identifier: Apache-2.0
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'))
sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
sys.path.insert(0,str(ROOT/'experiments'/'stage55_60'))
from core import graph, make_problem
from run56 import conservative_repair
from run57 import spectral_repair

class Stage55to60Tests(unittest.TestCase):
    def test_population_covariance_sign_matches_gauge(self):
        A,q,_,g,_=graph('grid',77123)
        p=make_problem(A,q,.05,1.0)
        obs=p['obs']; C=p['C']
        # Off-diagonal observation noise is zero, so signs must match gauge products.
        for a in range(len(obs)):
            for b in range(a+1,len(obs)):
                self.assertEqual(np.sign(C[a,b]), np.sign(g[obs[a]]*g[obs[b]]))

    def test_population_spectral_repair_recovers_all_edge_signs_with_one_anchor(self):
        A,q,_,_,_=graph('cross',77124)
        p=make_problem(A,q,.05,1.0)
        # Construct deterministic calibration matrix whose covariance is exactly C.
        L=np.linalg.cholesky(p['C'])
        d=L.shape[0]
        Y=np.vstack([np.sqrt(d)*L.T,-np.sqrt(d)*L.T])
        # spectral_repair only depends on covariance; candidate magnitudes are enough.
        candidate=np.abs(A)
        repaired,_,allacc,qacc=spectral_repair(candidate,A,q,Y)
        self.assertAlmostEqual(allacc,1.0)
        self.assertAlmostEqual(qacc,1.0)
        mask=A!=0
        self.assertTrue(np.array_equal(np.sign(repaired[mask]),np.sign(A[mask])))

    def test_conservative_repair_never_changes_query_edges(self):
        A,q,_,_,_=graph('tree',77125)
        p=make_problem(A,q,.05,1.0)
        rng=np.random.default_rng(1)
        Y=rng.multivariate_normal(np.zeros(48),p['C'],size=128)
        candidate=A.copy()
        # Deliberately flip exterior edges only.
        edges=np.argwhere(np.triu(candidate!=0,1))
        for i,j in edges[:5]:
            if i!=q and j!=q:
                candidate[i,j]*=-1;candidate[j,i]*=-1
        before=candidate[q].copy()
        repaired,*_=conservative_repair(candidate,A,q,Y,.10)
        self.assertTrue(np.array_equal(repaired[q],before))

    def test_query_gauge_flip_leaves_observed_covariance_unchanged(self):
        A,q,_,_,_=graph('chain',77126)
        flip=np.ones(49);flip[q]=-1
        B=A*flip[:,None]*flip[None,:]
        p=make_problem(A,q,.05,4.0);r=make_problem(B,q,.05,4.0)
        self.assertLess(np.max(np.abs(p['C']-r['C'])),1e-12)
        self.assertLess(np.max(np.abs(p['k']+r['k'])),1e-12)

if __name__=='__main__': unittest.main()
