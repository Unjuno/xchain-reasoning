# SPDX-License-Identifier: Apache-2.0
"""Analytic and leakage checks for observation-only reliability diagnostics."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import cho_factor, cho_solve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'stage49_51'))
sys.path.insert(0,str(ROOT/'experiments'/'stage52'))
from core import graph, make_problem, coefficient_path
from reliability import build_candidates, diagnostic_scores, select_predictions, corrupt_relations

class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.a,self.q,_,_,_=graph('grid',62980)
        self.p=make_problem(self.a,self.q,.05,1.)
        self.c=build_candidates(self.a,self.q,.05,self.p['noisevar'],[0.,.5,1.])
        self.y=np.random.default_rng(62981).normal(size=(8,48))

    def test_full_matches_original(self):
        np.testing.assert_allclose(self.c[-1].output_coef,coefficient_path(self.p,[16])[16][self.q],atol=1e-14)
        np.testing.assert_allclose(self.c[-1].exact_output_coef,self.p['bayes_coef'],atol=1e-14)

    def test_loo_matches_explicit_deletion(self):
        h=self.c[-1].observed_precision
        cov=self.p['C']; fast=diagnostic_scores(self.y,[self.c[-1]])['loo_nlpd'][:,0]
        losses=[]
        for j in range(48):
            rest=np.delete(np.arange(48),j)
            beta=np.linalg.solve(cov[np.ix_(rest,rest)],cov[rest,j])
            variance=cov[j,j]-cov[j,rest]@beta
            mean=self.y[:,rest]@beta
            losses.append(.5*(np.log(variance)+(self.y[:,j]-mean)**2/variance))
            pred=self.y[:,j]-(self.y@h.T)[:,j]/h[j,j]
            np.testing.assert_allclose(pred,mean,atol=1e-13)
        np.testing.assert_allclose(fast,np.mean(losses,axis=0),atol=1e-13)

    def test_own_observation_excluded_from_prediction(self):
        h=self.c[-1].observed_precision
        altered=self.y.copy();altered[:,7]+=50
        old=self.y[:,7]-(self.y@h.T)[:,7]/h[7,7]
        new=altered[:,7]-(altered@h.T)[:,7]/h[7,7]
        np.testing.assert_allclose(old,new,atol=1e-13)

    def test_query_edges_and_early_state_unchanged(self):
        for kind in ['intact','flip15','flip50','coherent50']:
            bad=corrupt_relations(self.a,self.q,kind,np.random.default_rng(1))
            np.testing.assert_array_equal(bad[self.q],self.a[self.q])
            np.testing.assert_array_equal(np.abs(bad),np.abs(self.a))
            c=build_candidates(bad,self.q,.05,self.p['noisevar'],[1.],steps=2)
            np.testing.assert_array_equal(c[0].output_coef,coefficient_path(self.p,[2])[2][self.q])

    def test_coherent_is_cycle_balanced_but_wrong(self):
        bad=corrupt_relations(self.a,self.q,'coherent50',np.random.default_rng(1))
        self.assertGreater(np.count_nonzero(bad!=self.a),0)
        gauge=np.zeros(49);gauge[0]=1;stack=[0]
        while stack:
            i=stack.pop()
            for j in np.flatnonzero(bad[i]):
                desired=gauge[i]*np.sign(bad[i,j])
                if gauge[j]==0: gauge[j]=desired;stack.append(j)
                else:self.assertEqual(gauge[j],desired)

    def test_no_truth_argument_and_deterministic_gate(self):
        import inspect
        self.assertEqual(list(inspect.signature(select_predictions).parameters),['observations','candidates','scores'])
        scores=diagnostic_scores(self.y,self.c)['loo_nlpd']
        p,i=select_predictions(self.y,self.c,scores)
        p2,i2=select_predictions(self.y.copy(),self.c,scores.copy())
        np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(i,i2)
        self.assertTrue(np.all((i>=0)&(i<len(self.c))))

    def test_observation_equivalent_worlds(self):
        signs=np.ones(49);signs[self.q]=-1
        b=self.a*signs[:,None]*signs[None,:]
        other=make_problem(b,self.q,.05,1.)
        np.testing.assert_array_equal(self.p['C'],other['C'])
        np.testing.assert_array_equal(self.p['k'],-other['k'])
        cc=build_candidates(b,self.q,.05,self.p['noisevar'],[0.,.5,1.])
        score_a=diagnostic_scores(self.y,self.c)['loo_nlpd']
        score_b=diagnostic_scores(self.y,cc)['loo_nlpd']
        np.testing.assert_array_equal(score_a,score_b)
        pa,ia=select_predictions(self.y,self.c,score_a)
        pb,ib=select_predictions(self.y,cc,score_b)
        np.testing.assert_array_equal(ia,ib)
        np.testing.assert_array_equal(pa,-pb)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError): build_candidates(self.a,self.q,.05,np.zeros(48),[1.])
        with self.assertRaises(ValueError): build_candidates(self.a,self.q,.05,self.p['noisevar'],[2.])
        with self.assertRaises(ValueError): diagnostic_scores(np.zeros((2,47)),self.c)
        with self.assertRaises(ValueError): corrupt_relations(self.a,self.q,'other',np.random.default_rng(1))

if __name__=='__main__':unittest.main()
