# SPDX-License-Identifier: Apache-2.0
"""Mathematical and implementation regressions; no network or credentials."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'stage49_51'))
from core import FAMILIES, graph, connected, make_problem, risks, subset_coef, coefficient_path


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.adj, self.q, self.edges, self.signs, self.dist = graph('grid', 49001)
        self.p = make_problem(self.adj, self.q, .05, 1.)

    def test_all_families_are_connected_and_reproducible(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                a, q, edges, _, _ = graph(family, 49001)
                self.assertTrue(connected(edges, 49))
                np.testing.assert_array_equal(a, a.T)
                np.testing.assert_array_equal(a, graph(family, 49001)[0])
                self.assertEqual(a[q, q], 0)

    def test_rewiring_preserves_degrees(self):
        g = graph('grid', 49001)[0]
        h = graph('grid_rewired', 49001)[0]
        np.testing.assert_array_equal(np.count_nonzero(g, axis=1), np.count_nonzero(h, axis=1))

    def test_unsupported_graph_size_and_family(self):
        with self.assertRaises(ValueError):
            graph('grid', 1, n=48)
        with self.assertRaises(ValueError):
            graph('not-a-graph', 1)

    def test_invalid_problem_inputs(self):
        for tau, noise in [(0, 1), (-1, 1), (1, 0), (1, float('nan'))]:
            with self.assertRaises(ValueError):
                make_problem(self.adj, self.q, tau, noise)
        with self.assertRaises(ValueError):
            make_problem(self.adj, 99, .05, 1)
        with self.assertRaises(ValueError):
            make_problem(self.adj[:2], 0, .05, 1)
        bad = self.adj.copy()
        bad[0, 1] += 1
        with self.assertRaises(ValueError):
            make_problem(bad, self.q, .05, 1)

    def test_query_truth_is_not_an_observation(self):
        self.assertNotIn(self.q, self.p['obs'])
        self.assertEqual(self.p['prec'][self.q], 0)
        np.testing.assert_array_equal(self.p['U'][self.q], np.zeros(48))

    def test_positive_definite_and_contractive(self):
        self.assertGreater(np.linalg.eigvalsh(self.p['Q']).min(), 0)
        self.assertLess(np.abs(self.p['T']).sum(axis=1).max(), 1)

    def test_bayes_fixed_point(self):
        a = self.p['allcoef']
        np.testing.assert_allclose(a, self.p['T'] @ a + self.p['U'], atol=2e-14)
        w = np.linalg.solve(self.p['C'], self.p['k'])
        np.testing.assert_allclose(w, self.p['bayes_coef'], atol=2e-14)

    def test_risk_decomposition(self):
        rng = np.random.default_rng(52102)
        for _ in range(10):
            r = risks(rng.normal(size=48), self.p)
            self.assertLess(r['excess_identity_error'], 2e-11)
            self.assertGreaterEqual(r['excess_exact'], 0)

    def test_causal_reach(self):
        for t, a in coefficient_path(self.p, [1, 2, 4, 8]).items():
            outside = self.dist[self.q, self.p['obs']] > t - 1
            np.testing.assert_array_equal(a[self.q, outside], np.zeros(outside.sum()))

    def test_coefficients_match_direct_state_updates(self):
        y = np.random.default_rng(52103).normal(size=48)
        z = np.zeros(49)
        for _ in range(16):
            z = self.p['T'] @ z + self.p['U'] @ y
        a = coefficient_path(self.p, [16])[16]
        np.testing.assert_allclose(a @ y, z, atol=5e-15)

    def test_shallow_restarts_do_not_transport_farther(self):
        paths = coefficient_path(self.p, [2, 16])
        for _ in range(8):
            a = np.zeros_like(self.p['U'])
            for __ in range(2):
                a = self.p['T'] @ a + self.p['U']
        np.testing.assert_array_equal(a, paths[2])
        self.assertGreater(np.max(np.abs(a - paths[16])), 1e-5)

    def test_corrupt_outer_edges_preserve_early_output(self):
        bad = self.adj.copy()
        for i, j in self.edges:
            if i != self.q and j != self.q:
                bad[i, j] *= -1
                bad[j, i] *= -1
        p_bad = {**self.p, 'T': bad / self.p['D'][:, None]}
        a = coefficient_path(self.p, [2, 16])
        b = coefficient_path(p_bad, [2, 16])
        np.testing.assert_array_equal(a[2][self.q], b[2][self.q])
        self.assertGreater(np.max(np.abs(a[16][self.q] - b[16][self.q])), 1e-5)

    def test_bayes_subset_has_no_extra_information(self):
        local = subset_coef(self.p, np.flatnonzero(self.dist[self.q] <= 1))
        self.assertGreaterEqual(risks(local, self.p)['mse_exact'] + 1e-13, self.p['bayes_var'])

    def test_full_bayes_sign_and_zero_prediction(self):
        self.assertGreaterEqual(risks(self.p['bayes_coef'], self.p)['acc_exact'], .5)
        self.assertEqual(risks(np.zeros(48), self.p)['acc_exact'], .5)

    def test_node_permutation_equivariance(self):
        order = np.random.default_rng(52104).permutation(49)
        q = int(np.flatnonzero(order == self.q)[0])
        p = make_problem(self.adj[np.ix_(order, order)], q, .05, 1)
        w = p['bayes_coef']
        back = np.empty(48)
        old_to_obs = {int(node): i for i, node in enumerate(self.p['obs'])}
        for i, node in enumerate(order[p['obs']]):
            back[old_to_obs[int(node)]] = w[i]
        np.testing.assert_allclose(back, self.p['bayes_coef'], atol=2e-14)


if __name__ == '__main__':
    unittest.main()
