# SPDX-License-Identifier: Apache-2.0
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'roadmap_resolution'))
import ising_core as ic
import run122

class RoadmapResolutionTests(unittest.TestCase):
    def test_graphs_are_valid(self):
        for fam in ('barbell','tri_ladder','torus3x4','cycle','cactus','ladder','grid'):
            edges,q=ic.graph(fam)
            self.assertTrue(0 <= q < ic.N)
            self.assertTrue(all(u != v for u,v in edges))
            self.assertEqual(len(edges), len(set(edges)))

    def test_bp_beliefs_are_finite(self):
        edges,q=ic.graph('grid')
        J=np.full(len(edges),0.5)
        obs=np.ones((8,ic.N),dtype=np.int8)
        b=ic.beliefs(edges,J,q,obs,0.2,(1,2,8))
        self.assertEqual(set(b),{1,2,8})
        self.assertTrue(all(np.isfinite(x).all() for x in b.values()))

    def test_stage122_edge_order_is_frozen(self):
        edges,q=run122.stage122_graph('cycle')
        self.assertEqual(edges[-1],(11,0))
        self.assertEqual(q,0)
        cactus,_=run122.stage122_graph('cactus')
        self.assertEqual(cactus[2],(2,0))

    def test_exact_map_binary(self):
        edges,q=run122.stage122_graph('cycle')
        J=np.full(len(edges),0.4)
        m=run122.exact_map(edges,J,q,0.2)
        self.assertEqual(len(m),2**11)
        self.assertTrue(np.isin(m,[-1,1]).all())

if __name__=='__main__': unittest.main()
