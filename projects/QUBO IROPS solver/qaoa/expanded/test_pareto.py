import unittest
from fixtures import small_cases
from solver import exact_reference
from expanded.pareto import dominates,frontier,exact_tradeoffs,point,coverage


class ParetoTests(unittest.TestCase):
    def test_competing_pnr_passenger_and_S3_objectives_are_retained(self):
        a={'assignment':{'a':'1'},'assigned_pnrs':2,'assigned_passengers':2,'objective':3.}
        b={'assignment':{'b':'2'},'assigned_pnrs':1,'assigned_passengers':4,'objective':2.}
        c={'assignment':{},'assigned_pnrs':1,'assigned_passengers':1,'objective':5.}
        self.assertEqual(frontier([c,a,b,a]),[a,b]);self.assertTrue(dominates(a,c));self.assertFalse(dominates(a,b))
        self.assertEqual(coverage([a],[a,b]),1)

    def test_exact_frontier_matches_independent_S3_reference(self):
        for instance in small_cases():
            result=exact_tradeoffs(instance);assignment,reference=exact_reference(instance)
            self.assertAlmostEqual(result['minimum_S3'],reference['objective'])
            self.assertEqual(coverage([point(instance,assignment)],result['frontier']),1)
            self.assertFalse(any(dominates(a,b) for a in result['frontier'] for b in result['frontier']))
        with self.assertRaises(ValueError):exact_tradeoffs(small_cases()[0],max_combinations=1)
