from dataclasses import replace
import unittest
from unittest.mock import patch
from data_contract import Instance, Party
from fixtures import option, small_cases
from solver import score, exact_reference, solve_decomposed
from allocation_refinement import augment_routes, two_for_one, refine_allocation, improve_route_costs
from qubo import compile_bqm, estimate_bqm_variables

S = 'S3_capacityStability'


def case(parties, choices, capacities):
    return Instance(tuple(Party(pid, pax, .5) for pid, pax in parties),
        tuple(option(oid, pid, [(k, 'Y') for k in legs], cost_level=cost) for oid, pid, legs, cost in choices),
        {(k, 'Y'): v for k, v in capacities.items()})


class AllocationRefinementTests(unittest.TestCase):
    def test_two_reroute_chain_reaches_independent_exact_optimum(self):
        i = case([('P',1),('Q',1),('U',1)], [('PA','P',['A'],.1),('PB','P',['B'],.1),
            ('QB','Q',['B'],.1),('QC','Q',['C'],.1),('UA','U',['A'],.1)], {'A':1,'B':1,'C':1})
        incumbent = {'P':'PA','Q':'QB'}
        found, d = augment_routes(i, incumbent, S, max_depth=2)
        exact, _ = exact_reference(i)
        self.assertEqual(found, {'P':'PB','Q':'QC','U':'UA'})
        self.assertAlmostEqual(score(i,found)['objective'], score(i,exact)['objective'])
        self.assertEqual(incumbent, {'P':'PA','Q':'QB'})
        self.assertEqual(d['accepted'], 1)

    def test_augmentation_moves_multiple_occupants_without_splitting_parties(self):
        i = case([('P',1),('Q',1),('U',2)], [('PA','P',['A'],.1),('PB','P',['B'],.1),
            ('QA','Q',['A'],.1),('QC','Q',['C'],.1),('UA','U',['A'],.1)], {'A':2,'B':1,'C':1})
        a, _ = augment_routes(i, {'P':'PA','Q':'QA'}, S, max_depth=2)
        self.assertEqual(score(i,a)['assigned_passengers'], 4)
        self.assertEqual(score(i,a)['capacity_violations'], 0)

    def test_budget_or_depth_failure_rolls_back_every_mutation(self):
        i = small_cases()[2]; initial = {'SYN_A':'SYN_A1'}
        a, d = augment_routes(i, initial, S, max_depth=0, node_budget=1)
        self.assertEqual(a, initial)
        self.assertLessEqual(d['nodes'], d['attempts'])
        self.assertGreater(d['pnrs_hitting_node_budget'], 0)

    def test_two_for_one_serves_two_pnrs_with_same_passengers(self):
        i = case([('Q',2),('P',1),('U',1)], [('QA','Q',['A'],.1),('PA','P',['A'],.1),('UA','U',['A'],.1)], {'A':2})
        a, d = two_for_one(i, {'Q':'QA'}, S)
        self.assertEqual(set(a), {'P','U'}); self.assertEqual(d['exchanges'], 1)
        exact, _ = exact_reference(i)
        self.assertAlmostEqual(score(i,a)['objective'], score(i,exact)['objective'])
        self.assertEqual(score(i,a)['assigned_passengers'], 2)

    def test_two_for_one_releases_multiple_legs(self):
        i = case([('Q',1),('P',1),('U',1)], [('QAB','Q',['A','B'],.1),('PA','P',['A'],.1),('UB','U',['B'],.1)], {'A':1,'B':1})
        a, _ = two_for_one(i, {'Q':'QAB'}, S)
        self.assertEqual((len(a),score(i,a)['assigned_passengers']), (2,2))

    def test_two_for_one_rejects_passenger_loss_and_collective_overbooking(self):
        for old_pax, capacity in [(3,3),(1,1)]:
            i = case([('Q',old_pax),('P',1),('U',1)], [('QA','Q',['A'],.1),('PA','P',['A'],.1),('UA','U',['A'],.1)], {'A':capacity})
            a, _ = two_for_one(i, {'Q':'QA'}, S)
            self.assertEqual(a, {'Q':'QA'})

    def test_no_move_may_worsen_the_declared_airline_objective(self):
        i = case([('Q',2),('P',1),('U',1)], [('QA','Q',['A'],.1),('PA','P',['A'],100),('UA','U',['A'],100)], {'A':2})
        a, _ = two_for_one(i, {'Q':'QA'}, S)
        self.assertEqual(a, {'Q':'QA'})
        i = case([('P',1)], [('PA','P',['A'],100)], {'A':1})
        a, _ = augment_routes(i, {}, S)
        self.assertEqual(a,{})

    def test_route_cost_improvement_preserves_bookings_and_respects_inventory(self):
        i = case([('P',1),('Q',1)], [('PA','P',['A'],.2),('PB','P',['B'],.1),('QB','Q',['B'],.1)], {'A':1,'B':1})
        first, d = improve_route_costs(i, {'P':'PA'}, S)
        self.assertEqual(first, {'P':'PB'}); self.assertEqual(d['rerouted_pnrs'],1)
        self.assertGreater(d['objective_reduction'],0)
        blocked, d = improve_route_costs(i, {'P':'PA','Q':'QB'}, S)
        self.assertEqual(blocked, {'P':'PA','Q':'QB'})
        self.assertEqual(d['rerouted_pnrs'],0)

    def test_refinement_is_deterministic_and_never_accepts_infeasible_incumbent(self):
        i = small_cases()[2]; initial = {'SYN_A':'SYN_A1'}
        first, d = refine_allocation(i, initial); second, _ = refine_allocation(i, initial)
        self.assertEqual(first, second)
        self.assertGreaterEqual(d['after']['assigned_pnrs'], d['before']['assigned_pnrs'])
        with self.assertRaises(ValueError):
            refine_allocation(i, {'SYN_A':'SYN_A1','SYN_B':'SYN_B1'})
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError): augment_routes(i,initial,S,node_budget=bad)


class BQMConstructionEfficiencyTests(unittest.TestCase):
    def test_estimator_matches_compiler_with_zero_and_nonzero_residual_capacities(self):
        for instance in small_cases():
            ids = [p.id for p in instance.parties]; options = instance.eligible()[0]
            for ceiling in (0,1,2,10):
                remaining = {k:min(v,ceiling) for k,v in instance.capacities.items()}
                bqm = compile_bqm(instance, ids, options, remaining, S)
                self.assertEqual(estimate_bqm_variables(instance,ids,options,remaining),bqm.bqm.num_variables)

    def test_multiple_options_of_one_pnr_do_not_inflate_capacity_users(self):
        i = case([('P',1)], [('PA','P',['A'],.1),('PB','P',['A'],.2)], {'A':1})
        self.assertEqual(estimate_bqm_variables(i,['P'],i.options,i.capacities),3)
        self.assertEqual(compile_bqm(i,['P'],i.options,i.capacities,S).bqm.num_variables,3)

    def test_full_compilation_happens_once_per_accepted_chunk(self):
        i = small_cases()[0]
        with patch('solver.compile_bqm', wraps=compile_bqm) as compiler:
            _, report = solve_decomposed(i, reads=4, sweeps=20)
        self.assertEqual(compiler.call_count,len(report['chunks']))
        self.assertLess(compiler.call_count,len(i.parties))
