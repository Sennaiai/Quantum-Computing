from dataclasses import replace
import unittest
import json
import numpy as np
from data_contract import CONFIG, Instance, Party
from fixtures import small_cases, scale_case, option
from solver import (score, loads_for, feasible, reconcile, exact_reference,
                    solve_decomposed, catalog)


class PipelineTests(unittest.TestCase):
    def test_exact_reference_matches_hand_solved_capacity_case(self):
        case = small_cases()[0]; sol, r = exact_reference(case)
        self.assertEqual(set(sol), {'SYN_P2', 'SYN_P3'})
        self.assertEqual(r['assigned_pnrs'], 2); self.assertEqual(r['capacity_violations'], 0)
        self.assertEqual(r['combinations'], 8)

    def test_whole_pnr_not_split_across_option_departure_blocks(self):
        case = small_cases()[2]
        sol, r = solve_decomposed(case, reads=12, sweeps=100)
        self.assertEqual(sum(c['pnrs'] for c in r['chunks']), len(case.parties))
        self.assertEqual(r['assigned_pnrs'], 2); self.assertTrue(feasible(case, sol))

    def test_cross_block_ledger_and_global_replacement(self):
        case = small_cases()[1]
        sol, r = solve_decomposed(case, reads=12, sweeps=100)
        self.assertEqual(len(r['chunks']), 2)
        self.assertEqual(sol, {'SYN_B': 'SYN_B1'})
        self.assertEqual(r['before_global_reconciliation']['assigned_pnrs'], 1)
        self.assertLess(r['objective'], r['before_global_reconciliation']['objective'])
        self.assertEqual(r['capacity_violations'], 0)

    def test_budget_chunks_keep_every_pnr_and_never_randomly_drop_rows(self):
        case = scale_case(24)
        sol, r = solve_decomposed(case, max_variables=24, reads=8, sweeps=80)
        serviceable = sum(bool(v) for v in catalog(case, 'S3_capacityStability').values())
        self.assertEqual(sum(c['pnrs'] for c in r['chunks']), serviceable)
        self.assertTrue(all(c['variables'] <= 24 for c in r['chunks']))
        self.assertTrue(feasible(case, sol))
        with self.assertRaises(ValueError): solve_decomposed(case, max_variables=1, reads=8, sweeps=80)

    def test_global_tail_can_recover_option_removed_from_top_k(self):
        case = small_cases()[2]
        sol, r = solve_decomposed(case, top_k=1, reads=12, sweeps=100)
        self.assertEqual(r['pruned_top_k_options'], 1)
        self.assertEqual(r['assigned_pnrs'], 2)
        self.assertEqual(sol['SYN_A'], 'SYN_A2')

    def test_no_eligible_options_remain_in_denominator_and_unassigned_objective(self):
        case = small_cases()[3]; sol, r = solve_decomposed(case, reads=8, sweeps=80)
        self.assertEqual(r['pnrs'], 5); self.assertEqual(r['assigned_pnrs'], 2)
        self.assertEqual(r['unassigned_pnrs'], 3); self.assertEqual(r['assignment_rate'], .4)
        self.assertEqual(r['excluded_options'], {'after_72_hours': 1, 'party_exceeds_leg_capacity': 1})
        self.assertAlmostEqual(r['objective'], r['assigned_scenario_cost']+30)

    def test_raw_and_repaired_diagnostics_are_separate(self):
        case = scale_case(48)
        sol, r = solve_decomposed(case, reads=4, sweeps=20)
        self.assertTrue(feasible(case, sol))
        self.assertIn('before_global_reconciliation', r)
        for chunk in r['chunks']:
            self.assertLessEqual(chunk['samples_with_zero_constraint_residual'], 4)
            self.assertIn('raw_best_one_hot_violations', chunk)
            self.assertIn('after_chunk_repair_assignments', chunk)
        self.assertLessEqual(r['objective'], r['before_global_reconciliation']['objective']+1e-9)

    def test_all_fixed_seeds_and_scenarios_return_feasible_solutions(self):
        for case in small_cases():
            for scenario in CONFIG['scenarios']:
                for seed in (7, 11):
                    sol, r = solve_decomposed(case, scenario, seed, reads=12, sweeps=100)
                    self.assertTrue(feasible(case, sol)); self.assertEqual(r['capacity_violations'], 0)
                    self.assertGreaterEqual(r['objective']+1e-8, exact_reference(case, scenario)[1]['objective'])

    def test_repair_conserves_party_size_and_separates_cabin_capacity(self):
        case = small_cases()[3]; sol = reconcile(case, {}, 'S3_capacityStability')
        loads = loads_for(case, sol)
        self.assertEqual(loads[('SYN_F1', 'C')], 2)
        self.assertEqual(loads[('SYN_F1', 'Y')], 1)
        self.assertEqual(loads[('SYN_F2', 'Y')], 2)
        self.assertEqual(score(case, sol)['assigned_passengers'], 3)

    def test_repair_drops_overload_without_hiding_unknown_assignments(self):
        case = small_cases()[0]
        sol = reconcile(case, {o.party: o.id for o in case.options}, 'S3_capacityStability')
        self.assertEqual(set(sol), {'SYN_P2', 'SYN_P3'})
        for bad in ({'unknown': 'SYN_O1'}, {'SYN_P1': 'ghost'}, {'SYN_P1': 'SYN_O2'}):
            with self.assertRaises(ValueError): score(case, bad)
            with self.assertRaises(ValueError): reconcile(case, bad, 'S3_capacityStability')

    def test_scoring_objective_is_identical_for_every_solver(self):
        case = small_cases()[2]
        for assignment in ({'SYN_A': 'SYN_A1'}, exact_reference(case)[0], reconcile(case, {}, 'S3_capacityStability')):
            r = score(case, assignment)
            self.assertAlmostEqual(r['objective'], sum(next(o for o in case.options if o.id == oid).cost('S3_capacityStability') for oid in assignment.values())+10*(len(case.parties)-len(assignment)))

    def test_seed_repeatability_and_validation_of_solver_parameters(self):
        case = scale_case(24)
        a, ar = solve_decomposed(case, seed=7, reads=8, sweeps=80)
        b, br = solve_decomposed(case, seed=7, reads=8, sweeps=80)
        self.assertEqual(a, b)
        self.assertEqual([c['best_sample_energy'] for c in ar['chunks']], [c['best_sample_energy'] for c in br['chunks']])
        for params in ({'top_k': 0}, {'reads': 0}, {'sweeps': 0}, {'seed': True}):
            with self.assertRaises(ValueError): solve_decomposed(case, **params)
        for fn in (catalog, score):
            with self.assertRaises(ValueError): fn(case, 'bad') if fn is catalog else fn(case, {}, 'bad')
        with self.assertRaises(ValueError): exact_reference(case, max_combinations=1)

    def test_all_unassigned_zero_candidate_case(self):
        case = Instance((Party('SYN_NONE', 2, .5),), (), {})
        sol, r = solve_decomposed(case)
        self.assertEqual(sol, {}); self.assertEqual(r['objective'], 10.)
        self.assertEqual(r['chunks'], []); self.assertEqual(r['assignment_rate'], 0.)

    def test_complete_report_is_strict_json_serializable(self):
        _, report = solve_decomposed(small_cases()[0], reads=8, sweeps=80)
        decoded = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(decoded['assigned_pnrs'], report['assigned_pnrs'])
