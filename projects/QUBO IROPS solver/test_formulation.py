from dataclasses import replace
from itertools import product
import unittest
import dimod
import numpy as np

from data_contract import CONFIG, Instance, Party
from fixtures import small_cases, option
from qubo import bounded_weights, compile_bqm, resources
from solver import exact_reference, score, feasible


class EncodingTests(unittest.TestCase):
    def test_bounded_slack_has_exact_integer_support(self):
        for limit in range(40):
            weights = bounded_weights(limit)
            values = {sum(w*x for w, x in zip(weights, bits)) for bits in product((0, 1), repeat=len(weights))}
            self.assertEqual(values, set(range(limit+1)))
        for bad in (-1, 2.5, True, np.nan):
            with self.assertRaises(ValueError): bounded_weights(bad)

    def test_every_binary_energy_matches_direct_constraint_formula(self):
        for case in small_cases():
            compiled = compile_bqm(case, [p.id for p in case.parties], case.eligible()[0],
                                   dict(case.capacities), 'S3_capacityStability')
            for sample, energy in dimod.ExactSolver().sample(compiled.bqm).data(['sample', 'energy']):
                self.assertAlmostEqual(energy, compiled.semantic_energy(sample), places=8)

    def test_all_ground_states_are_feasible_and_match_independent_reference_all_scenarios(self):
        for case in small_cases():
            for scenario in CONFIG['scenarios']:
                c = compile_bqm(case, [p.id for p in case.parties], case.eligible()[0], dict(case.capacities), scenario)
                reference, report = exact_reference(case, scenario)
                states = dimod.ExactSolver().sample(c.bqm).lowest(atol=1e-7)
                for sample in states.samples():
                    selected = c.decode(sample)
                    self.assertTrue(all(len(v) <= 1 for v in selected.values()))
                    assignment = {p: ids[0] for p, ids in selected.items()}
                    self.assertTrue(feasible(case, assignment))
                    self.assertAlmostEqual(score(case, assignment, scenario)['objective'], report['objective'], places=8)
                    self.assertAlmostEqual(c.semantic_energy(sample), report['objective'], places=8)

    def test_collective_overload_not_detectable_by_pairwise_conflicts(self):
        case = small_cases()[0]
        self.assertTrue(all(a.pax+b.pax <= 2 for i, a in enumerate(case.parties) for b in case.parties[i+1:]))
        c = compile_bqm(case, [p.id for p in case.parties], case.options, dict(case.capacities), 'S3_capacityStability')
        sample = {v: int(v.startswith('option:')) for v in c.bqm.variables}
        self.assertGreater(c.semantic_energy(sample), c.penalty)
        self.assertEqual(score(case, {o.party: o.id for o in case.options})['excess_seats'], 1)

    def test_ising_and_upper_triangular_qubo_offsets_all_states(self):
        case = small_cases()[0]
        c = compile_bqm(case, [p.id for p in case.parties], case.options, dict(case.capacities), 'S3_capacityStability')
        h, j, offset = c.bqm.to_ising(); q, qoff = c.bqm.to_qubo()
        for sample, energy in dimod.ExactSolver().sample(c.bqm).data(['sample', 'energy']):
            spins = {v: 2*int(x)-1 for v, x in sample.items()}
            value = offset+sum(h[v]*spins[v] for v in h)+sum(x*spins[a]*spins[b] for (a, b), x in j.items())
            binary = qoff+sum(value*sample[a]*sample[b] for (a, b), value in q.items())
            self.assertAlmostEqual(energy, value, places=8); self.assertAlmostEqual(energy, binary, places=8)

    def test_penalty_certificate_and_bad_samples_fail_closed(self):
        case = small_cases()[0]
        c = compile_bqm(case, [p.id for p in case.parties], case.options, dict(case.capacities), 'S3_capacityStability')
        self.assertGreater(c.penalty, c.objective_range_bound)
        with self.assertRaises(ValueError): compile_bqm(case, [p.id for p in case.parties], case.options, dict(case.capacities), 'S3_capacityStability', penalty=c.objective_range_bound)
        with self.assertRaises(ValueError): c.decode({})
        bad = {v: 2 for v in c.bqm.variables}
        with self.assertRaises(ValueError): c.semantic_energy(bad)
        with self.assertRaises(ValueError): compile_bqm(case, ['unknown'], (), dict(case.capacities), 'S3_capacityStability')
        with self.assertRaises(ValueError): compile_bqm(case, [case.parties[0].id], case.options, dict(case.capacities), 'S3_capacityStability')
        with self.assertRaises(ValueError): compile_bqm(case, [p.id for p in case.parties], case.options, {}, 'S3_capacityStability')

    def test_negative_candidate_costs_do_not_break_penalty_guarantee(self):
        case = Instance((Party('SYN_P', 1, 1.),),
                        (option('SYN_O', 'SYN_P', [('SYN_F', 'Y')], cvm=1., cost_level=0.),), {('SYN_F', 'Y'): 1})
        c = compile_bqm(case, ['SYN_P'], case.options, dict(case.capacities), 'S2_highCVM')
        self.assertLess(case.options[0].cost('S2_highCVM'), 0)
        self.assertEqual(c.decode(dimod.ExactSolver().sample(c.bqm).first.sample), {'SYN_P': ['SYN_O']})

    def test_resources_count_all_option_dummy_and_slack_variables(self):
        case = small_cases()[0]
        c = compile_bqm(case, [p.id for p in case.parties], case.options, dict(case.capacities), 'S3_capacityStability')
        r = resources(c)
        self.assertEqual(r['variables'], 8)
        self.assertEqual(r['option_variables']+r['unassigned_variables']+r['slack_variables'], r['variables'])
        self.assertEqual(r['logical_annealing_qubits_lower_bound'], r['variables'])
        self.assertEqual(r['p1_all_to_all_CX_count_before_cancellation'], 2*r['couplers'])

    def test_large_offsets_obey_explicit_roundoff_bound(self):
        parties = tuple(Party(f'SYN_P{i}', 200, .5) for i in range(6))
        options = tuple(option(f'SYN_O{i}', p.id, [('SYN_F', 'Y')], cost_level=.123456789) for i, p in enumerate(parties))
        case = Instance(parties, options, {('SYN_F', 'Y'): 1000})
        c = compile_bqm(case, [p.id for p in parties], options, dict(case.capacities), 'S3_capacityStability')
        rng = np.random.default_rng(409)
        for _ in range(100):
            sample = dict(zip(c.bqm.variables, rng.integers(0, 2, c.bqm.num_variables)))
            self.assertLessEqual(abs(c.bqm.energy(sample)-c.semantic_energy(sample)), c.roundoff_bound(sample))

    def test_native_int8_samples_cannot_overflow_large_passenger_totals(self):
        parties = tuple(Party(f'SYN_P{i}', 13, .5) for i in range(20))
        options = tuple(option(f'SYN_O{i}', p.id, [('SYN_F', 'Y')]) for i, p in enumerate(parties))
        case = Instance(parties, options, {('SYN_F', 'Y'): 200})
        c = compile_bqm(case, [p.id for p in parties], options, dict(case.capacities), 'S3_capacityStability')
        sample = {v: np.int8(v.startswith('option:')) for v in c.bqm.variables}
        expected = sum(o.cost('S3_capacityStability') for o in options)+c.penalty*(260-200)**2
        self.assertAlmostEqual(c.semantic_energy(sample), expected, places=7)
        self.assertLessEqual(abs(c.bqm.energy(sample)-expected), c.roundoff_bound(sample))
