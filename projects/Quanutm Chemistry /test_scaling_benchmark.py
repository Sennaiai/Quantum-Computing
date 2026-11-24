"""Scientific/reference and bounded-resource checks for active-space scaling."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pennylane as qml

from scaling_benchmark import (
    build_case, build_model, cas_reference, load_models, map_integrals,
    pauli_records, pbe_grid_control, qwc_groups, reference_space, resource_metrics,
    sector_basis, sector_matrix, validate_hierarchy, validate_space,
)


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {name: build_model(name) for name in load_models()}
        cls.small, cls.small_operator = build_case(cls.models['ammonia'], 2, 2)

    def test_molecular_counts_and_geometry_provenance(self):
        expected = {'ammonia': (10, 8, 16), 'carbonic_acid': (32, 22, 44),
                    'bicarbonate': (32, 21, 42)}
        for name, model in self.models.items():
            b = model.baseline
            self.assertEqual((b['electrons'], b['spatial_orbitals'], b['full_space_qubits']), expected[name])
            if name != 'ammonia':
                self.assertEqual(b['geometry_provenance']['random_seed'], 1729)
                self.assertEqual(b['geometry_provenance']['uff_status'], 0)
        self.assertEqual(self.models['bicarbonate'].baseline['charge'], -1)

    def test_active_space_capacity_parity_and_indices(self):
        self.assertEqual(validate_space(32, 22, 6, 6), (list(range(13)), list(range(13, 19))))
        for values in [(32, 22, 5, 6), (10, 8, 8, 8), (10, 8, 12, 6),
                       (10, 8, 6, 2), (10, 8, 0, 2), (10, 8, 4., 4)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError): validate_space(*values)

    def test_sector_projection_pauli_phases_and_dense_reference(self):
        # Paired XX+YY contributions cancel couplings that change occupation.
        toy = .2*qml.Identity(0) + .3*qml.Z(0) + .4*(qml.X(0)@qml.X(2)) + .4*(qml.Y(0)@qml.Y(2))
        basis, block = sector_matrix(toy, 2, 2)
        full = qml.matrix(toy, wire_order=range(4))
        np.testing.assert_allclose(block, full[np.ix_(basis, basis)], atol=1e-12, rtol=0)
        broken = qml.X(0)
        with self.assertRaisesRegex(ValueError, 'leaves'):
            sector_matrix(broken, 2, 2)

    def test_molecular_sector_matches_casci_and_hf(self):
        row, operator = build_case(self.models['ammonia'], 4, 4)
        self.assertAlmostEqual(row['sector_exact_hartree'], row['casci_hartree'], delta=1e-8)
        self.assertAlmostEqual(row['hf_qubit_hartree'], self.models['ammonia'].baseline['rhf_hartree'], delta=1e-8)
        basis, block = sector_matrix(operator, 4, 4)
        np.testing.assert_allclose(block, qml.matrix(operator, wire_order=range(8))[np.ix_(basis, basis)], atol=1e-12, rtol=0)

    def test_exact_checks_never_build_a_full_qubit_matrix(self):
        with patch('scaling_benchmark.qml.matrix', side_effect=AssertionError('Full matrix forbidden')):
            row, _ = build_case(self.models['ammonia'], 6, 6)
        self.assertEqual(row['sector_dimension'], 400)
        self.assertAlmostEqual(row['sector_exact_spin_squared'], 0., delta=1e-8)

    def test_resource_guards_precede_exponential_allocation(self):
        with self.assertRaisesRegex(ValueError, '12 qubits'):
            sector_matrix(None, 8, 8)
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            build_case(self.models['carbonic_acid'], 8, 8)
        self.assertEqual(len(sector_basis(6, 6)), 400)

    def test_large_operator_opt_in_does_not_diagonalize_qubits(self):
        row, _ = build_case(self.models['carbonic_acid'], 8, 8, allow_large=True)
        self.assertEqual(row['qubits'], 16)
        self.assertNotIn('sector_exact_hartree', row)
        self.assertIn('not performed', row['qubit_diagonalization'])

    def test_identity_constant_counted_once(self):
        tensors = self.small['active_integrals']
        _, no_core = map_integrals(0., tensors['one_body_hartree'], tensors['two_body_chemist_hartree'])
        basis, original = sector_matrix(self.small_operator, 2, 2)
        _, stripped = sector_matrix(no_core, 2, 2)
        np.testing.assert_allclose(original-stripped, tensors['core_energy_hartree']*np.eye(len(basis)), atol=1e-10, rtol=0)
        self.assertGreater(abs(self.small['identity_coefficient_hartree']-self.small['core_constant_hartree']), .01)

    def test_grouping_and_norm_metrics_have_defined_meaning(self):
        words = ['XI', 'ZI', 'IZ', 'ZZ']
        groups = qwc_groups(words)
        self.assertEqual(groups, qwc_groups(list(reversed(words))))
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(w for g in groups for w in g['words']), sorted(words))
        for group in groups:
            for word in group['words']:
                self.assertTrue(all(a == 'I' or a == b for a, b in zip(word, group['basis'])))
        toy = 2*qml.Identity(0)+.5*qml.X(0)-.25*qml.Z(1)
        metrics = resource_metrics(toy, 2)
        self.assertEqual(metrics['pauli_terms'], 3)
        self.assertEqual(metrics['identity_coefficient_hartree'], 2.)
        self.assertEqual(metrics['nonidentity_l1_hartree'], .75)
        self.assertEqual(metrics['greedy_qwc_settings'], 1)

    def test_nested_hierarchy_and_classical_reference(self):
        model = self.models['ammonia']
        _, reference = cas_reference(model, *reference_space(model))
        self.assertAlmostEqual(reference['casci_hartree'], -55.5102110930238, delta=1e-8)
        larger, _ = build_case(model, 4, 4)
        validate_hierarchy([self.small, larger], reference)
        invalid = dict(larger, casci_hartree=self.small['casci_hartree']+.01)
        with self.assertRaises(ValueError): validate_hierarchy([self.small, invalid], reference)

    def test_failed_references_are_rejected(self):
        with patch('scaling_benchmark.scf.RHF') as hf:
            hf.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, 'RHF'):
                build_model('ammonia')
        with patch('scaling_benchmark.mcscf.CASCI') as cas:
            cas.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, 'CASCI'):
                cas_reference(self.models['ammonia'], 2, 2)

    def test_pbe_grid_controls_at_all_model_geometries(self):
        for name, model in self.models.items():
            with self.subTest(name=name):
                self.assertLess(abs(pbe_grid_control(model)['difference_hartree']), 1e-6)

    def test_saved_models_replay_resources_and_reference_energies(self):
        evidence = json.loads((Path(__file__).parent/'benchmarks/scaling/results.json').read_text())
        self.assertEqual(evidence['profile'], 'default')
        self.assertEqual({m['baseline']['name'] for m in evidence['molecules']}, set(self.models))
        for molecule in evidence['molecules']:
            model = self.models[molecule['baseline']['name']]
            self.assertAlmostEqual(molecule['baseline']['rhf_hartree'], model.rhf.e_tot, delta=1e-8)
            self.assertEqual([r['qubits'] for r in molecule['cases']], [4, 8, 12])
            validate_hierarchy(molecule['cases'], molecule['reference'])
            for row in molecule['cases']:
                with self.subTest(molecule=model.name, qubits=row['qubits']):
                    tensors = row['active_integrals']
                    _, operator = map_integrals(tensors['core_energy_hartree'],
                                                tensors['one_body_hartree'], tensors['two_body_chemist_hartree'])
                    metrics = resource_metrics(operator, row['qubits'])
                    for key in ['pauli_terms', 'nonidentity_terms', 'max_pauli_weight', 'greedy_qwc_settings']:
                        self.assertEqual(metrics[key], row[key])
                    self.assertAlmostEqual(metrics['nonidentity_l1_hartree'], row['nonidentity_l1_hartree'], delta=1e-10)
                    _, block = sector_matrix(operator, row['active_electrons'], row['active_orbitals'])
                    self.assertAlmostEqual(np.linalg.eigvalsh(block)[0], row['casci_hartree'], delta=1e-8)
                    _, fresh = cas_reference(model, row['active_electrons'], row['active_orbitals'])
                    self.assertAlmostEqual(fresh['casci_hartree'], row['casci_hartree'], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
