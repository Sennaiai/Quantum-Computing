"""Independent spectral, dynamical, and convergence checks for CD transport."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'python_classes')]
from transport_benchmark import (
    TransportConfig, exact_ground_projector_agp, run_transport_benchmark,
    smooth_schedule, state_distance,
)

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.])


def fixture(theta=(1.,), interactions=()):
    return {'hamiltonian_A_interaction_ising': {'fixture': {
        'selected_features': list(range(len(theta))),
        'pauli_terms': [{'coefficient': float(c), 'pauli_string': word,
                         'locality': len(word.split())} for c, word in interactions]}}}


class SpectralTransportTests(unittest.TestCase):
    def test_smooth_schedule_has_zero_endpoint_velocity_and_is_monotonic(self):
        self.assertEqual(smooth_schedule(0), (0., 0.))
        self.assertEqual(smooth_schedule(1), (1., 0.))
        samples = [smooth_schedule(u) for u in np.linspace(0, 1, 21)]
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(samples, samples[1:])))
        for value in (True, -1e-3, 1.1, np.nan, '0.5'):
            with self.assertRaises(ValueError):
                smooth_schedule(value)

    def test_spectral_gauge_matches_exact_one_qubit_expression(self):
        for s in (0., .1, .25, .5, .9, 1.):
            H = -(1-s)*X+s*Z
            gauge, _, _ = exact_ground_projector_agp(H, X+Z)
            target = Y/(2*((1-s)**2+s*s))
            np.testing.assert_allclose(gauge, target, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(gauge, gauge.conj().T, atol=1e-12)

    def test_projector_gauge_is_basis_covariant_and_cancels_ground_transitions(self):
        H, D = -.7*X+.4*Z, X+Z
        unitary = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
        A, _, _ = exact_ground_projector_agp(H, D)
        rotated, values, vectors = exact_ground_projector_agp(unitary@H@unitary.conj().T,
                                                             unitary@D@unitary.conj().T)
        np.testing.assert_allclose(rotated, unitary@A@unitary.conj().T, atol=1e-12)
        G = D+1j*(A@H-H@A)
        _, eigvec = np.linalg.eigh(H)
        self.assertLess(np.linalg.norm((eigvec.conj().T@G@eigvec)[1:, 0]), 1e-12)

    def test_excited_degeneracy_allowed_but_ground_degeneracy_fails_closed(self):
        H = np.diag([-2., 0., 0., 1.])
        D = np.ones((4, 4))
        gauge, _, vectors = exact_ground_projector_agp(H, D)
        self.assertTrue(np.isfinite(gauge).all())
        G = D+1j*(gauge@H-H@gauge)
        self.assertLess(np.linalg.norm((vectors.conj().T@G@vectors)[1:, 0]), 1e-12)
        with self.assertRaisesRegex(ValueError, 'gap is unresolved'):
            exact_ground_projector_agp(np.diag([0., 0., 1., 2.]), D)
        with self.assertRaisesRegex(ValueError, 'gap is unresolved'):
            exact_ground_projector_agp(np.diag([0., 1e-12]), X, gap_tolerance=1e-10)

    def test_spectral_input_validation_and_global_phase_distance(self):
        state = np.array([1., 1j])/np.sqrt(2)
        self.assertLess(state_distance(state, 3*np.exp(.73j)*state), 1e-14)
        for a, b in ((np.array([1e308, 1e308]), state), (np.zeros(2), state), (np.ones(3), state), (np.array([np.nan, 1]), state)):
            with self.assertRaises(ValueError):
                state_distance(a, b)
        for H, D in ((np.eye(2), np.eye(3)), (np.ones(2), np.eye(2)),
                     (np.array([[0, 1], [0, 1]]), np.eye(2)), (np.diag([0, np.inf]), X)):
            with self.assertRaises(ValueError):
                exact_ground_projector_agp(H, D)


class ExperimentTests(unittest.TestCase):
    def test_one_qubit_exact_ansatz_transport_and_qsvt_snapshots(self):
        config = TransportConfig(durations=(.2, 1., 2.), grid_points=9,
                                 midpoint_steps=(16, 32, 64))
        result = run_transport_benchmark(fixture(), 'fixture', [1.], config)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['rows']), 9)
        rows = {(r['duration'], r['method']): r for r in result['rows']}
        self.assertLess(rows[(.2, 'bare')]['ground_fidelity'], .99)
        for T in config.durations:
            self.assertGreater(rows[(T, 'first_order_cd')]['ground_fidelity'], 1-1e-7)
            self.assertGreater(rows[(T, 'exact_ground_cd')]['ground_fidelity'], 1-1e-7)
            for method in ('first_order_cd', 'exact_ground_cd'):
                self.assertGreater(rows[(T, method)]['sampled_peak_control_norm'], 0.)
                self.assertGreater(rows[(T, method)]['integrated_control_hs_squared_quadrature'], 0.)
        self.assertEqual(len(result['qsvt_snapshots']), 3)
        for row in result['qsvt_snapshots']:
            self.assertLessEqual(row['reference']['recovered_evolution_error'], row['epsilon'])
        json.dumps(result, allow_nan=False)

    def test_interacting_two_qubit_variational_residual_and_independent_convergence(self):
        theta = [.4, .8]
        config = TransportConfig(durations=(.3, 1.), grid_points=9,
                                 midpoint_steps=(16, 32, 64), qsvt_snapshots=False)
        result = run_transport_benchmark(fixture(theta, ((.7, 'Z0 Z1'),)), 'fixture', theta, config)
        for row in result['residual_grid']:
            self.assertLessEqual(row['first_order_cd_hs_action'], row['bare_hs_action']+1e-10)
            self.assertLess(row['production_dense_agp_error'], 1e-10)
            self.assertLess(row['exact_ground_cd_ground_transition_residual'], 1e-9)
        for method in ('bare', 'first_order_cd', 'exact_ground_cd'):
            rows = [r for r in result['convergence'] if r['method'] == method]
            self.assertLess(rows[-1]['phase_aligned_state_distance'], rows[0]['phase_aligned_state_distance'])
            self.assertLess(rows[-1]['phase_aligned_state_distance'], config.midpoint_state_tolerance)
        self.assertGreater(result['observed_minimum_ground_gap'], 0.)
        self.assertIn('distinct from API', result['initial_state'])
        self.assertEqual(len(result['problem_hamiltonian_sha256']), 64)
        self.assertGreater(result['driver_excited_degenerate_pairs'], 0)

    def test_resource_budgets_and_gap_fail_before_integration(self):
        config = TransportConfig(max_system_qubits=1, qsvt_snapshots=False)
        with patch('transport_benchmark.hamiltonian_matrix', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(ValueError, 'qubit budget'):
                run_transport_benchmark(fixture((.4, .8)), 'fixture', [.4, .8], config)
        with self.assertRaisesRegex(ValueError, 'gap is unresolved'):
            run_transport_benchmark(fixture((0.,)), 'fixture', [0.], config)
        limited = TransportConfig(durations=(1.,), midpoint_steps=(16, 32), max_rhs_evaluations=32,
                                  qsvt_snapshots=False)
        with self.assertRaisesRegex(ValueError, 'evaluation budget'):
            run_transport_benchmark(fixture(), 'fixture', [1.], limited)

    def test_large_spectral_reference_rejected_before_diagonalization(self):
        with patch('transport_benchmark.np.linalg.eigh', side_effect=AssertionError('must not diagonalize')):
            with self.assertRaisesRegex(ValueError, 'reference budget'):
                exact_ground_projector_agp(np.eye(128), np.eye(128))

    def test_config_rejects_ambiguous_and_unbounded_inputs(self):
        with self.assertRaisesRegex(ValueError, 'config must'):
            run_transport_benchmark(fixture(), 'fixture', [1.], False)
        for kwargs in ({'durations': (True,)}, {'durations': (1., 1.)}, {'durations': (np.inf,)},
                       {'midpoint_steps': (32, 16)}, {'midpoint_steps': (True, 32)},
                       {'grid_points': 2}, {'rtol': 1e-16}, {'atol': np.nan},
                       {'gap_tolerance': 0}, {'max_system_qubits': 7}, {'qsvt_snapshots': 1}, {'norm_tolerance': 1.}):
            with self.assertRaises(ValueError):
                TransportConfig(**kwargs).validate()


if __name__ == '__main__':
    unittest.main()
