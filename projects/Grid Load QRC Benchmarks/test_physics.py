import math
import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, SparsePauliOp
from reservoirs import CVReservoir, SpinReservoir, spin_hamiltonian, finite_windows, echo_features
from physics_checks import gaussian_pennylane


class CVPhysicsTests(unittest.TestCase):
    def setUp(self): self.windows = np.random.default_rng(17).normal(size=(6, 8, 7))

    def test_gaussian_full_window_matches_independent_coherent_amplitudes(self):
        for seed in (11, 29, 47):
            r = CVReservoir(seed, kerr=0.)
            f, d = r.features(self.windows)
            np.testing.assert_allclose(f, r.gaussian_reference(self.windows), atol=1e-10)
            self.assertLess(d['max_norm_error'], 1e-12)

    def test_native_pennylane_gaussian_matches_features(self):
        r = CVReservoir(11, kerr=0.)
        np.testing.assert_allclose(r.features(self.windows[:1])[0][0], gaussian_pennylane(r, self.windows[0]), atol=1e-10)

    def test_kerr_single_mode_against_analytic_coherent_expectation(self):
        d = 30; alpha = .7+.2j; kappa = .23
        n = np.arange(d)
        coherent = np.array([np.exp(-abs(alpha)**2/2)*alpha**j/np.sqrt(float(math.factorial(j))) for j in n])
        state = coherent*np.exp(1j*kappa*n**2)
        value = np.sum(state[:-1].conj()*state[1:]*np.sqrt(n[1:]))
        expected = alpha*np.exp(1j*kappa)*np.exp(abs(alpha)**2*(np.exp(2j*kappa)-1))
        self.assertAlmostEqual(abs(value-expected), 0., places=12)
        # A linear phase rotation is not the Kerr transformation of this state.
        self.assertGreater(abs(value-alpha*np.exp(1j*kappa)), .02)

    def test_beamsplitter_one_photon_action_and_fock_sign(self):
        r = CVReservoir(11, kerr=0., cutoff=6); d = r.cutoff
        state = np.zeros(d*d, complex); state[d] = np.exp(-1j*r.phi[0])
        out = r.unitary@state
        expected = np.zeros(d*d, complex)
        expected[d] = np.cos(r.theta); expected[1] = np.exp(1j*r.phase)*np.sin(r.theta)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_kerr_multi_mode_convergence_and_non_gaussian_difference(self):
        r = CVReservoir(11); high = CVReservoir(11, cutoff=14)
        f, d = r.features(self.windows)
        np.testing.assert_allclose(f, high.features(self.windows)[0], atol=1e-8)
        self.assertLess(d['max_cutoff_boundary_probability'], 1e-7)
        gaussian = CVReservoir(11, kerr=0.).features(self.windows)[0]
        self.assertGreater(np.max(abs(f-gaussian)), 1e-4)

    def test_complete_kerr_state_against_full_hilbert_exponentials(self):
        r = CVReservoir(29, cutoff=6); d = r.cutoff
        a = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)
        modes = [np.kron(a, np.eye(d)), np.kron(np.eye(d), a)]
        number = [op.conj().T@op for op in modes]
        rotation = expm(1j*sum(r.phi[q]*number[q] for q in range(2)))
        bs = expm(r.theta*(np.exp(1j*r.phase)*modes[0]@modes[1].conj().T
                           -np.exp(-1j*r.phase)*modes[0].conj().T@modes[1]))
        kerr = expm(1j*sum(r.kappas[q]*(number[q]@number[q]) for q in range(2)))
        state = np.zeros(d*d, complex); state[0] = 1
        for row in r.encode(self.windows[:1])[0]:
            for q in range(2): state = expm(row[q]*(modes[q].conj().T-modes[q]))@state
            state = kerr@bs@rotation@state
        np.testing.assert_allclose(state, r.evolve(self.windows[:1])[0].ravel(), atol=1e-12)

    def test_window_reset_and_vacuum(self):
        for cls in (CVReservoir, SpinReservoir):
            r = cls(11); full = r.features(self.windows)[0]
            separate = np.vstack([r.features(w[None])[0] for w in self.windows])
            np.testing.assert_allclose(full, separate, atol=1e-12)
        np.testing.assert_allclose(CVReservoir(11).features(np.zeros((2, 8, 7)))[0], 0., atol=1e-12)

    def test_invalid_windows_and_cutoffs(self):
        for x in (np.zeros((2, 7)), np.zeros((0, 8, 7)), np.zeros((2, 8, 6)), np.full((2, 8, 7), np.nan)):
            with self.assertRaises(ValueError): finite_windows(x)
        for d in (True, 0, 21, 8.5):
            with self.assertRaises(ValueError): CVReservoir(11, cutoff=d)


class SpinPhysicsTests(unittest.TestCase):
    def test_explicit_hamiltonian_matches_qiskit_paulis(self):
        r = SpinReservoir(11, n=3); labels = []
        for q in range(r.n):
            for p, value in [('X', r.hx[q]), ('Z', r.hz[q])]:
                label = ['I']*r.n; label[r.n-1-q] = p; labels.append((''.join(label), value))
        for q, value in enumerate(r.coupling):
            label = ['I']*r.n; label[r.n-1-q] = label[r.n-2-q] = 'Z'; labels.append((''.join(label), value))
        np.testing.assert_allclose(r.hamiltonian, SparsePauliOp.from_list(labels).to_matrix(), atol=1e-14)
        np.testing.assert_allclose(r.unitary.conj().T@r.unitary, np.eye(8), atol=1e-12)

    def test_full_state_window_matches_qiskit(self):
        windows = np.random.default_rng(23).normal(size=(3, 8, 7))
        r = SpinReservoir(11)
        for window, state in zip(windows, r.evolve(windows)):
            np.testing.assert_allclose(state, Statevector.from_instruction(r.qiskit_circuit(window)).data, atol=1e-12)

    def test_all_readout_observables_match_qiskit(self):
        window = np.random.default_rng(23).normal(size=(1, 8, 7)); r = SpinReservoir(11, n=3)
        state = Statevector(r.evolve(window)[0]); expected = []
        for q in range(r.n):
            for p in ('X', 'Y', 'Z'):
                label = ['I']*r.n; label[r.n-1-q] = p
                expected.append(state.expectation_value(SparsePauliOp.from_list([(''.join(label), 1)])).real)
        for a, b in r.pairs:
            label = ['I']*r.n; label[r.n-1-a] = label[r.n-1-b] = 'Z'
            expected.append(state.expectation_value(SparsePauliOp.from_list([(''.join(label), 1)])).real)
        np.testing.assert_allclose(r.features(window)[0][0], expected, atol=1e-12)

    def test_product_formula_error_reduces_quadratically(self):
        r = SpinReservoir(11, n=3); window = np.random.default_rng(23).normal(size=(8, 7))
        exact = r.evolve(window[None])[0]
        errors = [np.linalg.norm(Statevector.from_instruction(r.qiskit_circuit(window, slices=s)).data-exact) for s in (1, 2, 4)]
        self.assertGreater(errors[0]/errors[1], 3.8); self.assertGreater(errors[1]/errors[2], 3.8)

    def test_uncoupled_ablation_and_seed_reproducibility(self):
        w = np.random.default_rng(23).normal(size=(3, 8, 7))
        a = SpinReservoir(11); b = SpinReservoir(11, coupled=False)
        self.assertTrue(np.all(b.coupling == 0))
        self.assertGreater(np.max(abs(a.features(w)[0]-b.features(w)[0])), .001)
        np.testing.assert_array_equal(a.features(w)[0], SpinReservoir(11).features(w)[0])
        np.testing.assert_array_equal(echo_features(w, 11), echo_features(w, 11))

    def test_invalid_registers_and_slices(self):
        for n in (True, 1, 10, 4.5):
            with self.assertRaises(ValueError): SpinReservoir(11, n=n)
        with self.assertRaises(ValueError): SpinReservoir(11).qiskit_circuit(np.zeros((8, 7)), slices=0)
