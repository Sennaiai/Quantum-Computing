"""Scientific regression checks; run with python -m unittest -v test_h2_benchmark."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pennylane as qml
from scipy.linalg import expm
from h2_benchmark import (COEFFICIENTS, WIRE_ORDER, classical_point,
                          coordinates_bohr, historical_analysis, historical_operator,
                          quantum_point, run_scan, validate_row)


class H2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_rows = run_scan([.74, 2.0])

    def test_historical_coupling_and_invariant_subspace(self):
        result = historical_analysis(); h = result['matrix']
        np.testing.assert_allclose(h,h.conj().T,atol=1e-14)
        self.assertAlmostEqual(result['coupling'],2*(COEFFICIENTS[8]-COEFFICIENTS[7]))
        outside = sorted(set(range(16))-{3,12})
        np.testing.assert_allclose(h[np.ix_(outside,[3,12])],0,atol=1e-14)
        self.assertAlmostEqual(result['analytic_min'],np.linalg.eigvalsh(result['block'])[0])
        self.assertAlmostEqual(result['expectation'],result['analytic_min'])

    def test_basis_wire_convention(self):
        op = historical_operator(); h = qml.matrix(op,wire_order=WIRE_ORDER)
        dev = qml.device('default.qubit',wires=4)
        @qml.qnode(dev)
        def expectation(bits):
            qml.BasisState(np.array(bits),wires=WIRE_ORDER)
            return qml.expval(op)
        for index in range(16):
            self.assertAlmostEqual(float(expectation([int(b) for b in f'{index:04b}'])),h[index,index].real)

    def test_two_parameter_formula_against_matrix_evolution(self):
        b = historical_analysis()['block']; ea,eb,v = b[0,0],b[1,1],b[0,1]
        center,delta=(ea+eb)/2,(ea-eb)/2
        for beta,gamma in [(0,0),(.37,-.81),(-1.2,.6)]:
            state = expm(-1j*beta*np.diag([ea,eb])) @ expm(-1j*gamma*np.array([[0,v],[v,0]])) @ [1,0]
            formula = center+delta*np.cos(2*v*gamma)+v*np.sin(2*v*gamma)*np.sin(beta*(ea-eb))
            self.assertAlmostEqual(float(np.real(np.vdot(state,b@state))),formula)

    def test_invalid_distance(self):
        for distance in [0,-1,np.nan,np.inf]:
            for function in [classical_point, quantum_point]:
                with self.subTest(distance=distance, function=function.__name__):
                    with self.assertRaises(ValueError): function(distance)

    def test_units_and_nuclear_repulsion(self):
        np.testing.assert_allclose(coordinates_bohr(.529177210903)[1], [0,0,1], atol=1e-14, rtol=0)
        self.assertAlmostEqual(self.reference_rows[0]['nuclear_repulsion_hartree'], .52917721092/.74, places=12)

    def test_explicit_integral_fermion_mapping_pipeline(self):
        molecule = qml.qchem.Molecule(['H', 'H'], qml.numpy.array(
            coordinates_bohr(.74), requires_grad=False), basis_name='sto-3g')
        constant, one, two = qml.qchem.electron_integrals(molecule)()
        self.assertEqual(one.shape, (2, 2))
        self.assertEqual(two.shape, (2, 2, 2, 2))
        self.assertAlmostEqual(np.asarray(constant).item(),
                               self.reference_rows[0]['nuclear_repulsion_hartree'], places=9)
        fermion = qml.qchem.fermionic_observable(constant, one, two)
        mapped = qml.matrix(qml.jordan_wigner(fermion), wire_order=range(4))
        sector = [3, 6, 9, 12]  # Nalpha=Nbeta=1, interleaved spin orbitals
        self.assertAlmostEqual(np.linalg.eigvalsh(mapped[np.ix_(sector, sector)])[0],
                               self.reference_rows[0]['fci_hartree'], delta=2e-7)
        self.assertAlmostEqual(mapped[12, 12].real,
                               self.reference_rows[0]['rhf_hartree'], delta=2e-7)

    def test_invalid_grid_and_empty_scan(self):
        for grid in [-1,10,2.5]:
            with self.assertRaises(ValueError): classical_point(.74, grid_level=grid)
        with self.assertRaises(ValueError): run_scan([])

    def test_molecular_reference_at_equilibrium_and_stretch(self):
        rows = self.reference_rows
        self.assertAlmostEqual(rows[0]['fci_hartree'],-1.1372838345,places=7)
        self.assertGreater(rows[1]['rhf_hartree']-rows[1]['fci_hartree'],
                           rows[0]['rhf_hartree']-rows[0]['fci_hartree'])

    def test_static_correlation_diagnostic(self):
        short, long = [r['fci_natural_occupations'] for r in self.reference_rows]
        self.assertGreater(short[0], long[0])
        self.assertLess(short[1], long[1])

    def test_committed_review_evidence_matches_recomputation(self):
        evidence = json.loads((Path(__file__).parent / 'benchmarks/h2/results.json').read_text())
        self.assertEqual(evidence['profile'], 'default')
        self.assertEqual(len(evidence['scan']), 7)
        for fresh in self.reference_rows:
            recorded = next(r for r in evidence['scan']
                            if r['distance_angstrom'] == fresh['distance_angstrom'])
            validate_row(recorded)
            for key in ['fci_hartree', 'rhf_hartree', 'rks_pbe_hartree', 'vqe_hartree']:
                with self.subTest(distance=fresh['distance_angstrom'], method=key):
                    self.assertAlmostEqual(recorded[key], fresh[key], delta=2e-7)

    def test_corrupted_results_are_rejected(self):
        for key,value in [('vqe_hartree',np.nan),('vqe_gap_hartree',-.01),
                          ('sector_leakage',.1),('vqe_spin_squared',2.),
                          ('vqe_particle_number',3.),('fci_converged',False),
                          ('variance_hartree2',.01),('fci_natural_occupations',[1.,0.])]:
            with self.subTest(key=key):
                row = dict(self.reference_rows[0]); row[key] = value
                if key == 'vqe_gap_hartree':
                    row['vqe_hartree'] = row['sector_exact_hartree'] + value
                with self.assertRaises(ValueError): validate_row(row)

    def test_inconsistent_reported_gap_is_rejected(self):
        row = dict(self.reference_rows[0]); row['vqe_gap_hartree'] = 0.001
        with self.assertRaises(AssertionError): validate_row(row)

    def test_independent_reference_mismatch_is_rejected(self):
        row = dict(self.reference_rows[0]); row['fci_hartree'] += .001
        with self.assertRaises(AssertionError): validate_row(row)

    def test_unconverged_reference_is_rejected(self):
        with patch('h2_benchmark.scf.RHF') as rhf:
            rhf.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, 'RHF failed'): classical_point(.74)
        with patch('h2_benchmark.fci.FCI') as fci:
            fci.return_value.kernel.return_value = (-1., None)
            fci.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, 'FCI failed'): classical_point(.74)
        with patch('h2_benchmark.dft.RKS') as rks:
            rks.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, 'RKS-PBE failed'): classical_point(.74)

    def test_failed_optimizer_is_rejected(self):
        with patch('h2_benchmark.minimize_scalar', return_value=SimpleNamespace(
                success=False, message='evaluation budget exhausted')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation budget'): quantum_point(.74)

    def test_pbe_grid_convergence(self):
        a=classical_point(.74,grid_level=3); b=classical_point(.74,grid_level=5)
        self.assertLess(abs(a['rks_pbe_hartree']-b['rks_pbe_hartree']),1e-6)


if __name__ == '__main__': unittest.main()
