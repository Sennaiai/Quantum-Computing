"""Independent mathematical checks; these do not execute native Beryllium."""
import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'python_classes')]
from classical_preprocessing import (AGPSettings, build_initial_hamiltonian, commutator,
    merge_terms, parse_pauli_string, recalculate_counterdiabatic_hamiltonian)
from hamiltonian_validation import HamiltonianValidation
from qsvt_helper import QSVTHelper, reflection_response
from quantum_reference import (PAULI, hamiltonian_matrix, lcu_operators,
    prepare_matrix, qsvt_matrix, verify_quantum_reference, verify_spectral_reference)


def payload(terms, nq):
    return {'selected_features': list(range(nq)), 'feature_vector': [.25]*nq,
            'pauli_terms': [{'coefficient': float(c), 'pauli_string': p} for c,p in terms]}


def settings(p, time=.3, epsilon=1e-6):
    return QSVTHelper(HamiltonianValidation(p), time, epsilon).to_settings()


class PauliAGPTests(unittest.TestCase):
    def test_all_single_pauli_commutators_against_independent_dense_matrices(self):
        for a,b in itertools.product('IXYZ',repeat=2):
            ta=[{'coefficient':.7,'pauli_string':a+'0'}]
            tb=[{'coefficient':-.4,'pauli_string':b+'0'}]
            actual=hamiltonian_matrix(commutator(ta,tb),1)
            A=.7*PAULI[a];B=-.4*PAULI[b]
            np.testing.assert_allclose(actual,A@B-B@A,atol=1e-14)

    def test_commutator_materializes_generator_and_merges_cancellations(self):
        left=[{'coefficient':1,'pauli_string':'X0'}, {'coefficient':2,'pauli_string':'Z0'}]
        right=[{'coefficient':1,'pauli_string':'Y0'}, {'coefficient':-.2,'pauli_string':'X0'}]
        actual=hamiltonian_matrix(commutator(iter(left),iter(right)),1)
        A=hamiltonian_matrix(left,1);B=hamiltonian_matrix(right,1)
        np.testing.assert_allclose(actual,A@B-B@A,atol=1e-14)
        self.assertEqual(merge_terms([{'coefficient':1,'pauli_string':'Z1 X0'},
                                     {'coefficient':-1,'pauli_string':'X0 Z1'}]),[])

    def test_dense_two_qubit_variational_agp_and_stationarity(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[10,20],
            'pauli_terms':[{'coefficient':-.7,'pauli_string':'Z0 Z1'}]}}}
        lam=.37;rate=.8
        p=recalculate_counterdiabatic_hamiltonian(curated,'fixture',[.2,.8],
            AGPSettings(interpolation=lam,lambda_dot=rate),include_components=True)
        H0=-np.kron(PAULI['X'],PAULI['I'])-np.kron(PAULI['I'],PAULI['X'])
        Hp=-.7*np.kron(PAULI['Z'],PAULI['Z'])+.2*np.kron(PAULI['Z'],PAULI['I'])+.8*np.kron(PAULI['I'],PAULI['Z'])
        H=(1-lam)*H0+lam*Hp;D=Hp-H0;C=H@D-D@H;C2=H@C-C@H
        alpha=-np.vdot(C,C).real/np.vdot(C2,C2).real
        np.testing.assert_allclose(hamiltonian_matrix(p['pauli_terms'],2),H+rate*1j*alpha*C,atol=1e-13)
        self.assertAlmostEqual(p['agp_metadata']['alpha1'],alpha)
        self.assertLess(abs(np.vdot(C2,D+alpha*C2).real),1e-12)
        np.testing.assert_allclose(hamiltonian_matrix(p['agp_components']['o1'],2),C,atol=1e-13)
        self.assertEqual(p['agp_metadata']['interaction_truncation']['dropped_count'],0)

    def test_one_qubit_matches_exact_adiabatic_gauge_potential(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        theta=.7;lam=.4
        p=recalculate_counterdiabatic_hamiltonian(curated,'fixture',[theta],AGPSettings(interpolation=lam),include_components=True)
        exact=theta/(2*((1-lam)**2+(lam*theta)**2))*PAULI['Y']
        np.testing.assert_allclose(hamiltonian_matrix(p['agp_components']['agp'],1),exact,atol=1e-13)

    def test_commuting_path_has_zero_gauge(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        p=recalculate_counterdiabatic_hamiltonian(curated,'fixture',[0.],AGPSettings(),include_components=True)
        self.assertTrue(p['agp_metadata']['commuting_path'])
        self.assertEqual(p['agp_components']['agp'],[])
        self.assertEqual(p['agp_metadata']['alpha1'],0.)

    def test_small_field_near_gap_retains_finite_exact_agp(self):
        # Absolute pruning of theta would falsely classify this path as
        # commuting and erase an order-one gauge potential.
        theta=1e-13;lam=1-1e-7
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        p=recalculate_counterdiabatic_hamiltonian(curated,'fixture',[theta],
            AGPSettings(interpolation=lam),include_components=True)
        exact=theta/(2*((1-lam)**2+(lam*theta)**2))*PAULI['Y']
        np.testing.assert_allclose(hamiltonian_matrix(p['agp_components']['agp'],1),exact,rtol=1e-13,atol=1e-13)
        self.assertFalse(p['agp_metadata']['commuting_path'])
        self.assertEqual(p['agp_components']['problem'][0]['coefficient'],theta)
        self.assertEqual(p['agp_metadata']['coefficient_pruning'],'none; exact zero cancellations only')
        self.assertEqual(p['agp_metadata']['hermiticity_imaginary_tolerance'],1e-12)
        self.assertEqual(len(merge_terms([{'coefficient':1e-30,'pauli_string':'Z0'}])),1)
        self.assertTrue(any(t['pauli_string']=='Z0' and t['coefficient'] != 0 for t in p['pauli_terms']))

    def test_nonzero_commutator_norm_underflow_fails_as_precision_limit(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        with self.assertRaisesRegex(ValueError,'insufficient float64 precision'):
            recalculate_counterdiabatic_hamiltonian(curated,'fixture',[1e-200])

    def test_commutator_budget_rejects_before_any_expansion(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        with patch('classical_preprocessing.multiply_pauli_words',side_effect=AssertionError('expansion occurred')):
            with self.assertRaisesRegex(ValueError,'budget exceeded before expansion'):
                recalculate_counterdiabatic_hamiltonian(curated,'fixture',[.5],
                    AGPSettings(max_commutator_products=1))
        with self.assertRaises(ValueError):AGPSettings(max_commutator_products=True).validate()
        with self.assertRaises(ValueError):AGPSettings(max_commutator_products=0).validate()

    def test_rejects_malformed_and_nonfinite_inputs(self):
        for word in ['X-1','X01','I0 X0','X0 Z0','Q0']:
            with self.subTest(word=word), self.assertRaises(ValueError):parse_pauli_string(word)
        for coefficient in [float('nan'),float('inf'),complex(1,float('inf')),True,'1']:
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                merge_terms([{'coefficient':coefficient,'pauli_string':'X0'}])
        for value in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):AGPSettings(lambda_dot=value).validate()
            with self.assertRaises(ValueError):AGPSettings(zero_tolerance=value).validate()
        with self.assertRaises(ValueError):AGPSettings(interaction_term_limit=2.5).validate()
        with self.assertRaises(ValueError):build_initial_hamiltonian(0)
        for key in ['interpolation','lambda_dot','zero_tolerance']:
            with self.subTest(key=key),self.assertRaises(ValueError):AGPSettings(**{key:True}).validate()
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[3],'pauli_terms':[]}}}
        with self.assertRaises(ValueError):recalculate_counterdiabatic_hamiltonian(curated,'fixture',[True])
        curated['hamiltonian_A_interaction_ising']['fixture']['pauli_terms']=[{'coefficient':1,'pauli_string':'I9'}]
        with self.assertRaises(ValueError):recalculate_counterdiabatic_hamiltonian(curated,'fixture',[.5])


class QuantumCircuitTests(unittest.TestCase):
    def test_prepare_tree_for_zero_padded_skewed_distributions_and_inverse(self):
        for weights,ni in [([1.],1),([.2,.8],1),([0.,.3,.7],2),([1e-24,0.,.2,.8-1e-24],2),([.1,.2,.3,.4],3)]:
            a=np.sqrt(weights);P=prepare_matrix(a,ni)
            np.testing.assert_allclose(P[:,0],np.pad(a,(0,2**ni-len(a))),atol=2e-15)
            np.testing.assert_allclose(P.T@P,np.eye(2**ni),atol=2e-15)

    def test_lcu_signed_identity_y_and_padding_against_direct_hamiltonian(self):
        p=payload([(-.3,''),(.7,'Y0'),(-.2,'X0 Z1')],2)
        ops=lcu_operators(p)
        H=-.3*np.eye(4)+.7*np.kron(PAULI['Y'],PAULI['I'])-.2*np.kron(PAULI['X'],PAULI['Z'])
        np.testing.assert_allclose(ops['block'][:4,:4],H/1.2,atol=2e-15)
        np.testing.assert_allclose(ops['block']@ops['block'],np.eye(16),atol=2e-15)
        np.testing.assert_allclose(ops['walk_inverse']@ops['walk'],np.eye(16),atol=2e-15)

    def test_phase_certificate_covers_cosine_and_sine_on_independent_grid(self):
        p=payload([(1.,'Z0')],1)
        s=settings(p,time=1.1)
        x=np.linspace(-1,1,2003)
        for name,fn in [('cosine',np.cos),('sine',np.sin)]:
            b=s[name];response=reflection_response(x,b['phase_angles']).real/b['scale']
            error=np.max(np.abs(response-fn(s['tau']*x)))
            self.assertLessEqual(error,b['approximation_error_bound']+b['synthesis_error_bound'])
        self.assertLessEqual(s['recovered_evolution_error_bound'],s['epsilon'])
        self.assertEqual(s['extra_ancillas'],2)
        self.assertFalse(s['native_validation']['live_api_execution_allowed'])
        self.assertAlmostEqual(s['success_amplitude'],1/(1/s['cosine']['scale']+1/s['sine']['scale']))

    def test_full_coherent_unitary_matches_expm_for_mixed_signed_and_identity_terms(self):
        fixtures=[payload([(1.,'Z0')],1), payload([(-.8,'')],1),
                  payload([(-.3,''),(.7,'Y0'),(-.2,'X0 Z1')],2),
                  payload([(.4,'X0'),(-.6,'Y0 Y1'),(.2,'Z1'),(.1,'X0 Z1')],2)]
        for p in fixtures:
            with self.subTest(terms=p['pauli_terms']):
                s=settings(p);report=verify_quantum_reference(p,s)
                self.assertLess(report['recovered_evolution_error'],s['epsilon'])
                self.assertLess(report['heralded_unitarity_error'],1e-12)
                self.assertGreaterEqual(report['minimum_success_probability']+1e-13,s['success_probability_lower_bound'])

    def test_noncommuting_cd_to_expm_end_to_end(self):
        curated={'hamiltonian_A_interaction_ising':{'fixture':{'selected_features':[10,20],
            'pauli_terms':[{'coefficient':-.7,'pauli_string':'Z0 Z1'}]}}}
        p=recalculate_counterdiabatic_hamiltonian(curated,'fixture',[.2,.8])
        s=settings(p,time=.2);report=verify_quantum_reference(p,s)
        self.assertLess(report['recovered_evolution_error'],s['recovered_evolution_error_bound'])

    def test_spectral_reduction_matches_full_ancilla_reference(self):
        fixtures=[payload([(-.8,'')],1),
                  payload([(-.3,''),(.7,'Y0'),(-.2,'X0 Z1')],2)]
        for p in fixtures:
            with self.subTest(terms=p['pauli_terms']):
                s=settings(p,time=.7)
                full=verify_quantum_reference(p,s)
                spectral=verify_spectral_reference(p,s)
                self.assertLess(abs(full['recovered_evolution_error']-spectral['recovered_evolution_error']),1e-12)
                self.assertLess(abs(full['minimum_success_probability']-spectral['minimum_success_probability']),1e-12)
                self.assertLessEqual(spectral['recovered_evolution_error'],s['recovered_evolution_error_bound'])
                self.assertFalse(spectral['native_compiler_verified'])

    def test_valid_small_tau_passes_final_certificate_after_phase_synthesis(self):
        # pyqsp's internal sampled reconstruction differs from the requested
        # final error budget. This valid case previously failed a fixed 1e-9
        # internal threshold despite satisfying the independent certificate.
        p=payload([(-.25,''),(.5,'X1'),(.75,'Y0 Z1'),(-.125,'Z0 X1')],2)
        s=settings(p,time=.1,epsilon=1e-6)
        report=verify_quantum_reference(p,s)
        self.assertAlmostEqual(s['lambda_value'],1.625)
        self.assertLessEqual(s['recovered_evolution_error_bound'],1e-6)
        self.assertLessEqual(report['recovered_evolution_error'],s['recovered_evolution_error_bound'])

    def test_degrees_resources_and_numerical_limits_fail_closed(self):
        p=payload([(1.,'Z0')],1);v=HamiltonianValidation(p)
        for time,eps in [(float('inf'),1e-6),(1.,float('nan')),(0.,1e-6),(1.,1e-12)]:
            with self.subTest(time=time,eps=eps),self.assertRaises(ValueError):QSVTHelper(v,time,eps)
        with self.assertRaises(ValueError):QSVTHelper(v,physical_time=1000,max_degree=20)
        with self.assertRaises(ValueError):lcu_operators(payload([(1.,'Z0')],5),max_dimension=32)


if __name__=='__main__':unittest.main()
