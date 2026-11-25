import unittest
import numpy as np
import dimod
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
from qaoa_core import QUBOStatevector,optimize


def model():
    return QUBOStatevector(dimod.BinaryQuadraticModel({'a':.7,'b':-1.3,'c':.2},
        {('a','b'):.9,('a','c'):-.37,('b','c'):1.1},.43,dimod.BINARY))


class QAOAPhysicsTests(unittest.TestCase):
    def test_every_basis_energy_matches_qubo_and_pauli_signs(self):
        m=model()
        for state in range(1 << m.n):
            sample=m.sample_dict(state); z=1-2*m.bits[state].astype(int)
            pauli=m.offset+sum(c*z[q] for q,c in m.z.items())+sum(c*z[a]*z[b] for (a,b),c in m.zz.items())
            self.assertAlmostEqual(m.energy[state],m.bqm.energy(sample),places=12)
            self.assertAlmostEqual(m.energy[state],pauli,places=12)
            self.assertAlmostEqual(float(Statevector.from_int(state,1 << m.n).expectation_value(m.hamiltonian()).real),m.energy[state],places=12)
        self.assertEqual(m.sample_dict(1),{'a':1,'b':0,'c':0})
        self.assertEqual(m.sample_dict(4),{'a':0,'b':0,'c':1})

    def test_dense_exponentials_and_qiskit_circuit_agree_all_depths(self):
        m=model(); generator=np.zeros((8,8),complex)
        for q in range(m.n):
            for i in range(8):generator[i^(1 << q),i]+=1
        for depth in (1,2,3,5):
            params=np.random.default_rng(depth).uniform(-1.5,2.,2*depth)
            dense=np.ones(8,complex)/np.sqrt(8)
            for gamma,beta in zip(params[:depth],params[depth:]):
                dense=expm(-1j*beta*generator)@expm(-1j*gamma*np.diag(m.cost))@dense
            state,energy=m.evolve(params)
            self.assertTrue(np.allclose(state,dense,atol=1e-12))
            circuit=np.asarray(Statevector.from_instruction(m.circuit(params)))
            self.assertAlmostEqual(abs(np.vdot(state,circuit)),1.,places=12)
            self.assertAlmostEqual(energy,float(np.vdot(dense,m.cost*dense).real),places=12)

    def test_adjoint_gradient_matches_independent_finite_differences(self):
        m=model()
        for p in (1,2,3,5):
            x=np.random.default_rng(100+p).uniform(.1,2.,2*p);value,gradient=m.evolve(x,gradient=True)
            finite=[];h=1e-6
            for j in range(len(x)):
                step=np.zeros_like(x);step[j]=h
                finite.append((m.evolve(x+step)[1]-m.evolve(x-step)[1])/(2*h))
            np.testing.assert_allclose(gradient,finite,atol=2e-8,rtol=2e-6)
            self.assertAlmostEqual(value,m.evolve(x)[1],places=12)

    def test_shots_are_seeded_and_do_not_use_ground_state_information(self):
        m=model();params=[.9,.4]
        first,prob=m.sample(params,1024,17);second,_=m.sample(params,1024,17)
        self.assertEqual(first,second);self.assertEqual(sum(first.values()),1024)
        self.assertAlmostEqual(float(prob.sum()),1.,places=12)
        self.assertTrue((prob>=0).all())

    def test_constant_hamiltonian_has_uniform_probabilities_and_zero_gradient(self):
        m=QUBOStatevector(dimod.BinaryQuadraticModel({'a':0.,'b':0.},{},7.,dimod.BINARY))
        state,_=m.evolve([1.,2.,.3,.4]);_,g=m.evolve([1.,2.,.3,.4],gradient=True)
        np.testing.assert_allclose(abs(state)**2,.25,atol=1e-12)
        np.testing.assert_allclose(g,0.,atol=1e-12)
        self.assertAlmostEqual(m.physical_energy([.4,.3]),7.)

    def test_invalid_dimensions_parameters_and_oversize_problems_fail_before_allocation(self):
        with self.assertRaises(ValueError):QUBOStatevector(dimod.BinaryQuadraticModel({}, {}, 0., dimod.BINARY))
        with self.assertRaises(ValueError):QUBOStatevector(dimod.BinaryQuadraticModel({j:0 for j in range(21)}, {}, 0., dimod.BINARY))
        with self.assertRaises(ValueError):QUBOStatevector(dimod.BinaryQuadraticModel({'a':1}, {}, 0., dimod.SPIN))
        m=model()
        for params in ([],[1],[1,2,3],[np.nan,0],[np.inf,0],[[1,2]]):
            with self.subTest(params=params),self.assertRaises(ValueError):m.evolve(params)
        with self.assertRaises(ValueError):m.sample([1,2],0,7)
        with self.assertRaises(ValueError):m.sample_dict(8)

    def test_continuation_includes_lower_depth_state_without_assuming_gamma_periodicity(self):
        m=model();p1,r1=optimize(m,1,7,starts=2,maxiter=15)
        p2,r2=optimize(m,2,7,starts=2,maxiter=15,previous=p1)
        self.assertLessEqual(r2['expected_energy'],r1['expected_energy']+1e-10)
        self.assertGreater(r2['optimization_evaluations'],0)
        self.assertIn('not a claimed period',r2['angle_domain'])
        p1_repeat,_=optimize(m,1,7,starts=2,maxiter=15)
        np.testing.assert_allclose(p1,p1_repeat,atol=0,rtol=0)


class SamplerInterfaceTests(unittest.TestCase):
    def test_binary_bqm_sampler_preserves_labels_energies_counts_and_repeatability(self):
        from qaoa_sampler import QAOASampler,basis_counts
        m=model();sampler=QAOASampler(8)
        a=sampler.sample(m.bqm,num_reads=128,depth=2,seed=17,starts=2,maxiter=10)
        b=sampler.sample(m.bqm,num_reads=128,depth=2,seed=17,starts=2,maxiter=10)
        self.assertEqual(basis_counts(a,m),basis_counts(b,m))
        self.assertEqual(sum(basis_counts(a,m).values()),128)
        for row in a.data(['sample','energy']):self.assertAlmostEqual(m.bqm.energy(row.sample),row.energy,places=10)
        with self.assertRaises(ValueError):QAOASampler(2).sample_prepared(m)
