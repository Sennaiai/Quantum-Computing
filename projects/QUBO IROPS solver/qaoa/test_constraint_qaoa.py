import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
from bootstrap import PARENT
from fixtures import small_cases
from measurements import compile_case
from constraint_qaoa import OneHotQAOA
from qaoa_sampler import QAOASampler


class ConstraintMixerTests(unittest.TestCase):
    def setUp(self):self.model=OneHotQAOA(compile_case(small_cases()[0]),16)

    def test_initial_state_and_evolution_preserve_exactly_one_route_or_dummy_per_pnr(self):
        m=self.model
        for p in (1,2,3):
            x=np.random.default_rng(7+p).uniform(.1,2.,2*p);state,_=m.evolve(x)
            self.assertLess(float(np.abs(state[~m.support]).sum()),1e-12)
            self.assertAlmostEqual(float(np.vdot(state,state).real),1.,places=12)
        self.assertAlmostEqual(float(np.vdot(m.initial_state(),m.initial_state()).real),1.,places=12)

    def test_qiskit_projector_circuit_and_independent_dense_generator_match(self):
        m=self.model;D=1 << m.n;B=np.zeros((D,D),complex)
        for group in m.groups:
            mask=sum(1 << bit for bit in group)
            for j in range(D):
                if (j & mask).bit_count()==1:
                    for bit in group:B[(j & ~mask)|(1 << bit),j]+=1/len(group)
        for bit in m.slack_bits:
            for j in range(D):B[j^(1 << bit),j]+=1
        params=np.array([.4,.8,.7,1.2]);dense=m.initial_state()
        for gamma,beta in zip(params[:2],params[2:]):dense=expm(-1j*beta*B)@(np.exp(-1j*gamma*m.cost)*dense)
        state,_=m.evolve(params);np.testing.assert_allclose(state,dense,atol=1e-11)
        independent=np.asarray(Statevector.from_instruction(m.circuit(params)))
        self.assertAlmostEqual(abs(np.vdot(state,independent))**2,1.,places=10)

    def test_projector_adjoint_gradient_matches_finite_difference(self):
        m=self.model;x=np.array([.9,.4,1.2,.8]);_,actual=m.evolve(x,gradient=True);finite=[];h=1e-6
        for j in range(len(x)):
            d=np.zeros_like(x);d[j]=h;finite.append((m.evolve(x+d)[1]-m.evolve(x-d)[1])/(2*h))
        np.testing.assert_allclose(actual,finite,atol=2e-8,rtol=1e-5)

    def test_sampler_uses_same_qubo_and_preserves_onehot_within_finite_shots(self):
        compiled=compile_case(small_cases()[0]);samples=QAOASampler(16).sample_constrained(compiled,num_reads=128,depth=2,seed=3,starts=2,maxiter=10)
        for row in samples.data(['sample']):
            for coeff,c in compiled.constraints[:3]:self.assertEqual(c+sum(int(a)*int(row.sample[v]) for v,a in coeff.items()),0)
