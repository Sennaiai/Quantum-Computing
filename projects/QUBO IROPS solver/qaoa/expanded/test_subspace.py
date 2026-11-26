import unittest
import numpy as np
from qiskit.quantum_info import Statevector
from case_catalog import load_cases
from measurements import compile_case
from constraint_qaoa import OneHotQAOA
from expanded.subspace import SubspaceQAOA
from expanded.circuit_audit import replay_circuit


class SubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=load_cases()[1]

    def test_compressed_evolution_equals_full_existing_engine_and_qiskit(self):
        rng=np.random.default_rng(123)
        for row,instance in self.cases:
            compiled=compile_case(instance);small=SubspaceQAOA(compiled);full=OneHotQAOA(compiled)
            self.assertEqual(len(set(map(int,small.states))),small.dimension)
            np.testing.assert_allclose(small.energy,full.energy[small.states],atol=1e-8)
            for depth in (1,3):
                params=rng.uniform(.1,2,2*depth);a,_=small.evolve(params);b,_=full.evolve(params)
                np.testing.assert_allclose(a,b[small.states],atol=1e-12)
                self.assertAlmostEqual(float(np.sum(abs(a)**2)),1,places=12)
                independent=np.asarray(Statevector.from_instruction(small.circuit(params)))
                np.testing.assert_allclose(abs(independent[small.states]),abs(a),atol=1e-10)
                self.assertAlmostEqual(abs(np.vdot(independent[small.states],a))**2,1,places=9)
                replay=replay_circuit(small,params)
                self.assertAlmostEqual(abs(np.vdot(independent[small.states],replay))**2,1,places=9)

    def test_adjoint_gradient_matches_finite_differences(self):
        model=SubspaceQAOA(compile_case(self.cases[-1][1]));x=np.array([.4,1.2,.3,.8]);value,gradient=model.evolve(x,gradient=True)
        numerical=[]
        for i in range(len(x)):
            step=np.zeros(len(x));step[i]=1e-6
            numerical.append((model.evolve(x+step)[1]-model.evolve(x-step)[1])/2e-6)
        np.testing.assert_allclose(gradient,numerical,atol=1e-8)

    def test_samples_use_physical_basis_indices_and_one_choice(self):
        model=SubspaceQAOA(compile_case(self.cases[-1][1]));counts=model.draw([.7,.8],1024,7)
        self.assertEqual(sum(counts.values()),1024)
        for basis in counts:
            sample=model.basis_sample(basis)
            for group in model.groups:self.assertEqual(sum(sample[model.labels[q]] for q in group),1)
        with self.assertRaises(ValueError):model.basis_sample(1<<model.n)

    def test_explicit_resource_budgets_are_enforced(self):
        compiled=compile_case(self.cases[-1][1])
        with self.assertRaises(ValueError):SubspaceQAOA(compiled,max_qubits=8)
        with self.assertRaises(ValueError):SubspaceQAOA(compiled,max_states=1)
