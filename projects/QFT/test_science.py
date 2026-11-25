"""Independent matrices, coherent arithmetic, oracle domains, and order recovery."""
import unittest
from itertools import product
from math import gcd
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFTGate
from qiskit.synthesis import adder_ripple_c04
from qft_core import *
from arithmetic import *
from cross_backend import *
from verification import coherent_probe,execute_state
from order_finding import *

class FourierTests(unittest.TestCase):
    def test_full_fourier_matrix_including_phase_and_wire_order(self):
        for n in range(1,7):
            expected=fourier_matrix(n);actual=Operator(qft(n)).data
            np.testing.assert_allclose(actual,expected,atol=8e-14)
            np.testing.assert_allclose(Operator(qft(n,inverse=True)).data,expected.conj().T,atol=8e-14)
            np.testing.assert_allclose(Operator(QFTGate(n)).data,expected,atol=8e-14)

    def test_counting_register_inverse_qft_full_matrix_against_fft(self):
        for n in (8,9):
            dim=2**n
            expected=np.fft.fft(np.eye(dim),axis=0)/np.sqrt(dim)
            np.testing.assert_allclose(Operator(qft(n,inverse=True)).data,expected,atol=3e-13)

    def test_pennylane_native_and_builtin_agree_with_independent_fourier_matrix(self):
        for n in range(1,6):
            np.testing.assert_allclose(pl_qft_matrix(n),fourier_matrix(n),atol=8e-14)
            np.testing.assert_allclose(pl_qft_matrix(n,builtin=True),fourier_matrix(n),atol=8e-14)

    def test_approximate_roundtrip_does_not_prove_correct_fourier_transform(self):
        full=fourier_matrix(5)
        approximate=Operator(qft(5,cutoff=0)).data
        np.testing.assert_allclose(approximate.conj().T@approximate,np.eye(32),atol=1e-12)
        self.assertGreater(np.linalg.norm(approximate-full,2),1.)
        np.testing.assert_allclose(Operator(qft(5,cutoff=4)).data,full,atol=1e-13)

    def test_constant_addition_all_basis_columns_including_wraparound_and_negative_k(self):
        for n in range(1,5):
            D=2**n
            for k in (-3,0,1,D-1,D+1):
                expected=np.zeros((D,D),complex)
                for x in range(D):expected[(x+k)%D,x]=1
                np.testing.assert_allclose(Operator(constant_adder(n,k)).data,expected,atol=2e-12)

    def test_controlled_branch_phase_regression(self):
        r=phase_regression()
        self.assertLess(r['basis_probability_error'],1e-12)
        self.assertGreater(r['controlled_operator_error'],1.)
        gate=constant_adder(3,5).to_gate().control(1)
        expected=np.zeros((16,16),complex)
        for x in range(8):
            expected[2*x,2*x]=1;expected[2*((x+5)%8)+1,2*x+1]=1
        np.testing.assert_allclose(Operator(gate).data,expected,atol=2e-12)

    def test_quantum_addition_and_ripple_reference_use_same_unitary(self):
        for n in (1,2,3):
            mapping={a+(b<<n):a+(((a+b)%2**n)<<n) for a,b in product(range(2**n),repeat=2)}
            q=coherent_probe(quantum_adder(n),mapping)
            ripple=adder_ripple_c04(n,kind='fixed')
            r=coherent_probe(ripple,mapping)
            self.assertLess(max(q['state_error_l2'],r['state_error_l2']),1e-10)

    def test_validation_rejects_boolean_bad_width_and_truncation(self):
        for args in [(0,{}),(11,{}),(True,{}),(3,{'cutoff':-1}),(3,{'cutoff':3}),(3,{'cutoff':True}),(3,{'inverse':1})]:
            with self.assertRaises(ValueError):qft(args[0],**args[1])
        with self.assertRaises(ValueError):fourier_matrix(7)
        with self.assertRaises(ValueError):constant_adder(3,True)

class ModularTests(unittest.TestCase):
    def test_modular_constant_add_all_valid_columns_and_clean_work(self):
        for n in (2,3):
            for N in range(2,2**n+1):
                for k in range(N):
                    matrix=Operator(modular_constant_adder(n,N,k)).data[:,:N]
                    expected=np.zeros_like(matrix)
                    for b in range(N):expected[(b+k)%N,b]=1
                    np.testing.assert_allclose(matrix,expected,atol=3e-12)

    def test_pennylane_modular_unitaries_and_device_states_preserve_coherent_phase(self):
        for n,N,k in [(2,3,2),(3,5,3),(3,7,5)]:
            circuit=modular_constant_adder(n,N,k)
            np.testing.assert_allclose(pl_modular_matrix(n,N,k),Operator(circuit).data,atol=3e-12)
            rng=np.random.default_rng(3);psi=rng.normal(size=N)+1j*rng.normal(size=N);psi/=np.linalg.norm(psi)
            initial=np.zeros(2**(n+2),complex);initial[:N]=psi
            expected=np.zeros_like(initial)
            for b in range(N):expected[(b+k)%N]=psi[b]
            np.testing.assert_allclose(pl_modular_state(n,N,k,initial),expected,atol=3e-12)

    def test_controlled_modular_addition_cleans_both_branches(self):
        n,N,k=3,7,5
        controlled=QuantumCircuit(n+3);controlled.append(modular_constant_adder(n,N,k).to_gate().control(),range(n+3))
        mapping={control+(b<<1):control+(((b+control*k)%N)<<1) for control in (0,1) for b in range(N)}
        self.assertLess(coherent_probe(controlled,mapping)['state_error_l2'],1e-10)

    def test_quantum_and_outplace_adders(self):
        n,N=2,3;D=2**n
        mapping={a+(b<<n):a+(((a+b)%N)<<n) for a in range(D) for b in range(N)}
        coherent_probe(modular_quantum_adder(n,N),mapping)
        coherent_probe(outplace_constant_adder(n,N,2),{a:a+(((a+2)%N)<<n) for a in range(N)})
        coherent_probe(outplace_quantum_adder(n,N),{a+(b<<n):a+(b<<n)+(((a+b)%N)<<(2*n)) for a in range(D) for b in range(N)})

    def test_constant_and_quantum_multiply_accumulate(self):
        n,N=2,3;D=2**n
        coherent_probe(multiply_accumulate(n,N,2),{a+(b<<n):a+(((b+2*a)%N)<<n) for a in range(D) for b in range(N)})
        coherent_probe(quantum_multiply_accumulate(n,N),{a+(b<<n)+(r<<(2*n)):a+(b<<n)+(((r+a*b)%N)<<(2*n)) for a,b in product(range(D),repeat=2) for r in range(N)})

    def test_coprime_multipliers_on_prime_and_composite_moduli(self):
        for n,N,k in [(3,7,3),(3,5,2),(4,15,2)]:
            coherent_probe(modular_multiplier(n,N,k),{a:(k*a)%N for a in range(N)})
        self.assertNotEqual(pow(2,15-2,15)*2%15,1)  # the old prime-only inverse is invalid here

    def test_controlled_composite_multiplier_preserves_relative_phase(self):
        n,N,k=4,15,2
        gate=modular_multiplier(n,N,k).to_gate().control(1)
        qc=QuantumCircuit(2*n+3);qc.append(gate,range(qc.num_qubits))
        mapping={c+2*a:c+2*((k*a)%N if c else a) for c in (0,1) for a in range(N)}
        coherent_probe(qc,mapping)

    def test_native_modular_exponentiation_coherent_inputs_and_workspace(self):
        for n,N,a,t in [(3,7,4,3),(3,5,2,2)]:
            coherent_probe(modular_exponentiation(n,N,a,t),{x:x+(pow(a,x,N)<<t) for x in range(2**t)})

    def test_invalid_domain_parameters_fail_without_silent_identity(self):
        for arguments in [(3,9,2),(0,3,1),(True,3,1),(3,1,2)]:
            with self.assertRaises(ValueError):modular_constant_adder(*arguments)
        with self.assertRaises(ValueError):modular_multiplier(4,15,3)
        with self.assertRaises(ValueError):modular_exponentiation(4,15,3,3)
        with self.assertRaises(ValueError):modular_exponentiation(5,21,2,10)

class OrderTests(unittest.TestCase):
    def test_compiled_multiplier_has_correct_full_unitary_extension(self):
        for a,N in [(1,15),(2,15),(2,21),(5,21)]:
            mapping=multiplier_permutation(a,N);self.assertEqual(len(set(mapping)),len(mapping))
            gate=Operator(compiled_controlled_multiplier(a,N)).data
            for y,out in enumerate(mapping):
                for control in (0,1):
                    expected=np.zeros(2*len(mapping));expected[2*(out if control else y)+control]=1
                    np.testing.assert_allclose(gate[:,2*y+control],expected,atol=1e-13)
            for y in range(N,len(mapping)):self.assertEqual(mapping[y],y)

    def test_controlled_powers_use_consistent_base(self):
        self.assertEqual(multiplier_permutation(1,15),list(range(16)))
        for a in (2,5):
            base=multiplier_permutation(a,21)
            for j in range(6):
                for y in range(32):
                    value=y
                    for _ in range(2**j):value=base[value]
                    target=multiplier_permutation(pow(a,2**j,21),21)[y]
                    self.assertEqual(value,target)

    def test_compiled_shor_matches_independent_modular_power_fft(self):
        result=run_order_finding(2,21,oracle='compiled_permutation')
        self.assertLess(result['state_error_l2_vs_exact_fft'],1e-10)
        self.assertEqual(result['recovery']['order'],6)
        self.assertEqual(result['factorization']['factors'],[3,7])

    def test_native_qft_arithmetic_drives_order_finding(self):
        result=run_order_finding(2,15,oracle='qft_arithmetic')
        self.assertLess(result['state_error_l2_vs_exact_fft'],1e-9)
        self.assertLess(result['workspace_leakage'],1e-10)
        self.assertEqual(result['recovery']['order'],4)
        self.assertEqual(result['factorization']['factors'],[3,5])

    def test_valid_order_can_fail_to_factor(self):
        result=run_order_finding(5,21,oracle='compiled_permutation')
        self.assertEqual(result['recovery']['order'],6)
        self.assertEqual(result['factorization']['status'],'trivial_square_root')
        self.assertEqual(result['factorization']['factors'],[])
        self.assertEqual(factors_from_order(4,21,3)['status'],'odd_order')

    def test_no_phase_evidence_does_not_trigger_classical_order_search(self):
        self.assertIsNone(recover_order({'000000000':4096},2,21,9)['order'])
        # A reduced denominator need not be the order; validate it rather than force a result.
        self.assertIsNone(recover_order({'100000000':4096},2,21,9)['order'])
        self.assertEqual(recover_order({'010101011':2000,'100000000':2096},2,21,9)['order'],6)

    def test_approximate_iqft_error_is_measured_not_hidden_by_roundtrip(self):
        r=run_order_finding(2,21,oracle='compiled_permutation',cutoff=0)
        self.assertGreater(r['state_error_l2_vs_exact_fft'],.1)
        self.assertGreater(r['phase_tv_vs_exact_fft'],.1)

    def test_order_parameters_and_counts_are_strict(self):
        for a,N in [(3,21),(1,21),(2,32),(True,21)]:
            with self.assertRaises(ValueError):order_circuit(a,N)
        with self.assertRaises(ValueError):recover_order({'00x':2},2,21,3)
        with self.assertRaises(ValueError):recover_order({'001':True},2,21,3)
        with self.assertRaises(ValueError):recover_order({},2,21,3)
        with self.assertRaises(ValueError):order_circuit(2,21,oracle='unknown')

if __name__=='__main__':unittest.main()
