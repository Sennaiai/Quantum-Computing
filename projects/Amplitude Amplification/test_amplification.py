"""Analytic, independent linear-algebra, predicate, and sampling regressions."""
import math
import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_aer import AerSimulator
from amplification import *

class AmplificationTests(unittest.TestCase):
    def test_every_lookup_basis_phase_including_endianness(self):
        for n in range(1,5):
            targets=[format(i,f'0{n}b') for i in range(0,2**n,3)]
            expected=np.ones(2**n);expected[[int(t,2) for t in targets]]=-1
            np.testing.assert_allclose(Operator(oracle_for_targets(n,targets)).data,np.diag(expected),atol=1e-12)

    def test_duplicate_and_malformed_targets_are_rejected(self):
        for n,targets in [(2,['01','01']),(2,['2a']),(2,['0']),(0,[]),(True,[]),(3,{'000'})]:
            with self.subTest(n=n,targets=targets),self.assertRaises(ValueError):oracle_for_targets(n,targets)

    def test_reflections_match_exact_dense_matrices_including_one_qubit(self):
        for n in (1,2,3,4):
            a=uniform_preparation(n);a.ry(.31,0);a.rz(.19,n-1)
            if n>1:a.cx(0,n-1)
            psi=Statevector.from_instruction(a).data
            expected=2*np.outer(psi,psi.conj())-np.eye(2**n)
            d=Operator(diffusion(a)).data
            np.testing.assert_allclose(d,expected,atol=2e-12)
            np.testing.assert_allclose(d@d,np.eye(2**n),atol=2e-12)

    def test_complex_nonuniform_preparation_matches_full_amplitude_reference(self):
        a=QuantumCircuit(3);a.ry(.4,0);a.ry(.9,1);a.h(2);a.cx(2,1);a.rz(.6,0)
        targets=['001','110'];psi=Statevector.from_instruction(a).data
        reduced=partial_trace(Statevector(psi),[0,2]).data
        self.assertLess(float(np.trace(reduced@reduced).real),.99)  # genuinely entangled preparation
        p=float(sum(abs(psi[int(t,2)])**2 for t in targets))
        for k in range(7):
            state=Statevector.from_instruction(build_amplification(a,oracle_for_targets(3,targets),k)).data
            np.testing.assert_allclose(state,dense_reference(psi,targets,k),atol=3e-12)
            self.assertAlmostEqual(sum(abs(state[int(t,2)])**2 for t in targets),analytic_probability(p,k),places=11)

    def test_first_peak_integer_policy_handles_dense_marking_and_empty_full_cases(self):
        for p,k in [(0.,0),(1.,0),(.5,0),(.75,0),(1/4,1),(2/32,3),(2/256,8)]:
            self.assertEqual(first_peak_iterations(p),k)
        for p in np.linspace(.001,.999,201):
            k=first_peak_iterations(float(p));peak=math.pi/(4*math.asin(math.sqrt(p)))-.5
            candidates={max(0,math.floor(peak)),max(0,math.ceil(peak))}
            self.assertIn(k,candidates)
            self.assertGreaterEqual(analytic_probability(p,k)+1e-13,max(analytic_probability(p,j) for j in candidates))

    def test_zero_iteration_and_empty_full_oracles(self):
        a=uniform_preparation(2)
        for targets in ([],['00','01','10','11']):
            result,_=simulate(a,oracle_for_targets(2,targets),targets,0,shots=256)
            self.assertEqual(result['exact_success'],float(bool(targets)))

    def test_overshooting_is_observed_not_hidden(self):
        p=2/32;k=first_peak_iterations(p)
        self.assertGreater(analytic_probability(p,k),analytic_probability(p,k+2))

    def test_seeded_shots_follow_the_analytic_probability(self):
        a=uniform_preparation(5);targets=['00011','10101'];o=oracle_for_targets(5,targets)
        for seed in (7,2718):
            r,_=simulate(a,o,targets,3,shots=4096,seed=seed)
            self.assertLessEqual(abs(r['sampled_success']-r['exact_success']),r['hoeffding_half_width'])
            self.assertEqual(sum(r['counts'].values()),4096)
            self.assertEqual(r['valid_shots']+r['invalid_shots'],r['shots'])
            self.assertLessEqual(r['binomial_95_interval'][0],r['sampled_success'])
            self.assertGreaterEqual(r['binomial_95_interval'][1],r['sampled_success'])

    def test_board_predicate_rejects_out_of_range_or_malformed_columns(self):
        for cols in ([-1], [1], [True], ['0'], [], [0,0], [4,2,0,3]):
            self.assertFalse(valid_board(cols))
        self.assertTrue(valid_board([1,3,0,2]))

    def test_queens_classical_solvers_and_non_power_of_two_embedding(self):
        self.assertEqual(queens_reference(4)['boards'],[[1,3,0,2],[2,0,3,1]])
        q=queens_reference(5)
        self.assertEqual(q['marked_count'],10)
        self.assertEqual(q['binary_search_space'],32768)
        self.assertEqual(q['in_range_assignments'],3125)
        self.assertEqual(q['first_peak_iterations'],44)
        self.assertLess(q['backtracking_column_attempts'],q['classical_enumeration_candidates'])

    def test_constraint_oracle_all_256_assignments_and_clean_workspace(self):
        # Random complex amplitudes detect unwanted basis permutations as well
        # as wrong phases. Every possible board is present in one probe.
        rng=np.random.default_rng(1234);psi=rng.normal(size=256)+1j*rng.normal(size=256);psi/=np.linalg.norm(psi)
        qc=QuantumCircuit(14);qc.initialize(psi,range(8));qc.compose(queens4_constraint_oracle(),inplace=True)
        sim=AerSimulator(method='statevector',max_parallel_threads=1)
        compiled=transpile(qc,sim,seed_transpiler=2718);compiled.save_statevector()
        state=np.asarray(sim.run(compiled).result().get_statevector()).reshape(-1,256)
        expected=psi.copy()
        for x in range(256):
            bits=format(x,'08b');cols=[int(bits[i:i+2],2) for i in range(0,8,2)]
            if len(set(cols))==4 and all(abs(cols[i]-cols[j])!=j-i for i in range(4) for j in range(i+1,4)):
                expected[x]*=-1
        np.testing.assert_allclose(state[0],expected,atol=1e-11)
        self.assertLess(np.linalg.norm(state[1:]),1e-11)

    def test_queens_constraint_run_matches_lookup_reference(self):
        ref=queens_reference(4)
        r,_=simulate(uniform_preparation(8),queens4_constraint_oracle(),ref['targets'],8,shots=2048)
        self.assertLess(r['state_error_l2'],1e-9)
        self.assertLess(r['workspace_leakage'],1e-12)
        self.assertGreater(r['exact_success'],.99)

    def test_nonunitary_unbound_and_over_budget_inputs_fail(self):
        from qiskit.circuit import Parameter
        bad=[]
        a=QuantumCircuit(1);a.reset(0);bad.append(a)
        a=QuantumCircuit(1,1);a.measure(0,0);bad.append(a)
        a=QuantumCircuit(1);a.ry(Parameter('t'),0);bad.append(a)
        for a in bad:
            with self.assertRaises(ValueError):diffusion(a)
        for k in (-1,True,1.5,129):
            with self.assertRaises(ValueError):build_amplification(uniform_preparation(2),oracle_for_targets(2,['01']),k)
        for p in (float('nan'),float('inf'),True,-.1,1.1):
            with self.assertRaises(ValueError):first_peak_iterations(p)
        with self.assertRaises(ValueError):build_amplification(uniform_preparation(2),QuantumCircuit(MAX_TOTAL_QUBITS+1),1)
        with self.assertRaises(ValueError):queens_reference(6)

    def test_classical_sampling_control_and_transpiled_resources(self):
        self.assertAlmostEqual(classical_success(.25,2),7/16)
        self.assertEqual(classical_success(1,0),0)
        cost=resource_counts(oracle_for_targets(3,['001']))
        self.assertGreater(cost['cx_count'],0)
        self.assertGreater(cost['depth'],0)


    def test_counter_increment_is_correct_for_every_counter_value_and_control(self):
        from amplification import _increment_if
        circuit=QuantumCircuit(5)
        _increment_if(circuit,[0],list(range(1,5)))
        expected=np.zeros((32,32),complex)
        for control in (0,1):
            for count in range(16):
                source=(count<<1)|control
                target=(((count+control)%16)<<1)|control
                expected[target,source]=1
        np.testing.assert_allclose(Operator(circuit).data,expected,atol=1e-12)

    def test_five_queens_constraint_oracle_marks_all_32768_inputs_and_uncomputes(self):
        size=32768
        rng=np.random.default_rng(2026)
        psi=rng.normal(size=size)+1j*rng.normal(size=size);psi/=np.linalg.norm(psi)
        oracle=queens_constraint_oracle(5)
        self.assertEqual(oracle.num_qubits,19)
        qc=QuantumCircuit(19)
        state=np.zeros(2**19,complex);state[:size]=psi
        qc.set_statevector(state)
        qc.compose(oracle,inplace=True)
        backend=AerSimulator(method='statevector',max_parallel_threads=1)
        compiled=transpile(qc,backend,optimization_level=1,seed_transpiler=2718)
        compiled.save_statevector()
        observed=np.asarray(backend.run(compiled).result().get_statevector()).reshape(16,size)
        # Independent vectorized predicate includes invalid binary column codes.
        x=np.arange(size)
        cols=np.column_stack([(x>>(3*(4-row)))&7 for row in range(5)])
        valid=np.all(cols<5,axis=1)
        for i in range(5):
            for j in range(i+1,5):
                valid &= (cols[:,i]!=cols[:,j]) & (np.abs(cols[:,i]-cols[:,j])!=j-i)
        self.assertEqual(int(valid.sum()),10)
        expected=psi*np.where(valid,-1,1)
        np.testing.assert_allclose(observed[0],expected,atol=2e-11)
        self.assertLess(np.linalg.norm(observed[1:]),1e-11)
        self.assertEqual([format(i,'015b') for i in np.flatnonzero(valid)],queens_reference(5)['targets'])

    def test_counter_predicate_supports_no_solution_instances(self):
        for n in (2,3):
            reference=queens_reference(n)
            self.assertEqual(reference['marked_count'],0)
            result,_=simulate(uniform_preparation(reference['system_qubits']),
                queens_constraint_oracle(n),[],2,shots=256)
            self.assertLess(result['state_error_l2'],1e-10)
            self.assertEqual(result['exact_success'],0.)
        self.assertEqual(queens_constraint_oracle(np.int64(4)).num_qubits,11)

    def test_cached_ideal_readout_agrees_with_direct_circuit_sampling(self):
        a=uniform_preparation(5);targets=['00011','10101']
        run,circuit=simulate(a,oracle_for_targets(5,targets),targets,3,shots=4096)
        measured=circuit.copy();measured.add_register(__import__('qiskit').ClassicalRegister(5))
        measured.measure(range(5),range(5))
        backend=AerSimulator(method='statevector',max_parallel_threads=1)
        counts=backend.run(transpile(measured,backend,seed_transpiler=2718),shots=4096,seed_simulator=2718).result().get_counts()
        direct=sum(counts.get(t,0) for t in targets)/4096
        self.assertLess(abs(direct-run['exact_success']),run['hoeffding_half_width'])
        self.assertLess(abs(direct-run['sampled_success']),2*run['hoeffding_half_width'])


class PublishedEvidenceTests(unittest.TestCase):
    def test_published_evidence_has_current_sources_and_valid_physics(self):
        import json
        from pathlib import Path
        from experiments import source_hashes
        report=json.loads((Path(__file__).parent/'results/summary.json').read_text())
        self.assertEqual(report['source_hashes'],source_hashes(),'Regenerate the published evidence after source changes')
        manifest=json.loads((Path(__file__).parent/'results/manifest.json').read_text())
        import hashlib
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent/'results'/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(set(manifest['files']),{'summary.json','amplification.svg'})
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['network_calls'],0)
        for run in (report['uniform'],report['nonuniform'],report['queens4']['constraint_run'],report['queens5']['lookup_run'],report['queens5']['constraint_run']):
            self.assertLess(run['state_error_l2'],1e-9)
            self.assertLess(run['workspace_leakage'],1e-12)
            self.assertAlmostEqual(run['analytic_success'],run['exact_success'],places=10)
            self.assertLessEqual(abs(run['sampled_success']-run['exact_success']),run['hoeffding_half_width'])
        self.assertEqual(report['queens4']['constraint_run']['workspace_qubits'],6)
        self.assertIn('pre-enumerated',report['queens5']['lookup_scope'])
        self.assertEqual(report['queens5']['constraint_run']['workspace_qubits'],4)
        self.assertEqual(report['queens5']['constraint_run']['iterations'],44)
        self.assertGreater(report['queens5']['constraint_run']['exact_success'],.999)
        self.assertEqual(report['queens5']['reference_observed_marked_states'],report['queens5']['classical']['targets'])

if __name__=='__main__':unittest.main()
