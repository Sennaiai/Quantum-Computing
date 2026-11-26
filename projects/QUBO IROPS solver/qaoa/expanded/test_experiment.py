import json,unittest
from datetime import datetime,timezone
import numpy as np
from qiskit.quantum_info import Statevector
from expanded.catalog import load_expanded_cases,build_expanded_cases
from expanded.subspace import SubspaceQAOA
from expanded.pareto import exact_tradeoffs
from expanded.measurements import DrawAudit
from measurements import compile_case,audit_counts
from qaoa_core import QUBOStatevector
from case_catalog import load_cases,SCENARIO
from solver import exact_reference,score
from data_contract import Instance,Party,Option,FEATURES


class ExpandedCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.provenance,cls.cases=load_expanded_cases()

    def test_actual_structural_sizes_anchor_and_disjointness(self):
        self.assertEqual([r['qubits'] for r,i in self.cases],[17,19,22,25,28,26])
        self.assertEqual([len(i.parties) for r,i in self.cases],[4,5,6,6,8,10])
        seen=set()
        for row,instance in self.cases:
            self.assertFalse(seen.intersection(instance.party_by_id));seen.update(instance.party_by_id)
            self.assertEqual(score(instance,row['incumbent'],SCENARIO)['capacity_violations'],0)
            self.assertNotIn(row['root_pnr'],row['incumbent'])
            model=SubspaceQAOA(compile_case(instance))
            self.assertEqual((model.n,model.dimension),(row['qubits'],row['subspace_states']))
        self.assertIn('No solver/reference outcomes',self.provenance['rich_selection'])

    def test_deterministic_selection_reproduces_published_cases(self):
        fresh=build_expanded_cases()
        self.assertEqual(json.loads(json.dumps(fresh)),self.provenance)

    def test_all_local_ground_energies_and_sampled_binary_semantics(self):
        for row,instance in self.cases:
            compiled=compile_case(instance);model=SubspaceQAOA(compiled);reference=exact_tradeoffs(instance)
            self.assertAlmostEqual(float(model.energy.min()),reference['minimum_S3'],places=7)
            rng=np.random.default_rng(97)
            for index in rng.choice(model.dimension,min(64,model.dimension),replace=False):
                sample=model.basis_sample(int(model.states[index]))
                self.assertLessEqual(abs(compiled.semantic_energy(sample)-model.energy[index]),max(1e-8,compiled.roundoff_bound(sample)))
            best=int(model.states[np.argmin(model.energy)])
            audit=DrawAudit(compiled,model,instance,reference).evaluate({best:1024})
            self.assertEqual(audit['valid_candidate_draws'],1024)
            self.assertEqual(audit['strict_encoding_draws'],1024)
            self.assertTrue(audit['raw_local_S3_optimum_hit'])

    def test_full_Qiskit_state_for_22_qubit_control(self):
        instance=self.cases[2][1];model=SubspaceQAOA(compile_case(instance));params=[.43,.72]
        state,_=model.evolve(params);full=np.asarray(Statevector.from_instruction(model.circuit(params)))
        self.assertAlmostEqual(abs(np.vdot(full[model.states],state))**2,1,places=8)
        self.assertAlmostEqual(float(np.sum(abs(full[model.states])**2)),1,places=8)

    def test_scalar_draw_audit_matches_baseline_dense_audit(self):
        for row,instance in load_cases()[1]:
            compiled=compile_case(instance);dense=QUBOStatevector(compiled.bqm);compact=SubspaceQAOA(compiled)
            counts=compact.draw([.8,.5],64,7);reference=exact_tradeoffs(instance)
            a=DrawAudit(compiled,compact,instance,reference).evaluate(counts)
            b=audit_counts(compiled,dense,instance,counts,reference['minimum_S3'])
            self.assertEqual(a['valid_candidate_draws'],round(b['raw_physical_feasible_fraction']*64))
            self.assertEqual(a['strict_encoding_draws'],round(b['raw_strict_feasible_fraction']*64))
            self.assertEqual(a['repaired_assignment'],b['repaired_assignment'])

    def test_uint64_basis_indices_preserve_qubits_above_31(self):
        parties=tuple(Party(f'P{i}',1,0.) for i in range(17))
        caps={(f'L{i}','Y'):1 for i in range(17)}
        options=tuple(Option(f'O{i}',p.id,((f'L{i}','Y'),),datetime(2026,1,1,tzinfo=timezone.utc),0.,(0.,)*len(FEATURES)) for i,p in enumerate(parties))
        model=SubspaceQAOA(compile_case(Instance(parties,options,caps)))
        self.assertEqual(model.n,34);self.assertTrue(any(int(k)>2**32 for k in model.states))
        for basis in model.draw([.1,.2],8,7):
            sample=model.basis_sample(basis)
            self.assertEqual(sum(sample[v] << i for i,v in enumerate(model.labels)),basis)
