from pathlib import Path
import hashlib,json,unittest
import numpy as np
import nbformat
from qiskit import qasm2
from qiskit.quantum_info import Statevector
from bootstrap import ROOT,PARENT
from qaoa_experiment import CONFIG,NOTEBOOK,source_hashes
from case_catalog import load_cases,parent_input_identity,SCENARIO
from qaoa_core import QUBOStatevector
from constraint_qaoa import OneHotQAOA
from measurements import compile_case,audit_counts
from solver import exact_reference


class PublishedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((ROOT/'results/qaoa_results.json').read_text())
        _,cases=load_cases();cls.cases={r['id']:(r,i) for r,i in cases}

    def test_manifest_sources_and_parent_binding(self):
        manifest=json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(set(manifest['files']),{'qaoa_results.json','qaoa_comparison.svg','example_qaoa.qasm'})
        for name,expected in manifest['files'].items():self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(),expected)
        self.assertEqual(self.report['source_hashes'],source_hashes())
        self.assertEqual(self.report['parent_identity'],parent_input_identity())
        self.assertEqual(self.report['status'],'passed')

    def test_every_sample_batch_and_repair_recomputed_from_published_counts(self):
        models={key:QUBOStatevector(compile_case(i).bqm,16) for key,(row,i) in self.cases.items()}
        constrained={key:OneHotQAOA(compile_case(i),16) for key,(row,i) in self.cases.items()}
        for run in self.report['runs']:
            row,instance=self.cases[run['case']];model=models[run['case']]
            _,reference=exact_reference(instance)
            counts={int(k):v for k,v in run['sample_counts'].items()}
            audit=audit_counts(compile_case(instance),model,instance,counts,reference['objective'])
            self.assertEqual(sum(counts.values()),CONFIG['shots'])
            for key,value in audit.items():
                if isinstance(value,float):self.assertAlmostEqual(value,run[key],places=10)
                else:self.assertEqual(value,run[key])
            self.assertEqual(run['repaired']['capacity_violations'],0)
            if run['backend'].startswith('qaoa'):
                self.assertAlmostEqual(run['circuit_fidelity'],1.,places=9)
                params=run['training']['parameters']
                quantum_model=constrained[run['case']] if run['backend'].startswith('qaoa_onehot') else model
                self.assertAlmostEqual(quantum_model.physical_energy(params),run['training']['expected_energy'],places=8)
                if run['backend'].startswith('qaoa_onehot'):self.assertEqual(run['raw_best_onehot_violations'],0)
                self.assertTrue(all(x>=0 for x in run['gate_counts'].values()))

    def test_all_predeclared_cases_seeds_depths_and_controls_are_present(self):
        expected={(key,seed,method) for key in self.cases for seed in CONFIG['seeds']
            for method in ['uniform','onehot_uniform','neal']+[f'{family}_p{p}' for family in ('qaoa','qaoa_onehot') for p in CONFIG['depths']]}
        actual={(r['case'],r['seed'],r['backend']) for r in self.report['runs']}
        self.assertEqual(actual,expected);self.assertEqual(len(self.report['runs']),len(expected))
        self.assertIn('compute/physical-shot budgets are not equated',self.report['fairness'])
        self.assertIn('no exact optimum',next(r for r in self.report['runs'] if r['backend']=='qaoa_p1')['training']['training'].lower())

    def test_global_admission_metrics_never_claim_invalid_or_degrading_allocations(self):
        for audit in self.report['global_patch_audits']:
            self.assertEqual(audit['admitted']['capacity_violations'],0)
            self.assertGreaterEqual(audit['admitted']['assigned_pnrs'],audit['before']['assigned_pnrs'])
            self.assertGreaterEqual(audit['admitted']['assigned_passengers'],audit['before']['assigned_passengers'])
            self.assertLessEqual(audit['admitted']['objective'],audit['before']['objective']+1e-7)
            self.assertIn('independent cases cannot be summed',audit['scope'])

    def test_exported_circuit_reproduces_selected_actual_qaoa_state(self):
        row,instance=self.cases['actual_01'];model=OneHotQAOA(compile_case(instance),16)
        run=next(r for r in self.report['runs'] if r['case']=='actual_01' and r['seed']==7 and r['backend']=='qaoa_onehot_p1')
        expected,_=model.evolve(run['training']['parameters'])
        loaded=Statevector.from_instruction(qasm2.loads((ROOT/'results/example_qaoa.qasm').read_text()))
        self.assertAlmostEqual(abs(np.vdot(loaded,expected))**2,1.,places=10)

    def test_notebook_is_concise_and_does_not_load_raw_or_private_data(self):
        nb=nbformat.read(ROOT/NOTEBOOK,as_version=4);nbformat.validate(nb)
        self.assertLess(sum(len(c.source.split()) for c in nb.cells if c.cell_type=='markdown'),180)
        self.assertEqual(len([c for c in nb.cells if c.cell_type=='code']),3)
        self.assertFalse(any(o.output_type=='error' for c in nb.cells for o in c.get('outputs',[])))
        self.assertNotIn('irops_clean','\n'.join(c.source for c in nb.cells))
