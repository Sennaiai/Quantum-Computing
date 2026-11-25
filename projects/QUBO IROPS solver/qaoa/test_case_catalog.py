import gzip,json,unittest
from pathlib import Path
import numpy as np
from bootstrap import ROOT,PARENT
from case_catalog import load_cases,build_cases,SCENARIO,parent_input_identity
from measurements import compile_case,masks
from qaoa_core import QUBOStatevector
from solver import exact_reference,score


class ActualCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data,cls.cases=load_cases()

    def test_actual_cases_have_declared_structural_sizes_and_real_identifiers(self):
        self.assertEqual([r['qubits'] for r,i in self.cases],[8,9,12,13])
        ids=set()
        for row,instance in self.cases:
            self.assertTrue(row['root_pnr'].startswith('PNR_'))
            self.assertNotIn(row['root_pnr'],row['incumbent'])
            self.assertFalse(ids & set(instance.party_by_id));ids.update(instance.party_by_id)
            self.assertTrue(all(o.id.startswith(('OPT_','EXT_')) for o in instance.options))
            self.assertTrue(all(f.startswith('LEG_') for f,c in instance.capacities))
            self.assertEqual(score(instance,row['incumbent'],SCENARIO)['capacity_violations'],0)
            compiled=compile_case(instance)
            self.assertEqual(compiled.bqm.num_variables,row['qubits'])
            self.assertTrue(compiled.slack)
        self.assertIn('No QAOA or exact-solution outcomes',self.data['selection'])

    def test_all_binary_case_energies_match_semantics_and_independent_feasible_optimum(self):
        for row,instance in self.cases:
            compiled=compile_case(instance);model=QUBOStatevector(compiled.bqm,16)
            strict,physical,objective=masks(compiled,model,instance)
            _,exact=exact_reference(instance)
            self.assertAlmostEqual(float(model.energy.min()),exact['objective'],places=7)
            self.assertTrue(strict[np.isclose(model.energy,model.energy.min(),atol=1e-8,rtol=0)].all())
            for index in range(len(model.states)):
                sample=model.sample_dict(index)
                self.assertLessEqual(abs(compiled.semantic_energy(sample)-model.energy[index]),max(1e-8,compiled.roundoff_bound(sample)))

    def test_case_generator_reproduces_frozen_actual_neighbourhoods_without_solver_tuning(self):
        fresh=build_cases()
        self.assertEqual(json.loads(json.dumps(fresh)),self.data)
        self.assertEqual(fresh['candidate_neighbourhoods_examined'],299)
