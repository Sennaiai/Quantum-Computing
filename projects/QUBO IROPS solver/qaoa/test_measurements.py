import hashlib,json,unittest
from collections import defaultdict
import numpy as np
from bootstrap import PARENT
from fixtures import small_cases
from data_contract import instance_payload
from solver import exact_reference,score
from qaoa_core import QUBOStatevector
from measurements import compile_case,masks,audit_counts,apply_global_patch


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.instance=small_cases()[0];self.compiled=compile_case(self.instance);self.model=QUBOStatevector(self.compiled.bqm)
        _,self.reference=exact_reference(self.instance)

    def test_strict_and_physical_feasibility_are_distinct(self):
        sample={v:0 for v in self.model.labels}
        for label in self.compiled.dummy_variables:sample[label]=1
        index=sum(sample[v] << q for q,v in enumerate(self.model.labels))
        result=audit_counts(self.compiled,self.model,self.instance,{index:1024},self.reference['objective'])
        self.assertEqual(result['raw_strict_feasible_fraction'],0)
        self.assertEqual(result['raw_physical_feasible_fraction'],1)
        self.assertEqual(result['raw_best_physical_objective'],30)
        self.assertEqual(result['repaired']['capacity_violations'],0)

    def test_invalid_zero_state_is_not_reported_as_a_valid_raw_solution(self):
        result=audit_counts(self.compiled,self.model,self.instance,{0:1024},self.reference['objective'])
        self.assertEqual(result['raw_physical_feasible_fraction'],0)
        self.assertIsNone(result['raw_best_physical_objective'])
        self.assertIsNone(result['raw_best_physical_objective_gap'])
        self.assertFalse(result['raw_physical_reference_hit'])
        self.assertEqual(result['raw_best_onehot_violations'],3)
        self.assertEqual(result['repaired']['assigned_pnrs'],2)

    def test_exhaustive_masks_agree_with_independent_decoded_checks(self):
        strict,physical,obj=masks(self.compiled,self.model,self.instance)
        for index in range(len(self.model.states)):
            sample=self.model.sample_dict(index);decoded=self.compiled.decode(sample)
            onehot=all(sum(sample[v] for v in coeff)==1 for coeff,c in self.compiled.constraints[:3])
            used=defaultdict(int)
            for pid,oids in decoded.items():
                for oid in oids:
                    for leg in self.instance.option_by_id[oid].legs:used[leg]+=self.instance.party_by_id[pid].pax
            valid=onehot and all(n<=self.instance.capacities[k] for k,n in used.items())
            self.assertEqual(bool(physical[index]),valid)
            self.assertEqual(bool(strict[index]),all(c+sum(int(a)*sample[v] for v,a in coeff.items())==0 for coeff,c in self.compiled.constraints))
            self.assertAlmostEqual(obj[index],sum(c*sample[v] for v,c in self.compiled.objective.items()))

    def test_bad_counts_are_rejected(self):
        for counts in ({},{0:0},{-1:1},{len(self.model.states):1},{0:-1}):
            with self.assertRaises(ValueError):audit_counts(self.compiled,self.model,self.instance,counts,self.reference['objective'])


class GlobalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.full=small_cases()[2];self.anchor={'SYN_A':'SYN_A1'}
        self.row={'instance':instance_payload(self.full),'outside_loads_sha256':hashlib.sha256(b'[]').hexdigest()}

    def test_improving_patch_is_globally_audited_and_does_not_mutate_incumbent(self):
        chosen={'SYN_A':'SYN_A2','SYN_B':'SYN_B1'}
        result,audit=apply_global_patch(self.full,self.anchor,self.row,chosen)
        self.assertTrue(audit['accepted']);self.assertEqual(len(result),2)
        self.assertEqual(score(self.full,result)['capacity_violations'],0)
        self.assertEqual(self.anchor,{'SYN_A':'SYN_A1'})

    def test_overcapacity_ghost_and_nonimproving_patches_cannot_be_admitted(self):
        with self.assertRaises(ValueError):apply_global_patch(self.full,self.anchor,self.row,{'SYN_A':'SYN_A1','SYN_B':'SYN_B1'})
        with self.assertRaises(ValueError):apply_global_patch(self.full,self.anchor,self.row,{'ghost':'SYN_A1'})
        result,audit=apply_global_patch(self.full,self.anchor,self.row,{})
        self.assertFalse(audit['accepted']);self.assertEqual(result,self.anchor)

    def test_stale_residual_context_is_rejected(self):
        wrong={**self.row,'outside_loads_sha256':'stale'}
        with self.assertRaisesRegex(ValueError,'Stale residual'):
            apply_global_patch(self.full,self.anchor,wrong,{'SYN_A':'SYN_A2'})
