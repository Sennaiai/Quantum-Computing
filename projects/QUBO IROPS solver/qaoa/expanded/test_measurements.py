import unittest
from case_catalog import load_cases
from measurements import compile_case
from expanded.subspace import SubspaceQAOA
from expanded.pareto import exact_tradeoffs
from expanded.measurements import DrawAudit,paired_S3_comparison


class ExpandedMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance=load_cases()[1][0][1];cls.compiled=compile_case(cls.instance)
        cls.model=SubspaceQAOA(cls.compiled);cls.reference=exact_tradeoffs(cls.instance)

    def test_classical_repair_is_not_counted_as_raw_feasibility_or_frontier_discovery(self):
        audit=DrawAudit(self.compiled,self.model,self.instance,self.reference).evaluate({0:1024})
        self.assertEqual(audit['valid_candidate_draws'],0)
        self.assertEqual(audit['observed_frontier'],[])
        self.assertEqual(audit['exact_local_frontier_points_found'],0)
        self.assertFalse(audit['raw_local_S3_optimum_hit'])
        self.assertEqual(audit['repaired']['capacity_violations'],0)

    def test_invalid_sample_counts_cannot_be_published(self):
        inspector=DrawAudit(self.compiled,self.model,self.instance,self.reference)
        for counts in ({},{0:0},{0:-1},{-1:1},{1<<self.model.n:1}):
            with self.assertRaises(ValueError):inspector.evaluate(counts)

    def test_paired_comparison_matches_case_and_seed_instead_of_unpaired_means(self):
        report={'config':{'depths':[1]},'runs':[]}
        for seed,a,b in ((7,1.,2.),(11,4.,3.),(29,2.,2.)):
            report['runs'].extend([{'case':'a','seed':seed,'backend':'neal','raw_best_S3':b},
                {'case':'a','seed':seed,'backend':'onechoice_QAOA_p1','raw_best_S3':a}])
        value=paired_S3_comparison(report)[0]
        self.assertEqual((value['QAOA_lower_raw_S3'],value['equal_raw_S3'],value['annealing_lower_raw_S3']),(1,1,1))
