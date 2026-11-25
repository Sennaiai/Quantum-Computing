from pathlib import Path
import hashlib,json,unittest
import nbformat
from experiments import source_hashes,NOTEBOOKS
ROOT=Path(__file__).resolve().parent

class EvidenceTests(unittest.TestCase):
    def test_published_checksums_and_sources(self):
        manifest=json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(set(manifest['files']),{'arithmetic.json','shor.json','arithmetic.svg','order_finding.svg'})
        for name,digest in manifest['files'].items():self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(),digest)
        for name in ('arithmetic.json','shor.json'):
            report=json.loads((ROOT/'results'/name).read_text())
            self.assertEqual(report['source_hashes'],source_hashes(),'Regenerate published evidence after intentional source changes')
            self.assertEqual(report['status'],'passed')

    def test_published_arithmetic_agrees_with_reference_contracts(self):
        report=json.loads((ROOT/'results/arithmetic.json').read_text())
        self.assertEqual(len(report['coherent_arithmetic']),13)
        for row in report['coherent_arithmetic']:
            self.assertLess(row['state_error_l2'],1e-9)
            self.assertLess(row['outside_promised_output_probability'],1e-10)
        self.assertLess(report['controlled_phase_regression']['basis_probability_error'],1e-12)
        self.assertGreater(report['controlled_phase_regression']['controlled_operator_error'],1.)
        self.assertGreater(report['approximation'][0]['operator_error_2'],1.)
        self.assertLess(report['approximation'][0]['roundtrip_error_2'],1e-12)
        for row in report['adder_comparison']:
            self.assertEqual(row['qft']['qubits'],2*row['data_width'])
            self.assertLess(max(row['qft_coherent_error'],row['ripple_coherent_error']),1e-10)

    def test_published_shor_factors_and_failures_are_both_retained(self):
        report=json.loads((ROOT/'results/shor.json').read_text())
        self.assertEqual(report['gate_arithmetic_N15']['factorization']['factors'],[3,5])
        self.assertEqual(report['compiled_N21_base2']['factorization']['factors'],[3,7])
        self.assertEqual(report['compiled_N21_base5_negative_control']['recovery']['order'],6)
        self.assertEqual(report['compiled_N21_base5_negative_control']['factorization']['status'],'trivial_square_root')
        for name in ('gate_arithmetic_N15','compiled_N15','compiled_N21_base2','compiled_N21_base5_negative_control'):
            r=report[name];self.assertLess(r['state_error_l2_vs_exact_fft'],1e-9);self.assertLess(r['workspace_leakage'],1e-10)
            self.assertEqual(sum(r['counts'].values()),r['shots'])
            self.assertAlmostEqual(sum(r['phase_probabilities']),1.,places=9)
        self.assertGreater(report['approximate_IQFT_N21']['phase_tv_vs_exact_fft'],.05)

    def test_only_two_output_free_source_notebooks(self):
        self.assertEqual({p.name for p in ROOT.glob('*.ipynb')},set(NOTEBOOKS))
        for name in NOTEBOOKS:
            notebook=nbformat.read(ROOT/name,as_version=4);nbformat.validate(notebook)
            self.assertTrue(all(c.get('execution_count') is None and not c.get('outputs') for c in notebook.cells if c.cell_type=='code'))
            self.assertNotIn('!pip','\n'.join(c.source for c in notebook.cells))
