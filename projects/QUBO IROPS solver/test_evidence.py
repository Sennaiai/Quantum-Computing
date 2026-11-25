from pathlib import Path
import hashlib
import json
import unittest
import nbformat
import pandas as pd
from experiment import ROOT, NOTEBOOK, CONFIG, load_public_catalog, source_hashes
from solver import score


class ActualDataEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.instance, cls.provenance = load_public_catalog()

    def test_actual_catalog_and_deidentification_contract(self):
        i, p = self.instance, self.provenance
        self.assertFalse(p['synthetic'])
        self.assertEqual((len(i.parties), len(i.options), sum(p.pax for p in i.parties)), (7653, 57963, 11822))
        self.assertTrue(all(p.id.startswith('PNR_') for p in i.parties))
        self.assertTrue(all(o.id.startswith('OPT_') for o in i.options))
        self.assertTrue(all(f.startswith('LEG_') for f, c in i.capacities))
        self.assertEqual(p['input_audit']['negative_capacities_clipped'], 622)
        self.assertGreater(p['input_audit']['options_with_original_duration_over_72h'], 0)

    def test_published_manifest_and_source_identities(self):
        manifest = json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(set(manifest['files']), {'actual_catalog_results.json', 'assignments.csv.gz', 'allocation_resources.svg'})
        for name, value in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(), value)
        result = json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        self.assertEqual(result['source_hashes'], source_hashes())
        self.assertEqual(result['status'], 'passed')

    def test_every_seed_allocation_and_capacity_recomputed_from_saved_assignments(self):
        result = json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        df = pd.read_csv(ROOT/'results/assignments.csv.gz', dtype='string')
        self.assertEqual(list(df.PNR), [p.id for p in self.instance.parties])
        self.assertFalse(df.PNR.duplicated().any())
        self.assertEqual([r['seed'] for r in result['runs']], CONFIG['seeds'])
        for row in result['runs']:
            column = f"seed_{row['seed']}"
            selected = {str(p): str(o) for p, o in zip(df.PNR, df[column]) if not pd.isna(o)}
            actual = score(self.instance, selected, result['scenario'])
            for field in actual: self.assertAlmostEqual(actual[field], row[field], places=7)
            self.assertEqual(actual['capacity_violations'], 0)
            self.assertLessEqual(row['resources']['peak_variables'], CONFIG['max_bqm_variables'])
            for c in row['chunks']:
                self.assertGreater(c['penalty'], c['objective_range_bound'])
                self.assertLessEqual(c['best_sample_energy_identity_error'], c['energy_identity_roundoff_bound'])

    def test_notebook_is_concise_and_does_not_load_private_source_tables(self):
        self.assertEqual({p.name for p in ROOT.glob('*.ipynb')}, {NOTEBOOK})
        nb = nbformat.read(ROOT/NOTEBOOK, as_version=4); nbformat.validate(nb)
        source = '\n'.join(c.source for c in nb.cells)
        self.assertNotIn('!pip', source); self.assertNotIn('irops_clean', source)
        self.assertLess(sum(len(c.source.split()) for c in nb.cells if c.cell_type == 'markdown'), 180)
        self.assertFalse(any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', [])))

    def test_external_client_comparison_is_not_fabricated(self):
        result = json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        self.assertFalse(result['client_comparison']['outperformance_claim'])
        self.assertIn('independently', result['client_comparison']['scope'])

    def test_end_to_end_runtime_covers_all_solver_runs(self):
        result = json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        timing = result['timing']
        self.assertEqual(timing['seed_count'], len(CONFIG['seeds']))
        self.assertGreaterEqual(timing['end_to_end_seconds'], sum(r['seconds'] for r in result['runs']))
        self.assertIn('exports', timing['scope'])
