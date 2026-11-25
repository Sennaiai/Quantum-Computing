import hashlib
import json
from pathlib import Path
import unittest
import nbformat
import numpy as np
import pandas as pd
from benchmark import NOTEBOOKS, source_hashes, metrics
from data_contract import ROOT, SEEDS, load_data, prepare


class EvidenceTests(unittest.TestCase):
    def test_published_sources_and_file_checksums(self):
        manifest = json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(set(manifest['files']), {'cv.json', 'spin.json', 'cv.svg', 'spin.svg',
                                               'cv_predictions.csv', 'spin_predictions.csv', 'physics.json'})
        for name, digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(), digest)
        for study in ('cv', 'spin'):
            report = json.loads((ROOT/'results'/f'{study}.json').read_text())
            self.assertEqual(report['source_hashes'], source_hashes())
            self.assertEqual(report['status'], 'passed')

    def test_every_published_model_metric_recomputed_from_aligned_predictions(self):
        data = prepare(load_data()); y = data.y[data.test_start:]
        scale = np.mean(abs(np.diff(data.y[:data.initial_train])))
        for study in ('cv', 'spin'):
            report = json.loads((ROOT/'results'/f'{study}.json').read_text())
            predictions = pd.read_csv(ROOT/'results'/f'{study}_predictions.csv')
            np.testing.assert_array_equal(predictions.actual_MW, y)
            self.assertEqual(list(pd.to_datetime(predictions.target_utc, utc=True)), list(data.times[data.test_start:]))
            for name, row in report['models'].items():
                actual = metrics(y, predictions[name].to_numpy(), scale)
                for metric in actual: self.assertAlmostEqual(actual[metric], row['test'][metric], places=6)
                if name.endswith('_ensemble'):
                    member = predictions[row['members']].mean(axis=1)
                    np.testing.assert_allclose(predictions[name], member, atol=1e-7)
                    self.assertEqual(len(row['members']), len(SEEDS))

    def test_cross_study_targets_splits_and_classical_benchmarks_identical(self):
        cv, spin = [json.loads((ROOT/'results'/f'{s}.json').read_text()) for s in ('cv', 'spin')]
        self.assertEqual(cv['splits'], spin['splits'])
        for name in ('persistence', 'daily', 'weekly', 'raw_ridge', 'random_forest', 'gradient_boosting'):
            for metric in ('MAE_MW', 'RMSE_MW', 'MASE'):
                self.assertAlmostEqual(cv['models'][name]['test'][metric], spin['models'][name]['test'][metric], places=6)

    def test_only_two_scientific_notebooks_and_no_install_upload_cells(self):
        self.assertEqual({p.name for p in ROOT.glob('*.ipynb')}, set(NOTEBOOKS))
        for name in NOTEBOOKS:
            nb = nbformat.read(ROOT/name, as_version=4); nbformat.validate(nb)
            source = '\n'.join(c.source for c in nb.cells)
            self.assertNotIn('!pip', source); self.assertNotIn('files.upload', source)
            self.assertLess(sum(len(c.source.split()) for c in nb.cells if c.cell_type == 'markdown'), 180)
            for c in nb.cells:
                self.assertFalse(any(o.output_type == 'error' for o in c.get('outputs', [])))

    def test_physics_resource_evidence(self):
        p = json.loads((ROOT/'results/physics.json').read_text())
        self.assertEqual(len(p['cv_validation']), 6); self.assertEqual(len(p['spin_validation']), 3)
        for row in p['cv_validation']:
            self.assertLess(row['max_feature_error_cutoff10_vs14'], 1e-6)
            self.assertLess(row['max_cutoff_boundary_probability'], 1e-7)
        for row in p['spin_validation']:
            self.assertLess(row['exact_qiskit_state_error_l2'], 1e-10)
            self.assertLess(row['strang_product_formula'][-1]['state_error_l2'], row['strang_product_formula'][0]['state_error_l2'])
