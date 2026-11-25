import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from data_contract import ROOT, GRID, SEEDS, WINDOW, load_data, prepare, settlement_start, split_description
from benchmark import metrics, select_model, model_factory, RIDGE_GRID, paired_day_bootstrap


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.frame = load_data(); cls.data = prepare(cls.frame)

    def test_complete_source_checksum_schema_and_utc_chronology(self):
        meta = json.loads((ROOT/'data/provenance.json').read_text())
        self.assertEqual(len(self.frame), meta['rows']); self.assertEqual(len(self.frame), 14448)
        self.assertTrue(self.frame.index.is_unique)
        self.assertTrue(np.all(np.diff(self.frame.index.asi8) == 1800*10**9))
        self.assertEqual(self.frame.groupby('SETTLEMENT_DATE').size().value_counts().to_dict(), {48: 299, 46: 1, 50: 1})

    def test_spring_and_autumn_settlement_conversion(self):
        spring = [settlement_start('2025-03-30', p) for p in range(1, 47)]
        autumn = [settlement_start('2025-10-26', p) for p in range(1, 51)]
        self.assertEqual(len(set(spring+autumn)), 96)
        self.assertEqual((settlement_start('2025-03-31', 1)-spring[-1]).total_seconds(), 1800)
        self.assertEqual((settlement_start('2025-10-27', 1)-autumn[-1]).total_seconds(), 1800)
        for date, period in [('2025-03-30', 47), ('2025-10-26', 51), ('2025-01-01', True)]:
            with self.assertRaises(ValueError): settlement_start(date, period)

    def test_feature_target_alignment_and_only_available_lags(self):
        d = self.data
        np.testing.assert_array_equal(d.y, self.frame[GRID[0]].to_numpy()[336:])
        np.testing.assert_array_equal(d.persistence, self.frame[GRID[0]].to_numpy()[335:-1])
        np.testing.assert_array_equal(d.daily, self.frame[GRID[0]].to_numpy()[288:-48])
        np.testing.assert_array_equal(d.weekly, self.frame[GRID[0]].to_numpy()[:-336])
        self.assertTrue(np.all(d.source_times < d.times))
        np.testing.assert_allclose(d.scaler.inverse_transform(d.windows[0]), self.frame[list(GRID)].to_numpy()[328:336])

    def test_future_perturbation_cannot_change_training_encoder_or_features(self):
        frame = self.frame.copy(); cutoff = 336+self.data.initial_train
        frame.iloc[cutoff:, frame.columns.get_indexer(GRID)] *= 10
        changed = prepare(frame)
        np.testing.assert_array_equal(changed.scaler.mean_, self.data.scaler.mean_)
        np.testing.assert_array_equal(changed.windows[:self.data.initial_train], self.data.windows[:self.data.initial_train])
        np.testing.assert_array_equal(changed.raw[:self.data.initial_train], self.data.raw[:self.data.initial_train])

    def test_target_t_cannot_enter_its_own_features(self):
        frame = self.frame.copy(); t = 336+23
        frame.iloc[t, frame.columns.get_indexer(GRID)] += 10000
        changed = prepare(frame)
        np.testing.assert_array_equal(changed.raw[23], self.data.raw[23])
        # The encoder mean would legitimately change if perturbing its training
        # sample; compare unscaled windows using each fitted scaler instead.
        np.testing.assert_allclose(changed.scaler.inverse_transform(changed.windows[23]),
                                   self.data.scaler.inverse_transform(self.data.windows[23]))

    def test_missing_nonfinite_duplicate_and_unsorted_inputs_fail(self):
        import tempfile
        for frame in (self.frame.reset_index(drop=True).iloc[::-1],
                      pd.concat([self.frame.iloc[:1000], self.frame.iloc[:1]]).reset_index(drop=True)):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)/'bad.csv'; frame.to_csv(p, index=False)
                with self.assertRaises(ValueError): load_data(p, verify=False)
        bad = self.frame.copy(); bad.iloc[0, bad.columns.get_loc(GRID[0])] = np.nan
        with self.assertRaises(ValueError): prepare(bad)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_data().iloc[:1600]; cls.data = prepare(df)

    def test_folds_chronological_disjoint_test_and_cover_validation(self):
        d = self.data; indices = []
        for tr, va in d.folds():
            self.assertLess(tr[-1], va[0]); self.assertLess(va[-1], d.test_start)
            indices.extend(va)
        self.assertEqual(indices, list(range(d.initial_train, d.test_start)))

    def test_test_labels_cannot_change_selected_params_or_validation_scores(self):
        d = self.data; altered = copy.deepcopy(d); altered.y[d.test_start:] += 1e8
        a = select_model('ridge', RIDGE_GRID, d.raw, d)
        b = select_model('ridge', RIDGE_GRID, d.raw, altered)
        self.assertEqual(a, b)

    def test_all_fits_exclude_heldout_labels_and_scaler_is_train_only(self):
        d = self.data; sizes = []
        factory = model_factory
        def tracked(kind, params):
            model = factory(kind, params); original = model.fit
            def fit(x, y):
                sizes.append(len(x)); self.assertLessEqual(len(x), d.test_start)
                fitted = original(x, y)
                np.testing.assert_allclose(fitted.steps[0][1].mean_, np.mean(x, axis=0))
                return fitted
            model.fit = fit
            return model
        with patch('benchmark.model_factory', side_effect=tracked): select_model('ridge', RIDGE_GRID, d.raw, d)
        self.assertEqual(len(sizes), 9)

    def test_metrics_known_numbers_and_bootstrap_paired_identity(self):
        out = metrics(np.array([1., 3.]), np.array([2., 1.]), 2.)
        self.assertEqual(out['MAE_MW'], 1.5); self.assertEqual(out['MASE'], .75)
        self.assertAlmostEqual(out['RMSE_MW'], np.sqrt(2.5))
        y = np.arange(480.); p = y+2
        result = paired_day_bootstrap(y, p, p)
        self.assertEqual(result['conditional_95pct_interval_MW'], [0., 0.])

    def test_fixed_config_matches_experiment_contract(self):
        protocol = json.loads((ROOT/'protocol.json').read_text())
        self.assertEqual(protocol['seeds'], list(SEEDS)); self.assertEqual(protocol['window'], WINDOW)
        self.assertEqual(protocol['ridge_alpha_grid'], [r['alpha'] for r in RIDGE_GRID])
        self.assertFalse(model_factory('boost', {'max_leaf_nodes': 15}).early_stopping)

    def test_split_report_serializes_strict_json(self):
        report = split_description(self.data)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
