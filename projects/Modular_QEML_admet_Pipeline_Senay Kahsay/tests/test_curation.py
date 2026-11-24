"""Chemical encoding, statistical references, and full offline curation checks."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from admet_curation import (
    CurationConfig, FIXTURE_NAME, _estimate, discover_triads, estimate_pairwise,
    fingerprint_datasets, molecule_problem_input, public_smiles_fixture,
    run_complete_curation, select_features, smiles_to_morgan_count_fp,
)


class CurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = CurationConfig(top_k=4)
        cls.curation = run_complete_curation(config=cls.config)

    def test_count_encoding_preserves_multiplicity_and_atom_order(self):
        ethane, _ = smiles_to_morgan_count_fp("CC", self.config)
        self.assertEqual(int(ethane.sum()), 3)
        self.assertEqual(int(ethane.max()), 2)
        benzene, _ = smiles_to_morgan_count_fp("c1ccccc1", self.config)
        self.assertEqual(int(benzene.max()), 6)
        left, canonical_left = smiles_to_morgan_count_fp("CCO", self.config)
        right, canonical_right = smiles_to_morgan_count_fp("OCC", self.config)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(canonical_left, canonical_right)
        for invalid in (None, float("nan"), 42, "", "  ", "not-a-smiles"):
            self.assertIsNone(smiles_to_morgan_count_fp(invalid, self.config))

    def test_invalid_smiles_keep_label_and_source_row_alignment(self):
        frame = pd.DataFrame({"Drug": ["CC", None, "CO", "", "CN", "CCC"],
                              "Y": [0, 1, 1, 0, 0, 1]}, index=list("abcdef"))
        actual = fingerprint_datasets({"alignment": frame}, self.config)["alignment"]
        self.assertEqual(actual["source_row_positions"], [0, 2, 4, 5])
        self.assertEqual(actual["source_indices"], ["a", "c", "e", "f"])
        np.testing.assert_array_equal(actual["y"], [0, 1, 0, 1])
        self.assertEqual([r["source_row_position"] for r in actual["rejected_rows"]], [1, 3])

    def test_label_schema_sample_and_memory_guards(self):
        frame = public_smiles_fixture()[FIXTURE_NAME].iloc[:8].copy()
        for invalid in (float("nan"), float("inf"), -1, 2):
            bad = frame.astype({"Y": float})
            bad.loc[0, "Y"] = invalid
            with self.assertRaisesRegex(ValueError, "binary"):
                fingerprint_datasets({"bad": bad}, self.config)
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            fingerprint_datasets({"one_class": frame.assign(Y=0)}, self.config)
        with self.assertRaisesRegex(ValueError, "Drug and Y"):
            fingerprint_datasets({"missing": frame.drop(columns="Y")}, self.config)
        with self.assertRaisesRegex(ValueError, "more valid molecules"):
            fingerprint_datasets({"small": frame.iloc[:3]}, self.config)
        with self.assertRaisesRegex(ValueError, "sample budget"):
            fingerprint_datasets({"large": frame}, replace(self.config, max_samples=4))
        with self.assertRaisesRegex(ValueError, "fingerprint-cell budget"):
            fingerprint_datasets({"large": frame}, replace(self.config, max_fingerprint_cells=100))

    def test_discrete_mi_reference_and_stable_feature_ties(self):
        # Two equal informative count columns: exact I(X;Y)=ln(2) nats.
        y = np.tile([0, 1], 8)
        X = np.c_[2*y, y, np.zeros(len(y)), np.ones(len(y))]
        selected = select_features({"reference": {"X": X, "y": y}}, self.config)["reference"]
        np.testing.assert_array_equal(selected["selected_features"], [0, 1, 2, 3])
        np.testing.assert_allclose(selected["mi_scores"][:2], np.log(2), atol=1e-14)
        np.testing.assert_array_equal(selected["X_selected_scaled"][:, 0], y)
        np.testing.assert_array_equal(selected["X_selected_scaled"][:, 2:], 0)
        self.assertEqual(selected["mi_units"], "nats")
        self.assertEqual(selected["fit_scope"], "exploratory_full_table")

    def test_constant_guard_and_gaussian_ksg_reference(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(2500, 1))
        y = 0.65*x + np.sqrt(1-0.65**2)*rng.normal(size=x.shape)
        estimate = _estimate(self.config, ("gaussian_reference",), x, y)
        expected_bits = -0.5*np.log2(1-0.65**2)
        self.assertAlmostEqual(estimate, expected_bits, delta=0.08)
        with patch("admet_curation.mi", side_effect=AssertionError("must not call KSG")):
            self.assertEqual(_estimate(self.config, ("constant",), x, np.zeros_like(x)), 0)

    def test_signed_co_information_against_discrete_reference(self):
        binary = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (512, 1))
        for kind, expected in (("redundant", 1.0), ("xor", -1.0)):
            X = (np.repeat(binary[:, :1], 3, axis=1) if kind == "redundant"
                 else np.c_[binary, np.logical_xor(binary[:, 0], binary[:, 1])])
            data = {"toy": {"X_selected_scaled": X, "selected_features": np.arange(3)}}
            pairwise = estimate_pairwise(data, self.config)
            data["toy"].update(retained_pairs=[{"feature_i": 0, "feature_j": 1}],
                               active_feature_positions=[0, 1, 2])
            triads = discover_triads(data, pairwise, self.config)["toy"]
            score = triads["triad_diagnostics"][0]["triadic_interaction"]
            self.assertAlmostEqual(score, expected, delta=0.15)
            self.assertEqual(len(triads["retained_triads"]), int(expected > 0))

    def test_full_fixture_determinism_and_global_rng_isolation(self):
        np.random.seed(867)
        expected = np.random.random(5)
        np.random.seed(867)
        repeated = run_complete_curation(config=self.config)
        actual = np.random.random(5)
        np.testing.assert_array_equal(actual, expected)
        for group, matrix in (("pairwise_mi_data", "raw_mi_matrix"),
                              ("conditional_pair_data", "conditional_mi_matrix")):
            np.testing.assert_array_equal(self.curation[group][FIXTURE_NAME][matrix], repeated[group][FIXTURE_NAME][matrix])
        self.assertEqual(self.curation["triadic_interaction_data"][FIXTURE_NAME]["triad_diagnostics"],
                         repeated["triadic_interaction_data"][FIXTURE_NAME]["triad_diagnostics"])
        self.assertTrue(repeated["fixture_labels_are_synthetic"])

    def test_runtime_adapter_uses_molecule_counts_not_mi_scores(self):
        adapter, theta, provenance = molecule_problem_input(self.curation, FIXTURE_NAME, 13)
        data = self.curation["selected_feature_data"][FIXTURE_NAME]
        active = self.curation["interaction_hypergraph"][FIXTURE_NAME]["active_features"]
        np.testing.assert_array_equal(theta, data["X_selected_scaled"][13, active])
        self.assertFalse(np.allclose(theta, data["selected_mi_scores"][active]))
        self.assertEqual(provenance["source_row_position"], 13)
        self.assertEqual(provenance["active_features"], 4)
        terms = adapter["hamiltonian_A_interaction_ising"][FIXTURE_NAME]["pauli_terms"]
        self.assertEqual(len(terms), 10)
        self.assertEqual({t["locality"] for t in terms}, {2, 3})
        self.assertEqual(len({t["pauli_string"] for t in terms}), len(terms))

    def test_runtime_adapter_rejects_invalid_indices_mapping_and_values(self):
        for index in (-1, 96, 0.5, True):
            with self.assertRaisesRegex(ValueError, "sample_index"):
                molecule_problem_input(self.curation, FIXTURE_NAME, index)
        bad = deepcopy(self.curation)
        bad["interaction_hypergraph"][FIXTURE_NAME]["original_feature_map"][0] = -1
        with self.assertRaisesRegex(ValueError, "mapping"):
            molecule_problem_input(bad, FIXTURE_NAME, 13)
        bad = deepcopy(self.curation)
        bad["selected_feature_data"][FIXTURE_NAME]["X_selected_scaled"][13, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            molecule_problem_input(bad, FIXTURE_NAME, 13)
        bad = deepcopy(self.curation)
        bad["interaction_hypergraph"][FIXTURE_NAME]["active_features"] = []
        with self.assertRaisesRegex(ValueError, "No retained interaction"):
            molecule_problem_input(bad, FIXTURE_NAME, 13)

    def test_budget_preflight_never_truncates_interactions(self):
        with patch("admet_curation._estimate", side_effect=AssertionError("budget must fail first")):
            with self.assertRaisesRegex(ValueError, "pair/estimator budget"):
                estimate_pairwise(self.curation["selected_feature_data"], replace(self.config, max_pairs=2))
            with self.assertRaisesRegex(ValueError, "candidate-triad budget"):
                discover_triads(self.curation["conditional_pair_data"], self.curation["pairwise_mi_data"],
                                replace(self.config, max_candidate_triads=1))
            with self.assertRaisesRegex(ValueError, "estimator-call budget"):
                discover_triads(self.curation["conditional_pair_data"], self.curation["pairwise_mi_data"],
                                replace(self.config, max_estimator_calls=13))

    def test_default_and_provided_modes_do_not_load_tdc(self):
        with patch("admet_curation.load_tdc_datasets", side_effect=AssertionError("offline mode downloaded data")):
            provided = run_complete_curation(mode="provided", datasets=public_smiles_fixture(), config=self.config)
        self.assertEqual(provided["mode"], "provided")
        with self.assertRaisesRegex(ValueError, "Injected datasets"):
            run_complete_curation(datasets=public_smiles_fixture())
        with self.assertRaisesRegex(ValueError, "requires datasets"):
            run_complete_curation(mode="provided")

    def test_config_rejects_ambiguous_or_nonfinite_parameters(self):
        for kwargs in ({"seed": True}, {"top_k": 0}, {"top_k": 1025}, {"n_neighbors": 2.5},
                       {"information_base": 1}, {"triad_threshold": float("nan")},
                       {"max_pairs": 0}, {"seed": 2**32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CurationConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
