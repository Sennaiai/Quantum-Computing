from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes"), str(ROOT / "tests")]
from admet_curation import CurationConfig
from classical_preprocessing import AGPSettings, recalculate_counterdiabatic_hamiltonian
from data_quality import clean_endpoint_table, split_cleaned_dataset
from supervised_encoding import (fit_hamiltonian_schema, transform_molecules,
                                 validate_frozen_schema, _schema_digest)
from test_data_quality import diverse_table


class FrozenEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleaned = clean_endpoint_table(diverse_table(), "fixture")["cleaned"]
        cls.split = split_cleaned_dataset(cls.cleaned)
        cls.schema = fit_hamiltonian_schema(cls.split["train"], "fixture")

    def test_only_train_ids_and_statistics_enter_the_frozen_schema(self):
        self.assertEqual(self.schema["train_molecule_ids"], self.split["manifest"]["train"])
        self.assertEqual(self.schema["fit_scope"], "training_only")
        self.assertEqual(len(self.schema["training_labels_sha256"]),64)
        self.assertFalse(set(self.schema["train_molecule_ids"]) & set(self.split["manifest"]["test"]))
        reordered = fit_hamiltonian_schema(self.split["train"].iloc[::-1], "fixture")
        self.assertEqual(self.schema, reordered)
        self.assertTrue(validate_frozen_schema(json.loads(json.dumps(self.schema))))

    def test_transform_never_accesses_labels_or_refits_statistics(self):
        test = self.split["test"].copy()
        without = test.drop(columns="Y")
        test["Y"] = [object() for _ in range(len(test))]
        with patch("supervised_encoding.run_complete_curation", side_effect=AssertionError("Inference tried to refit")):
            expected = transform_molecules(self.schema, without)
            actual = transform_molecules(self.schema, test)
        self.assertEqual(expected, actual)
        self.assertTrue(all("Y" not in row for row in actual["records"]))
        self.assertTrue(all(len(row["full_counts"]) == 1024 for row in actual["records"]))

    def test_all_selected_local_fields_survive_empty_interaction_screen(self):
        config = CurationConfig(top_k=4, conditional_mi_threshold=1e9, triad_threshold=1e9)
        schema = fit_hamiltonian_schema(self.split["train"], "fixture", config)
        family = schema["adapter"]["hamiltonian_A_interaction_ising"]["fixture"]
        self.assertEqual(family["pauli_terms"], [])
        self.assertEqual(len(family["selected_features"]), 4)
        payload = recalculate_counterdiabatic_hamiltonian(schema["adapter"], "fixture", [0., 0., 0., 0.], AGPSettings())
        self.assertEqual(len(payload["selected_features"]), 4)
        self.assertEqual(len(payload["pauli_terms"]), 4)

    def test_training_ranges_clip_unseen_counts_and_flag_constant_columns(self):
        config = CurationConfig(n_bits=16, radius=0, top_k=16, conditional_mi_threshold=1e9, triad_threshold=1e9)
        schema = fit_hamiltonian_schema(self.split["train"], "fixture", config)
        novel = pd.DataFrame({"Drug": ["C"*40, "[SiH4]", "[SeH2]", "[PH3]", "[Na+]", "I"]})
        output = transform_molecules(schema, novel)
        self.assertTrue(any(r["clipping"]["above_training_range_wires"] for r in output["records"]))
        self.assertTrue(any(r["clipping"]["outside_constant_training_wires"] for r in output["records"]))
        for row in output["records"]:
            self.assertTrue(all(0 <= value <= 1 for value in row["theta"]))
            self.assertTrue(all(row["theta"][wire] == 0 for wire in row["clipping"]["constant_training_wires"]))

    def test_duplicate_sources_combine_but_distinct_molecules_remain_distinct(self):
        table = pd.DataFrame({"Drug": ["CCO", "OCC", "CCN"], "Y": [object(), object(), object()]})
        output = transform_molecules(self.schema, table)
        self.assertEqual(len(output["records"]), 2)
        self.assertEqual(output["diagnostics"]["identity_duplicates_combined"], 1)
        self.assertEqual(sorted(len(r["source_row_positions"]) for r in output["records"]), [1, 2])

    def test_corruption_identity_and_endpoint_mismatch_rejected(self):
        mutated = deepcopy(self.schema);mutated["scaling"]["feature_max"][0] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):validate_frozen_schema(mutated)
        table = self.split["test"].copy();table.loc[0, "molecule_id"] = "0"*64
        with self.assertRaisesRegex(ValueError, "molecule_id"):transform_molecules(self.schema, table)
        table = self.split["test"].copy();table.loc[0, "endpoint"] = "other"
        with self.assertRaisesRegex(ValueError, "endpoint"):transform_molecules(self.schema, table)

    def test_readout_schema_contains_pair_closure_for_connected_triads(self):
        # Add a declared triad to a valid frozen family and verify that missing
        # pair moments cannot be passed off as a complete connected readout.
        schema = deepcopy(self.schema)
        family = schema["adapter"]["hamiltonian_A_interaction_ising"]["fixture"]
        family["pauli_terms"] = [{"coefficient": .2, "pauli_string": "Z0 Z1 Z2", "locality": 3}]
        schema["hamiltonian_pair_supports"] = []
        schema["hamiltonian_triad_supports"] = [[0, 1, 2]]
        schema["readout_supports"] = {"singles": [[i] for i in range(4)], "pairs": [], "triads": [[0, 1, 2]]}
        schema["schema_id"] = schema["encoding_fit_id"] = _schema_digest(schema)
        with self.assertRaisesRegex(ValueError, "close"):validate_frozen_schema(schema)
        schema["readout_supports"]["pairs"] = [[0, 1], [0, 2], [1, 2]]
        schema["schema_id"] = schema["encoding_fit_id"] = _schema_digest(schema)
        self.assertTrue(validate_frozen_schema(schema))


if __name__ == "__main__":
    unittest.main()
