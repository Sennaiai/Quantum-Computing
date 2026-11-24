"""Published figures/results must identify the tested code and pass acceptance."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes")]
from admet_submission_entry import source_hashes
from data_benchmark import BBB_SHA256, BBB_BYTES
from runtime_input_protocol import canonical_json


class PublishedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / "results"
        cls.report = json.loads((cls.directory / "benchmark_summary.json").read_text())
        cls.manifest = json.loads((cls.directory / "manifest.json").read_text())

    def test_published_checksums_and_portable_svg_figures(self):
        expected = {"benchmark_summary.json", "transport.svg", "information.svg"}
        self.assertEqual(set(self.manifest["files"]), expected)
        for name, digest in self.manifest["files"].items():
            path = self.directory / name
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            if path.suffix == ".svg":
                self.assertEqual(ET.parse(path).getroot().tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIs(self.manifest["raw_dataset_committed"], False)
        self.assertEqual(self.manifest["triple_alpha_calls"], 0)

    def test_published_evidence_matches_current_implementation(self):
        self.assertEqual(self.report["source_hashes"], source_hashes(),
                         "Published evidence is stale: regenerate run_benchmarks.py and publish its four result files")
        digest = hashlib.sha256(canonical_json(self.report["source_hashes"]).encode()).hexdigest()
        self.assertEqual(digest, self.report["source_tree_sha256"])
        self.assertEqual(digest, self.manifest["source_tree_sha256"])
        self.assertIn("run_benchmarks.py", self.report["source_hashes"])
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["triple_alpha_calls"], 0)
        self.assertEqual(self.report["native_compilation"], "unverified")

    def test_real_data_provenance_quality_and_all_runtime_handoffs(self):
        data = self.report["real_data"]
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["source"]["sha256"], BBB_SHA256)
        self.assertEqual(data["source"]["bytes"], BBB_BYTES)
        self.assertEqual(data["raw_rows"], 2039)
        self.assertEqual(data["valid_rows"]+len(data["rejected_rows"]), data["raw_rows"])
        self.assertEqual(data["valid_rows"]-data["unique_canonical_molecules"], data["duplicate_observations"])
        self.assertEqual(data["fit_scope"], "exploratory_full_table")
        self.assertEqual([p["top_k"] for p in data["profiles"]], [4, 8])
        for profile in data["profiles"]:
            pipeline = profile["pipeline"]
            self.assertEqual(pipeline["status"], "passed")
            self.assertFalse(pipeline["live_api_execution_allowed"])
            self.assertGreater(pipeline["cd_terms"], 10)
            self.assertLessEqual(pipeline["reference"]["recovered_evolution_error"], pipeline["error_bound"]+1e-11)
            self.assertLessEqual(pipeline["error_bound"], 1e-6)
            self.assertEqual(len(profile["seed_runs"]), 5)
            self.assertTrue(all(0 <= j <= 1 for j in profile["pair_jaccard_between_seeds"]))
        for forbidden in ("canonical_smiles", "smiles", "X", "y", "datasets"):
            self.assertNotIn(forbidden, data)

    def test_transport_calibration_cost_and_independent_refinement(self):
        experiment = self.report["transport"]
        self.assertEqual(experiment["status"], "passed")
        self.assertEqual(experiment["num_system_qubits"], 4)
        self.assertEqual(experiment["endpoint_ground_multiplicity"], 1)
        self.assertGreater(experiment["observed_minimum_ground_gap"], 0)
        self.assertEqual(len(experiment["rows"]), 15)
        for row in experiment["rows"]:
            self.assertTrue(0 <= row["ground_fidelity"] <= 1)
            self.assertGreaterEqual(row["excitation_energy"], 0)
            self.assertLessEqual(row["max_norm_error"], experiment["config"]["norm_tolerance"])
            self.assertGreaterEqual(row["integrated_control_hs_squared_quadrature"], 0)
            if row["method"] == "exact_ground_cd":
                self.assertLessEqual(row["ground_infidelity"], experiment["config"]["exact_infidelity_tolerance"])
        for method in ("bare", "first_order_cd", "exact_ground_cd"):
            rows = [r for r in experiment["convergence"] if r["method"] == method]
            self.assertEqual([r["midpoint_steps"] for r in rows], [32, 64, 128])
            self.assertLess(rows[-1]["phase_aligned_state_distance"], rows[0]["phase_aligned_state_distance"])
            self.assertLessEqual(rows[-1]["phase_aligned_state_distance"], experiment["config"]["midpoint_state_tolerance"])
        for row in experiment["qsvt_snapshots"]:
            reference = row["reference"]
            self.assertLessEqual(reference["recovered_evolution_error"], reference["certified_recovered_evolution_error_bound"]+1e-11)
            self.assertLessEqual(reference["certified_recovered_evolution_error_bound"], row["epsilon"])

    def test_supervised_cleaning_all_molecule_physics_and_unfitted_readout(self):
        learning=self.report["supervised_learning"]
        self.assertEqual(learning["data_source_sha256"],BBB_SHA256)
        self.assertEqual(learning["source_hashes"],source_hashes())
        self.assertEqual(learning["raw_rows"],2039)
        self.assertEqual(learning["cleaned_labelled_identities"],1965)
        self.assertEqual(learning["all_valid_inference_identities"],1975)
        self.assertEqual(learning["prepared_molecules"],1975)
        self.assertEqual(learning["concordant_duplicate_rows_collapsed"],54)
        self.assertEqual(learning["quarantined_label_identities"],10)
        self.assertEqual(learning["quarantined_labelled_rows"],20)
        self.assertTrue(learning["scaffold_groups_disjoint"])
        self.assertEqual(sum(learning["split_sizes"].values()),1965)
        self.assertEqual(learning["unique_physics"]+learning["physics_reuse_count"],1975)
        self.assertEqual(learning["source_named_input_roundtrips"],learning["unique_physics"])
        self.assertLessEqual(learning["maximum_recovered_operator_error"],learning["maximum_recovered_error_bound"]+1e-11)
        self.assertLessEqual(learning["maximum_recovered_error_bound"],1e-6)
        self.assertEqual(learning["status"],"awaiting_api_results")
        self.assertFalse(learning["quantum_model_fitted"])
        self.assertIsNone(learning["quantum_prediction_metrics"])
        self.assertIsNone(learning["readout_spec"]["fitted_coefficients"])
        for name,result in learning["classical_test_results"].items():
            self.assertIn(name,{"theta","full_counts"})
            self.assertEqual(result["evaluation_split"],"untouched_test")
            self.assertEqual(result["metrics"]["samples"],learning["split_sizes"]["test"])
            self.assertTrue(0<=result["metrics"]["roc_auc"]<=1)
        for flag in ("native_compiler_verified","native_input_binding_verified","native_output_mapping_verified"):
            self.assertIs(learning[flag],False)
        self.assertEqual(learning["triple_alpha_calls"],0)
