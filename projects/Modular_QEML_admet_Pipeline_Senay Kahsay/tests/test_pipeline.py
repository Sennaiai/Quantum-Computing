"""Offline integration and artifact provenance tests using real chemical stages."""
from pathlib import Path
import json
import errno
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes")]
import admet_submission_entry as entry
from admet_curation import CurationConfig, fingerprint_datasets, public_smiles_fixture
from runtime_input_protocol import decode_runtime_input_values


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("admet_curation.load_tdc_datasets", side_effect=AssertionError("Unexpected data download")):
            cls.payload = entry.build_molecular_payload()

    def test_full_molecule_path_retains_all_interactions_and_provenance(self):
        p = self.payload
        self.assertEqual(p["sample_index"], 13)
        self.assertTrue(p["fixture_labels_are_synthetic"])
        self.assertGreater(len(p["pauli_terms"]), 10)
        self.assertEqual(p["agp_metadata"]["interaction_truncation"]["dropped_count"], 0)
        provenance = p["curation_provenance"]
        self.assertEqual(provenance["fit_scope"], "exploratory_full_table")
        self.assertEqual(len(provenance["source_table_sha256"]), 64)
        self.assertEqual(len(provenance["scaling"]["active_feature_min"]), len(p["selected_features"]))
        self.assertTrue(any(p["feature_vector"]))

    def test_bundle_roundtrip_reproducibility_and_source_immutability(self):
        before = entry.source_hashes()
        with tempfile.TemporaryDirectory() as tmp:
            bundle, settings = entry.prepare_runtime_inputs(self.payload, Path(tmp))
            self.assertEqual(entry.verify_runtime_bundle(bundle)["source_hashes"], before)
            self.assertFalse(entry.verify_runtime_bundle(bundle)["live_api_execution_allowed"])
            document = settings["synchronization"]["runtime_document"]
            decoded = decode_runtime_input_values(document["inputs"])
            self.assertEqual(decoded["num_terms"], len(self.payload["pauli_terms"]))
            self.assertEqual(len(decoded["sine_phases"]), len(settings["sine"]["phase_angles"]))
            measured = settings["mathematical_reference"]["recovered_evolution_error"]
            self.assertLessEqual(measured, settings["recovered_evolution_error_bound"]+1e-11)
            again, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp))
            self.assertEqual(bundle, again)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
        self.assertEqual(entry.source_hashes(), before)

    def test_artifact_and_source_mutations_fail_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp))
            with patch.object(entry, "source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "sources differ"):
                    entry.verify_runtime_bundle(bundle)
            path = bundle / "runtime_inputs.json"
            value = json.loads(path.read_text())
            value["inputs"][5] += .1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "checksum"):
                entry.verify_runtime_bundle(bundle)

    def test_failed_package_write_does_not_publish_partial_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(entry, "atomic_write_json", side_effect=OSError("disk fixture failure")):
                with self.assertRaises(OSError):
                    entry.prepare_runtime_inputs(self.payload, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_output_cannot_overwrite_circuit_sources(self):
        with self.assertRaisesRegex(ValueError, "outside source"):
            entry.prepare_runtime_inputs(self.payload, entry.QBE_SOURCE_DIR)

    def test_explicit_reference_skip_is_recorded_and_never_enables_live_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, settings = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            self.assertEqual(settings["mathematical_reference"]["status"], "skipped_explicitly")
            self.assertFalse(entry.verify_runtime_bundle(bundle)["live_api_execution_allowed"])

    def test_source_table_hash_tracks_input_labels(self):
        original = public_smiles_fixture()
        changed = {name: frame.copy() for name, frame in original.items()}
        name = next(iter(changed))
        changed[name].loc[0, "Y"] = 1
        a = fingerprint_datasets(original, CurationConfig(top_k=4))[name]
        b = fingerprint_datasets(changed, CurationConfig(top_k=4))[name]
        self.assertNotEqual(a["source_table_sha256"], b["source_table_sha256"])
        self.assertEqual(a["fingerprint_matrix_sha256"], b["fingerprint_matrix_sha256"])


    @staticmethod
    def _rewrite_manifest(bundle, mutate, *, readdress=False):
        path = bundle / "manifest.json"
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(entry.canonical_json(value) + "\n")
        if readdress:
            destination = bundle.parent / entry.sha256_bytes(entry.canonical_json(value).encode())
            bundle.rename(destination)
            return destination
        return bundle

    def test_manifest_content_address_and_native_policy_cannot_be_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            self._rewrite_manifest(bundle, lambda m: m["environment"].update(python="modified"))
            with self.assertRaisesRegex(ValueError, "content address"):
                entry.verify_runtime_bundle(bundle)
        for key in entry._POLICY_FLAGS:
            with tempfile.TemporaryDirectory() as tmp:
                bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
                bundle = self._rewrite_manifest(bundle, lambda m: m.update({key: True}), readdress=True)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Offline bundle"):
                    entry.verify_runtime_bundle(bundle)

    def test_manifest_schema_module_and_reference_coherence(self):
        mutations = [lambda m: m.update(schema_version=2.0),
                     lambda m: m.update(local_contract_status="unchecked"),
                     lambda m: m.update(environment=[]),
                     lambda m: m.update(environment={"python": "3.11"}),
                     lambda m: m.update(source_hashes={}),
                     lambda m: m["module_hashes"].update({"main.qbe": "0" * 64}),
                     lambda m: m.update(mathematical_reference={"status": "fabricated"})]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
                bundle = self._rewrite_manifest(bundle, mutate, readdress=True)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    entry.verify_runtime_bundle(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            (bundle / "manifest.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "manifest fields"):
                entry.verify_runtime_bundle(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            path = bundle / "manifest.json"
            path.write_text(path.read_text().replace('"schema_version":2', '"schema_version":2,"schema_version":2'))
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                entry.verify_runtime_bundle(bundle)

    def test_historical_mode_always_checks_settings_and_stream_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            with patch.object(entry, "source_hashes", side_effect=AssertionError("No current-source access")), \
                 patch.object(entry, "QBE_SOURCE_DIR", Path(tmp) / "unavailable-sources"):
                entry.verify_runtime_bundle(bundle, require_current_sources=False)
            settings_path = bundle / "qsvt_settings.json"
            value = json.loads(settings_path.read_text())
            value["cosine"]["phase_angles"][0] += .1
            settings_path.write_text(entry.canonical_json(value) + "\n")
            digest = entry.sha256_bytes(settings_path.read_bytes())
            bundle = self._rewrite_manifest(bundle,
                lambda m: m["file_hashes"].update({"qsvt_settings.json": digest}), readdress=True)
            with self.assertRaisesRegex(ValueError, "Hamiltonian and phases"):
                entry.verify_runtime_bundle(bundle, require_current_sources=False)

    def test_pending_bundle_requires_explicit_expected_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            package_id = bundle.name
            pending = bundle.parent / ".pending-fixture"
            bundle.rename(pending)
            with self.assertRaisesRegex(ValueError, "content address"):
                entry.verify_runtime_bundle(pending)
            entry.verify_runtime_bundle(pending, expected_package_id=package_id)

    def test_identical_publication_race_reuses_verified_winner(self):
        original_rename = Path.rename
        raced = []
        def concurrent_publish(path, target):
            if path.name.startswith(".pending-") and not raced:
                shutil.copytree(path, target)
                raced.append(target)
                raise OSError(errno.EEXIST, "Synthetic concurrent publisher")
            return original_rename(path, target)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Path, "rename", new=concurrent_publish):
                bundle, _ = entry.prepare_runtime_inputs(self.payload, Path(tmp), dense_reference=False)
            self.assertEqual(raced, [bundle])
            entry.verify_runtime_bundle(bundle)
            self.assertEqual([path.resolve() for path in Path(tmp).iterdir()], [bundle])

    def test_reference_argument_types_and_protected_outputs_fail_before_synthesis(self):
        with patch.object(entry, "prepare_qsvt_settings", side_effect=AssertionError("No synthesis")):
            for kwargs in ({"dense_reference": 0}, {"dense_reference": "false"},
                           {"max_reference_dimension": 0}, {"max_reference_dimension": -1},
                           {"max_reference_dimension": True}, {"max_reference_dimension": float("nan")},
                           {"physical_time": True}):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                        entry.prepare_runtime_inputs(self.payload, Path(tmp), **kwargs)
                    self.assertEqual(list(Path(tmp).iterdir()), [])
            for output in (ROOT, ROOT.parent, ROOT / "tests", ROOT / ".git" / "output"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    entry.prepare_runtime_inputs(self.payload, output)

    def test_provided_tables_and_csv_use_the_same_pipeline_without_downloads(self):
        frame = next(iter(public_smiles_fixture().values())).copy()
        config = CurationConfig(top_k=4)
        with patch("admet_curation.load_tdc_datasets", side_effect=AssertionError("No download")):
            provided = entry.build_molecular_payload(data_mode="provided", datasets={"Provided_ADMET": frame},
                                                     config=config)
        self.assertEqual(provided["data_mode"], "provided")
        self.assertEqual(provided["dataset"], "Provided_ADMET")
        self.assertEqual(provided["feature_vector"], self.payload["feature_vector"])
        self.assertEqual(provided["curation_provenance"]["source_table_sha256"],
                         self.payload["curation_provenance"]["source_table_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "public_fixture.csv"
            frame.to_csv(source, index=False)
            output = Path(tmp) / "bundles"
            with patch("admet_curation.load_tdc_datasets", side_effect=AssertionError("No download")), \
                 contextlib.redirect_stdout(io.StringIO()):
                entry.main(["--data-csv", str(source), "--top-k", "4", "--output-dir", str(output)])
            bundles = list(output.iterdir())
            self.assertEqual(len(bundles), 1)
            entry.verify_runtime_bundle(bundles[0])
            saved = json.loads((bundles[0] / "hamiltonian.json").read_text())
            self.assertEqual(saved["data_mode"], "provided")
            self.assertEqual(saved["dataset"], "Provided_ADMET")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(["--data-mode", "provided"])


if __name__ == "__main__":
    unittest.main()
