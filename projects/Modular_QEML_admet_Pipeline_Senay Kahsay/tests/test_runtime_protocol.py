import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_runtime_fixtures import ROOT, payload, settings
from hamiltonian_validation import HamiltonianValidation, parse_pauli_string
import runtime_input_protocol as protocol
from runtime_sync_validation import (QBE_MODULES, module_source_hashes,
                                      verify_main_source_contract, verify_runtime_synchronization)


class RuntimeValidationTests(unittest.TestCase):
    def test_canonical_words_and_lcu_metadata(self):
        p = payload()
        p["pauli_terms"][2]["pauli_string"] = "Z1 Y0"
        v = HamiltonianValidation(p)
        self.assertTrue(v.validate())
        self.assertEqual(v.pauli_strings[2], "Y0 Z1")
        self.assertEqual(v.lambda_value, 1.625)
        self.assertAlmostEqual(sum(v.weights), 1.0)
        self.assertEqual(parse_pauli_string("I0 I1", 2), (0, 0))

    def test_rejects_invalid_pauli_words(self):
        for word in ("X0 Z0", "I0 X0", "X-1", "X+1", "X01", "X2", "A0", "X", 1):
            with self.subTest(word=word), self.assertRaises(ValueError):
                parse_pauli_string(word, 2)

    def test_rejects_invalid_payloads(self):
        changes = [lambda p: p.update(selected_features=[]),
                   lambda p: p.update(selected_features=[1, 1]),
                   lambda p: p.update(pauli_terms=[]),
                   lambda p: p.update(feature_vector=[0.0]),
                   lambda p: p.update(feature_vector=[-0.1, 1.0]),
                   lambda p: p.update(feature_vector=[0.0, 1.1]),
                   lambda p: p.update(feature_vector=[True, 1.0]),
                   lambda p: p.update(feature_vector=[float("nan"), 1.0]),
                   lambda p: p["pauli_terms"][0].update(coefficient=float("inf")),
                   lambda p: p["pauli_terms"][0].update(coefficient=True),
                   lambda p: p["pauli_terms"][0].update(coefficient=1j),
                   lambda p: p["pauli_terms"][0].update(coefficient="1.0"),
                   lambda p: p["pauli_terms"][1].update(pauli_string="I0 I1")]
        for change in changes:
            p = payload()
            change(p)
            with self.subTest(change=change), self.assertRaises(ValueError):
                HamiltonianValidation(p)
        p = payload()
        for term in p["pauli_terms"]:
            term["coefficient"] = 0.0
        with self.assertRaises(ValueError):
            HamiltonianValidation(p)

    def test_runtime_requires_features_and_integer_sample_index(self):
        p = payload()
        del p["feature_vector"]
        HamiltonianValidation(p)
        with self.assertRaises(ValueError):
            protocol.build_runtime_input_document(p, settings())
        for sample in (True, -1, 0.5):
            p = payload()
            p["sample_index"] = sample
            with self.assertRaises(ValueError):
                protocol.build_runtime_input_document(p, settings())

    def test_rejects_stale_validator_and_invalid_branch_metadata(self):
        p, s = payload(), settings()
        validator = HamiltonianValidation(p)
        p["pauli_terms"][0]["coefficient"] = -0.5
        with self.assertRaises(ValueError):
            protocol.build_runtime_input_document(p, s, validator)
        for branch, key, value in (("cosine", "phase_angles", [float("nan"), 0, 0]),
                                   ("sine", "polynomial_degree", 2),
                                   ("cosine", "scale", 0),
                                   ("cosine", "synthesis_error_bound", -1)):
            s = settings()
            s[branch][key] = value
            with self.assertRaises(ValueError):
                protocol.build_runtime_input_document(payload(), s)
        for key, value in (("lambda_value", 2), ("tau", 1), ("success_amplitude", 0.5),
                           ("branch_rotation_angle", 0.5), ("epsilon", float("inf")),
                           ("physical_time", 0), ("algorithm", "cosine_only"),
                           ("recovered_evolution_error_bound", .01),
                           ("ideal_success_probability", .9),
                           ("success_probability_lower_bound", .9)):
            s = settings()
            s[key] = value
            with self.assertRaises(ValueError):
                protocol.build_runtime_input_document(payload(), s)


class RuntimeProtocolTests(unittest.TestCase):
    def test_hand_calculated_golden_stream_and_independent_decoder(self):
        s = settings()
        expected = [4., 2., 2., 3., 4., 0., 1.,
                    -.25, 0., 0., .5, 0., 1., .75, 2., 3., -.125, 3., 1.,
                    .1, .2, .3, .4, .5, .6, .7, s["branch_rotation_angle"]]
        document = protocol.build_runtime_input_document(payload(), s)
        self.assertEqual(document["inputs"], expected)
        self.assertEqual(document["input_count"], 27)
        self.assertEqual(protocol.expected_input_count(4, 2, 3, 4), 27)
        decoded = protocol.validate_runtime_input_document(json.loads(protocol.canonical_json(document)))
        self.assertEqual(decoded["coefficients"], [-.25, .5, .75, -.125])
        self.assertEqual(decoded["pauli_rows"], [[0, 0], [0, 1], [2, 3], [3, 1]])
        self.assertEqual(decoded["feature_vector"], [0., 1.])
        self.assertEqual(decoded["cosine_phases"], [.1, .2, .3])
        self.assertEqual(decoded["sine_phases"], [.4, .5, .6, .7])
        self.assertEqual(document["num_total_qubits"], 6)

    def test_decoder_rejects_corrupted_dimensions_values_and_length(self):
        original = protocol.build_runtime_input_values(payload(), settings())
        mutations = [(0, True), (0, 4.5), (1, 0), (2, 1000000), (3, 2),
                     (5, -0.1), (8, 4), (8, 0.5), (19, float("nan")),
                     (-1, 4)]
        for index, value in mutations:
            stream = original.copy()
            stream[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                protocol.decode_runtime_input_values(stream)
        for stream in (original[:-1], original + [0.0], [], [1e300] * 5):
            with self.assertRaises(ValueError):
                protocol.decode_runtime_input_values(stream)
        duplicate = original.copy()
        duplicate[11:13] = [0., 0.]
        with self.assertRaises(ValueError):
            protocol.decode_runtime_input_values(duplicate)

    def test_document_checksum_and_metadata_mutations(self):
        original = protocol.build_runtime_input_document(payload(), settings())
        for key, value in (("protocol_version", 1), ("inputs_sha256", "0" * 64),
                           ("input_count", 26), ("num_total_qubits", 7), ("layout", [])):
            doc = copy.deepcopy(original)
            doc[key] = value
            with self.assertRaises(ValueError):
                protocol.validate_runtime_input_document(doc)
        doc = copy.deepcopy(original)
        doc["inputs"][5] = .5
        with self.assertRaises(ValueError):
            protocol.validate_runtime_input_document(doc)

    def test_strict_atomic_persistence_and_failed_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "inputs.json"
            doc = protocol.write_runtime_input_payload(target, payload(), settings())
            self.assertEqual(json.loads(target.read_text()), doc)
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                protocol.atomic_write_json(target, {"bad": float("nan")})
            self.assertEqual(target.read_bytes(), before)
            with patch("runtime_input_protocol.os.replace", side_effect=OSError("simulated")):
                with self.assertRaises(OSError):
                    protocol.atomic_write_json(target, {"changed": True})
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(target.parent.glob("*.tmp")), [])
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_source_hashes_and_honest_local_validation_scope(self):
        report = verify_runtime_synchronization(payload(), ROOT / "qbe_modules", settings())
        self.assertEqual(report["status"], "local_contract_verified")
        self.assertFalse(report["compiler_verified"])
        self.assertFalse(report["deployment_verified"])
        self.assertFalse(report["service_contract_verified"])
        self.assertEqual(set(report["module_hashes"]), set(QBE_MODULES))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in QBE_MODULES:
                shutil.copyfile(ROOT / "qbe_modules" / name, source / name)
            (source / "math.qbe").write_text("# altered source\n")
            with self.assertRaises(ValueError):
                module_source_hashes(source, report["module_hashes"])

    def test_contract_ignores_comments_and_detects_reordered_reads(self):
        main = (ROOT / "qbe_modules" / "main.qbe").read_text()
        self.assertTrue(verify_main_source_contract(main + "\n# input -> fake\n"))
        changed = main.replace("input -> num num_cos_phases", "# input -> num num_cos_phases")
        with self.assertRaises(ValueError):
            verify_main_source_contract(changed)
        changed = main.replace("num num_coefs", "num PLACEHOLDER", 1).replace("num num_qubits", "num num_coefs", 1).replace("num PLACEHOLDER", "num num_qubits", 1)
        with self.assertRaises(ValueError):
            verify_main_source_contract(changed)

    def test_independent_decoder_detects_semantically_wrong_packer(self):
        values = protocol.build_runtime_input_values(payload(), settings())
        values[5:7] = [1., 0.]
        with patch("runtime_input_protocol.build_runtime_input_values", return_value=values):
            with self.assertRaisesRegex(ValueError, "feature_vector"):
                verify_runtime_synchronization(payload(), ROOT / "qbe_modules", settings())


if __name__ == "__main__":
    unittest.main()
