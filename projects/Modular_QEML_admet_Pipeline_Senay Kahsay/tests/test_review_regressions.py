"""Adversarial public-input regressions discovered in independent review."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes"), str(ROOT / "tests")]
from admet_curation import CurationConfig, fingerprint_datasets
from classical_preprocessing import commutator, scale_terms
from hamiltonian_validation import HamiltonianValidation
from qsvt_helper import QSVTHelper
from runtime_input_protocol import canonical_json, validate_qsvt_settings
from test_runtime_fixtures import payload, settings
from triple_alpha_client import TripleAlphaClient


class ScientificInputReviewTests(unittest.TestCase):
    def test_complex_or_text_labels_cannot_be_silently_cast_into_binary_labels(self):
        for values in ([0+1j, 1+2j, 0+3j, 1+4j],
                       [0+0j, 1+0j, 0+0j, 1+0j],
                       ["0", "1", "0", "1"]):
            frame = pd.DataFrame({"Drug": ["C", "CC", "CO", "CN"], "Y": values})
            with self.subTest(values=values), patch("admet_curation.smiles_to_morgan_count_fp") as fingerprint:
                with self.assertRaisesRegex(ValueError, "real binary"):
                    fingerprint_datasets({"bad_labels": frame}, CurationConfig(top_k=2))
                fingerprint.assert_not_called()

    def test_feature_identifiers_are_stable_json_scalars(self):
        for invalid in (float("nan"), float("inf"), object(), (1, 2), True, -1, 1.0, "", " "):
            p = payload()
            p["selected_features"][0] = invalid
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                HamiltonianValidation(p)
        p = payload()
        p["selected_features"] = [np.int64(4), "feature_8"]
        self.assertEqual(HamiltonianValidation(p).selected_features, [4, "feature_8"])

    def test_scientific_numeric_scalars_have_a_strict_portable_json_form(self):
        self.assertEqual(canonical_json({"feature": np.int64(4), "weight": np.float32(.5)}),
                         '{"feature":4,"weight":0.5}')
        with self.assertRaises(ValueError):
            canonical_json({"weight": np.float32(np.inf)})
        with self.assertRaises(TypeError):
            canonical_json({"unrecognized": object()})

    def test_boolean_class_labels_remain_valid_binary_observations(self):
        frame = pd.DataFrame({"Drug": ["C", "CC", "CO", "CN"], "Y": [False, True, False, True]})
        result = fingerprint_datasets({"bool_labels": frame}, CurationConfig(top_k=2))["bool_labels"]
        np.testing.assert_array_equal(result["y"], [0, 1, 0, 1])

    def test_small_hamiltonian_norm_does_not_disable_norm_and_time_invariants(self):
        p, s = payload(), settings()
        for term in p["pauli_terms"]:
            term["coefficient"] *= 1e-18
        validator = HamiltonianValidation(p)
        s.update(lambda_value=validator.lambda_value, tau=validator.lambda_value*s["physical_time"])
        validate_qsvt_settings(s, validator)
        for key, value in (("lambda_value", 0.), ("lambda_value", -validator.lambda_value),
                           ("lambda_value", 2*validator.lambda_value), ("tau", 0.),
                           ("tau", -s["tau"]), ("tau", 2*s["tau"])):
            bad = deepcopy(s)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_qsvt_settings(bad, validator)
        bad = deepcopy(s)
        bad.update(physical_time=1e-308, tau=1e-320)
        with self.assertRaisesRegex(ValueError, "underflowed"):
            validate_qsvt_settings(bad, validator)

    def test_unrepresentable_heralding_probability_fails_explicitly(self):
        s = settings()
        for name in ("cosine", "sine"):
            s[name]["scale"] = 1e-300
        s.update(success_amplitude=5e-301, ideal_success_probability=0.,
                 success_probability_lower_bound=0., branch_rotation_angle=np.pi/2)
        with self.assertRaisesRegex(ValueError, "underflow"):
            validate_qsvt_settings(s, HamiltonianValidation(payload()))

    def test_phase_helper_rejects_boolean_or_text_physical_parameters(self):
        for time, epsilon in ((True, 1e-6), ("0.1", 1e-6), (.1, True), (.1, "1e-6")):
            with self.subTest(time=time, epsilon=epsilon), self.assertRaises(ValueError):
                QSVTHelper(SimpleNamespace(lambda_value=1.), physical_time=time, epsilon=epsilon)

    def test_noncommuting_products_and_scaling_cannot_silently_underflow(self):
        left = [{"coefficient": 1e-200, "pauli_string": "X0"}]
        right = [{"coefficient": 1e-200, "pauli_string": "Z0"}]
        with self.assertRaisesRegex(ValueError, "underflowed"):
            commutator(left, right)
        with self.assertRaisesRegex(ValueError, "underflowed"):
            scale_terms(left, 1e-200)
        self.assertEqual(commutator(left, left), [])
        self.assertEqual(scale_terms(left, 0.)[0]["coefficient"], 0.)
        for scale in (1e308, 1e308+1e308j):
            a = [{"coefficient": scale, "pauli_string": "X0"}]
            b = [{"coefficient": scale, "pauli_string": "Z0"}]
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "overflowed"):
                commutator(a, b)


class APIDeadlineReviewTests(unittest.TestCase):
    def client(self, *, clock=lambda: 0.):
        transport = Mock(side_effect=AssertionError("No transport call expected"))
        flags = {name: True for name in ("compiler_verified", "gate_convention_verified",
                                         "deployment_verified", "service_contract_verified")}
        return TripleAlphaClient("https://example.invalid", "test-placeholder", enable_live=True,
                                 native_validation=flags, transport=transport, clock=clock), transport

    def test_nonfinite_or_boolean_deadlines_are_rejected_before_transport(self):
        for invalid in (float("nan"), float("inf"), True):
            client, transport = self.client()
            with self.subTest(deadline=invalid), self.assertRaises(ValueError):
                client.wait_for("test-job", {"successful"}, deadline=invalid)
            with self.assertRaises(ValueError):
                client.create_job(10, deadline=invalid)
            transport.assert_not_called()

    def test_nonfinite_injected_clock_cannot_remove_the_lifecycle_deadline(self):
        client, transport = self.client(clock=lambda: float("nan"))
        from runtime_input_protocol import build_runtime_input_values
        with self.assertRaises(ValueError):
            client.execute(build_runtime_input_values(payload(), settings()))
        transport.assert_not_called()
