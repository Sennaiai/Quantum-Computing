"""Exact discrete references, pinned cache, and offline real-pipeline checks."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"python_classes")]
import data_benchmark as benchmark
from admet_curation import public_smiles_fixture


class DiscreteInformationTests(unittest.TestCase):
    def test_independent_redundant_and_xor_references(self):
        x = np.tile([0, 0, 1, 1], 8)
        y = np.tile([0, 1, 0, 1], 8)
        self.assertAlmostEqual(benchmark.discrete_entropy(x), 1)
        self.assertAlmostEqual(benchmark.discrete_mi(x, y), 0)
        self.assertAlmostEqual(benchmark.discrete_mi(x, x), 1)
        self.assertAlmostEqual(benchmark.discrete_co_information(x, x, x), 1)
        self.assertAlmostEqual(benchmark.discrete_co_information(x, y, x^y), -1)
        self.assertAlmostEqual(benchmark.discrete_mi(x, y, x^y), 1)
        self.assertAlmostEqual(benchmark.discrete_co_information(3*x+7, y, x^y), -1)

    def test_invalid_discrete_inputs_rejected(self):
        for values in ([], [0, np.nan], [[1, 2, np.inf]], ["a", "b"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.discrete_entropy(values)
        with self.assertRaises(ValueError):
            benchmark.discrete_mi([0, 1], [0, 1, 0])
        with self.assertRaises(ValueError):
            benchmark.discrete_entropy([0, 1], base=1)


class PublicCacheTests(unittest.TestCase):
    raw = b"Drug_ID\tDrug\tY\n1\tC\t0\n"

    def setUp(self):
        self.pins = patch.multiple(benchmark, BBB_BYTES=len(self.raw),
            BBB_SHA256=hashlib.sha256(self.raw).hexdigest())
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def test_explicit_download_is_bounded_pinned_atomic_and_cache_first(self):
        response = Mock()
        response.iter_content.return_value = [self.raw[:7], self.raw[7:]]
        transport = Mock(return_value=response)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cache"/"bbb.tab"
            self.assertEqual(benchmark.download_bbb(path, transport=transport), path)
            self.assertEqual(path.read_bytes(), self.raw)
            self.assertEqual(benchmark.download_bbb(path, transport=Mock(side_effect=AssertionError("network"))), path)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["bbb.tab"])
        self.assertEqual(transport.call_args.args[0], benchmark.BBB_URL)
        self.assertEqual(transport.call_args.kwargs["timeout"], 45)
        response.close.assert_called_once()

    def test_corrupt_existing_cache_is_not_replaced_or_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bbb.tab"
            path.write_bytes(self.raw[:-1]+b"x")
            transport = Mock(side_effect=AssertionError("network"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                benchmark.download_bbb(path, transport=transport)
            transport.assert_not_called()

    def test_changed_or_oversized_response_never_publishes(self):
        for raw in (b"bad response", self.raw+b"oversize"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                response = Mock()
                response.iter_content.return_value = [raw]
                destination = Path(tmp)/"cache"/"bbb.tab"
                with self.assertRaises(ValueError):
                    benchmark.download_bbb(destination, transport=Mock(return_value=response))
                self.assertFalse(destination.exists())
                response.close.assert_called_once()


class DataExperimentTests(unittest.TestCase):
    def test_same_pipeline_comparison_and_heralded_expm_without_download(self):
        frame = next(iter(public_smiles_fixture().values()))
        with tempfile.TemporaryDirectory() as tmp, patch("requests.get", side_effect=AssertionError("network")):
            report = benchmark.benchmark_table(frame, output_dir=Path(tmp), profiles=(3,),
                                               jitter_seeds=(42, 43), permutations=3)
            self.assertEqual(report["status"], "passed")
            profile = report["profiles"][0]
            self.assertEqual(profile["config"]["conditional_mi_threshold"], .10)
            self.assertEqual(profile["config"]["triad_threshold"], .15)
            self.assertEqual(profile["jitter_seeds"], [42, 43])
            self.assertEqual(len(profile["pair_comparisons"]), 3)
            self.assertEqual(len(profile["pair_jaccard_between_seeds"]), 1)
            self.assertEqual(len(profile["label_permutation_diagnostic"]["permuted_fixed_feature_mean_mi_bits"]), 3)
            pipeline = profile["pipeline"]
            self.assertEqual(pipeline["status"], "passed")
            self.assertFalse(pipeline["live_api_execution_allowed"])
            self.assertLessEqual(pipeline["reference"]["recovered_evolution_error"], pipeline["error_bound"]+1e-11)
            self.assertEqual(json.loads((Path(tmp)/"data_benchmark.json").read_text()), report)

    def test_empty_graph_is_reported_without_lowering_thresholds(self):
        frame = pd.DataFrame({"Drug": ["C"]*16, "Y": [0, 1]*8})
        with tempfile.TemporaryDirectory() as tmp:
            report = benchmark.benchmark_table(frame, output_dir=Path(tmp), profiles=(3,),
                                               jitter_seeds=(42,), permutations=1)
            self.assertEqual(report["status"], "diagnostic_only_no_retained_features")
            self.assertEqual(report["profiles"][0]["pipeline"]["status"], "no_retained_interaction_features")
            self.assertFalse(report["profiles"][0]["pipeline"]["thresholds_changed"])

    def test_resource_profiles_are_explicit_and_bounded(self):
        for profiles, seeds, permutations in (((9,), (42,), 1), ((4, 4), (42,), 1),
                                            ((4,), (43,), 1), ((4,), (42,), 0), ([[4]], (42,), 1)):
            with self.subTest(profiles=profiles, seeds=seeds, permutations=permutations), self.assertRaises(ValueError):
                benchmark._validated_settings(profiles, seeds, permutations)


if __name__ == "__main__":
    unittest.main()
