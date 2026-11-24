"""Supported reduced benchmark configurations must still produce valid figures."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "admet-test-matplotlib"))
from admet_curation import public_smiles_fixture
from benchmark_reporting import plot_information
from data_benchmark import benchmark_table


class ReportingTests(unittest.TestCase):
    def test_single_seed_marks_support_stability_undefined_without_crashing(self):
        table = next(iter(public_smiles_fixture().values()))
        with tempfile.TemporaryDirectory() as tmp:
            report = benchmark_table(table, output_dir=Path(tmp)/"work", dataset_name="fixture",
                                     profiles=(3,), jitter_seeds=(42,), permutations=1)
            self.assertEqual(report["profiles"][0]["pair_jaccard_between_seeds"], [])
            path = Path(tmp)/"figure.svg"
            plot_information(report, path)
            self.assertEqual(ET.parse(path).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            self.assertIn("Undefined", path.read_text())
            self.assertTrue(all(line == line.rstrip() for line in path.read_text().splitlines()))
