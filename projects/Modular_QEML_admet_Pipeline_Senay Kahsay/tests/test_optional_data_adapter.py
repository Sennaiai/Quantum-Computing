"""Optional external loader diagnostics and download-free core behavior."""
from pathlib import Path
import builtins
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from admet_curation import load_tdc_datasets


class OptionalDataTests(unittest.TestCase):
    def test_missing_external_package_has_actionable_csv_fallback(self):
        original = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "tdc.single_pred":
                raise ModuleNotFoundError("deliberately unavailable optional loader")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "Drug,Y CSV"):
                load_tdc_datasets(("BBB_Martins",), Path("unused"))

    def test_unknown_dataset_rejected_before_optional_import(self):
        with self.assertRaisesRegex(ValueError, "Unsupported TDC"):
            load_tdc_datasets(("not_a_tdc_endpoint",), Path("unused"))
