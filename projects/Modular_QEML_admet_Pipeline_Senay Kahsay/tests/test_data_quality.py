from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from admet_curation import public_smiles_fixture
from data_quality import (clean_endpoint_table, molecule_identity, sha256_json,
                          split_cleaned_dataset, validate_split_manifest)


def diverse_table():
    # Invented labels, solely a leakage/contract fixture. Every scaffold has
    # two chemically distinct molecules, one per class.
    pairs = [
        ("c1ccccc1", "Cc1ccccc1"), ("c1ccncc1", "Cc1ccncc1"),
        ("c1ccoc1", "Cc1ccoc1"), ("c1ccsc1", "Cc1ccsc1"),
        ("C1CCCCC1", "CC1CCCCC1"), ("C1CCCC1", "CC1CCCC1"),
        ("C1CCNCC1", "CC1CCNCC1"), ("C1COCCN1", "CC1COCCN1"),
        ("c1cncnc1", "Cc1cncnc1"), ("c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1")]
    return pd.DataFrame([(smiles, label) for pair in pairs for label, smiles in enumerate(pair)], columns=["Drug", "Y"])


class IdentityCleaningTests(unittest.TestCase):
    def test_duplicate_conflict_and_invalid_rows_are_auditable_without_label_invention(self):
        table = pd.DataFrame({"Drug": ["CCO", "[CH3:1][CH2:2][OH:3]", "CN", "CN", "C", "bad", "O"],
                              "Y": [0, 0, 0, 1, np.nan, 1, 1]})
        result = clean_endpoint_table(table, "fixture")
        self.assertEqual(len(result["cleaned"]), 2)
        self.assertEqual(len(result["inference"]), 4)
        self.assertNotIn("Y", result["inference"])
        audit = result["audit"]
        self.assertEqual(audit["concordant_duplicate_rows_collapsed"], 1)
        self.assertEqual(audit["conflicting_label_identities"], 1)
        self.assertEqual(audit["conflicting_labelled_rows_quarantined"], 2)
        self.assertEqual(audit["invalid_smiles_rows"], 1)
        self.assertEqual(audit["invalid_label_rows"], 1)
        ethanol = result["cleaned"].loc[result["cleaned"]["Drug"] == "CCO"].iloc[0]
        self.assertEqual(ethanol["source_row_positions"], [0, 1])
        self.assertEqual(ethanol["Y"], 0)
        conflict = result["inference"].loc[result["inference"]["Drug"] == "CN"].iloc[0]
        self.assertEqual(conflict["label_status"], "conflicting_labels")

    def test_identity_retains_stereo_charge_isotopes_and_disconnected_fragments(self):
        self.assertNotEqual(molecule_identity("F[C@H](Cl)Br")[1], molecule_identity("F[C@@H](Cl)Br")[1])
        self.assertNotEqual(molecule_identity("[13CH4]")[1], molecule_identity("C")[1])
        self.assertNotEqual(molecule_identity("C[NH3+]")[1], molecule_identity("CN")[1])
        self.assertNotEqual(molecule_identity("CCO.Cl")[1], molecule_identity("CCO")[1])
        self.assertEqual(molecule_identity("[CH3:1][CH2:2][OH:3]")[1], molecule_identity("CCO")[1])

    def test_scaffolds_are_strict_and_canonical_atom_order_invariant(self):
        self.assertEqual(molecule_identity("c1ccccc1")[2], molecule_identity("Cc1ccccc1")[2])
        self.assertNotEqual(molecule_identity("c1ccccc1")[2], molecule_identity("C1CCCCC1")[2])
        self.assertEqual(molecule_identity("CCO")[2], molecule_identity("CCCCN")[2])
        smiles = "C[C@]12C=CC(=O)C=C1[C@@H](F)C[C@@H]1[C@@H]2[C@@H](O)C[C@@]2(C)[C@H]1CC[C@]2(O)C(=O)CO"
        identity = molecule_identity(smiles)
        self.assertEqual(identity, molecule_identity(identity[0]))
        mol = Chem.MolFromSmiles(smiles)
        reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        self.assertEqual(identity, molecule_identity(Chem.MolToSmiles(reversed_mol, canonical=False, isomericSmiles=True)))

    def test_complex_and_text_labels_never_become_training_labels(self):
        table = pd.DataFrame({"Drug": ["C", "CC", "CCC", "CCCC"], "Y": [0+1j, "1", 0, 1]})
        result = clean_endpoint_table(table, "fixture")
        self.assertEqual(result["audit"]["invalid_label_rows"], 2)
        self.assertEqual(len(result["cleaned"]), 2)
        self.assertEqual(len(result["inference"]), 4)

    def test_fused_ring_stereoisomers_share_scaffold_but_keep_distinct_identity(self):
        # MolHash retains stereochemistry at fused ring junctions unless it
        # is removed explicitly. A constitutional scaffold must not leak
        # across partitions because stereo was changed or unspecified.
        stereo = "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        alternatives = (stereo.replace("@", ""), stereo.replace("@@", "@"))
        original = molecule_identity(stereo)
        for alternative in alternatives:
            other = molecule_identity(alternative)
            self.assertNotEqual(original[1], other[1])
            self.assertEqual(original[2], other[2])


class ScaffoldSplitTests(unittest.TestCase):
    def setUp(self):
        self.cleaned = clean_endpoint_table(diverse_table(), "fixture")["cleaned"]

    def test_partition_is_deterministic_disjoint_and_contains_both_classes(self):
        one = split_cleaned_dataset(self.cleaned)
        two = split_cleaned_dataset(self.cleaned.sample(frac=1, random_state=77))
        self.assertEqual(one["manifest"], two["manifest"])
        for i, left in enumerate(("train", "validation", "test")):
            self.assertEqual(set(one[left]["Y"]), {0, 1})
            for right in ("train", "validation", "test")[i+1:]:
                self.assertFalse(set(one[left]["molecule_id"]) & set(one[right]["molecule_id"]))
                self.assertFalse(set(one[left]["scaffold_id"]) & set(one[right]["scaffold_id"]))
        self.assertTrue(validate_split_manifest(one["manifest"]))

    def test_too_few_scaffolds_fails_without_random_fallback(self):
        table = next(iter(public_smiles_fixture().values()))
        cleaned = clean_endpoint_table(table, "fixture")["cleaned"]
        with self.assertRaisesRegex(ValueError, "scaffold"):
            split_cleaned_dataset(cleaned)

    def test_manifest_corruption_and_leakage_fail_even_if_rehashed(self):
        manifest = split_cleaned_dataset(self.cleaned)["manifest"]
        corrupted = deepcopy(manifest);corrupted["train"].append(corrupted["validation"][0])
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_split_manifest(corrupted)
        corrupted["train"] = sorted(corrupted["train"])
        corrupted["split_id"] = sha256_json({k:v for k,v in corrupted.items() if k != "split_id"})
        with self.assertRaises(ValueError):
            validate_split_manifest(corrupted)

    def test_invalid_config_and_forged_identity_rejected(self):
        for kwargs in ({"seed": True}, {"fractions": (.8, .2, 0)}, {"fractions": (.5, .2, .2)}):
            with self.assertRaises(ValueError):split_cleaned_dataset(self.cleaned, **kwargs)
        bad = self.cleaned.copy();bad.loc[0, "molecule_id"] = "0"*64
        with self.assertRaisesRegex(ValueError, "identity"):
            split_cleaned_dataset(bad)


if __name__ == "__main__":
    unittest.main()
