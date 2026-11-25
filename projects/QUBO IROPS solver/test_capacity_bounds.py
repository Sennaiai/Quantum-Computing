import copy
import json
from pathlib import Path
import unittest
from data_contract import ROOT
from experiment import load_public_catalog
from capacity_bounds import verify_capacity_bounds


class CapacityCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance, cls.provenance = load_public_catalog()
        cls.certificate = json.loads((ROOT/'data/capacity_bounds.json').read_text())

    def test_every_option_has_an_exact_rational_dual_bound(self):
        bounds = verify_capacity_bounds(self.instance, self.certificate, self.provenance['sha256'])
        self.assertEqual(bounds['pnrs']['integer_upper_bound'], 4557)
        self.assertEqual(bounds['passengers']['integer_upper_bound'], 5930)
        self.assertFalse(bounds['pnrs']['seventy_percent_possible'])
        self.assertFalse(bounds['passengers']['seventy_percent_possible'])

    def test_wrong_catalog_and_weakened_certificate_are_rejected(self):
        with self.assertRaises(ValueError): verify_capacity_bounds(self.instance, self.certificate, 'different')
        bad = copy.deepcopy(self.certificate)
        bad['bounds']['pnrs']['pnr_dual'] = {}; bad['bounds']['pnrs']['leg_dual'] = []
        with self.assertRaises(ValueError): verify_capacity_bounds(self.instance, bad, self.provenance['sha256'])
        bad = copy.deepcopy(self.certificate)
        bad['bounds']['passengers']['rational_upper_bound'] = '1'
        with self.assertRaises(ValueError): verify_capacity_bounds(self.instance, bad, self.provenance['sha256'])
