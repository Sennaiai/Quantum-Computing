import unittest
from data_contract import ROOT
from run_private import private_output


class PrivateOutputTests(unittest.TestCase):
    def test_private_outputs_cannot_target_public_source_or_results(self):
        for path in (ROOT, ROOT/'results', ROOT/'fixtures', ROOT.parent/'other-project'):
            with self.assertRaises(ValueError): private_output(path)
        self.assertEqual(private_output(ROOT/'artifacts/private/run'), ROOT/'artifacts/private/run')
