from pathlib import Path
import importlib.util
import unittest
from experiment import NOTEBOOK

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


class DiscoveryTests(unittest.TestCase):
    def test_dedicated_actual_catalog_ci_and_private_artifact_exclusion(self):
        spec = importlib.util.spec_from_file_location('irops_execution_policy', REPO/'.github/scripts/execute_notebooks.py')
        policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        self.assertIn('dedicated IROPS', policy.skip_reason(ROOT/NOTEBOOK))
        self.assertIn('generated', policy.skip_reason(ROOT/'artifacts/private/legacy.ipynb'))
        workflow = (REPO/'.github/workflows/irops-qubo.yml').read_text()
        for text in ('projects/QUBO IROPS solver/**', 'projects/QUBO IROPS solver/requirements.txt', 'python run_checks.py'):
            self.assertIn(text, workflow)
