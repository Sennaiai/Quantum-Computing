import importlib.util
from pathlib import Path
import unittest
from benchmark import NOTEBOOKS

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


class WorkflowTests(unittest.TestCase):
    def test_forecasting_has_dedicated_scientific_execution(self):
        script = REPO/'.github/scripts/execute_notebooks.py'
        spec = importlib.util.spec_from_file_location('forecast_execution_policy', script)
        policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        for name in NOTEBOOKS:
            self.assertIn('dedicated forecasting', policy.skip_reason(ROOT/name))
        self.assertIn('generated', policy.skip_reason(ROOT/'artifacts/result.ipynb'))
        workflow = (REPO/'.github/workflows/forecasting-notebooks.yml').read_text()
        for text in ('projects/Grid Load QRC Benchmarks/**', 'python run_checks.py',
                     'projects/Grid Load QRC Benchmarks/requirements.txt'):
            self.assertIn(text, workflow)
