from pathlib import Path
import importlib.util,unittest
from experiments import NOTEBOOKS
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]

class WorkflowTests(unittest.TestCase):
    def test_dedicated_qft_execution_and_paths(self):
        script=REPO/'.github/scripts/execute_notebooks.py'
        spec=importlib.util.spec_from_file_location('qft_policy',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name in NOTEBOOKS:self.assertIn('dedicated QFT',module.skip_reason(ROOT/name))
        self.assertIn('generated',module.skip_reason(ROOT/'artifacts/example.ipynb'))
        text=(REPO/'.github/workflows/qft-notebooks.yml').read_text()
        self.assertIn('projects/QFT/**',text);self.assertIn('python run_checks.py',text)
        self.assertIn('projects/QFT/requirements.txt',text)
