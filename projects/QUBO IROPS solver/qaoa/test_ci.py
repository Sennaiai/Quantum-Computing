import importlib.util
from pathlib import Path
import unittest
from bootstrap import ROOT
from qaoa_experiment import NOTEBOOK


class DedicatedWorkflowTests(unittest.TestCase):
    def test_qaoa_notebook_has_its_own_workflow_and_is_skipped_by_generic_execution(self):
        repo=ROOT.parents[2]
        spec=importlib.util.spec_from_file_location('qaoa_notebook_policy',repo/'.github/scripts/execute_notebooks.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertIn('dedicated IROPS QAOA',module.skip_reason(ROOT/NOTEBOOK))
        workflow=(repo/'.github/workflows/irops-qaoa.yml').read_text()
        self.assertIn('python run_qaoa_checks.py',workflow)
        self.assertIn('qaoa/requirements.txt',workflow)
        self.assertNotIn('irops_clean',workflow)
