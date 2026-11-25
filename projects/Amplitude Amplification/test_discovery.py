"""The scientific workflow must follow the project's path and own its execution."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parents[1]

class DiscoveryTests(unittest.TestCase):
    def test_project_routes_to_dedicated_scientific_job(self):
        script=REPOSITORY/'.github/scripts/execute_notebooks.py'
        spec=importlib.util.spec_from_file_location('amplitude_routing',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        notebook=ROOT/'Grovers_Algorithm_for_Quantum_Search_and_Acceleration.ipynb'
        self.assertIn('dedicated amplitude',module.skip_reason(notebook))
        self.assertIn('generated',module.skip_reason(ROOT/'artifacts/result.ipynb'))
        text=(REPOSITORY/'.github/workflows/amplitude-notebook.yml').read_text()
        self.assertIn('projects/Amplitude Amplification/**',text)
        self.assertIn('projects/Amplitude Amplification/requirements.txt',text)
        self.assertIn('python run_checks.py',text)
        self.assertTrue((ROOT/'requirements.txt').is_file())
