"""The repository move must not execute artifacts or silently promote coursework."""
import importlib.util
from pathlib import Path
import re
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/execute_notebooks.py'
spec = importlib.util.spec_from_file_location('notebook_execution_policy', SCRIPT)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class DiscoveryTests(unittest.TestCase):
    def reason(self, relative):
        return policy.skip_reason(policy.ROOT / 'projects' / relative)

    def test_generated_notebooks_are_not_executed(self):
        self.assertIn('generated', self.reason('Quanutm Chemistry /artifacts/h2/result.ipynb'))
        self.assertIn('generated', self.reason('QFT/.ipynb_checkpoints/example.ipynb'))

    def test_h2_has_a_dedicated_execution_job(self):
        self.assertIn('dedicated H2', self.reason(
            'Quanutm Chemistry /Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb'))

    def test_qft_has_scientific_workflow_and_legacy_shor_is_excluded(self):
        self.assertIn('dedicated QFT', self.reason('QFT/QFT_Arithmetic_Validation.ipynb'))
        self.assertIn('coursework', self.reason('Shor_Senay_Kahsay.ipynb'))

    def test_hcn_has_a_dedicated_execution_job(self):
        self.assertIn('dedicated HCN', self.reason(
            'Quanutm Chemistry /Chemistry_HCN_VQE_Study.ipynb'))

    def test_admet_uses_scientific_workflow_and_excludes_generated_notebooks(self):
        self.assertIn('dedicated ADMET', self.reason(
            'Quantum Molecular Learning for ADMET/ADMET_Triple_Alpha_End_to_End.ipynb'))
        self.assertIn('generated', self.reason(
            'Quantum Molecular Learning for ADMET/artifacts/validation/result.ipynb'))

    def test_admet_workflow_paths_follow_the_current_project_directory(self):
        project = policy.ROOT / 'projects/Quantum Molecular Learning for ADMET'
        text = (policy.ROOT / '.github/workflows/admet-pipeline.yml').read_text()
        self.assertTrue(project.is_dir())
        self.assertIn('projects/Quantum Molecular Learning for ADMET/**', text)
        dependencies = re.findall(r'cache-dependency-path:\s*"([^"]+)"', text)
        self.assertEqual(len(dependencies), 2)
        for relative in dependencies:
            self.assertTrue((policy.ROOT / relative).is_file(), relative)
        self.assertNotIn('Modular_QEML_admet_Pipeline_Senay Kahsay', text)

    def test_previous_offline_and_resource_policies_are_preserved(self):
        self.assertIsNone(self.reason('Optimization/Unit_Commitment_MOPSO.ipynb'))
        self.assertIn('dedicated scaling', self.reason(
            'Quanutm Chemistry /Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb'))
        self.assertIn('external data', self.reason('Modular_QEML_admet_Pipeline/example.ipynb'))

    def test_workflow_directories_survive_actions_whitespace_trimming(self):
        for name in ['h2-notebook.yml', 'hcn-notebook.yml', 'scaling-notebook.yml', 'admet-pipeline.yml']:
            text = (policy.ROOT / '.github/workflows' / name).read_text()
            directories = re.findall(r'working-directory:\s*"([^"]+)"', text)
            self.assertTrue(directories, f'No working directory in {name}')
            for directory in directories:
                self.assertTrue((policy.ROOT / directory.strip()).is_dir(),
                                f'Actions trims terminal whitespace: {name}: {directory!r}')


if __name__ == '__main__':
    unittest.main()
