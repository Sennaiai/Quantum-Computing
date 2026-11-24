"""The repository move must not execute artifacts or silently promote coursework."""
import importlib.util
from pathlib import Path
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

    def test_relocated_coursework_keeps_its_unvalidated_status(self):
        self.assertIn('coursework', self.reason('QFT/Qiskit Arithemtic .ipynb'))
        self.assertIn('coursework', self.reason('Shor_Senay_Kahsay.ipynb'))

    def test_hcn_has_a_dedicated_execution_job(self):
        self.assertIn('dedicated HCN', self.reason(
            'Quanutm Chemistry /Chemistry_HCN_VQE_Study.ipynb'))

    def test_previous_offline_and_resource_policies_are_preserved(self):
        self.assertIsNone(self.reason('Optimization/Unit_Commitment_MOPSO.ipynb'))
        self.assertIn('resource-intensive', self.reason(
            'Quanutm Chemistry /Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb'))
        self.assertIn('external data', self.reason('Modular_QEML_admet_Pipeline/example.ipynb'))


if __name__ == '__main__':
    unittest.main()
