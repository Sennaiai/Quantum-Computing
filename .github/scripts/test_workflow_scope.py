"""Keep scientific experiments scoped to their own project changes."""
from fnmatch import fnmatchcase
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC = {
    'h2-notebook.yml': 'projects/Quanutm Chemistry /**',
    'hcn-notebook.yml': 'projects/Quanutm Chemistry /**',
    'scaling-notebook.yml': 'projects/Quanutm Chemistry /**',
    'hhl-notebook.yml': 'projects/HHL/**',
    'admet-pipeline.yml': 'projects/Quantum Molecular Learning for ADMET/**',
    'amplitude-notebook.yml': 'projects/Amplitude Amplification/**',
    'qft-notebooks.yml': 'projects/QFT/**',
    'forecasting-notebooks.yml': 'projects/Grid Load QRC Benchmarks/**',
    'irops-qubo.yml': 'projects/QUBO IROPS solver/**',
    'irops-qaoa.yml': 'projects/QUBO IROPS solver/**',
}


def event_paths(text, event):
    block = re.search(rf'^  {event}:\n(.*?)(?=^  [a-z_]+:|^permissions:)', text, re.M | re.S)
    if block is None: raise AssertionError(f'Missing {event} trigger')
    return re.findall(r'^      - "([^"]+)"$', block.group(1), re.M)


class ProjectScientificScopeTests(unittest.TestCase):
    def test_only_project_files_trigger_automatic_scientific_execution(self):
        for workflow, project in SCIENTIFIC.items():
            text = (ROOT/'.github/workflows'/workflow).read_text()
            for event in ('pull_request', 'push'):
                with self.subTest(workflow=workflow, event=event):
                    exclusions = ['!projects/QUBO IROPS solver/README.md'] if workflow == 'irops-qaoa.yml' else (['!projects/QUBO IROPS solver/qaoa/**', '!projects/QUBO IROPS solver/README.md'] if workflow == 'irops-qubo.yml' else [])
                    self.assertEqual(event_paths(text, event), [project]+exclusions)
                    self.assertTrue(fnmatchcase(project.removesuffix('**')+'experiment.py', project))
                    self.assertFalse(fnmatchcase('.github/scripts/execute_notebooks.py', project))
                    self.assertFalse(fnmatchcase('README.md', project))
                    if workflow not in ('irops-qubo.yml', 'irops-qaoa.yml'):
                        self.assertFalse(fnmatchcase('projects/QUBO IROPS solver/solver.py', project))

    def test_manual_scientific_reruns_remain_available(self):
        for workflow in SCIENTIFIC:
            with self.subTest(workflow=workflow):
                self.assertIn('  workflow_dispatch:', (ROOT/'.github/workflows'/workflow).read_text())

    def test_shared_checks_validate_scope_without_closed_project_execution(self):
        text = (ROOT/'.github/workflows/quality-control.yml').read_text()
        self.assertIn('python3 -m unittest discover -s .github/scripts -p test_workflow_scope.py -v', text)
        self.assertNotIn('run_checks.py', text)

    def test_irops_artifact_contains_only_scientific_evidence(self):
        text = (ROOT/'.github/workflows/irops-qubo.yml').read_text()
        match = re.search(r'          path: \|\n((?:            .+\n)+)', text)
        self.assertIsNotNone(match)
        names = {line.strip().removeprefix('projects/QUBO IROPS solver/artifacts/validation/') for line in match.group(1).splitlines()}
        self.assertEqual(names, {'validation_summary.json', 'QUBO_IROPS_Allocation_Experiment.ipynb',
                                 'actual_catalog_results.json', 'assignments.csv.gz', 'allocation_resources.svg'})

    def test_qaoa_extension_does_not_reexecute_completed_classical_experiment(self):
        def matches(patterns, path):
            result = False
            for pattern in patterns:
                if fnmatchcase(path, pattern.removeprefix('!')): result = not pattern.startswith('!')
            return result
        for event in ('pull_request','push'):
            baseline = event_paths((ROOT/'.github/workflows/irops-qubo.yml').read_text(),event)
            quantum = event_paths((ROOT/'.github/workflows/irops-qaoa.yml').read_text(),event)
            self.assertFalse(matches(baseline,'projects/QUBO IROPS solver/qaoa/qaoa_core.py'))
            self.assertTrue(matches(quantum,'projects/QUBO IROPS solver/qaoa/qaoa_core.py'))
            self.assertFalse(matches(baseline,'projects/QUBO IROPS solver/README.md'))
            self.assertFalse(matches(quantum,'projects/QUBO IROPS solver/README.md'))
            self.assertTrue(matches(baseline,'projects/QUBO IROPS solver/qubo.py'))
            self.assertTrue(matches(quantum,'projects/QUBO IROPS solver/qubo.py'))

    def test_qaoa_artifact_has_no_runtime_cache_or_private_inputs(self):
        text = (ROOT/'.github/workflows/irops-qaoa.yml').read_text()
        match = re.search(r'          path: \|\n((?:            .+\n)+)',text)
        self.assertIsNotNone(match)
        names = {line.strip().removeprefix('projects/QUBO IROPS solver/qaoa/artifacts/validation/') for line in match.group(1).splitlines()}
        self.assertEqual(names, {'validation_summary.json','QAOA_IROPS_Neighbourhoods.ipynb',
                                 'qaoa_results.json','qaoa_comparison.svg','example_qaoa.qasm'})


if __name__ == '__main__': unittest.main()
