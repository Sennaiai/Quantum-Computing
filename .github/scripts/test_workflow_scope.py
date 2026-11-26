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

    def test_public_structure_checks_are_independent_of_private_workflows(self):
        text = (ROOT/'.github/workflows/quality-control.yml').read_text()
        self.assertNotIn('Sennaiai/Quality-Control-', text)
        self.assertIn('run: python .github/scripts/check_notebook_structure.py', text)
        self.assertTrue((ROOT/'.github/scripts/check_notebook_structure.py').is_file())

    def test_shared_notebook_execution_is_manual_only(self):
        text = (ROOT/'.github/workflows/quality-control.yml').read_text()
        jobs = text.split('jobs:', 1)[1]
        execution = jobs.split('  execute-notebooks:', 1)[1]
        self.assertIn("    if: github.event_name == 'workflow_dispatch'", execution)
        self.assertIn('run: python .github/scripts/execute_notebooks.py', execution)
        automatic = jobs.split('  execute-notebooks:', 1)[0]
        self.assertIn('  workflow-scope:', automatic)
        self.assertIn('run: python3 -m unittest discover', automatic)
        self.assertNotIn('execute_notebooks.py', automatic)
        self.assertNotIn('setup-micromamba', automatic)

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

    def test_expanded_qaoa_artifact_and_execution_are_explicitly_scoped(self):
        import ast
        text = (ROOT/'.github/workflows/irops-qaoa.yml').read_text()
        self.assertIn('run: python -m expanded.run_checks',text)
        blocks = re.findall(r'          path: \|\n((?:            .+\n)+)',text)
        self.assertEqual(len(blocks),2)
        prefix='projects/QUBO IROPS solver/qaoa/expanded/artifacts/validation/'
        names={line.strip().removeprefix(prefix) for line in blocks[1].splitlines()}
        self.assertEqual(names,{'validation_summary.json','Larger_IROPS_QAOA.ipynb','expanded_results.json','allocation_pareto.svg','example_large_qaoa.qasm'})
        # Check routing before CI installs notebook dependencies: stdlib only.
        tree=ast.parse((ROOT/'.github/scripts/execute_notebooks.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='skip_reason')
        namespace={'ROOT':ROOT}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook-routing>','exec'),namespace)
        path=ROOT/'projects/QUBO IROPS solver/qaoa/expanded/Larger_IROPS_QAOA.ipynb'
        self.assertIn('dedicated IROPS QAOA',namespace['skip_reason'](path))


if __name__ == '__main__': unittest.main()
