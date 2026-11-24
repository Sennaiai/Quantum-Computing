"""Validate and execute one chemistry study or the complete public project."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Study:
    name: str
    notebook: str
    test_module: str
    prefix: str
    profiles: tuple[str, ...] = ('default', 'smoke')


STUDIES = {
    'h2': Study('h2', 'Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb',
                'test_h2_benchmark', 'H2', ('default', 'smoke', 'extended')),
    'hcn': Study('hcn', 'Chemistry_HCN_VQE_Study.ipynb', 'test_hcn_benchmark', 'HCN'),
    'scaling': Study('scaling', 'Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb',
                     'test_scaling_benchmark', 'SCALING'),
}


def output_directory(study):
    output = Path(os.environ.get(f'{study.prefix}_OUTPUT_DIR', f'artifacts/{study.name}'))
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    benchmarks = ROOT / 'benchmarks'
    if output == ROOT or output == benchmarks or benchmarks in output.parents:
        raise ValueError('Generated output must not overwrite source or saved benchmarks')
    return output


def execute_study(study):
    path = ROOT / study.notebook
    output = output_directory(study)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, kernel_name='python3', timeout=600,
                            resources={'metadata': {'path': str(ROOT)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client.execute(env={**os.environ, f'{study.prefix}_OUTPUT_DIR': str(output)})
    output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output / path.name)
    return notebook, output


def publish_review(study, notebook, output):
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
        for item in cell.get('outputs', []):
            if item.get('output_type') == 'stream':
                text = item['text'].replace(str(output), f'artifacts/{study.name}')
                # Keep warning categories/messages but remove machine-specific
                # environment prefixes from their source locations.
                item['text'] = re.sub(r'(?m)^/[^\n:]+/site-packages/', '', text)
    nbformat.write(notebook, ROOT / study.notebook)
    destination = ROOT / 'benchmarks' / study.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / 'results.json', destination / 'results.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['all', *STUDIES], default='all')
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--save-review', action='store_true',
                        help='Refresh default notebook outputs and numerical JSON after successful execution')
    args = parser.parse_args(argv)
    selected = list(STUDIES.values()) if args.study == 'all' else [STUDIES[args.study]]
    if args.save_review and args.tests_only:
        parser.error('--save-review requires notebook execution')
    for study in selected:
        profile = os.environ.get(f'{study.prefix}_PROFILE', 'default')
        if profile not in study.profiles:
            parser.error(f'Unknown {study.prefix}_PROFILE: {profile}')
        if args.save_review and (profile != 'default' or
                (study.name == 'scaling' and os.environ.get('SCALING_LARGE_OPERATORS') == '1')):
            parser.error('--save-review requires default profiles without large-operator opt-in')
        try:
            output_directory(study)
        except ValueError as error:
            parser.error(str(error))
    modules = [study.test_module for study in selected] + ['test_run_checks', 'test_notebook_discovery']
    optimize = ['-OO' if sys.flags.optimize > 1 else '-O'] if sys.flags.optimize else []
    subprocess.run([sys.executable, *optimize, '-m', 'unittest', '-v', *modules], cwd=ROOT, check=True)
    if args.tests_only:
        return
    executed = []
    for study in selected:
        notebook, output = execute_study(study)
        executed.append((study, notebook, output))
        print(f'PASS {study.name}: fresh-kernel execution ({output})')
    # Never refresh any published study if an earlier execution failed.
    if args.save_review:
        for study, notebook, output in executed:
            publish_review(study, notebook, output)
        print('Saved verified notebook outputs and benchmark JSON')


if __name__ == '__main__':
    main()
