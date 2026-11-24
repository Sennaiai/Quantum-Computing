"""Execute H2 tests and notebook, saving an executed review artifact."""
import os
import argparse
import shutil
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-review', action='store_true',
                        help='Save verified default-profile outputs into the review notebook and benchmarks/h2')
    args = parser.parse_args()
    if args.save_review and os.environ.get('H2_PROFILE', 'default') != 'default':
        parser.error('--save-review requires the default profile')
    root = Path(__file__).resolve().parent
    output = Path(os.environ.get('H2_OUTPUT_DIR', 'artifacts/h2'))
    if not output.is_absolute():
        output = root / output
    os.environ['H2_OUTPUT_DIR'] = str(output)
    subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_h2_benchmark'],
                   cwd=root, check=True)
    path = root / 'Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, kernel_name='python3', timeout=600,
                            resources={'metadata':{'path':str(root)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client.execute()
    output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output/path.name)
    if args.save_review:
        # Keep GitHub's notebook view useful without machine-specific timings.
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
            for item in cell.get('outputs', []):
                if item.get('output_type') == 'stream':
                    item['text'] = item['text'].replace(str(output), 'artifacts/h2')
        nbformat.write(notebook, path)
        review = root / 'benchmarks' / 'h2'
        review.mkdir(parents=True, exist_ok=True)
        for filename in ['results.json', 'potential_energy_curve.png', 'natural_occupations.png']:
            shutil.copy2(output / filename, review / filename)
    print(f'PASS scientific tests and fresh-kernel notebook execution: {output}')


if __name__ == '__main__':
    main()
