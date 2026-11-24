"""Run HCN scientific tests and execute the notebook in the same Python environment."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-review', action='store_true',
                        help='Refresh verified default notebook outputs and benchmarks/hcn')
    args = parser.parse_args()
    if args.save_review and os.environ.get('HCN_PROFILE', 'default') != 'default':
        parser.error('--save-review requires the default profile')
    root = Path(__file__).resolve().parent
    output = Path(os.environ.get('HCN_OUTPUT_DIR', 'artifacts/hcn'))
    if not output.is_absolute():
        output = root / output
    os.environ['HCN_OUTPUT_DIR'] = str(output)
    subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_hcn_benchmark'],
                   cwd=root, check=True)
    path = root / 'Chemistry_HCN_VQE_Study.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, kernel_name='python3', timeout=600,
                            resources={'metadata': {'path': str(root)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client.execute()
    output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output / path.name)
    if args.save_review:
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
            for item in cell.get('outputs', []):
                if item.get('output_type') == 'stream':
                    item['text'] = item['text'].replace(str(output), 'artifacts/hcn')
        nbformat.write(notebook, path)
        review = root / 'benchmarks' / 'hcn'
        review.mkdir(parents=True, exist_ok=True)
        for filename in ['results.json', 'potential_energy_curve.png', 'convergence.png']:
            shutil.copy2(output / filename, review / filename)
    print(f'PASS HCN scientific tests and fresh-kernel execution: {output}')


if __name__ == '__main__':
    main()
