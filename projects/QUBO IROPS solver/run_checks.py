"""Verify the QUBO experiment and execute the actual catalog in a fresh kernel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'): os.environ[key] = '1'


def deny_connections():
    def denied(*args, **kwargs): raise RuntimeError('IROPS public acceptance denies Python network connections')
    socket.create_connection = denied; socket.socket.connect = denied; socket.socket.connect_ex = denied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/validation')
    args = parser.parse_args(); output = args.output_dir.resolve()
    if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
        parser.error('Evidence must stay in artifacts/ or outside source project')
    output.mkdir(parents=True, exist_ok=True)
    for key in ('MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR'):
        path = output/key.lower(); path.mkdir(exist_ok=True); os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from experiment import NOTEBOOK, source_hashes
    before = source_hashes(); summary = output/'validation_summary.json'
    summary.write_text(json.dumps({'status': 'incomplete', 'source_hashes': before}, indent=2)+'\n')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    deny_connections()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py'))
    if not result.wasSuccessful(): raise SystemExit(1)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb = nbformat.read(ROOT/NOTEBOOK, as_version=4); nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == 'code': c.outputs = []; c.execution_count = None
    nb.cells.insert(0, nbformat.v4.new_code_cell('from run_checks import deny_connections\ndeny_connections()'))
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try: NotebookClient(nb, km=manager, timeout=1200, resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nb.cells.pop(0)
    code = [c for c in nb.cells if c.cell_type == 'code']
    if not all(c.execution_count is not None for c in code): raise RuntimeError('A notebook code cell was skipped')
    nbformat.write(nb, output/NOTEBOOK)
    for name in ('actual_catalog_results.json', 'assignments.csv.gz', 'allocation_resources.svg'):
        shutil.copyfile(ROOT/'artifacts/notebook'/name, output/name)
    if before != source_hashes(): raise RuntimeError('Scientific sources changed during acceptance')
    report = {'status': 'passed', 'tests': result.testsRun, 'failures': 0, 'skipped_tests': len(result.skipped),
              'executed_notebooks': [NOTEBOOK], 'source_hashes': before,
              'dataset': 'Provided de-identified actual catalog, not a synthetic benchmark',
              'client_baseline': 'Client performs comparison independently; outside this solver',
              'versions': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    summary.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'tests', 'executed_notebooks')}, indent=2))


if __name__ == '__main__': main()
