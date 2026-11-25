"""Scientific tests and full notebooks in fresh, offline Python kernels."""
from __future__ import annotations

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
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'


def deny_connections():
    def denied(*args, **kwargs):
        raise RuntimeError('Public forecasting acceptance denies Python network connections')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/validation')
    args = parser.parse_args(); output = args.output_dir.resolve()
    if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
        parser.error('Evidence must stay in artifacts/ or outside the source project')
    output.mkdir(parents=True, exist_ok=True)
    for key in ('MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR'):
        path = output/key.lower(); path.mkdir(exist_ok=True); os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from benchmark import NOTEBOOKS, source_hashes
    before = source_hashes()
    summary_path = output/'validation_summary.json'
    summary_path.write_text(json.dumps({'status': 'incomplete', 'source_hashes': before}, indent=2)+'\n')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    deny_connections()
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    executed = []
    for name in NOTEBOOKS:
        nb = nbformat.read(ROOT/name, as_version=4); nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type == 'code': c.outputs = []; c.execution_count = None
        nb.cells.insert(0, nbformat.v4.new_code_cell('from run_checks import deny_connections\ndeny_connections()'))
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        try:
            NotebookClient(nb, km=manager, timeout=1200,
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        finally:
            if manager.has_kernel: manager.shutdown_kernel(now=True)
        nb.cells.pop(0); nbformat.write(nb, output/name); executed.append(name)
        print('PASS', name, flush=True)
    from data_contract import load_data, prepare
    from physics_checks import resource_report
    physics = resource_report(prepare(load_data()).windows)
    (output/'physics.json').write_text(json.dumps(physics, indent=2, allow_nan=False)+'\n')
    for study in ('cv', 'spin'):
        for suffix in ('.json', '.svg', '_predictions.csv'):
            shutil.copy2(ROOT/'artifacts/notebook'/f'{study}{suffix}', output/f'{study}{suffix}')
    if before != source_hashes(): raise RuntimeError('Executable source changed during acceptance')
    summary = {'status': 'passed', 'tests': result.testsRun, 'failures': 0,
               'skipped_tests': len(result.skipped), 'executed_notebooks': executed,
               'source_hashes': before, 'backend': 'Ideal NumPy/SciPy states, PennyLane Gaussian and Qiskit references',
               'network_policy': 'Python connections denied in tests and notebook kernels',
               'versions': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('status', 'tests', 'executed_notebooks')}, indent=2))


if __name__ == '__main__': main()
