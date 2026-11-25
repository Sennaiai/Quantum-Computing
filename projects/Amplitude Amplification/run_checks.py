"""Run physics regressions, measured experiments, and the fresh-kernel notebook."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent
for setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(setting,'1')


def deny_connections():
    def denied(*args,**kwargs):raise RuntimeError('Public experiment forbids external Python socket connections')
    socket.create_connection=denied
    socket.socket.connect=denied
    socket.socket.connect_ex=denied


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/validation')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
        parser.error('Generated output must be in artifacts/ or outside the source project')
    output.mkdir(parents=True,exist_ok=True)
    for setting in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        path=output/setting.lower();path.mkdir(exist_ok=True);os.environ[setting]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from experiments import NOTEBOOK, source_hashes, run_experiments, write_report, plot_report
    before=source_hashes()
    write_report({'status':'incomplete','source_hashes':before},output/'validation_summary.json')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    deny_connections()
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    report=run_experiments();write_report(report,output/'benchmark_summary.json');plot_report(report,output/'amplification.svg')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook=nbformat.read(ROOT/NOTEBOOK,as_version=4);nbformat.validate(notebook)
    notebook.cells.insert(0,nbformat.v4.new_code_cell(
        'from run_checks import deny_connections\ndeny_connections()\n'))
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(notebook,km=manager,timeout=600,
            resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    notebook.cells.pop(0);nbformat.write(notebook,output/NOTEBOOK)
    if source_hashes()!=before:raise RuntimeError('Sources changed during acceptance')
    write_report({'status':'passed','tests':result.testsRun,'failures':0,'skipped_tests':len(result.skipped),
        'executed_notebooks':[NOTEBOOK],'source_hashes':before,
        'python_socket_connections':'denied in tests and notebook kernel',
        'backend':'ideal Aer statevector; seeded sampling','hardware':'not run'},output/'validation_summary.json')
    print(json.dumps({'status':'passed','tests':result.testsRun,'executed_notebooks':1},indent=2))

if __name__=='__main__':main()
