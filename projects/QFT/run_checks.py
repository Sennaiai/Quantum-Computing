"""Execute QFT physics, arithmetic and both notebooks in fresh offline kernels."""
from __future__ import annotations
import argparse,json,os,socket,subprocess,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(setting,'1')

def deny_connections():
    def denied(*args,**kwargs):raise RuntimeError('Public QFT experiments deny Python network connections')
    socket.create_connection=denied;socket.socket.connect=denied;socket.socket.connect_ex=denied


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/validation')
    args=parser.parse_args();output=args.output_dir.resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
        parser.error('Generated evidence must stay in artifacts/ or outside the source project')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        path=output/name.lower();path.mkdir(exist_ok=True);os.environ[name]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from experiments import (NOTEBOOKS,source_hashes,run_arithmetic_experiments,run_shor_experiments,
        plot_arithmetic,plot_shor,write_json)
    before=source_hashes();write_json({'status':'incomplete','source_hashes':before},output/'validation_summary.json')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    deny_connections()
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    arithmetic=run_arithmetic_experiments();shor=run_shor_experiments()
    write_json(arithmetic,output/'arithmetic.json');write_json(shor,output/'shor.json')
    plot_arithmetic(arithmetic,output/'arithmetic.svg');plot_shor(shor,output/'order_finding.svg')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    executed=[]
    for name in NOTEBOOKS:
        notebook=nbformat.read(ROOT/name,as_version=4);nbformat.validate(notebook)
        notebook.cells.insert(0,nbformat.v4.new_code_cell('from run_checks import deny_connections\ndeny_connections()\n'))
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        try:NotebookClient(notebook,km=manager,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        notebook.cells.pop(0);nbformat.write(notebook,output/name);executed.append(name)
        print('PASS',name,flush=True)
    if source_hashes()!=before:raise RuntimeError('Source tree changed during scientific acceptance')
    write_json({'status':'passed','tests':result.testsRun,'failures':0,'skipped_tests':len(result.skipped),
        'source_hashes':before,'executed_notebooks':executed,
        'backend':'ideal Aer and PennyLane default.qubit; no hardware execution',
        'network_policy':'Python socket connections denied in tests and notebook kernels'},output/'validation_summary.json')
    print(json.dumps({'status':'passed','tests':result.testsRun,'executed_notebooks':executed},indent=2))

if __name__=='__main__':main()
