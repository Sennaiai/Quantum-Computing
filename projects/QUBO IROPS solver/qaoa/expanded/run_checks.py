"""Fresh offline notebook, independent scientific tests and evidence replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,unittest
from bootstrap import ROOT as QAOA_ROOT
from expanded.catalog import ROOT
from expanded.experiment import source_hashes

NOTEBOOK='Larger_IROPS_QAOA.ipynb'


def notebook_source(path):
    nb=json.loads(Path(path).read_text())
    return hashlib.sha256(json.dumps([(c['cell_type'],c['source']) for c in nb['cells']],ensure_ascii=False).encode()).hexdigest()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/validation');args=parser.parse_args()
    output=args.output_dir.resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):parser.error('Use artifacts/ or a location outside source')
    output.mkdir(parents=True,exist_ok=True)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
    for key in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        path=output/key.lower();path.mkdir(exist_ok=True);os.environ[key]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    before=source_hashes();nb_before=notebook_source(ROOT/NOTEBOOK)
    (output/'validation_summary.json').write_text(json.dumps({'status':'incomplete'},indent=2)+'\n')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    from run_checks import deny_connections
    deny_connections()
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb=nbformat.read(ROOT/NOTEBOOK,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    nb.cells.insert(0,nbformat.v4.new_code_cell('import bootstrap\nfrom run_checks import deny_connections\ndeny_connections()'))
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(nb,km=manager,timeout=900,resources={'metadata':{'path':str(QAOA_ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nb.cells.pop(0)
    if not all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):raise ArithmeticError('Skipped notebook cell')
    nbformat.write(nb,output/NOTEBOOK)
    import shutil
    for name in ('expanded_results.json','allocation_pareto.svg','example_large_qaoa.qasm'):shutil.copy2(ROOT/'artifacts/notebook'/name,output/name)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py',top_level_dir=str(QAOA_ROOT)))
    if not result.wasSuccessful() or result.skipped:raise SystemExit(1)
    from expanded.evidence import audit_evidence
    replay=audit_evidence(output)
    if before!=source_hashes() or nb_before!=notebook_source(ROOT/NOTEBOOK):raise RuntimeError('Sources changed during acceptance')
    summary={'status':'passed','tests':result.testsRun,'failures':0,'skipped_tests':0,'source_hashes':before,'notebook_source_sha256':nb_before,
        'verification_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'evidence_replay':replay,
        'versions':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ('status','tests','failures','skipped_tests','evidence_replay')},indent=2))


if __name__=='__main__':main()
