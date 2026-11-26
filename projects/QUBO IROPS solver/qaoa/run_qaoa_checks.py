"""Independent QAOA scientific tests followed by fresh offline notebook execution."""
from pathlib import Path
import os,sys,json,shutil,subprocess,unittest
from bootstrap import ROOT,PARENT


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/validation')
    args=parser.parse_args();output=args.output_dir.resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):parser.error('Generated evidence must stay in artifacts/ or outside source')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
    for name in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        path=output/name.lower();path.mkdir(exist_ok=True);os.environ[name]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from qaoa_experiment import source_hashes,NOTEBOOK
    from case_catalog import parent_input_identity
    from run_checks import deny_connections
    before=source_hashes();parent=parent_input_identity()
    (output/'validation_summary.json').write_text(json.dumps({'status':'incomplete','source_hashes':before},indent=2)+'\n')
    subprocess.run([sys.executable,'-m','pip','check'],check=True);deny_connections()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py'))
    if not result.wasSuccessful() or result.skipped:raise SystemExit(1)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb=nbformat.read(ROOT/NOTEBOOK,as_version=4);nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    nb.cells.insert(0,nbformat.v4.new_code_cell('import bootstrap\nfrom run_checks import deny_connections\ndeny_connections()'))
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(nb,km=manager,timeout=900,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nb.cells.pop(0)
    if not all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):raise ArithmeticError('A notebook cell was skipped')
    nbformat.write(nb,output/NOTEBOOK)
    for name in ('qaoa_results.json','qaoa_comparison.svg','example_qaoa.qasm'):shutil.copy2(ROOT/'artifacts/notebook'/name,output/name)
    if before!=source_hashes() or parent!=parent_input_identity():raise RuntimeError('Scientific sources changed during acceptance')
    summary={'status':'passed','tests':result.testsRun,'failures':0,'skipped_tests':len(result.skipped),'executed_notebooks':[NOTEBOOK],
        'source_hashes':before,'parent_identity':parent,'versions':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','tests','failures','skipped_tests')},indent=2))


if __name__=='__main__':main()
