"""Larger actual airline QUBOs: allocation deltas, exact local Pareto references."""
from pathlib import Path
from time import perf_counter
import hashlib,json,os,platform
import numpy as np
import pandas as pd
import neal
from qiskit import qasm2,transpile
from bootstrap import PARENT
from case_catalog import parent_input_identity,SCENARIO
from qaoa_experiment import source_hashes as baseline_source_hashes
from qaoa_core import optimize
from qaoa_sampler import basis_counts
from measurements import compile_case,apply_global_patch
from solver import score
from qubo import resources
from expanded.catalog import load_expanded_cases,ROOT
from expanded.subspace import SubspaceQAOA
from expanded.circuit_audit import verify
from expanded.pareto import exact_tradeoffs,point,frontier,coverage
from expanded.measurements import DrawAudit,allocation_summary,paired_S3_comparison

CONFIG=json.loads((ROOT/'config.json').read_text())
METHODS=['uniform','onechoice_untrained','neal']+[f'onechoice_QAOA_p{p}' for p in CONFIG['depths']]
SCIENTIFIC=('catalog.py','subspace.py','pareto.py','measurements.py','experiment.py','circuit_audit.py','config.json',
    'data/expanded_neighbourhoods.json.gz','data/expanded_provenance.json')


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SCIENTIFIC}


def global_audit(full,anchor,case,run):
    patched,audit=apply_global_patch(full,anchor,case,run['repaired_assignment'])
    default={**audit,'admitted':score(full,patched,SCENARIO)}
    best=default
    for candidate in run['observed_frontier']:
        patched,audit=apply_global_patch(full,anchor,case,candidate['assignment'])
        value={**audit,'admitted':score(full,patched,SCENARIO)}
        metric=lambda x:(x['admitted']['assigned_pnrs'],x['admitted']['assigned_passengers'],-x['admitted']['objective'])
        if metric(value)>metric(best):best=value
    return {'case':run['case'],'backend':run['backend'],'seed':run['seed'],'S3_selected':default,'Pareto_selected':best,
        'scope':'Each patch against the same fixed incumbent; independent gains cannot be summed. Pareto selection keeps the existing S3/PNR/passenger admission guard.'}


def run_experiment(output_dir,progress=True):
    output=Path(output_dir).resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):raise ValueError('Evidence must be outside source or under artifacts/')
    output.mkdir(parents=True,exist_ok=True);tick=perf_counter()
    before=source_hashes();baseline=baseline_source_hashes();parent=parent_input_identity()
    provenance,cases=load_expanded_cases();rows=[];references=[];examples={}
    for case,instance in cases:
        compiled=compile_case(instance);model=SubspaceQAOA(compiled,CONFIG['max_qubits'],CONFIG['max_subspace_states'])
        reference=exact_tradeoffs(instance)
        if abs(model.energy.min()-reference['minimum_S3'])>1e-7:raise ArithmeticError('Reachable QUBO ground energy disagrees with exact constrained enumeration')
        inspector=DrawAudit(compiled,model,instance,reference)
        references.append({'case':case['id'],'pnrs':len(instance.parties),'qubits':model.n,'reachable_states':model.dimension,
            'full_basis_states':1<<model.n,'incumbent':point(instance,case['incumbent']),'reference':reference,'resources':resources(compiled),
            'option_counts':{p.id:sum(o.party==p.id for o in instance.options) for p in instance.parties}})
        for seed in CONFIG['seeds']:
            start=perf_counter();raw=np.random.default_rng(seed).integers(0,1<<model.n,size=CONFIG['shots'],dtype=np.uint64)
            indices,counts=np.unique(raw,return_counts=True)
            metrics=inspector.evaluate({int(k):int(n) for k,n in zip(indices,counts)})
            rows.append({'case':case['id'],'seed':seed,'backend':'uniform','seconds':perf_counter()-start,**metrics})
            start=perf_counter();metrics=inspector.evaluate(model.draw([0.,0.],CONFIG['shots'],seed))
            rows.append({'case':case['id'],'seed':seed,'backend':'onechoice_untrained','seconds':perf_counter()-start,**metrics})
            start=perf_counter();samples=neal.SimulatedAnnealingSampler().sample(compiled.bqm,num_reads=CONFIG['shots'],num_sweeps=CONFIG['neal_sweeps'],seed=seed)
            metrics=inspector.evaluate(basis_counts(samples,model))
            rows.append({'case':case['id'],'seed':seed,'backend':'neal','seconds':perf_counter()-start,**metrics})
            previous=None
            for depth in CONFIG['depths']:
                start=perf_counter();params,training=optimize(model,depth,seed,CONFIG['starts'],CONFIG['maxiter'],previous)
                measurement_seed=(seed+1009*depth) % 2**32
                metrics=inspector.evaluate(model.draw(params,CONFIG['shots'],measurement_seed))
                fidelity=verify(model,params);circuit=model.circuit(params)
                rows.append({'case':case['id'],'seed':seed,'backend':f'onechoice_QAOA_p{depth}','depth':depth,
                    'training':training,'measurement_seed':measurement_seed,'circuit_block_fidelity':fidelity,
                    'gate_counts':{str(k):int(v) for k,v in circuit.count_ops().items()},'seconds':perf_counter()-start,**metrics})
                if case['id']=='rich_10' and seed==7 and depth==1:examples['qasm']=qasm2.dumps(transpile(circuit,basis_gates=['h','rz','rx','cx'],optimization_level=1,seed_transpiler=7))
                previous=params
            (output/'checkpoint.json').write_text(json.dumps({'status':'incomplete','source_hashes':before,'runs':rows},allow_nan=False)+'\n')
            if progress:print(f"{case['id']} | {model.n} qubits, {model.dimension:,} reachable states | seed {seed} complete | {perf_counter()-tick:.1f}s",flush=True)
    from experiment import load_experiment_catalogs
    _,full,_,_,_=load_experiment_catalogs()
    frame=pd.read_csv(PARENT/'results/assignments.csv.gz',dtype='string')
    anchor={str(p):str(o) for p,o in zip(frame.PNR,frame.seed_7) if not pd.isna(o)}
    case_by_id={c['id']:c for c,i in cases}
    patches=[global_audit(full,anchor,case_by_id[r['case']],r) for r in rows]
    report={'status':'passed','source_hashes':before,'baseline_source_hashes':baseline,'parent_identity':parent,'config':CONFIG,
        'methods':METHODS,'provenance':provenance,'cases':references,'runs':rows,'global_patch_audits':patches,
        'scope':'Exact local references only; fixed outside bookings; no full-population QAOA allocation rate or speedup claim.',
        'fairness':'Same unchanged QUBO, penalties, residual capacities, final sample counts and deterministic repair. Exact-expectation QAOA training uses additional classical compute; budgets are not time- or physical-shot-matched.',
        'pareto_scope':'Maximize PNRs/passengers and minimize S3 objective. Exact local frontier includes all feasible assignments in the restricted option catalog; observed frontier contains sampled feasible draws before repair. Training remains S3 only.',
        'simulation':'Exact invariant-subspace representation of the existing one-choice/slack-X circuit, not amplitude truncation. Qiskit gate-block matrices independently replay every optimized circuit; small cases also use full-state circuit checks.'}
    report['summary']=allocation_summary(report)
    report['paired_S3_comparison']=paired_S3_comparison(report)
    if before!=source_hashes() or baseline!=baseline_source_hashes() or parent!=parent_input_identity():raise RuntimeError('Scientific inputs changed during execution')
    plot(report,output/'allocation_pareto.svg')
    (output/'example_large_qaoa.qasm').write_text(examples['qasm']+'\n')
    report['timing']={'end_to_end_seconds':perf_counter()-tick,'python':platform.python_version(),'system':platform.system(),'architecture':platform.machine(),
        'requested_blas_threads':os.environ.get('OPENBLAS_NUM_THREADS','unspecified'),'scope':'Frozen cases, references, all sampling/training, Qiskit block checks, global audits and exports; excludes installation, kernel startup and one-time case generation.'}
    (output/'expanded_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    selected=['onechoice_untrained','neal']+[f'onechoice_QAOA_p{p}' for p in CONFIG['depths']]
    table={r['method']:r for r in report['summary']};x=np.arange(len(selected))
    axes[0].bar(x,[table[k]['local_S3_optimum_runs'] for k in selected],color='#246b84')
    total=table[selected[0]]['runs'];axes[0].set_ylim(0,total+1)
    axes[0].set_xticks(x,['Untrained','Annealing']+[f'QAOA p{p}' for p in CONFIG['depths']],rotation=30,ha='right')
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel(f'Runs finding a local S3 optimum (out of {total})')
    axes[0].set_title('Before common classical repair')
    cases=report['cases'];x=np.arange(len(cases))
    axes[1].bar(x-.18,[c['incumbent']['assigned_pnrs'] for c in cases],width=.36,label='Incumbent',color='#246b84')
    axes[1].bar(x+.18,[c['reference']['maximum_pnrs'] for c in cases],width=.36,label='Exact local maximum',color='#b36526')
    axes[1].set_xticks(x,[c['case'].replace('expanded_','E').replace('rich_','R') for c in cases])
    axes[1].set_ylabel('PNRs reallocated within the neighbourhood')
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title('Allocation headroom with outside bookings fixed');axes[1].legend(fontsize=8)
    fig.suptitle('Larger actual airline QUBOs: solution quality and allocation headroom');fig.tight_layout();fig.savefig(path);plt.close(fig)
    path=Path(path);path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/experiment');args=parser.parse_args()
    result=run_experiment(args.output_dir);print(json.dumps({'summary':result['summary'],'timing':result['timing']},indent=2))
