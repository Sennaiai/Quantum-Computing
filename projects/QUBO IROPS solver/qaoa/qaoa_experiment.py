"""Matched real-data QUBOs: exact reference, random sampling, thermal SA and QAOA."""
from pathlib import Path
from time import perf_counter
import hashlib,json,os,platform
import numpy as np
import pandas as pd
import neal
from qiskit import transpile,qasm2
from qiskit.quantum_info import Statevector
from bootstrap import ROOT,PARENT
from case_catalog import load_cases,parent_input_identity,SCENARIO
from qaoa_core import QUBOStatevector
from constraint_qaoa import OneHotQAOA
from qaoa_sampler import QAOASampler,basis_counts
from measurements import compile_case,masks,audit_counts,apply_global_patch
from solver import exact_reference,score
from qubo import resources

CONFIG=json.loads((ROOT/'config.json').read_text())
NOTEBOOK='QAOA_IROPS_Neighbourhoods.ipynb'


def source_hashes():
    paths=list(ROOT.glob('*.py'))+[ROOT/'config.json',ROOT/'requirements.txt']+list((ROOT/'data').glob('*'))
    result={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    if (ROOT/NOTEBOOK).exists():
        nb=json.loads((ROOT/NOTEBOOK).read_text());cells=[(c['cell_type'],''.join(c['source']),c.get('metadata',{}).get('tags',[])) for c in nb['cells']]
        result[NOTEBOOK]=hashlib.sha256(json.dumps(cells,ensure_ascii=False).encode()).hexdigest()
    return dict(sorted(result.items()))


def summarize(report):
    fields=['raw_strict_feasible_fraction','raw_physical_feasible_fraction','raw_ground_fraction','raw_best_physical_objective_gap','seconds']
    rows=[]
    for backend in ['uniform','onehot_uniform','neal']+[f'{family}_p{p}' for family in ('qaoa','qaoa_onehot') for p in CONFIG['depths']]:
        subset=[r for r in report['runs'] if r['backend']==backend]
        row={'backend':backend,'runs':len(subset)}
        for name in fields:
            values=[r[name] for r in subset if r[name] is not None]
            row[name]=float(np.mean(values)) if values else None
        row['raw_physical_sample_runs']=sum(r['raw_physical_sample_present'] for r in subset)
        row['raw_reference_hit_runs']=sum(r['raw_physical_reference_hit'] for r in subset)
        row['repaired_reference_hits']=sum(abs(r['repaired']['objective']-r['reference_objective'])<1e-7 for r in subset)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(output_dir=None,progress=True):
    tick=perf_counter();before=source_hashes();parent_before=parent_input_identity()
    provenance,cases=load_cases();rows=[];case_metrics=[]
    for case,instance in cases:
        compiled=compile_case(instance);model=QUBOStatevector(compiled.bqm,CONFIG['max_qubits']);onehot_model=OneHotQAOA(compiled,CONFIG['max_qubits'])
        strict,physical,objective=masks(compiled,model,instance)
        _,reference=exact_reference(instance);minimum=float(model.energy.min())
        if abs(minimum-reference['objective'])>1e-7:raise ArithmeticError('QUBO ground energy disagrees with independent constrained enumeration')
        ground=np.isclose(model.energy,minimum,rtol=0,atol=1e-8)
        if not strict[ground].all():raise ArithmeticError('A QUBO ground state violates its declared constraints')
        case_metrics.append({'case':case['id'],'qubits':model.n,'pnrs':len(instance.parties),'options':len(instance.options),
            'variable_order':list(model.labels),'pauli_hamiltonian':[(term,float(value.real)) for term,value in model.hamiltonian().to_list()],
            'reference':reference,'incumbent':score(instance,case['incumbent'],SCENARIO),'resources':resources(compiled),
            'strict_feasible_basis_states':int(strict.sum()),'physical_feasible_basis_states':int(physical.sum()),'basis_states':len(model.states)})
        for seed in CONFIG['seeds']:
            started=perf_counter();counts=np.random.default_rng(seed).multinomial(CONFIG['shots'],np.full(len(model.states),1/len(model.states)))
            uniform={int(k):int(counts[k]) for k in np.flatnonzero(counts)}
            metrics=audit_counts(compiled,model,instance,uniform,reference['objective'])
            rows.append({'case':case['id'],'seed':seed,'backend':'uniform','depth':0,**metrics,
                         'reference_objective':reference['objective'],'seconds':perf_counter()-started})
            started=perf_counter();initial_prob=np.abs(onehot_model.initial_state())**2
            initial_prob/=initial_prob.sum()
            initial_counts=np.random.default_rng(seed).multinomial(CONFIG['shots'],initial_prob)
            metrics=audit_counts(compiled,model,instance,{int(k):int(initial_counts[k]) for k in np.flatnonzero(initial_counts)},reference['objective'])
            rows.append({'case':case['id'],'seed':seed,'backend':'onehot_uniform','depth':0,**metrics,
                         'reference_objective':reference['objective'],'seconds':perf_counter()-started})
            started=perf_counter();samples=neal.SimulatedAnnealingSampler().sample(compiled.bqm,num_reads=CONFIG['shots'],num_sweeps=CONFIG['neal_sweeps'],seed=seed)
            counts=basis_counts(samples,model)
            metrics=audit_counts(compiled,model,instance,counts,reference['objective'])
            rows.append({'case':case['id'],'seed':seed,'backend':'neal','depth':0,**metrics,'sweeps':CONFIG['neal_sweeps'],
                         'reference_objective':reference['objective'],'seconds':perf_counter()-started})
            for family,quantum_model in [("qaoa",model),("qaoa_onehot",onehot_model)]:
                previous=None
                for depth in CONFIG['depths']:
                    started=perf_counter()
                    samples=QAOASampler(CONFIG['max_qubits']).sample_prepared(quantum_model,num_reads=CONFIG['shots'],depth=depth,seed=seed,starts=CONFIG['starts'],maxiter=CONFIG['maxiter'],previous=previous)
                    params=np.array(samples.info['parameters']);training=samples.info['training']
                    counts=basis_counts(samples,model);state,_=quantum_model.evolve(params);probabilities=np.abs(state)**2
                    circuit=quantum_model.circuit(params);independent=np.asarray(Statevector.from_instruction(circuit));state,_=quantum_model.evolve(params)
                    fidelity=float(abs(np.vdot(independent,state))**2)
                    if abs(fidelity-1)>1e-9:raise ArithmeticError('Optimized QAOA circuit disagrees with statevector engine')
                    translated=transpile(circuit,basis_gates=['h','rz','rx','cx'],optimization_level=1,seed_transpiler=seed)
                    metrics=audit_counts(compiled,model,instance,counts,reference['objective'])
                    rows.append({'case':case['id'],'seed':seed,'backend':f'{family}_p{depth}','mixer':quantum_model.mixer_name,'depth':depth,**metrics,
                        'reference_objective':reference['objective'],'training':training,'circuit_fidelity':fidelity,
                        'exact_ground_probability':float(probabilities[ground].sum()),
                        'exact_physical_feasible_probability':float(probabilities[physical].sum()),
                        'gate_counts':{str(k):int(v) for k,v in circuit.count_ops().items()},
                        'all_to_all_transpiled_depth':int(translated.depth()),
                        'all_to_all_transpiled_cx':int(translated.count_ops().get('cx',0)),
                        'seconds':perf_counter()-started})
                    previous=params
        if progress:print(f"{case['id']}: {model.n} qubits, exact reference + random/SA/QAOA depths {CONFIG['depths']} completed",flush=True)
    # Exercise the extension's global admission gate on one predeclared seed.
    # Each patch uses the same anchor independently; no double counting of cases.
    from experiment import load_experiment_catalogs
    _,full,_,_,_=load_experiment_catalogs()
    anchor_frame=pd.read_csv(PARENT/'results/assignments.csv.gz',dtype='string')
    anchor={str(p):str(o) for p,o in zip(anchor_frame.PNR,anchor_frame.seed_7) if not pd.isna(o)}
    global_audits=[]
    for case,instance in cases:
        for method in ['uniform','onehot_uniform','neal']+[f'{family}_p{p}' for family in ('qaoa','qaoa_onehot') for p in CONFIG['depths']]:
            result=next(r for r in rows if r['case']==case['id'] and r['seed']==7 and r['backend']==method)
            patched,audit=apply_global_patch(full,anchor,case,result['repaired_assignment'])
            final=score(full,patched,SCENARIO)
            if final['capacity_violations']:raise ArithmeticError('Global QAOA patch audit failed')
            global_audits.append({'case':case['id'],'backend':method,'seed':7,**audit,'admitted':final})
    report={'status':'passed','source_hashes':before,'parent_identity':parent_before,'config':CONFIG,'case_provenance':provenance,
        'cases':case_metrics,'runs':rows,'global_patch_audits':global_audits,'summary':None,
        'scope':'Four structurally selected actual residual neighbourhoods. Local allocation is not a replacement estimate for full-population 72.19% allocation. No hardware, noise, speedup or industry-outperformance claim.',
        'fairness':'Identical QUBOs, residual capacities, final sample counts and repair per case. One-hot initial-state controls separate subspace preparation from variational optimization. QAOA training uses exact expectations and extra classical optimization; compute/physical-shot budgets are not equated.',
        'hardware_resources':'Qiskit gate counts and transpilation assume all-to-all connectivity; no device routing, calibration or hardware execution.'}
    report['summary']=summarize(report).to_dict('records')
    if before!=source_hashes() or parent_before!=parent_input_identity():raise RuntimeError('Benchmark inputs changed during execution')
    if output_dir is not None:
        output=Path(output_dir).resolve()
        if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):raise ValueError('Write generated evidence under artifacts/ or outside the project')
        output.mkdir(parents=True,exist_ok=True)
        plot(report,output/'qaoa_comparison.svg')
        example_row,example_instance=cases[0]
        example_model=OneHotQAOA(compile_case(example_instance),CONFIG['max_qubits'])
        example=next(r for r in rows if r['case']==example_row['id'] and r['seed']==7 and r['backend']=='qaoa_onehot_p1')
        qc=transpile(example_model.circuit(example['training']['parameters']),basis_gates=['h','rz','rx','cx'],optimization_level=1,seed_transpiler=7)
        exported=qasm2.dumps(qc)
        loaded=Statevector.from_instruction(qasm2.loads(exported));expected,_=example_model.evolve(example['training']['parameters'])
        if abs(abs(np.vdot(loaded,expected))**2-1)>1e-9:raise ArithmeticError('Exported QASM circuit changed the QAOA state')
        (output/'example_qaoa.qasm').write_text(exported+'\n')
    report['timing']={'end_to_end_seconds':perf_counter()-tick,'python':platform.python_version(),'system':platform.system(),
        'architecture':platform.machine(),'requested_blas_threads':os.environ.get('OPENBLAS_NUM_THREADS','unspecified'),
        'scope':'Frozen-case loading, all exact/random/SA/QAOA runs, optimization, Qiskit verification, audits and plot; excludes installation, kernel startup and one-time case generation'}
    if output_dir is not None:(output/'qaoa_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full_table=summarize(report)
    selected=['uniform','onehot_uniform','neal','qaoa_p1','qaoa_p5','qaoa_onehot_p1','qaoa_onehot_p5']
    table=full_table.set_index('backend').loc[selected].reset_index()
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));x=np.arange(len(table))
    axes[0].bar(x-.18,table.raw_physical_feasible_fraction*100,width=.36,label='Physical feasibility',color='#246b84')
    axes[0].bar(x+.18,table.raw_ground_fraction*100,width=.36,label='Exact QUBO ground state',color='#b36526')
    axes[0].set_xticks(x,['Uniform','1-hot init','SA','X p1','X p5','1-hot p1','1-hot p5'],rotation=40,ha='right')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Observed raw samples (%)')
    axes[0].set_title('Before common repair; 1,024 samples/run');axes[0].legend(fontsize=8)
    for family,label,color in [('qaoa','X mixer','#246b84'),('qaoa_onehot','One-hot projector','#b36526')]:
        values=[np.mean([r['exact_ground_probability'] for r in report['runs'] if r['backend']==f'{family}_p{p}'])*100 for p in CONFIG['depths']]
        axes[1].plot(CONFIG['depths'],values,'o-',label=label,color=color)
    axes[1].set_xticks(CONFIG['depths']);axes[1].set_xlabel('QAOA depth');axes[1].set_ylabel('Mean exact ground-state probability (%)')
    axes[1].set_title('Ideal statevector; four cases, five seeds');axes[1].legend(fontsize=8)
    fig.suptitle('Same airline QUBOs • QAOA, thermal SA and exact references');fig.tight_layout();fig.savefig(path);plt.close(fig)
    path=Path(path)
    if path.suffix=='.svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/experiment')
    args=parser.parse_args();report=run_experiment(args.output_dir);print(summarize(report).to_string(index=False))
