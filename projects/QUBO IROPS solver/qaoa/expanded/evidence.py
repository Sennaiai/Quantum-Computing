"""Replay stored scientific evidence without repeating parameter optimization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from qiskit import qasm2,transpile
from bootstrap import PARENT
from case_catalog import parent_input_identity,SCENARIO
from qaoa_experiment import source_hashes as baseline_source_hashes
from measurements import compile_case
from expanded.catalog import load_expanded_cases
from expanded.experiment import source_hashes,CONFIG,METHODS,global_audit
from expanded.subspace import SubspaceQAOA
from expanded.circuit_audit import verify
from expanded.pareto import exact_tradeoffs
from expanded.measurements import DrawAudit,allocation_summary,paired_S3_comparison


def compare(a,b,path=''):
    if isinstance(a,float):
        if not np.isclose(a,b,rtol=1e-10,atol=1e-8):raise AssertionError((path,a,b))
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path)
        for key in a:compare(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError(path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif a!=b:raise AssertionError((path,a,b))


def audit_evidence(directory,replay_global=True):
    directory=Path(directory);report=json.loads((directory/'expanded_results.json').read_text())
    assert report['status']=='passed' and report['config']==CONFIG
    assert report['source_hashes']==source_hashes()
    assert report['baseline_source_hashes']==baseline_source_hashes()
    assert report['parent_identity']==parent_input_identity()
    provenance,cases=load_expanded_cases();assert report['provenance']==provenance
    models={};references={};inspectors={}
    for case,instance in cases:
        compiled=compile_case(instance);model=SubspaceQAOA(compiled);reference=exact_tradeoffs(instance)
        models[case['id']]=model;references[case['id']]=reference;inspectors[case['id']]=DrawAudit(compiled,model,instance,reference)
        saved=next(x for x in report['cases'] if x['case']==case['id']);compare(reference,saved['reference'])
        assert abs(model.energy.min()-reference['minimum_S3'])<1e-7
    expected={(case['id'],seed,method) for case,instance in cases for seed in CONFIG['seeds'] for method in METHODS}
    assert {(r['case'],r['seed'],r['backend']) for r in report['runs']}==expected
    assert len(report['runs'])==len(expected)==144
    circuits=0
    for run in report['runs']:
        counts={int(k):n for k,n in run['sample_counts'].items()}
        metrics=inspectors[run['case']].evaluate(counts);compare(metrics,{k:run[k] for k in metrics})
        assert metrics['shots']==CONFIG['shots'] and metrics['repaired']['capacity_violations']==0
        if run['backend'].startswith('onechoice_QAOA'):
            model=models[run['case']];params=run['training']['parameters']
            compare(model.physical_energy(params),run['training']['expected_energy'])
            compare(verify(model,params),run['circuit_block_fidelity']);circuits+=1
    assert circuits==90
    compare(allocation_summary(report),report['summary']);compare(paired_S3_comparison(report),report['paired_S3_comparison'])
    if replay_global:
        from experiment import load_experiment_catalogs
        _,full,_,_,_=load_experiment_catalogs()
        frame=pd.read_csv(PARENT/'results/assignments.csv.gz',dtype='string')
        anchor={str(p):str(o) for p,o in zip(frame.PNR,frame.seed_7) if not pd.isna(o)}
        case_by_id={c['id']:c for c,i in cases}
        assert len(report['global_patch_audits'])==144
        for run,saved in zip(report['runs'],report['global_patch_audits']):
            compare(global_audit(full,anchor,case_by_id[run['case']],run),saved)
            for key in ('S3_selected','Pareto_selected'):
                a=saved[key];assert a['admitted']['capacity_violations']==0
                assert a['admitted']['assigned_pnrs']>=a['before']['assigned_pnrs']
                assert a['admitted']['assigned_passengers']>=a['before']['assigned_passengers']
                assert a['admitted']['objective']<=a['before']['objective']+1e-8
    example=next(r for r in report['runs'] if r['case']=='rich_10' and r['seed']==7 and r['backend']=='onechoice_QAOA_p1')
    model=models['rich_10'];expected_qc=transpile(model.circuit(example['training']['parameters']),basis_gates=['h','rz','rx','cx'],optimization_level=1,seed_transpiler=7)
    loaded=qasm2.loads((directory/'example_large_qaoa.qasm').read_text());assert len(loaded.data)==len(expected_qc.data)
    for a,b in zip(expected_qc.data,loaded.data):
        assert a.operation.name==b.operation.name
        assert [expected_qc.find_bit(q).index for q in a.qubits]==[loaded.find_bit(q).index for q in b.qubits]
        np.testing.assert_allclose(a.operation.params,b.operation.params,atol=1e-12)
    return {'status':'passed','sample_batches_replayed':144,'circuit_block_replays':circuits,'global_patch_audits':144 if replay_global else 0}
