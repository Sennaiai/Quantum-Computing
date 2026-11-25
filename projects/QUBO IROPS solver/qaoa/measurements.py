"""Audit raw binary samples, common repair, and optional global patch admission."""
from hashlib import sha256
import json
import numpy as np
from bootstrap import PARENT
from solver import reconcile,score,loads_for
from qubo import compile_bqm
from case_catalog import SCENARIO


def compile_case(instance):
    return compile_bqm(instance,[p.id for p in instance.parties],instance.options,instance.capacities,SCENARIO)


def masks(compiled,model,instance):
    strict=np.ones(len(model.states),bool);onehot=np.ones(len(model.states),bool)
    for j,(coeff,constant) in enumerate(compiled.constraints):
        residual=np.full(len(model.states),constant,dtype=np.int64)
        for label,value in coeff.items():residual += int(value)*model.bits[:,model.index[label]].astype(np.int64)
        strict &= residual==0
        if j<len(instance.parties):onehot &= residual==0
    physical=onehot.copy()
    for key,capacity in instance.capacities.items():
        used=np.zeros(len(model.states),dtype=np.int64)
        for label,option in compiled.option_variables.items():
            if key in option.legs:used += instance.party_by_id[option.party].pax*model.bits[:,model.index[label]].astype(np.int64)
        physical &= used<=capacity
    objective=np.zeros(len(model.states))
    for label,value in compiled.objective.items():objective += value*model.bits[:,model.index[label]]
    if np.any(strict & ~physical):raise ArithmeticError('QUBO feasibility failed physical accounting')
    return strict,physical,objective


def audit_counts(compiled,model,instance,counts,reference_objective):
    if not counts or any(not isinstance(k,int) or k<0 or k>=len(model.states) or not isinstance(v,int) or v<=0 for k,v in counts.items()):
        raise ValueError('Invalid sampled basis counts')
    strict,physical,objective=masks(compiled,model,instance);shots=sum(counts.values())
    minimum=float(model.energy.min());ground=np.isclose(model.energy,minimum,rtol=0,atol=1e-8)
    best=min(counts,key=lambda k:(model.energy[k],k));sample=model.sample_dict(best)
    compiled.validate_sample(sample)
    if abs(compiled.semantic_energy(sample)-model.energy[best])>max(1e-8,compiled.roundoff_bound(sample)):
        raise ArithmeticError('Circuit basis energy disagrees with semantic QUBO')
    physically_valid=[k for k in counts if physical[k]]
    chosen=min(physically_valid,key=lambda k:(objective[k],model.energy[k],k)) if physically_valid else best
    selected_sample=model.sample_dict(chosen)
    decoded=compiled.decode(selected_sample)
    proposed={p:min(ids,key=lambda oid:(instance.option_by_id[oid].cost(SCENARIO),oid)) for p,ids in decoded.items()}
    repaired=reconcile(instance,proposed,SCENARIO,improve=False)
    audited=score(instance,repaired,SCENARIO)
    if audited['capacity_violations']:raise ArithmeticError('Common sample repair failed physical audit')
    observed=[objective[k] for k in counts if physical[k]]
    feasible_best=float(min(observed)) if observed else None
    return {'shots':shots,'raw_best_encoded_energy':float(model.energy[best]),
        'raw_strict_feasible_fraction':sum(n for k,n in counts.items() if strict[k])/shots,
        'raw_physical_feasible_fraction':sum(n for k,n in counts.items() if physical[k])/shots,
        'raw_ground_fraction':sum(n for k,n in counts.items() if ground[k])/shots,
        'raw_best_physical_objective':feasible_best,
        'raw_best_physical_objective_gap':feasible_best-reference_objective if observed else None,
        'raw_best_multiple_option_pnrs':sum(len(v)>1 for v in compiled.decode(sample).values()),
        'raw_best_onehot_violations':sum(constant+sum(int(a)*sample[v] for v,a in coeff.items())!=0 for coeff,constant in compiled.constraints[:len(instance.parties)]),
        'raw_best_decoded_capacity_violations':score(instance,{p:min(ids,key=lambda oid:(instance.option_by_id[oid].cost(SCENARIO),oid)) for p,ids in compiled.decode(sample).items()},SCENARIO)['capacity_violations'],
        'raw_physical_sample_present':bool(physically_valid),
        'raw_physical_reference_hit':bool(observed and min(observed)<=reference_objective+1e-7),
        'selection_policy':'Best physically valid observed assignment by S3 objective; fallback to best encoded-energy sample; identical deterministic repair for all samplers',
        'repaired_assignment':repaired,'repaired':audited,
        'sample_counts':{str(k):v for k,v in sorted(counts.items())}}


def apply_global_patch(full,anchor,row,local_assignment):
    ids={p['id'] for p in row['instance']['parties']}
    if not set(local_assignment)<=ids:raise ValueError('Patch includes a PNR outside its neighbourhood')
    outside={p:o for p,o in anchor.items() if p not in ids};outside_loads=loads_for(full,outside)
    digest=sha256(json.dumps(sorted((f,c,n) for (f,c),n in outside_loads.items()),separators=(',',':')).encode()).hexdigest()
    if digest!=row['outside_loads_sha256']:raise ValueError('Stale residual context; rebuild neighbourhood before applying patch')
    allowed={o['id'] for o in row['instance']['options']}
    if not set(local_assignment.values())<=allowed:raise ValueError('Patch uses an option outside its compiled QUBO')
    candidate={**outside,**local_assignment};before=score(full,anchor,SCENARIO);after=score(full,candidate,SCENARIO)
    if after['capacity_violations']:raise ValueError('Patch exceeds global inventory')
    accepted=(after['assigned_pnrs']>=before['assigned_pnrs'] and after['assigned_passengers']>=before['assigned_passengers']
              and after['objective']<before['objective']-1e-9)
    return (candidate if accepted else dict(anchor)),{'accepted':accepted,'before':before,'candidate':after,
        'scope':'One independently audited local patch against fixed outside bookings; independent cases cannot be summed'}
