"""Larger residual neighbourhoods selected by structure, before solver outcomes."""
from collections import defaultdict,Counter
from pathlib import Path
import gzip,hashlib,json,math
import pandas as pd
from bootstrap import PARENT
from case_catalog import SCENARIO,parent_input_identity,instance_from_payload
from data_contract import Instance,instance_payload,instance_fingerprint
from experiment import load_experiment_catalogs
from solver import catalog,loads_for,score
from qubo import compile_bqm,estimate_bqm_variables

SPECS=((4,16,18),(5,19,20),(6,21,22))
MAX_SUBSPACE=262144
ROOT=Path(__file__).resolve().parent


def build_expanded_cases():
    _,full,_,_,_=load_experiment_catalogs()
    frame=pd.read_csv(PARENT/'results/assignments.csv.gz',dtype='string')
    anchor={str(p):str(o) for p,o in zip(frame.PNR,frame.seed_7) if not pd.isna(o)}
    choices=catalog(full,SCENARIO);parties=full.party_by_id;options=full.option_by_id
    loads=loads_for(full,anchor);occupants=defaultdict(set);roots_by_leg=defaultdict(set)
    unassigned=sorted(set(parties)-set(anchor))
    for p,oid in anchor.items():
        for key in options[oid].legs:occupants[key].add(p)
    for p in unassigned:
        for o in choices[p][:5]:
            for key in o.legs:roots_by_leg[key].add(p)
    picked=[];used=set();examined=0;hist=Counter()
    for number,lo,hi in SPECS:
        found=None
        for root in unassigned:
            if root in used:continue
            for proposed in choices[root][:5]:
                neighbours=set()
                for key in proposed.legs:neighbours.update(occupants[key]);neighbours.update(roots_by_leg[key])
                neighbours-=used|{root}
                # Include another unassigned root where connected; then release blockers.
                other_roots=sorted((p for p in neighbours if p not in anchor),key=lambda p:(parties[p].pax,parties[p].cvm,p))
                blockers=sorted((p for p in neighbours if p in anchor),key=lambda p:(parties[p].pax,parties[p].cvm,p))
                ordered=other_roots[:1]+blockers+other_roots[1:]
                ids=[root]+ordered[:number-1]
                if len(ids)!=number:continue
                outside_loads=dict(loads)
                for p in ids:
                    if p in anchor:
                        for key in options[anchor[p]].legs:outside_loads[key]-=parties[p].pax
                outside_loads={k:v for k,v in outside_loads.items() if v}
                residual={k:v-outside_loads.get(k,0) for k,v in full.capacities.items()}
                selected=[]
                for p in ids:
                    valid=[o for o in choices[p] if all(parties[p].pax<=residual[k] for k in o.legs)]
                    keep=[]
                    if p in anchor:keep.append(options[anchor[p]])
                    if p==root and proposed in valid:keep.append(proposed)
                    for o in valid:
                        if o not in keep:keep.append(o)
                        if len(keep)>=2:break
                    selected.extend(keep[:2])
                if not any(o.party==root for o in selected):continue
                keys=sorted({k for o in selected for k in o.legs})
                local=Instance(tuple(parties[p] for p in ids),tuple(selected),{k:residual[k] for k in keys},'expanded_actual_neighbourhood')
                n=estimate_bqm_variables(local,ids,local.options,local.capacities);examined+=1;hist[(number,n)]+=1
                if not lo<=n<=hi:continue
                compiled=compile_bqm(local,ids,local.options,local.capacities,SCENARIO)
                if not compiled.slack:continue
                count=math.prod(1+sum(o.party==p for o in selected) for p in ids)*2**sum(len(x) for x in compiled.slack.values())
                if count>MAX_SUBSPACE:continue
                incumbent={p:anchor[p] for p in ids if p in anchor}
                if score(local,incumbent,SCENARIO)['capacity_violations']:raise ArithmeticError('Invalid residual incumbent')
                found={'id':f'expanded_{len(picked)+1:02}','root_pnr':root,'instance':instance_payload(local),'incumbent':incumbent,
                    'qubits':n,'subspace_states':count,'outside_loads_sha256':hashlib.sha256(json.dumps(sorted((f,c,n) for (f,c),n in outside_loads.items()),separators=(',',':')).encode()).hexdigest()}
                break
            if found:break
        if found is None:raise ValueError(f'No structural case for {number} PNRs and {lo}-{hi} qubits; sizes={dict(hist)}')
        picked.append(found);used.update(p['id'] for p in found['instance']['parties'])
    rich,rich_examined=build_rich_cases(full,anchor,choices,occupants,loads,used)
    picked.extend(rich)
    return {'schema_version':1,'parent_identity':parent_input_identity(),'expanded_fingerprint':instance_fingerprint(full),'anchor_seed':7,'scenario':SCENARIO,
        'selection':'First disjoint case per declared (PNRs, qubit band): (4,16–18),(5,19–20),(6,21–22). Roots sorted by ID; first five S3 routes; connected neighbours ordered unassigned first, then party size/CVM/ID; at most two options per PNR including incumbent. No solver or exact-reference outcomes used.',
        'rich_selection':'Additional first disjoint cases with (PNRs, max options per PNR)=(6,4),(8,3),(10,2), 23–36 logical qubits and reachable dimension <=262144. Assigned occupants come from the root first five S3 routes and a second hop through their first three alternatives. Include the first different-party-size neighbour where available, then party size/CVM/ID order. No solver/reference outcomes used; initial three controls retained.',
        'candidate_neighbourhoods_examined':examined,'rich_candidates_examined':rich_examined,'max_subspace_states':MAX_SUBSPACE,'cases':picked,
        'scope':'Outside bookings fixed; cases use independent copies of the same incumbent and cannot be summed.'}


def build_rich_cases(full,anchor,choices,occupants,loads,used):
    parties=full.party_by_id;options=full.option_by_id;used=set(used);picked=[];examined=0;hist=Counter()
    for number,top_k in ((6,4),(8,3),(10,2)):
        found=None
        for root in sorted(set(parties)-set(anchor)-used):
            neighbours=set()
            for o in choices[root][:5]:
                for key in o.legs:neighbours.update(occupants[key])
            first=sorted(neighbours,key=lambda p:(parties[p].pax,parties[p].cvm,p))
            for p in first:
                for o in choices[p][:3]:
                    for key in o.legs:neighbours.update(occupants[key])
            neighbours-=used|{root}
            ordered=sorted(neighbours,key=lambda p:(parties[p].pax,parties[p].cvm,p))
            diverse=next((p for p in ordered if parties[p].pax!=parties[root].pax),None)
            if diverse is not None:ordered=[diverse]+[p for p in ordered if p!=diverse]
            ids=[root]+ordered[:number-1]
            if len(ids)!=number:continue
            outside_loads=dict(loads)
            for p in ids:
                if p in anchor:
                    for key in options[anchor[p]].legs:outside_loads[key]-=parties[p].pax
            outside_loads={k:v for k,v in outside_loads.items() if v}
            residual={k:v-outside_loads.get(k,0) for k,v in full.capacities.items()}
            selected=[]
            for p in ids:
                valid=[o for o in choices[p] if all(parties[p].pax<=residual[k] for k in o.legs)]
                keep=[options[anchor[p]]] if p in anchor else []
                for o in valid:
                    if o not in keep:keep.append(o)
                    if len(keep)>=top_k:break
                selected.extend(keep[:top_k])
            if not any(o.party==root for o in selected):continue
            keys=sorted({k for o in selected for k in o.legs})
            local=Instance(tuple(parties[p] for p in ids),tuple(selected),{k:residual[k] for k in keys},'rich_actual_neighbourhood')
            n=estimate_bqm_variables(local,ids,local.options,local.capacities);examined+=1;hist[(number,n)]+=1
            if not 23<=n<=36:continue
            compiled=compile_bqm(local,ids,local.options,local.capacities,SCENARIO)
            dim=math.prod(1+sum(o.party==p for o in selected) for p in ids)*2**sum(len(x) for x in compiled.slack.values())
            if dim>MAX_SUBSPACE:continue
            incumbent={p:anchor[p] for p in ids if p in anchor}
            found={'id':f'rich_{number:02}','root_pnr':root,'instance':instance_payload(local),'incumbent':incumbent,
                'qubits':n,'subspace_states':dim,'outside_loads_sha256':hashlib.sha256(json.dumps(sorted((f,c,n) for (f,c),n in outside_loads.items()),separators=(',',':')).encode()).hexdigest()}
            break
        if found is None:raise ValueError(f'No rich structural case for {number} PNRs; sizes={dict(hist)}')
        picked.append(found);used.update(p['id'] for p in found['instance']['parties'])
    return picked,examined


def load_expanded_cases():
    path=ROOT/'data/expanded_neighbourhoods.json.gz'
    meta=json.loads((ROOT/'data/expanded_provenance.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Expanded case checksum mismatch')
    value=json.loads(gzip.decompress(path.read_bytes()))
    if value['parent_identity']!=parent_input_identity():raise ValueError('Expanded parent input identity changed')
    return value,[(row,instance_from_payload(row['instance'])) for row in value['cases']]


if __name__=='__main__':
    value=build_expanded_cases();(ROOT/'data').mkdir(exist_ok=True);path=ROOT/'data/expanded_neighbourhoods.json.gz'
    path.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    (ROOT/'data/expanded_provenance.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selection':value['selection'],'source':'Authorized de-identified actual catalog and published seed-7 incumbent'},indent=2)+'\n')
    print(json.dumps({'cases':[{k:r[k] for k in ('id','qubits','subspace_states')}|{'pnrs':len(r['instance']['parties'])} for r in value['cases']],'examined':value['candidate_neighbourhoods_examined']},indent=2))
