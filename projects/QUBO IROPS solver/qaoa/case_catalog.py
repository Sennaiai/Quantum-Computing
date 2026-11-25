"""Deterministic, outcome-blind actual-data neighbourhoods for QAOA/SA comparison."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import gzip
import hashlib
import json
import pandas as pd
from bootstrap import ROOT, PARENT
from data_contract import (Party, Option, Instance, FEATURES, utc, instance_payload, instance_fingerprint)
from experiment import load_experiment_catalogs, source_hashes as parent_source_hashes
from solver import catalog, loads_for, score
from qubo import compile_bqm

SCENARIO = 'S3_capacityStability'
BANDS = [(6,8),(9,10),(11,12),(13,16)]


def instance_from_payload(data):
    return Instance(tuple(Party(p['id'],p['pax'],p['cvm']) for p in data['parties']),
        tuple(Option(o['id'],o['party'],tuple(tuple(k) for k in o['legs']),utc(o['departure']),o['delay_minutes'],
                     tuple(o['features'][name] for name in FEATURES)) for o in data['options']),
        {(r['flight'],r['cabin']):r['seats'] for r in data['capacities']},data['name'])


def parent_input_identity():
    return {'scientific_sources':parent_source_hashes(),
        'assignments_sha256':hashlib.sha256((PARENT/'results/assignments.csv.gz').read_bytes()).hexdigest(),
        'results_sha256':hashlib.sha256((PARENT/'results/actual_catalog_results.json').read_bytes()).hexdigest()}


def build_cases():
    _,full,network,_,_=load_experiment_catalogs()
    frame=pd.read_csv(PARENT/'results/assignments.csv.gz',dtype='string')
    anchor={str(p):str(o) for p,o in zip(frame.PNR,frame.seed_7) if not pd.isna(o)}
    choices=catalog(full,SCENARIO); options=full.option_by_id; parties=full.party_by_id
    loads=loads_for(full,anchor); occupants=defaultdict(set)
    for pid,oid in anchor.items():
        for key in options[oid].legs:occupants[key].add(pid)
    picked={}; used_parties=set(); examined=0
    for root in sorted(set(parties)-set(anchor)):
        if root in used_parties:continue
        for proposed in choices[root][:5]:
            blocked=[k for k in proposed.legs if loads.get(k,0)+parties[root].pax>full.capacities[k]]
            offenders=sorted(set().union(*(occupants[k] for k in blocked))-used_parties,
                             key=lambda p:(parties[p].pax,parties[p].cvm,p)) if blocked else []
            for number in (1,2):
                ids=tuple([root]+offenders[:number])
                if len(ids)<2 or any(p in used_parties for p in ids):continue
                outside=dict(anchor)
                for pid in ids:outside.pop(pid,None)
                outside_loads=loads_for(full,outside)
                remaining={k:v-outside_loads.get(k,0) for k,v in full.capacities.items()}
                selected=[]
                for pid in ids:
                    valid=[o for o in choices[pid] if all(parties[pid].pax<=remaining[k] for k in o.legs)]
                    keep=[]
                    if pid in anchor:keep.append(options[anchor[pid]])
                    if pid==root and proposed in valid:keep.append(proposed)
                    for o in valid:
                        if o not in keep:keep.append(o)
                        if len(keep)>=2:break
                    selected.extend(keep[:2])
                if not any(o.party==root for o in selected):continue
                keys=sorted({k for o in selected for k in o.legs})
                local=Instance(tuple(parties[p] for p in ids),tuple(selected),{k:remaining[k] for k in keys},'actual_recovery_neighbourhood')
                compiled=compile_bqm(local,ids,local.options,local.capacities,SCENARIO);examined+=1
                if not compiled.slack or compiled.bqm.num_interactions==0:continue
                band=next((j for j,(lo,hi) in enumerate(BANDS) if lo<=compiled.bqm.num_variables<=hi),None)
                if band is None or band in picked:continue
                incumbent={p:anchor[p] for p in ids if p in anchor}
                if score(local,incumbent,SCENARIO)['capacity_violations']:raise ArithmeticError('Invalid local incumbent')
                picked[band]={'id':f'actual_{band+1:02}','root_pnr':root,'instance':instance_payload(local),
                    'incumbent':incumbent,'qubits':compiled.bqm.num_variables,
                    'outside_loads_sha256':hashlib.sha256(json.dumps(sorted((f,c,n) for (f,c),n in outside_loads.items()),separators=(',',':')).encode()).hexdigest()}
                used_parties.update(ids)
                break
            if root in used_parties:break
        if len(picked)==len(BANDS):break
    if len(picked)!=len(BANDS):raise ValueError('Not all predeclared structural size bands were found')
    return {'schema_version':1,'parent_identity':parent_input_identity(),'expanded_fingerprint':instance_fingerprint(full),
        'scenario':SCENARIO,'anchor_seed':7,'selection':'First disjoint-PNR case in each 6–8/9–10/11–12/13–16-qubit band; root PNR IDs ascending, first five S3 routes, first one/two blocking occupants by party size/CVM/ID. No QAOA or exact-solution outcomes used in selection.',
        'candidate_neighbourhoods_examined':examined,'cases':[picked[j] for j in sorted(picked)],
        'scope':'Actual expanded-catalog local residual problems with every outside booking fixed; independent cases are not additive global allocations.'}


def load_cases():
    path=ROOT/'data/neighbourhoods.json.gz'; manifest=json.loads((ROOT/'data/provenance.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('QAOA case checksum mismatch')
    data=json.loads(gzip.decompress(path.read_bytes()))
    if data['parent_identity']!=parent_input_identity():raise ValueError('Parent scientific inputs changed; regenerate and revalidate neighbourhoods')
    return data,[(row,instance_from_payload(row['instance'])) for row in data['cases']]


if __name__=='__main__':
    data=build_cases();path=ROOT/'data/neighbourhoods.json.gz'
    path.write_bytes(gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    (ROOT/'data/provenance.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source':'Authorized public de-identified actual catalog and published seed-7 incumbent; no synthetic main experiment',
        'selection':data['selection'],'expanded_fingerprint':data['expanded_fingerprint']},indent=2)+'\n')
    print(json.dumps({'case_count':len(data['cases']),'qubits':[row['qubits'] for row in data['cases']],
                      'examined':data['candidate_neighbourhoods_examined']},indent=2))
