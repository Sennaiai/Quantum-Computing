"""Exact local feasible tradeoffs; no reference solutions are used by training."""
from itertools import product
import math
from solver import score
from case_catalog import SCENARIO


def dominates(a,b):
    return (a['assigned_pnrs']>=b['assigned_pnrs'] and a['assigned_passengers']>=b['assigned_passengers']
        and a['objective']<=b['objective']+1e-9
        and (a['assigned_pnrs']>b['assigned_pnrs'] or a['assigned_passengers']>b['assigned_passengers'] or a['objective']<b['objective']-1e-9))


def point(instance,assignment):
    result=score(instance,assignment,SCENARIO)
    if result['capacity_violations']:raise ValueError('A Pareto point must be physically feasible')
    return {'assignment':dict(sorted(assignment.items())),'assigned_pnrs':result['assigned_pnrs'],
        'assigned_passengers':result['assigned_passengers'],'objective':result['objective']}


def frontier(points):
    # Equal metrics retain one deterministic representative, not duplicate draws.
    unique={}
    for value in sorted(points,key=lambda p:tuple(sorted(p['assignment'].items()))):
        key=(value['assigned_pnrs'],value['assigned_passengers'])
        if key not in unique or value['objective']<unique[key]['objective']:unique[key]=value
    values=list(unique.values())
    return sorted([p for p in values if not any(dominates(q,p) for q in values)],
        key=lambda p:(-p['assigned_pnrs'],-p['assigned_passengers'],p['objective'],tuple(p['assignment'].items())))


def exact_tradeoffs(instance,max_combinations=2000000):
    eligible,_=instance.eligible();groups=[[None]+[o for o in eligible if o.party==p.id] for p in instance.parties]
    combinations=math.prod(map(len,groups))
    if combinations>max_combinations:raise ValueError('Exact frontier enumeration exceeds budget')
    points=[]
    for values in product(*groups):
        assignment={p.id:o.id for p,o in zip(instance.parties,values) if o is not None}
        audit=score(instance,assignment,SCENARIO)
        if audit['capacity_violations']==0:points.append(point(instance,assignment))
    minimum=min(p['objective'] for p in points)
    return {'combinations':combinations,'feasible_assignments':len(points),'frontier':frontier(points),'minimum_S3':minimum,
        'maximum_pnrs':max(p['assigned_pnrs'] for p in points),'maximum_passengers':max(p['assigned_passengers'] for p in points),
        'scope':'Exact only for this restricted local option catalog with outside bookings fixed; PNR/passenger maxima may differ.'}


def coverage(observed,reference):
    return sum(any(p['assigned_pnrs']==q['assigned_pnrs'] and p['assigned_passengers']==q['assigned_passengers']
        and abs(p['objective']-q['objective'])<1e-8 for p in observed) for q in reference)
