"""Sample accounting without a dense full-Hilbert-space feasibility mask."""
import numpy as np
from solver import score,reconcile
from case_catalog import SCENARIO
from expanded.pareto import point,frontier,coverage


class DrawAudit:
    def __init__(self,compiled,model,instance,reference):
        self.compiled=compiled;self.model=model;self.instance=instance;self.reference=reference;self.cache={}

    def inspect(self,basis):
        if basis in self.cache:return self.cache[basis]
        sample=self.model.basis_sample(basis);compiled=self.compiled
        residual=[c+sum(a*sample[v] for v,a in coeff.items()) for coeff,c in compiled.constraints]
        onechoice=not any(residual[:len(self.instance.parties)])
        decoded=compiled.decode(sample)
        assignment={p:min(ids,key=lambda oid:(self.instance.option_by_id[oid].cost(SCENARIO),oid)) for p,ids in decoded.items()}
        audit=score(self.instance,assignment,SCENARIO)
        physical=onechoice and audit['capacity_violations']==0
        energy=float(compiled.bqm.energy(sample));semantic=compiled.semantic_energy(sample)
        if abs(energy-semantic)>max(1e-8,compiled.roundoff_bound(sample)):raise ArithmeticError('QUBO sample energy identity failed')
        value={'energy':energy,'strict':not any(residual),'physical':physical,'assignment':assignment,
            'point':point(self.instance,assignment) if physical else None}
        self.cache[basis]=value;return value

    def evaluate(self,counts):
        if not counts or any(not isinstance(k,int) or not isinstance(v,int) or v<1 for k,v in counts.items()):raise ValueError('Invalid sample counts')
        inspected={k:self.inspect(k) for k in counts};valid=[k for k in counts if inspected[k]['physical']]
        best=min(valid,key=lambda k:(inspected[k]['point']['objective'],inspected[k]['energy'],k)) if valid else min(counts,key=lambda k:(inspected[k]['energy'],k))
        repaired=reconcile(self.instance,inspected[best]['assignment'],SCENARIO,improve=False)
        final=score(self.instance,repaired,SCENARIO)
        if final['capacity_violations']:raise ArithmeticError('Common repair produced invalid assignment')
        observed=frontier([inspected[k]['point'] for k in valid]);minimum=self.reference['minimum_S3']
        return {'shots':sum(counts.values()),'valid_candidate_draws':sum(v for k,v in counts.items() if inspected[k]['physical']),
            'strict_encoding_draws':sum(v for k,v in counts.items() if inspected[k]['strict']),
            'exact_QUBO_ground_draws':sum(v for k,v in counts.items() if abs(inspected[k]['energy']-minimum)<1e-7),
            'raw_local_S3_optimum_hit':any(abs(inspected[k]['point']['objective']-minimum)<1e-8 for k in valid),
            'raw_best_S3':min((inspected[k]['point']['objective'] for k in valid),default=None),
            'observed_frontier':observed,'exact_local_frontier_points_found':coverage(observed,self.reference['frontier']),
            'exact_local_frontier_points':len(self.reference['frontier']),'repaired_assignment':repaired,'repaired':final,
            'sample_counts':{str(k):int(v) for k,v in sorted(counts.items())},
            'selection_policy':'Best observed physically valid S3 candidate, fallback best encoded energy; same deterministic repair as baseline'}


def allocation_summary(report):
    rows=[]
    for method in report['methods']:
        runs=[r for r in report['runs'] if r['backend']==method]
        patches=[p for p in report['global_patch_audits'] if p['backend']==method]
        rows.append({'method':method,'runs':len(runs),'valid_candidate_draws':sum(r['valid_candidate_draws'] for r in runs),
            'generated_draws':sum(r['shots'] for r in runs),'local_S3_optimum_runs':sum(r['raw_local_S3_optimum_hit'] for r in runs),
            'exact_frontier_points_found':sum(r['exact_local_frontier_points_found'] for r in runs),
            'exact_frontier_points_total':sum(r['exact_local_frontier_points'] for r in runs),
            'max_added_PNRs_in_one_patch':max(p['S3_selected']['admitted']['assigned_pnrs']-p['S3_selected']['before']['assigned_pnrs'] for p in patches),
            'max_added_passengers_in_one_patch':max(p['S3_selected']['admitted']['assigned_passengers']-p['S3_selected']['before']['assigned_passengers'] for p in patches),
            'max_added_PNRs_from_observed_Pareto':max(p['Pareto_selected']['admitted']['assigned_pnrs']-p['Pareto_selected']['before']['assigned_pnrs'] for p in patches),
            'max_added_passengers_from_observed_Pareto':max(p['Pareto_selected']['admitted']['assigned_passengers']-p['Pareto_selected']['before']['assigned_passengers'] for p in patches)})
    return rows


def paired_S3_comparison(report):
    """Descriptive matched case/seed comparisons; no IID significance claim."""
    annealing={(r['case'],r['seed']):r for r in report['runs'] if r['backend']=='neal'}
    rows=[]
    for depth in report['config']['depths']:
        wins=ties=losses=0
        for run in report['runs']:
            if run['backend']!=f'onechoice_QAOA_p{depth}':continue
            a=run['raw_best_S3'];b=annealing[(run['case'],run['seed'])]['raw_best_S3']
            if a is None and b is None:ties+=1
            elif b is None or (a is not None and a<b-1e-8):wins+=1
            elif a is None or a>b+1e-8:losses+=1
            else:ties+=1
        rows.append({'depth':depth,'QAOA_lower_raw_S3':wins,'equal_raw_S3':ties,'annealing_lower_raw_S3':losses,
            'scope':'Best physically valid observed draw, before repair; matched case/seed, 1024 final draws per method; training compute not matched.'})
    return rows
