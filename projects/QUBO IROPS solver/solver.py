"""Whole-PNR time decomposition, simulated annealing and certified reconciliation."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from time import perf_counter

import neal

from data_contract import CONFIG, integer
from qubo import compile_bqm, resources, estimate_bqm_variables


def catalog(instance, scenario):
    if scenario not in CONFIG['scenarios']: raise ValueError('Unknown scenario')
    if scenario in instance.scenario_catalogs: return instance.scenario_catalogs[scenario]
    options, _ = instance.eligible()
    by_party = {p.id: [] for p in instance.parties}
    for o in options: by_party[o.party].append(o)
    for p in by_party: by_party[p].sort(key=lambda o: (o.cost(scenario), o.id))
    instance.scenario_catalogs[scenario] = by_party
    return by_party


def loads_for(instance, assignment):
    parties = instance.party_by_id; options = instance.eligible_by_id
    loads = defaultdict(int)
    for p, oid in assignment.items():
        if p not in parties or oid not in options or options[oid].party != p:
            raise ValueError('Unknown, ineligible or cross-PNR assignment')
        for key in options[oid].legs: loads[key] += parties[p].pax
    return dict(loads)


def score(instance, assignment, scenario='S3_capacityStability'):
    if scenario not in CONFIG['scenarios']: raise ValueError('Unknown scenario')
    loads = loads_for(instance, assignment); by_id = instance.option_by_id
    violation = [max(0, v-instance.capacities[k]) for k, v in loads.items()]
    parties = instance.party_by_id
    cost = sum(by_id[oid].cost(scenario) for oid in assignment.values())
    missing = len(instance.parties)-len(assignment)
    passengers = sum(p.pax for p in instance.parties)
    assigned_passengers = sum(parties[p].pax for p in assignment)
    return {'pnrs': len(instance.parties), 'assigned_pnrs': len(assignment), 'unassigned_pnrs': missing,
            'assignment_rate': len(assignment)/len(instance.parties),
            'pnr_allocation_percent': 100*len(assignment)/len(instance.parties),
            'passengers': passengers, 'assigned_passengers': assigned_passengers,
            'passenger_allocation_percent': 100*assigned_passengers/passengers,
            'captured_cvm': sum(parties[p].cvm for p in assignment),
            'assigned_scenario_cost': float(cost),
            'objective': float(cost+CONFIG['unassigned_cost']*missing),
            'capacity_violations': sum(x > 0 for x in violation),
            'excess_seats': int(sum(violation))}


def feasible(instance, assignment):
    return all(v <= instance.capacities[k] for k, v in loads_for(instance, assignment).items())


def can_place(option, pax, loads, capacities):
    return all(loads.get(key, 0)+pax <= capacities[key] for key in option.legs)


def reconcile(instance, assignment, scenario, capacities=None, party_ids=None, improve=True, diagnostics=None):
    """Canonical hard feasibility + deterministic fill and bounded one-exchange.

    Capacity overrides are the current global residual ledger during chunk repair.
    All changes after hard repair improve the same declared objective.
    """
    caps = instance.capacities if capacities is None else capacities
    if set(caps) != set(instance.capacities): raise ValueError('Incomplete repair capacity ledger')
    for key, value in caps.items(): integer(value, 'repair capacity', 0, instance.capacities[key])
    parties = instance.party_by_id
    ids = set(parties) if party_ids is None else set(party_ids)
    if not ids <= set(parties): raise ValueError('Unknown repair PNR')
    choices = catalog(instance, scenario); options = instance.eligible_by_id
    loads_for(instance, assignment)  # Reject ghost/cross-PNR options instead of ignoring them.
    if not set(assignment) <= ids: raise ValueError('Repair received a PNR outside its chunk')
    selected = {p: oid for p, oid in assignment.items() if options[oid].cost(scenario) < CONFIG['unassigned_cost']}
    loads = defaultdict(int, loads_for(instance, selected))
    occupants = defaultdict(set)
    for p, oid in selected.items():
        for k in options[oid].legs: occupants[k].add(p)
    def remove(p):
        oid = selected.pop(p)
        for k in options[oid].legs:
            loads[k] -= parties[p].pax; occupants[k].remove(p)
    def place(p, option):
        selected[p] = option.id
        for k in option.legs:
            loads[k] += parties[p].pax; occupants[k].add(p)
    while True:
        overloaded = {k for k, value in loads.items() if value > caps[k]}
        if not overloaded: break
        offenders = set().union(*(occupants[k] for k in overloaded))
        drop = min(offenders, key=lambda p: (parties[p].cvm, -options[selected[p]].cost(scenario), p))
        remove(drop)
    order = sorted(ids-set(selected), key=lambda p: (len(choices[p]), -parties[p].cvm, p))
    for p in order:
        for option in choices[p]:
            if option.cost(scenario) < CONFIG['unassigned_cost'] and can_place(option, parties[p].pax, loads, caps):
                place(p, option)
                break
    # One deterministic pass: no claim of global/local optimality beyond this neighbourhood.
    if improve:
        total_trials, limited = 0, 0
        for p in sorted(ids-set(selected), key=lambda p: (-parties[p].cvm, p)):
            best = None; trials = 0; exhausted = False
            for option in choices[p]:
                if option.cost(scenario) >= CONFIG['unassigned_cost']: continue
                tight = [k for k in option.legs if loads[k]+parties[p].pax > caps[k]]
                offenders = sorted(set().union(*(occupants[k] for k in tight)), key=lambda q: (parties[q].cvm, q)) if tight else []
                for q in offenders:
                    old = options[selected[q]]
                    delta_load = defaultdict(int)
                    for k in old.legs: delta_load[k] -= parties[q].pax
                    if any(loads[k]+delta_load[k]+parties[p].pax > caps[k] for k in option.legs): continue
                    for k in option.legs: delta_load[k] += parties[p].pax
                    for alternative in [*choices[q], None]:
                        trials += 1
                        if trials > CONFIG['max_exchange_trials_per_pnr']:
                            exhausted = True; break
                        if alternative is not None:
                            if alternative.cost(scenario) >= CONFIG['unassigned_cost'] or any(loads[k]+delta_load[k]+parties[q].pax > caps[k] for k in alternative.legs): continue
                            replacement = alternative.cost(scenario)
                        else:
                            replacement = CONFIG['unassigned_cost']
                        delta = option.cost(scenario)+replacement-CONFIG['unassigned_cost']-old.cost(scenario)
                        key = (delta, q, option.id, '' if alternative is None else alternative.id)
                        if delta < -1e-9 and (best is None or key < best[0]): best = (key, q, option, alternative)
                    if exhausted: break
                if exhausted: break
            total_trials += min(trials, CONFIG['max_exchange_trials_per_pnr'])
            limited += int(exhausted)
            if best is not None:
                _, q, proposed, alternative = best
                remove(q); place(p, proposed)
                if alternative is not None: place(q, alternative)
        if diagnostics is not None: diagnostics.update(exchange_trials=total_trials, pnrs_hitting_trial_budget=limited)
    loads = loads_for(instance, selected)
    if any(value > caps[key] for key, value in loads.items()): raise ArithmeticError('Reconciliation failed capacity certification')
    return selected


def exact_reference(instance, scenario='S3_capacityStability', max_combinations=2_000_000):
    """Independent constrained enumeration; no BQM, penalty or SA is used."""
    tick = perf_counter(); parties = tuple(instance.parties)
    raw, _ = instance.eligible(); choices = [[None]+[o for o in raw if o.party == p.id] for p in parties]
    combinations = math.prod(len(x) for x in choices)
    if combinations > max_combinations: raise ValueError('Exact reference exceeds bounded enumeration budget')
    best_cost = math.inf; best_assignment = {}; feasible_count = 0
    for selection in product(*choices):
        used = defaultdict(int); cost = 0.; assigned = {}; valid = True
        for party, option in zip(parties, selection):
            if option is None: cost += CONFIG['unassigned_cost']; continue
            assigned[party.id] = option.id; cost += option.cost(scenario)
            for leg in option.legs:
                used[leg] += party.pax
                if used[leg] > instance.capacities[leg]: valid = False; break
            if not valid: break
        if valid:
            feasible_count += 1
            if cost < best_cost-1e-10:
                best_cost, best_assignment = cost, assigned
    return best_assignment, {'combinations': combinations, 'feasible_combinations': feasible_count,
                             'seconds': perf_counter()-tick, **score(instance, best_assignment, scenario)}


def solve_decomposed(instance, scenario='S3_capacityStability', seed=7, *, top_k=None,
                     max_variables=None, reads=None, sweeps=None, reconcile_global=True):
    """Whole PNR chunks, exact-capacity QUBOs, residual ledger and tail refill."""
    tick = perf_counter()
    top_k = integer(CONFIG['top_k'] if top_k is None else top_k, 'top K', 1, 100)
    budget = integer(CONFIG['max_bqm_variables'] if max_variables is None else max_variables, 'BQM variable budget', 1, 10000)
    reads = integer(CONFIG['num_reads'] if reads is None else reads, 'reads', 1, 10000)
    sweeps = integer(CONFIG['num_sweeps'] if sweeps is None else sweeps, 'sweeps', 1, 100000)
    seed = integer(seed, 'seed', 0, 2**31-1)
    all_choices = catalog(instance, scenario)
    parties = instance.party_by_id
    remaining = dict(instance.capacities); assignment = {}; logs = []
    groups = defaultdict(list)
    for p, choices in all_choices.items():
        if choices:
            first = min(o.departure.timestamp() for o in choices)
            key = int(first//(CONFIG['block_hours']*3600))
            groups[key].append(p)
    sampler = neal.SimulatedAnnealingSampler()

    def process(ids):
        options = [o for p in ids for o in all_choices[p][:top_k]]
        compiled = compile_bqm(instance, ids, options, remaining, scenario)
        if compiled.bqm.num_variables > budget: raise ArithmeticError('Whole-PNR chunk exceeds its accepted BQM budget')
        started = perf_counter()
        sample_set = sampler.sample(compiled.bqm, num_reads=reads, num_sweeps=sweeps,
                     seed=int((seed+104729*len(logs)) % (2**31-1)))
        sampler_seconds = perf_counter()-started
        sample = dict(sample_set.first.sample)
        identity_error = float(abs(compiled.bqm.energy(sample)-compiled.semantic_energy(sample)))
        roundoff = compiled.roundoff_bound(sample)
        if identity_error > roundoff:
            raise ArithmeticError('BQM energy differs from its declared residual objective')
        decoded = compiled.decode(sample)
        proposed = {p: min((o for o in options if o.id in oids), key=lambda o: (o.cost(scenario), o.id)).id
                    for p, oids in decoded.items()}
        raw_load = loads_for(instance, proposed)
        repair_tick = perf_counter()
        repaired = reconcile(instance, proposed, scenario, capacities=remaining, party_ids=ids, improve=False)
        log = {'pnrs': len(ids), **resources(compiled), 'sampler_seconds': sampler_seconds,
               'chunk_repair_seconds': perf_counter()-repair_tick,
               'reads': reads, 'sweeps': sweeps, 'best_sample_energy': float(sample_set.first.energy),
               'best_sample_energy_identity_error': identity_error,
               'energy_identity_roundoff_bound': roundoff,
               'raw_best_multiple_option_pnrs': sum(len(v) > 1 for v in decoded.values()),
               'raw_best_one_hot_violations': int(sum(sum(coeff[v]*int(sample[v]) for v in coeff) != 1
                                                for coeff, constant in compiled.constraints[:len(ids)])),
               'raw_best_overloaded_legs': sum(v > remaining[k] for k, v in raw_load.items()),
               'raw_proposed_assignments': len(proposed), 'after_chunk_repair_assignments': len(repaired),
               'samples_with_zero_constraint_residual': sum(int(row.num_occurrences)
                    for row in sample_set.data(['sample', 'num_occurrences'])
                    if all(constant+sum(a*int(row.sample[v]) for v, a in coeff.items()) == 0 for coeff, constant in compiled.constraints))}
        assignment.update(repaired)
        for key, load in loads_for(instance, repaired).items(): remaining[key] -= load
        if min(remaining.values(), default=0) < 0: raise ArithmeticError('Negative global capacity ledger')
        logs.append(log)

    for key in sorted(groups):
        chunk = []
        for p in sorted(groups[key], key=lambda p: (-parties[p].cvm, p)):
            proposed = [*chunk, p]
            opts = [o for q in proposed for o in all_choices[q][:top_k]]
            count = estimate_bqm_variables(instance, proposed, opts, remaining)
            if count > budget and chunk:
                process(chunk); chunk = []
                count = estimate_bqm_variables(instance, [p], all_choices[p][:top_k], remaining)
            if count > budget: raise ValueError('One whole PNR exceeds BQM budget; raise the budget or explicitly change top K')
            chunk.append(p)
        if chunk: process(chunk)
    before_global = dict(assignment)
    reconciliation = {}
    if reconcile_global: assignment = reconcile(instance, assignment, scenario, diagnostics=reconciliation)
    if not feasible(instance, assignment): raise ArithmeticError('Final solution is not capacity feasible')
    final_metrics = score(instance, assignment, scenario)
    return assignment, {**final_metrics,
             'before_global_reconciliation': score(instance, before_global, scenario),
             'top_k': top_k, 'pruned_top_k_options': sum(max(0, len(v)-top_k) for v in all_choices.values()),
             'global_reconciliation': reconcile_global, 'chunks': logs,
             'global_reconciliation_diagnostics': reconciliation,
             'excluded_options': instance.eligible()[1], 'seconds': perf_counter()-tick}
