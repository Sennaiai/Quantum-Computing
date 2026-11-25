"""Bounded feasibility-preserving route exchanges after the QUBO experiment.

These classical stages are reported separately from simulated annealing. Every
accepted move increases PNR allocation, preserves passenger allocation, and
improves the declared S3 objective; they do not prove global optimality.
"""
from collections import defaultdict
from time import perf_counter
from data_contract import CONFIG, integer
from solver import catalog, loads_for, can_place, score, feasible


def _audit_improvement(instance, before, after, scenario):
    old = score(instance, before, scenario); new = score(instance, after, scenario)
    if new['capacity_violations']: raise ArithmeticError('Refinement violated capacity')
    if new['assigned_pnrs'] < old['assigned_pnrs'] or new['assigned_passengers'] < old['assigned_passengers']:
        raise ArithmeticError('Refinement reduced allocation')
    if new['objective'] > old['objective']+1e-7: raise ArithmeticError('Refinement worsened the declared objective')
    return new


def augment_routes(instance, assignment, scenario, *, max_depth=4, node_budget=512, passes=2):
    """Add a whole PNR by bounded chains of reroutes; retain all served PNRs."""
    max_depth = integer(max_depth, 'augmentation depth', 0, 12)
    node_budget = integer(node_budget, 'augmentation node budget', 1, 1000000)
    passes = integer(passes, 'augmentation passes', 0, 100)
    if not feasible(instance, assignment): raise ValueError('Augmentation requires a feasible incumbent')
    tick = perf_counter(); choices = catalog(instance, scenario); options = instance.eligible_by_id
    parties = instance.party_by_id; capacities = instance.capacities
    selected = dict(assignment); used = defaultdict(int, loads_for(instance, selected))
    occupants = defaultdict(set)
    for pid, oid in selected.items():
        for key in options[oid].legs: occupants[key].add(pid)
    undo = []; objective = score(instance, selected, scenario)['objective']
    diagnostics = {'attempts': 0, 'accepted': 0, 'nodes': 0, 'pnrs_hitting_node_budget': 0, 'passes': 0}

    def apply(pid, oid, record=True):
        nonlocal objective
        old = selected.get(pid)
        if record: undo.append((pid, old))
        objective -= options[old].cost(scenario) if old is not None else CONFIG['unassigned_cost']
        if old is not None:
            for key in options[old].legs:
                used[key] -= parties[pid].pax; occupants[key].remove(pid)
            del selected[pid]
        if oid is not None:
            selected[pid] = oid
            for key in options[oid].legs:
                used[key] += parties[pid].pax; occupants[key].add(pid)
        objective += options[oid].cost(scenario) if oid is not None else CONFIG['unassigned_cost']

    def rollback(mark):
        while len(undo) > mark:
            pid, oid = undo.pop(); apply(pid, oid, record=False)

    nodes = 0
    def try_party(pid, depth, locked, excluded=None):
        nonlocal nodes
        ordered = sorted(choices[pid], key=lambda o: (
            sum(max(0, used[k]+parties[pid].pax-capacities[k]) for k in o.legs), o.cost(scenario), o.id))
        for option in ordered:
            if option.id == excluded: continue
            if nodes >= node_budget: return False
            nodes += 1; mark = len(undo); apply(pid, option.id)
            if clear(option.legs, depth, locked|{pid}): return True
            rollback(mark)
        return False

    def clear(keys, depth, locked):
        tight = [key for key in keys if used[key] > capacities[key]]
        if not tight: return True
        if depth <= 0 or nodes >= node_budget: return False
        key = max(tight, key=lambda k: (used[k]-capacities[k], k))
        candidates = sorted(occupants[key]-locked, key=lambda p: (-len(choices[p]), -parties[p].pax, p))
        for pid in candidates:
            if len(choices[pid]) <= 1: continue
            if nodes >= node_budget: return False
            mark = len(undo); old = selected[pid]; apply(pid, None)
            if try_party(pid, depth-1, locked, old) and clear(keys, depth, locked|{pid}): return True
            rollback(mark)
        return False

    for _ in range(passes):
        accepted = 0; diagnostics['passes'] += 1
        order = sorted(set(parties)-set(selected), key=lambda p: (len(choices[p]), -parties[p].pax, -parties[p].cvm, p))
        for pid in order:
            diagnostics['attempts'] += 1; nodes = 0; mark = len(undo); previous = objective
            if try_party(pid, max_depth, set()) and objective < previous-1e-9:
                accepted += 1; diagnostics['accepted'] += 1; undo.clear()
            else: rollback(mark)
            diagnostics['nodes'] += nodes
            diagnostics['pnrs_hitting_node_budget'] += int(nodes >= node_budget)
        if not accepted: break
    if not set(assignment) <= set(selected): raise ArithmeticError('Augmentation evicted an incumbent PNR')
    _audit_improvement(instance, assignment, selected, scenario)
    diagnostics['seconds'] = perf_counter()-tick
    return selected, diagnostics


def two_for_one(instance, assignment, scenario, *, passes=2, option_limit=64):
    """Replace one booking with two, without losing passengers or S3 quality.

    This is a bounded classical neighbourhood, not a client/industry benchmark.
    Customer-value tradeoffs remain represented by the supplied S3 weights.
    """
    passes = integer(passes, 'exchange passes', 0, 100); option_limit = integer(option_limit, 'exchange option budget', 2, 10000)
    if not feasible(instance, assignment): raise ValueError('Exchange requires a feasible incumbent')
    tick = perf_counter(); selected = dict(assignment); parties = instance.party_by_id
    options = instance.eligible_by_id; capacities = instance.capacities
    used = defaultdict(int, loads_for(instance, selected)); by_leg = defaultdict(list)
    for option in options.values():
        for key in option.legs: by_leg[key].append(option)
    def update(pid, oid, sign):
        for key in options[oid].legs: used[key] += sign*parties[pid].pax
    diagnostics = {'exchanges': 0, 'candidate_sets_hitting_budget': 0, 'tested_pairs': 0, 'passes': 0}
    for _ in range(passes):
        moves = 0; diagnostics['passes'] += 1
        order = sorted(tuple(selected), key=lambda p: (-len(options[selected[p]].legs), -parties[p].pax, p))
        for old_pid in order:
            if old_pid not in selected: continue
            old = options[selected.pop(old_pid)]; update(old_pid, old.id, -1)
            candidates = {o.id: o for key in old.legs for o in by_leg[key]
                if o.party not in selected and o.party != old_pid and can_place(o, parties[o.party].pax, used, capacities)}
            ordered = sorted(candidates.values(), key=lambda o: (-parties[o.party].pax, o.cost(scenario), -parties[o.party].cvm, o.id))
            seen = defaultdict(int); kept = []
            for option in ordered:
                if seen[option.party] >= 2: continue
                seen[option.party] += 1; kept.append(option)
            if len(kept) > option_limit:
                diagnostics['candidate_sets_hitting_budget'] += 1; kept = kept[:option_limit]
            best = None
            for j, first in enumerate(kept):
                update(first.party, first.id, 1)
                for second in kept[j+1:]:
                    if first.party == second.party: continue
                    diagnostics['tested_pairs'] += 1
                    gain = parties[first.party].pax+parties[second.party].pax-parties[old_pid].pax
                    if gain < 0 or not can_place(second, parties[second.party].pax, used, capacities): continue
                    delta = first.cost(scenario)+second.cost(scenario)-old.cost(scenario)-CONFIG['unassigned_cost']
                    priority_gain = parties[first.party].cvm+parties[second.party].cvm-parties[old_pid].cvm
                    rank = (-gain, delta, -priority_gain, first.id, second.id)
                    if delta < -1e-9 and (best is None or rank < best[0]): best = (rank, first, second)
                update(first.party, first.id, -1)
            if best is None:
                selected[old_pid] = old.id; update(old_pid, old.id, 1)
            else:
                _, first, second = best
                for option in (first, second): selected[option.party] = option.id; update(option.party, option.id, 1)
                moves += 1; diagnostics['exchanges'] += 1
        if not moves: break
    _audit_improvement(instance, assignment, selected, scenario)
    diagnostics['seconds'] = perf_counter()-tick
    return selected, diagnostics



def improve_route_costs(instance, assignment, scenario):
    """Choose a cheaper feasible itinerary without changing the served PNR set."""
    if not feasible(instance, assignment): raise ValueError('Route improvement requires a feasible incumbent')
    tick = perf_counter(); selected = dict(assignment); options = instance.eligible_by_id
    parties = instance.party_by_id; choices = catalog(instance, scenario)
    used = defaultdict(int, loads_for(instance, selected)); moved = 0
    for pid in sorted(selected):
        old = options[selected[pid]]
        for key in old.legs: used[key] -= parties[pid].pax
        chosen = old
        for option in choices[pid]:
            if option.cost(scenario) >= old.cost(scenario)-1e-9: break
            if can_place(option, parties[pid].pax, used, instance.capacities):
                chosen = option; break
        selected[pid] = chosen.id; moved += int(chosen.id != old.id)
        for key in chosen.legs: used[key] += parties[pid].pax
    final = _audit_improvement(instance, assignment, selected, scenario)
    return selected, {'rerouted_pnrs': moved, 'objective_reduction': score(instance, assignment, scenario)['objective']-final['objective'],
                      'seconds': perf_counter()-tick}

def refine_allocation(instance, assignment, scenario='S3_capacityStability', config=None):
    """Fixed-budget hybrid refinement, with independent per-stage acceptance."""
    config = CONFIG['allocation_refinement'] if config is None else config
    before = dict(assignment); selected = dict(assignment); stages = []; tick = perf_counter()
    cycles = integer(config['cycles'], 'refinement cycles', 1, 10)
    for cycle in range(cycles):
        start_count = len(selected)
        selected, diagnostics = improve_route_costs(instance, selected, scenario)
        stages.append({'cycle': cycle, 'method': 'feasible_route_cost_improvement', 'diagnostics': diagnostics, **score(instance, selected, scenario)})
        selected, diagnostics = augment_routes(instance, selected, scenario, max_depth=config['max_depth'],
            node_budget=config['node_budget'], passes=config['augmentation_passes'])
        stages.append({'cycle': cycle, 'method': 'bounded_augmenting_routes', 'diagnostics': diagnostics, **score(instance, selected, scenario)})
        selected, diagnostics = two_for_one(instance, selected, scenario,
            passes=config['exchange_passes'], option_limit=config['exchange_option_limit'])
        stages.append({'cycle': cycle, 'method': 'bounded_two_for_one', 'diagnostics': diagnostics, **score(instance, selected, scenario)})
        if len(selected) == start_count: break
    final = _audit_improvement(instance, before, selected, scenario)
    return selected, {'before': score(instance, before, scenario), 'after': final, 'stages': stages,
        'seconds': perf_counter()-tick, 'objective_policy': 'Increase PNRs, never reduce passengers, strictly improve S3 objective per accepted move',
        'scope': 'Classical postprocessing of the QUBO solution; bounded search and candidate limits do not guarantee global optimality'}
