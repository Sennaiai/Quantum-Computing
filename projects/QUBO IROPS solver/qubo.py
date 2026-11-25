"""Exact bounded-capacity BQM encoding, energy identities and resource counts."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math

import dimod
import networkx as nx

from data_contract import CONFIG, Instance, Option, integer, finite


def bounded_weights(limit):
    """Represent every integer 0..limit without encoding values above limit."""
    remaining = integer(limit, 'slack bound'); weights = []; value = 1
    while remaining:
        part = min(value, remaining); weights.append(part)
        remaining -= part; value *= 2
    return tuple(weights)


def add_square(bqm, coefficients, constant, weight):
    """Add weight*(constant + sum a_i*x_i)^2, using x_i^2=x_i."""
    items = list(coefficients.items())
    bqm.offset += weight*constant*constant
    for i, (v, a) in enumerate(items):
        bqm.add_linear(v, weight*(a*a+2*constant*a))
        for u, b in items[i+1:]: bqm.add_quadratic(v, u, 2*weight*a*b)


@dataclass
class Compiled:
    bqm: dimod.BinaryQuadraticModel
    option_variables: dict[str, Option]
    dummy_variables: dict[str, str]
    objective: dict[str, float]
    constraints: tuple[tuple[dict[str, int], int], ...]
    slack: dict[tuple[str, str], tuple[tuple[str, int], ...]]
    penalty: float
    objective_range_bound: float
    capacities_omitted_as_redundant: int

    def semantic_energy(self, sample):
        self.validate_sample(sample)
        obj = sum(c*int(sample[v]) for v, c in self.objective.items())
        residual = sum((constant+sum(a*int(sample[v]) for v, a in coeff.items()))**2
                       for coeff, constant in self.constraints)
        return obj+self.penalty*residual

    def validate_sample(self, sample):
        if set(sample) != set(self.bqm.variables): raise ValueError('Sample variables do not match the BQM')
        if any(sample[v] not in (0, 1) for v in sample): raise ValueError('Non-binary sample')

    def roundoff_bound(self, sample):
        """Conservative floating-point allowance for cancelling expanded terms."""
        import numpy as np
        magnitude = abs(self.bqm.offset)
        magnitude += sum(abs(c)*int(sample[v]) for v, c in self.bqm.linear.items())
        magnitude += sum(abs(c)*int(sample[a])*int(sample[b]) for (a, b), c in self.bqm.quadratic.items())
        return max(1e-9, float(16*np.finfo(float).eps*magnitude))

    def decode(self, sample):
        self.validate_sample(sample)
        selected = defaultdict(list)
        for v, o in self.option_variables.items():
            if sample[v]: selected[o.party].append(o.id)
        return dict(selected)


def compile_bqm(instance, party_ids, options, remaining, scenario, penalty=None):
    if scenario not in CONFIG['scenarios']: raise ValueError('Unknown scenario')
    parties = instance.party_by_id
    party_ids = tuple(party_ids); options = tuple(options)
    if not party_ids or len(set(party_ids)) != len(party_ids) or not set(party_ids) <= set(parties):
        raise ValueError('BQM requires unique known PNRs')
    if len({o.id for o in options}) != len(options): raise ValueError('Duplicate BQM option')
    canonical = instance.eligible_by_id
    for o in options:
        if o.party not in party_ids or canonical.get(o.id) != o:
            raise ValueError('Unknown or ineligible BQM option')
    if set(remaining) != set(instance.capacities): raise ValueError('Incomplete capacity ledger')
    for key, cap in remaining.items():
        integer(cap, 'remaining capacity', 0, instance.capacities[key])
    groups = {p: {} for p in party_ids}
    objective, opt_vars, dummy, by_leg = {}, {}, {}, defaultdict(dict)
    for o in options:
        v = 'option:'+o.id; opt_vars[v] = o
        groups[o.party][v] = 1; objective[v] = finite(o.cost(scenario), 'scenario cost')
        for key in o.legs: by_leg[key][v] = parties[o.party].pax
    for p in party_ids:
        v = 'unassigned:'+p; dummy[v] = p
        groups[p][v] = 1; objective[v] = CONFIG['unassigned_cost']
    bound = sum(abs(c) for c in objective.values())
    weight = bound+1. if penalty is None else finite(penalty, 'penalty', 0)
    if weight <= bound:
        raise ValueError('Penalty must exceed the full linear-objective range bound')
    bqm = dimod.BinaryQuadraticModel(objective, {}, 0., dimod.BINARY)
    constraints = [(coeff, -1) for coeff in groups.values()]
    slacks, omitted = {}, 0
    for j, (key, coefficients) in enumerate(sorted(by_leg.items())):
        users = {opt_vars[v].party for v in coefficients}
        # At most one option per PNR: this upper bound certifies redundancy.
        if sum(parties[p].pax for p in users) <= remaining[key]:
            omitted += 1; continue
        weights = bounded_weights(remaining[key])
        labels = tuple((f'slack:{j}:{i}', w) for i, w in enumerate(weights))
        slacks[key] = labels
        coeff = dict(coefficients); coeff.update(labels)
        for v, _ in labels: bqm.add_variable(v, 0.)
        constraints.append((coeff, -remaining[key]))
    for coeff, constant in constraints: add_square(bqm, coeff, constant, weight)
    if not all(math.isfinite(x) for x in (bqm.offset, *bqm.linear.values(), *bqm.quadratic.values())):
        raise ValueError('BQM coefficients exceed a finite numerical range; rescale input costs')
    return Compiled(bqm, opt_vars, dummy, objective, tuple(constraints), slacks, weight, bound, omitted)


def resources(compiled):
    bqm = compiled.bqm; n = bqm.num_variables
    graph = nx.Graph(); graph.add_nodes_from(bqm.variables)
    graph.add_edges_from((a, b) for (a, b), value in bqm.quadratic.items() if value != 0)
    h, j, offset = bqm.to_ising()
    values = [abs(x) for x in (*h.values(), *j.values()) if x != 0]
    qubo_values = [abs(x) for x in (*bqm.linear.values(), *bqm.quadratic.values()) if x != 0]
    # dimod convention is s=2*x-1, not the computational-basis Z convention.
    return {'variables': n, 'option_variables': len(compiled.option_variables),
            'unassigned_variables': len(compiled.dummy_variables),
            'slack_variables': sum(len(v) for v in compiled.slack.values()),
            'couplers': graph.number_of_edges(), 'maximum_degree': max(dict(graph.degree()).values(), default=0),
            'density': float(nx.density(graph)), 'components': nx.number_connected_components(graph),
            'minimum_linear_quadratic_nonzero_bias': min(qubo_values, default=0.),
            'maximum_linear_quadratic_bias': max(qubo_values, default=0.),
            'ising_dynamic_range': max(values)/min(values) if values else 0.,
            'ising_offset': float(offset), 'penalty': compiled.penalty,
            'objective_range_bound': compiled.objective_range_bound,
            'capacity_constraints_omitted_as_redundant': compiled.capacities_omitted_as_redundant,
            'dense_qubo_float64_payload_bytes': 8*n*n,
            'logical_annealing_qubits_lower_bound': n,
            'p1_cost_layer_RZZ_terms': graph.number_of_edges(),
            'p1_all_to_all_CX_count_before_cancellation': 2*graph.number_of_edges(),
            'resource_scope': 'Logical representation estimates only; no device embedding, chain-strength, noise, calibration or hardware execution model'}


def estimate_bqm_variables(instance, party_ids, options, remaining):
    """Exact size on the validated solver ledger, before constructing couplers.

    The authoritative compiler validates the full ledger when a chunk is accepted.
    This helper only receives canonical options and whole PNRs from the solver.
    """
    options = tuple(options); users = defaultdict(set); parties = instance.party_by_id
    for option in options:
        for key in option.legs: users[key].add(option.party)
    count = len(options)+len(party_ids)
    for key, members in users.items():
        capacity = integer(remaining[key], 'remaining capacity', 0, instance.capacities[key])
        if sum(parties[p].pax for p in members) > capacity:
            count += len(bounded_weights(capacity))
    return count
