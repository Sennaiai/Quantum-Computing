"""Bounded, reproducible recovery routes on the supplied de-identified network."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import gzip
import json

from data_contract import CONFIG, FEATURES, Instance, Option, cabin, finite, identifier, integer

SCALES = ('ARRIVAL_DELAY_MINS', 'EXCESS_TRAVEL_TIME_MINS', 'CONN_PENALTY',
          'CAPACITY_STRESS_PROXY', 'BOTTLENECK_RATIO')


def validate_network(data, instance, expected_catalog_sha256=None):
    if data.get('schema_version') != 1: raise ValueError('Unsupported network schema')
    if expected_catalog_sha256 is not None and data.get('base_catalog_sha256') != expected_catalog_sha256:
        raise ValueError('Connection metadata refers to a different base catalog')
    flights = {}
    for row in data['flights']:
        row = dict(row)
        fid = identifier(row['id'])
        if fid in flights: raise ValueError('Duplicate network flight')
        for key in ('origin', 'destination'): identifier(row[key])
        for key in ('departure', 'arrival'): row[key] = integer(row[key], 'flight UTC seconds', -2**53, 2**53)
        row['total_availability'] = finite(row['total_availability'], 'original total availability')
        if row['arrival'] <= row['departure'] or row['origin'] == row['destination']:
            raise ValueError('Invalid flight endpoints or chronological order')
        flights[fid] = dict(row)
    if set(flights) != {f for f, c in instance.capacities}: raise ValueError('Network/inventory flight universes differ')
    if any((fid, cab) not in instance.capacities for fid in flights for cab in ('C', 'Y')): raise ValueError('Both cabin inventories must be explicit')
    airports = {r[k] for r in flights.values() for k in ('origin', 'destination')}
    if set(data['journeys']) != set(instance.party_by_id): raise ValueError('Network/PNR universes differ')
    journeys = {}
    for pid, row in data['journeys'].items():
        row = dict(row)
        if row['origin'] not in airports or row['destination'] not in airports or row['origin'] == row['destination']:
            raise ValueError('Invalid recovery journey endpoints')
        for key in ('departure', 'arrival'): row[key] = integer(row[key], 'original UTC seconds', -2**53, 2**53)
        row['duration'] = finite(row['duration'], 'original journey duration', 0)
        row['connections'] = integer(row['connections'], 'original connections', 0)
        row = dict(row); row['cabin'] = cabin(row['cabin']); journeys[pid] = row
    if set(data['feature_ranges']) != set(SCALES): raise ValueError('Incomplete frozen feature scales')
    for name, bounds in data['feature_ranges'].items():
        if len(bounds) != 2 or finite(bounds[0], name) > finite(bounds[1], name): raise ValueError('Invalid frozen feature range')
    return {'flights': flights, 'journeys': journeys, 'feature_ranges': data['feature_ranges']}


def load_network(path, instance, expected_catalog_sha256=None):
    path = Path(path); raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return validate_network(json.loads(raw), instance, expected_catalog_sha256)


def validate_policy(policy):
    fields = {'min_connection_minutes', 'max_layover_minutes', 'recovery_minutes', 'max_legs', 'path_budget_per_journey'}
    if set(policy) != fields: raise ValueError('Incomplete connection policy')
    policy = {name: integer(policy[name], name, 1, 1000000) for name in fields}
    if policy['max_legs'] > 3 or policy['min_connection_minutes'] > policy['max_layover_minutes']:
        raise ValueError('Invalid connection count/wait bounds')
    return dict(policy)


def route_violations(instance, network, party, legs, policy):
    flights = network['flights']; journey = network['journeys'][party]
    path = [flights[f] for f, c in legs]; reasons = []
    if not path: return ['empty_route']
    if len(path) > policy['max_legs']: reasons.append('too_many_legs')
    if path[0]['origin'] != journey['origin'] or path[-1]['destination'] != journey['destination']: reasons.append('wrong_endpoints')
    airports = [path[0]['origin']]+[r['destination'] for r in path]
    if len(set(airports)) != len(airports): reasons.append('airport_cycle')
    if any(a['destination'] != b['origin'] for a, b in zip(path, path[1:])): reasons.append('disconnected_route')
    waits = [(b['departure']-a['arrival'])/60 for a, b in zip(path, path[1:])]
    if any(w < policy['min_connection_minutes'] or w > policy['max_layover_minutes'] for w in waits): reasons.append('connection_wait')
    if path[0]['departure'] < journey['departure']: reasons.append('early_departure')
    if path[-1]['arrival'] > journey['departure']+policy['recovery_minutes']*60: reasons.append('late_recovery')
    if (path[-1]['arrival']-journey['arrival'])/60 > CONFIG['max_arrival_delay_minutes']: reasons.append('late_arrival')
    if any(instance.capacities[k] < instance.party_by_id[party].pax for k in legs): reasons.append('party_capacity')
    return reasons


def option_from_route(instance, network, pid, legs, oid):
    flights = network['flights']; journey = network['journeys'][pid]; pax = instance.party_by_id[pid].pax
    depart = flights[legs[0][0]]['departure']; arrive = flights[legs[-1][0]]['arrival']
    delay = (arrive-journey['arrival'])/60; duration = (arrive-depart)/60
    def normalize(name, value):
        lo, hi = network['feature_ranges'][name]
        return (value-lo)/(hi-lo) if hi > lo else 0.
    # CVM_NORM is already present in the supplied scoring features, including for private raw-CVM inputs.
    priority = network['priorities'][pid]
    displacement = -1 if journey['cabin'] == 'C' and any(c == 'Y' for f, c in legs) else (
                    1 if journey['cabin'] == 'Y' and any(c == 'C' for f, c in legs) else 0)
    values = {'ARRIVAL_DELAY_MINS_NORM': normalize('ARRIVAL_DELAY_MINS', delay),
        'EXCESS_TRAVEL_TIME_MINS_NORM': normalize('EXCESS_TRAVEL_TIME_MINS', max(0., duration-journey['duration'])),
        'CONN_PENALTY_NORM': normalize('CONN_PENALTY', max(0, len(legs)-1-journey['connections'])),
        'CAPACITY_STRESS_PROXY_NORM': normalize('CAPACITY_STRESS_PROXY', sum(pax/max(1., flights[f]['total_availability']) for f, c in legs)),
        'BOTTLENECK_RATIO_NORM': normalize('BOTTLENECK_RATIO', pax/min(instance.capacities[k] for k in legs)),
        'CVM_NORM': priority, 'MIXED_CABIN_FLAG': float(len({c for f, c in legs}) > 1), 'CABIN_DISP': displacement}
    return Option(oid, pid, tuple(legs), datetime.fromtimestamp(depart, timezone.utc), delay, tuple(float(values[k]) for k in FEATURES))


def bind_and_verify_features(instance, network):
    network = dict(network); priorities = {}; ix = FEATURES.index('CVM_NORM')
    for o in instance.options:
        if o.party in priorities and priorities[o.party] != o.features[ix]: raise ValueError('Inconsistent normalized priority')
        priorities[o.party] = o.features[ix]
    if set(priorities) != set(instance.party_by_id): raise ValueError('Missing original normalized priority')
    network['priorities'] = priorities; maximum = 0.
    for o in instance.options:
        actual = option_from_route(instance, network, o.party, o.legs, o.id)
        error = max(abs(a-b) for a, b in zip(actual.features, o.features)); maximum = max(maximum, error)
        if error > 1e-7 or abs(actual.delay_minutes-o.delay_minutes) > 1e-7 or actual.departure != o.departure:
            raise ValueError('Network reconstruction does not reproduce original scenario features')
    return network, maximum


def expand_connections(instance, network, policy=None):
    tick = perf_counter(); policy = validate_policy(CONFIG['connection_generation'] if policy is None else policy)
    network, feature_error = bind_and_verify_features(instance, network)
    flights = network['flights']; byorigin = defaultdict(list)
    for flight in flights.values(): byorigin[flight['origin']].append(flight)
    for origin in byorigin: byorigin[origin].sort(key=lambda r: (r['departure'], r['arrival'], r['id']))
    departures = {origin: [r['departure'] for r in rows] for origin, rows in byorigin.items()}
    @lru_cache(maxsize=20000)
    def paths(origin, destination, departure, original_cabin, pax):
        deadline = departure+policy['recovery_minutes']*60; found = []
        def walk(location, earliest, latest, path, visited):
            rows = byorigin.get(location, []); times = departures.get(location, [])
            for flight in rows[bisect_left(times, earliest):bisect_right(times, latest)]:
                if flight['destination'] in visited or flight['arrival'] > deadline: continue
                for cab in (original_cabin, 'C' if original_cabin == 'Y' else 'Y'):
                    if instance.capacities[(flight['id'], cab)] < pax: continue
                    proposed = path+((flight['id'], cab),)
                    if flight['destination'] == destination:
                        found.append(proposed)
                        if len(found) > policy['path_budget_per_journey']: raise ValueError('Connection path budget exceeded; no silent route truncation')
                    elif len(proposed) < policy['max_legs']:
                        walk(flight['destination'], flight['arrival']+policy['min_connection_minutes']*60,
                             min(deadline, flight['arrival']+policy['max_layover_minutes']*60), proposed, visited|{flight['destination']})
        walk(origin, departure, deadline, (), {origin})
        return tuple(found)
    kept = []; removed = Counter(); existing = defaultdict(set)
    for o in instance.options:
        reasons = route_violations(instance, network, o.party, o.legs, policy)
        if reasons: removed.update(reasons)
        else: kept.append(o); existing[o.party].add(o.legs)
    extra = []; counts = Counter()
    for party in instance.parties:
        j = network['journeys'][party.id]; candidates = []
        for legs in paths(j['origin'], j['destination'], j['departure'], j['cabin'], party.pax):
            if legs in existing[party.id]: continue
            option = option_from_route(instance, network, party.id, legs, 'pending')
            if option.delay_minutes > CONFIG['max_arrival_delay_minutes']: continue
            candidates.append(option)
        candidates.sort(key=lambda o: (o.cost('S3_capacityStability'), len(o.legs), o.legs))
        from dataclasses import replace
        for o in candidates:
            option = replace(o, id=f'EXT_{len(extra):08}')
            if option.id in instance.option_by_id: raise ValueError('Generated option ID collides with original catalog')
            extra.append(option); counts[len(option.legs)] += 1
    expanded = Instance(instance.parties, tuple(kept)+tuple(extra), dict(instance.capacities), 'provided_catalog_expanded_connections_v1')
    if len({(o.party, o.legs) for o in expanded.options}) != len(expanded.options): raise ValueError('Duplicate physical route/cabin option')
    audit = {'original_options': len(instance.options), 'retained_original_options': len(kept), 'removed_original_options': len(instance.options)-len(kept),
             'removed_reasons': dict(removed), 'added_options': len(extra), 'added_leg_counts': dict(sorted(counts.items())),
             'total_options': len(expanded.options), 'base_feature_maximum_absolute_error': feature_error,
             'policy': policy, 'seconds': perf_counter()-tick,
             'scope': 'All supplied flights and feasible cabin patterns; at most two stops, no airport cycles. The prototype transfer threshold is not an airport-specific commercial MCT certification.'}
    return expanded, network, audit


def route_quality(instance, assignment, network):
    values = []; downgraded = upgraded = mixed = 0
    for pid, oid in assignment.items():
        o = instance.option_by_id[oid]; j = network['journeys'][pid]; rows = [network['flights'][f] for f, c in o.legs]
        pax = instance.party_by_id[pid].pax
        down = j['cabin'] == 'C' and any(c == 'Y' for f, c in o.legs)
        up = j['cabin'] == 'Y' and any(c == 'C' for f, c in o.legs)
        downgraded += int(down); upgraded += int(up); mixed += int(len({c for f, c in o.legs}) > 1)
        values.append((len(o.legs)-1, (rows[-1]['arrival']-rows[0]['departure'])/60,
                       (rows[-1]['arrival']-j['departure'])/60, pax))
    n = len(values); passengers = sum(v[3] for v in values)
    return {'mean_connections': sum(v[0] for v in values)/n if n else None,
        'mean_journey_minutes': sum(v[1] for v in values)/n if n else None,
        'passenger_weighted_mean_journey_minutes': sum(v[1]*v[3] for v in values)/passengers if passengers else None,
        'mean_recovery_minutes': sum(v[2] for v in values)/n if n else None,
        'downgraded_pnrs': downgraded, 'upgraded_pnrs': upgraded, 'mixed_cabin_pnrs': mixed,
        'direct_pnrs': sum(v[0] == 0 for v in values), 'connecting_pnrs': sum(v[0] > 0 for v in values),
        'interpretation': 'Observable service proxies; no measured customer-satisfaction or monetary-profit claim'}
