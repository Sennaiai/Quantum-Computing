"""Validated solver inputs; no itinerary generation or silent feature imputation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType
from functools import cached_property, lru_cache
import gzip
import math

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())
FEATURES = tuple(CONFIG['scenarios']['S3_capacityStability'])


def integer(value, name, minimum=0, maximum=10**6):
    if isinstance(value, bool): raise ValueError(f'{name} must be an integer')
    try: valid = math.isfinite(float(value)) and float(value) == int(value)
    except (TypeError, ValueError, OverflowError): valid = False
    if not valid or not minimum <= int(value) <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum},{maximum}]')
    return int(value)


def finite(value, name, minimum=None):
    try: x = float(value)
    except (TypeError, ValueError): raise ValueError(f'{name} must be finite') from None
    if not math.isfinite(x) or (minimum is not None and x < minimum):
        raise ValueError(f'{name} must be finite and meet its lower bound')
    return x


def identifier(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError('Identifiers must be nonempty, whitespace-free at their boundaries')
    return value


def cabin(value):
    if not isinstance(value, str): raise ValueError('Missing cabin code')
    token = value.strip().upper()
    if token in ('C', 'J', 'BUS', 'BUSINESS'): return 'C'
    if token in ('Y', 'M', 'E', 'ECO', 'ECONOMY', 'COACH'): return 'Y'
    raise ValueError('Unsupported cabin code; an explicit business mapping is required')


@lru_cache(maxsize=100000)
def utc(value):
    # DTMZ input columns are documented UTC, including naive strings.
    try: t = pd.to_datetime(value, utc=True, errors='raise')
    except (ValueError, TypeError): raise ValueError('Invalid UTC departure timestamp') from None
    if pd.isna(t) or not isinstance(t, pd.Timestamp): raise ValueError('Missing UTC departure timestamp')
    return t.to_pydatetime()


def validate_config(config):
    integer(config['max_arrival_delay_minutes'], 'maximum arrival delay', 0)
    finite(config['unassigned_cost'], 'unassigned cost', 1e-12)
    for key in ('block_hours', 'top_k', 'max_bqm_variables', 'num_reads', 'num_sweeps'):
        integer(config[key], key, 1)
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('A nonempty, unique seed list is required')
    for seed in config['seeds']: integer(seed, 'seed', 0, 2**31-1)
    for weights in config['scenarios'].values():
        if set(weights) != set(FEATURES): raise ValueError('Scenario feature schemas differ')
        for value in weights.values(): finite(value, 'scenario weight')


validate_config(CONFIG)


@dataclass(frozen=True)
class Party:
    id: str
    pax: int
    cvm: float


@dataclass(frozen=True)
class Option:
    id: str
    party: str
    legs: tuple[tuple[str, str], ...]
    departure: datetime
    delay_minutes: float
    features: tuple[float, ...]

    @cached_property
    def scenario_costs(self):
        return MappingProxyType({scenario: sum(x*weights[name] for name, x in zip(FEATURES, self.features))
                for scenario, weights in CONFIG['scenarios'].items()})

    def cost(self, scenario):
        if scenario not in CONFIG['scenarios']: raise ValueError('Unknown scenario')
        return self.scenario_costs[scenario]


@dataclass(frozen=True)
class Instance:
    parties: tuple[Party, ...]
    options: tuple[Option, ...]
    capacities: dict[tuple[str, str], int]
    name: str = 'instance'

    @cached_property
    def party_by_id(self): return MappingProxyType({p.id: p for p in self.parties})

    @cached_property
    def option_by_id(self): return MappingProxyType({o.id: o for o in self.options})

    @cached_property
    def scenario_catalogs(self): return {}

    def __post_init__(self):
        if not self.parties: raise ValueError('At least one impacted PNR is required')
        ids = [identifier(p.id) for p in self.parties]
        if len(set(ids)) != len(ids): raise ValueError('Duplicate PNR identifier')
        for p in self.parties:
            integer(p.pax, 'party size', 1); finite(p.cvm, 'CVM', 0)
        options = [identifier(o.id) for o in self.options]
        if len(set(options)) != len(options): raise ValueError('OPTION_ID must be globally unique')
        caps = {}
        for (flight, cab), value in self.capacities.items():
            key = (identifier(flight), cabin(cab))
            if key in caps: raise ValueError('Duplicate capacity after cabin normalization')
            caps[key] = integer(value, 'capacity')
        object.__setattr__(self, 'capacities', MappingProxyType(caps))
        id_set = set(ids)
        for o in self.options:
            if o.party not in id_set: raise ValueError('Option refers to an unknown PNR')
            if not o.legs or len({d for d, c in o.legs}) != len(o.legs):
                raise ValueError('A candidate needs nonempty, non-repeated flight legs')
            for leg in o.legs:
                if leg not in caps: raise ValueError('Candidate references unknown flight/cabin capacity')
            if not isinstance(o.departure, datetime) or o.departure.tzinfo is None or o.departure.utcoffset() is None:
                raise ValueError('Candidate departure must be timezone-aware')
            finite(o.delay_minutes, 'arrival delay')
            if len(o.features) != len(FEATURES): raise ValueError('Incomplete scenario feature vector')
            for x in o.features: finite(x, 'scenario feature')

    @cached_property
    def eligibility(self):
        party = self.party_by_id
        out, reasons = [], Counter()
        for o in self.options:
            if o.delay_minutes > CONFIG['max_arrival_delay_minutes']:
                reasons['after_72_hours'] += 1
            elif any(party[o.party].pax > self.capacities[k] for k in o.legs):
                reasons['party_exceeds_leg_capacity'] += 1
            else: out.append(o)
        return tuple(out), dict(reasons)

    @cached_property
    def eligible_by_id(self): return MappingProxyType({o.id: o for o in self.eligibility[0]})

    def eligible(self): return self.eligibility[0], dict(self.eligibility[1])


def dump_instance(instance, path):
    value = {'name': instance.name,
             'parties': [dict(id=p.id, pax=p.pax, cvm=p.cvm) for p in instance.parties],
             'options': [dict(id=o.id, party=o.party, legs=o.legs,
                       departure=o.departure.isoformat(), delay_minutes=o.delay_minutes,
                       features=dict(zip(FEATURES, o.features))) for o in instance.options],
             'capacities': [dict(flight=f, cabin=c, seats=v) for (f, c), v in instance.capacities.items()]}
    p = Path(path)
    if p.suffix == '.gz':
        raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
        p.write_bytes(gzip.compress(raw, mtime=0))
    else: p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_instance(path):
    p = Path(path)
    d = json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_text())
    parties = tuple(Party(identifier(p['id']), integer(p['pax'], 'party size', 1), finite(p['cvm'], 'CVM', 0)) for p in d['parties'])
    options = tuple(Option(identifier(o['id']), identifier(o['party']), tuple((identifier(f), cabin(c)) for f, c in o['legs']),
                    utc(o['departure']), finite(o['delay_minutes'], 'arrival delay'),
                    tuple(finite(o['features'][key], key) for key in FEATURES)) for o in d['options'])
    capacities = {}
    for r in d['capacities']:
        key = (identifier(r['flight']), cabin(r['cabin']))
        if key in capacities: raise ValueError('Duplicate capacity key in JSON')
        capacities[key] = integer(r['seats'], 'capacity')
    return Instance(parties, options, capacities, d.get('name', 'instance'))


def load_private_csvs(directory, reliable=False, capacity_policy='strict'):
    """Read the four original processed tables; return canonical instance + audit.

    Raw COPA files are not these processed inputs. No private data is generated
    or published by the public notebook. Gurobi's original model stays external.
    """
    if capacity_policy not in ('strict', 'conservative'):
        raise ValueError('Unknown capacity policy')
    suffix = '_reliable' if reliable else ''
    names = ('pnr_candidates_clean', 'candidate_segments', 'pnr_option_features_step2', 'available_flights')
    tables = []
    for name in names:
        p = Path(directory)/(name+suffix+'.csv')
        if not p.is_file(): raise FileNotFoundError(f'Missing required processed table: {p.name}')
        tables.append(pd.read_csv(p, dtype={'RECLOC': 'string', 'OPTION_ID': 'string', 'DEP_KEY': 'string'}, low_memory=False))
    candidates, segments, features, available = tables
    def require(df, columns, name):
        if not set(columns).issubset(df.columns): raise ValueError(f'Incomplete {name} schema')
        if df[list(columns)].isna().any().any(): raise ValueError(f'Missing values in {name}')
    require(candidates, ('RECLOC', 'OPTION_ID'), 'candidate')
    require(features, ('RECLOC', 'OPTION_ID', 'ARRIVAL_DELAY_MINS', *FEATURES), 'feature')
    departure_table = features if 'DEP_FIRST_DTMZ' in features else candidates
    require(departure_table, ('OPTION_ID', 'DEP_FIRST_DTMZ'), 'departure')
    require(available, ('DEP_KEY', 'C_AVAIL_CNT', 'Y_AVAIL_CNT'), 'capacity')
    cab = 'CABIN_USED' if 'CABIN_USED' in segments else 'CABIN'
    require(segments, ('OPTION_ID', 'DEP_KEY', cab), 'segment')
    for df in (candidates, features):
        if df.OPTION_ID.duplicated().any(): raise ValueError('Duplicate OPTION_ID in processed table')
    if set(candidates.OPTION_ID) != set(features.OPTION_ID) or set(segments.OPTION_ID) != set(features.OPTION_ID):
        raise ValueError('Candidate/feature/segment option universes differ')
    def pax_column(df):
        cols = [c for c in ('PAX_CNT', 'PAX_COUNT') if c in df]
        if not cols: raise ValueError('Party-size column is required')
        for c in cols: require(df, (c,), 'party size')
        if len(cols) == 2 and not df[cols[0]].equals(df[cols[1]]):
            if not (pd.to_numeric(df[cols[0]]) == pd.to_numeric(df[cols[1]])).all(): raise ValueError('Conflicting party-size aliases')
        return cols[0]
    fp, cp = pax_column(features), pax_column(candidates)
    raw_cvm = 'CVM' if 'CVM' in features else 'CVM_NORM'
    require(features, (raw_cvm,), 'CVM')
    merged = candidates.set_index('OPTION_ID')
    departures = departure_table.set_index('OPTION_ID')['DEP_FIRST_DTMZ'].to_dict()
    parties, options, legs, caps = {}, [], {}, {}
    audit = {'capacity_policy': capacity_policy, 'negative_capacities_clipped': 0,
             'capacity_conflicts_resolved': 0, 'cvm_source': raw_cvm}
    audit['departure_source'] = 'features' if departure_table is features else 'candidates'
    if 'ORIG_TRAVEL_TIME_MINS' in features:
        duration = pd.to_numeric(features.ORIG_TRAVEL_TIME_MINS, errors='raise')
        audit['options_with_original_duration_over_72h'] = int((duration > 4320).sum())
    delays = pd.to_numeric(features.ARRIVAL_DELAY_MINS, errors='raise')
    audit['options_arriving_over_72h_before_original'] = int((delays < -4320).sum())
    audit['ranking_boundary'] = 'Supplied normalized features are preserved; original-journey segmentation and feature quality are external to this solver'
    for row in available.to_dict('records'):
        for c in ('C', 'Y'):
            value = finite(row[c+'_AVAIL_CNT'], 'capacity')
            if value < 0 and capacity_policy == 'conservative':
                audit['negative_capacities_clipped'] += 1; value = 0
            value = integer(value, 'capacity'); key = (identifier(row['DEP_KEY']), c)
            if key in caps and caps[key] != value:
                if capacity_policy == 'strict': raise ValueError('Conflicting available capacity records')
                audit['capacity_conflicts_resolved'] += 1; value = min(value, caps[key])
            caps[key] = value
    for row in segments.to_dict('records'):
        key = identifier(row['OPTION_ID']); leg = (identifier(row['DEP_KEY']), cabin(row[cab]))
        legs.setdefault(key, []).append(leg)
    for row in features.to_dict('records'):
        oid, rec = identifier(row['OPTION_ID']), identifier(row['RECLOC'])
        pax = integer(row[fp], 'party size', 1); cvm = finite(row[raw_cvm], 'CVM', 0)
        source = merged.loc[oid]
        if str(source.RECLOC) != rec or integer(source[cp], 'party size', 1) != pax:
            raise ValueError('Candidate and feature PNR/party-size metadata disagree')
        party = Party(rec, pax, cvm)
        if rec in parties and parties[rec] != party: raise ValueError('PNR metadata differs across its options')
        parties[rec] = party
        options.append(Option(oid, rec, tuple(legs[oid]), utc(departures[oid]),
                       finite(row['ARRIVAL_DELAY_MINS'], 'arrival delay'),
                       tuple(finite(row[k], k) for k in FEATURES)))
    return Instance(tuple(parties.values()), tuple(options), caps, 'private_processed_input'), audit
