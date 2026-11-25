"""Strict conversion of supplied processed UTC tables into anonymous route metadata."""
from pathlib import Path
from functools import lru_cache
import gzip
import json
import math
import pandas as pd
from data_contract import integer, finite, cabin


def build_connection_metadata(directory, mapping, base_catalog_sha256, reliable=False):
    directory = Path(directory); suffix = '_reliable' if reliable else ''
    frames = {name: pd.read_csv(directory/(name+suffix+'.csv'), dtype={'RECLOC': 'string', 'OPTION_ID': 'string', 'DEP_KEY': 'string'}, low_memory=False)
        for name in ('pnr_candidates_clean', 'candidate_segments', 'pnr_option_features_step2', 'available_flights')}
    candidates = frames['pnr_candidates_clean']; features = frames['pnr_option_features_step2']
    segments = frames['candidate_segments']; available = frames['available_flights']
    required = {
        'pnr_candidates_clean': ('RECLOC', 'OPTION_ID', 'DEP_FIRST_DTMZ', 'ARR_FINAL_DTMZ', 'CABIN_ORIG'),
        'pnr_option_features_step2': ('RECLOC', 'OPTION_ID', 'DEPARTURE_DELAY_MINS', 'ARRIVAL_DELAY_MINS', 'ORIG_TRAVEL_TIME_MINS', 'ORIG_NUM_CONNS',
            'EXCESS_TRAVEL_TIME_MINS', 'CONN_PENALTY', 'CAPACITY_STRESS_PROXY', 'BOTTLENECK_RATIO'),
        'candidate_segments': ('OPTION_ID', 'SEG_SEQ', 'ORIG_CD', 'DEST_CD'),
        'available_flights': ('DEP_KEY', 'ORIG_CD', 'DEST_CD', 'DEP_DTMZ', 'ARR_DTMZ', 'TOTAL_AVAIL')}
    for name, columns in required.items():
        if not set(columns) <= set(frames[name]) or frames[name][list(columns)].isna().any().any():
            raise ValueError('Incomplete connection source metadata: '+name)
    if candidates.OPTION_ID.duplicated().any() or features.OPTION_ID.duplicated().any() or available.DEP_KEY.duplicated().any():
        raise ValueError('Duplicate connection source identity')
    if set(candidates.OPTION_ID) != set(features.OPTION_ID) or set(candidates.OPTION_ID) != set(segments.OPTION_ID):
        raise ValueError('Connection source option universes differ')
    merged = candidates.merge(features, on=['RECLOC', 'OPTION_ID'], suffixes=('', '_feature'), validate='one_to_one')
    if len(merged) != len(candidates): raise ValueError('Connection source PNR identities disagree')
    shift = integer(mapping['time_shift_seconds'], 'time shift', -2**53, 2**53)
    @lru_cache(maxsize=100000)
    def seconds(value):
        stamp = pd.to_datetime(value, utc=True, errors='raise')
        if pd.isna(stamp): raise ValueError('Missing UTC metadata timestamp')
        raw = stamp.timestamp()+shift
        return integer(raw, 'UTC metadata seconds', -2**53, 2**53)
    airports = {code: f'APT_{j:03}' for j, code in enumerate(sorted(set(available.ORIG_CD)|set(available.DEST_CD)))}
    flights = [dict(id=mapping['flights'][str(r['DEP_KEY'])], origin=airports[r['ORIG_CD']], destination=airports[r['DEST_CD']],
        departure=seconds(r['DEP_DTMZ']), arrival=seconds(r['ARR_DTMZ']), total_availability=finite(r['TOTAL_AVAIL'], 'original total availability'))
        for r in available.to_dict('records')]
    ordered = segments.sort_values(['OPTION_ID', 'SEG_SEQ'])
    endpoints = ordered.groupby('OPTION_ID').agg(origin=('ORIG_CD', 'first'), destination=('DEST_CD', 'last'))
    merged = merged.join(endpoints, on='OPTION_ID', validate='one_to_one')
    merged['original_departure'] = [seconds(v)-integer(float(d)*60, 'departure delay seconds', -2**53, 2**53)
        for v, d in zip(merged.DEP_FIRST_DTMZ, merged.DEPARTURE_DELAY_MINS)]
    merged['original_arrival'] = [seconds(v)-integer(float(d)*60, 'arrival delay seconds', -2**53, 2**53)
        for v, d in zip(merged.ARR_FINAL_DTMZ, merged.ARRIVAL_DELAY_MINS)]
    metadata_columns = ['original_departure', 'original_arrival', 'origin', 'destination', 'ORIG_TRAVEL_TIME_MINS', 'ORIG_NUM_CONNS', 'CABIN_ORIG']
    if (merged.groupby('RECLOC')[metadata_columns].nunique() != 1).any().any():
        raise ValueError('Original journey metadata differs across its supplied options')
    journeys = {}
    for row in merged.groupby('RECLOC', sort=True).first().reset_index().to_dict('records'):
        journeys[mapping['pnrs'][str(row['RECLOC'])]] = dict(origin=airports[row['origin']], destination=airports[row['destination']],
            departure=int(row['original_departure']), arrival=int(row['original_arrival']),
            duration=finite(row['ORIG_TRAVEL_TIME_MINS'], 'original duration', 0),
            connections=integer(row['ORIG_NUM_CONNS'], 'original connection count', 0), cabin=cabin(row['CABIN_ORIG']))
    names = ('ARRIVAL_DELAY_MINS', 'EXCESS_TRAVEL_TIME_MINS', 'CONN_PENALTY', 'CAPACITY_STRESS_PROXY', 'BOTTLENECK_RATIO')
    scales = {}
    for name in names:
        values = pd.to_numeric(features[name], errors='raise')
        if not all(math.isfinite(x) for x in values): raise ValueError('Nonfinite feature scale input')
        scales[name] = [float(values.min()), float(values.max())]
    return {'schema_version': 1, 'base_catalog_sha256': base_catalog_sha256, 'flights': flights, 'journeys': journeys, 'feature_ranges': scales,
        'source': 'Supplied processed UTC flight/segment/candidate tables; all identifiers including airports replaced and all dates shifted by the original public-catalog offset',
        'scoring': 'Frozen original min/max feature scales; no refitting or clipping when adding routes; every original feature vector is reconstructed and checked before expansion',
        'limits': 'Original journey/ranking anomalies remain external. Airport-specific transfer rules and monetary client costs are unavailable.'}


def write_network(data, path):
    Path(path).write_bytes(gzip.compress(json.dumps(data, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
