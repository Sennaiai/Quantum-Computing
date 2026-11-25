"""Create an authorized public solver catalog without original booking/flight IDs."""
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from data_contract import (ROOT, FEATURES, Instance, Party, dump_instance, load_private_csvs)


def deidentify(instance):
    pnrs = {p: f'PNR_{i:06}' for i, p in enumerate(sorted(p.id for p in instance.parties))}
    options = {p: f'OPT_{i:06}' for i, p in enumerate(sorted(o.id for o in instance.options))}
    flights = {p: f'LEG_{i:06}' for i, p in enumerate(sorted({f for f, c in instance.capacities}))}
    index = FEATURES.index('CVM_NORM'); priorities = {}
    for o in instance.options:
        value = o.features[index]
        if o.party in priorities and priorities[o.party] != value: raise ValueError('Normalized priority differs across a PNR')
        priorities[o.party] = value
    for p in instance.parties:
        if p.id not in priorities: raise ValueError('An original PNR without normalized priority needs an explicit input value')
    raw_order = sorted(instance.parties, key=lambda p: (p.cvm, p.id))
    norm_order = sorted(instance.parties, key=lambda p: (priorities[p.id], p.id))
    if [p.id for p in raw_order] != [p.id for p in norm_order]: raise ValueError('Priority normalization changes PNR ordering')
    # Also preserve ties, including sorting by PAX in the original greedy policy.
    for a, b in zip(raw_order, raw_order[1:]):
        if (a.cvm == b.cvm) != (priorities[a.id] == priorities[b.id]): raise ValueError('Priority ties changed during normalization')
    start = min(o.departure for o in instance.options).astimezone(timezone.utc)
    shift = datetime(2025, 1, 1, tzinfo=timezone.utc)-start.replace(hour=0, minute=0, second=0, microsecond=0)
    clean_parties = tuple(Party(pnrs[p.id], p.pax, priorities[p.id]) for p in instance.parties)
    clean_options = tuple(replace(o, id=options[o.id], party=pnrs[o.party],
                         legs=tuple((flights[f], c) for f, c in o.legs), departure=o.departure+shift) for o in instance.options)
    clean_caps = {(flights[f], c): value for (f, c), value in instance.capacities.items()}
    clean = Instance(clean_parties, clean_options, clean_caps, 'provided_catalog_deidentified')
    for before, after in zip(instance.options, clean.options):
        assert before.features == after.features
        assert before.delay_minutes == after.delay_minutes
        assert int(before.departure.timestamp()//43200)+int(shift.total_seconds()//43200) == int(after.departure.timestamp()//43200)
    mapping = {'pnrs': pnrs, 'options': options, 'flights': flights, 'time_shift_seconds': shift.total_seconds()}
    audit = {'source': 'User-supplied processed IROPS candidate tables; public de-identified release authorized by repository owner',
             'synthetic': False, 'pnrs': len(clean.parties), 'options': len(clean.options),
             'passengers': sum(p.pax for p in clean.parties), 'flight_cabin_resources': len(clean.capacities),
             'transform': 'Replace all original identifiers by monotone anonymous labels; shift all UTC departures by a common whole-day offset; replace raw CVM by supplied normalized CVM',
             'preserved': 'Party sizes, candidate membership, scenario features/costs, flight/cabin incidence, capacities, relative times, 12-hour grouping, priority order/ties and deterministic label tie-breaking',
             'priority_units': 'Normalized CVM, unitless; captured priority is not raw client CVM',
             'scope': 'PNRs in the supplied candidate catalog; original impacted/no-candidate PNR population and upstream itinerary generation remain external'}
    return clean, mapping, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    source, audit = load_private_csvs(args.input_dir, capacity_policy='conservative')
    public, mapping, provenance = deidentify(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = args.output_dir/'irops_catalog.json.gz'; dump_instance(public, catalog)
    provenance.update(input_audit=audit, sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
       original_file_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.input_dir.glob('*.csv'))})
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    private = ROOT/'artifacts/private'; private.mkdir(parents=True, exist_ok=True)
    (private/'deidentification_map.json').write_text(json.dumps(mapping, separators=(',', ':'))+'\n')
    print(json.dumps({k: provenance[k] for k in ('synthetic', 'pnrs', 'options', 'passengers', 'sha256')}, indent=2))


if __name__ == '__main__': main()
