"""Optional local execution on processed inputs; outputs remain outside public evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
from data_contract import ROOT, CONFIG, load_private_csvs, instance_fingerprint
from solver import solve_decomposed, score
from allocation_refinement import refine_allocation
from connection_curation import build_connection_metadata
from connections import validate_network, expand_connections, route_quality, route_violations


def private_output(path):
    output = Path(path).resolve(); repo = ROOT.parents[1]
    if output.is_relative_to(repo) and not output.is_relative_to(ROOT/'artifacts/private'):
        raise ValueError('Private outputs inside the repository must stay under artifacts/private/')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/private')
    parser.add_argument('--reliable', action='store_true')
    parser.add_argument('--capacity-policy', choices=('strict', 'conservative'), default='strict')
    parser.add_argument('--scenario', choices=tuple(CONFIG['scenarios']), default='S3_capacityStability')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--expand-connections', action='store_true', help='Enumerate recovery routes using metadata in the same four processed tables')
    args = parser.parse_args()
    try: output = private_output(args.output_dir)
    except ValueError as e: parser.error(str(e))
    tick = perf_counter()
    instance, audit = load_private_csvs(args.input_dir, args.reliable, args.capacity_policy)
    network = None; connection_audit = None
    if args.expand_connections:
        fingerprint = instance_fingerprint(instance)
        mapping = {'pnrs': {p.id:p.id for p in instance.parties},
                   'flights': {f:f for f,c in instance.capacities}, 'time_shift_seconds': 0}
        metadata = build_connection_metadata(args.input_dir, mapping, fingerprint, reliable=args.reliable)
        network = validate_network(metadata, instance, fingerprint)
        instance, network, connection_audit = expand_connections(instance, network)
    assignment, report = solve_decomposed(instance, args.scenario, args.seed)
    assignment, refinement = refine_allocation(instance, assignment, args.scenario)
    final = score(instance, assignment, args.scenario)
    if final['capacity_violations']: raise ArithmeticError('Private final capacity audit failed')
    if network is not None and any(route_violations(instance, network, p, instance.option_by_id[o].legs, CONFIG['connection_generation']) for p,o in assignment.items()):
        raise ArithmeticError('Private final route audit failed')
    data = {'status': 'passed', 'source': 'local processed inputs; not public benchmark data',
            'input_audit': audit,
            'decomposed_sa': report, 'connection_audit': connection_audit,
            'allocation_refinement': refinement, 'final': final,
            'service_proxies': route_quality(instance, assignment, network) if network is not None else None}
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    pd.DataFrame([{'RECLOC': p.id, 'OPTION_ID': assignment.get(p.id)} for p in instance.parties]).to_csv(output/'solution.csv', index=False)
    data['end_to_end_seconds'] = perf_counter()-tick
    data['timing_scope'] = 'Processed input loading, optional connection generation, QUBO solving, refinement, audits and solution export; excludes installation/interpreter startup'
    (output/'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'passed', 'pnrs': len(instance.parties),
                      'assigned_pnrs': final['assigned_pnrs'], 'assigned_passengers': final['assigned_passengers'], 'capacity_violations': final['capacity_violations']}))


if __name__ == '__main__': main()
