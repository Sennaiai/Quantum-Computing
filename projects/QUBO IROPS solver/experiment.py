"""Matched-catalog QUBO allocation and connection-expansion ablation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from data_contract import ROOT, CONFIG, Instance, load_instance, instance_fingerprint
from solver import solve_decomposed, score
from allocation_refinement import refine_allocation
from capacity_bounds import verify_public_bounds, verify_capacity_bounds
from connections import load_network, expand_connections, route_quality, route_violations

NOTEBOOK = 'QUBO_IROPS_Allocation_Experiment.ipynb'


def source_hashes():
    files = list(ROOT.glob('*.py'))+[ROOT/'config.json', ROOT/'requirements.txt']+list((ROOT/'data').glob('*'))
    out = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    if (ROOT/NOTEBOOK).exists():
        nb = json.loads((ROOT/NOTEBOOK).read_text())
        source = [(c['cell_type'], ''.join(c['source']), c.get('metadata', {}).get('tags', [])) for c in nb['cells']]
        out[NOTEBOOK] = hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()
    return dict(sorted(out.items()))


def load_public_catalog():
    provenance = json.loads((ROOT/'data/provenance.json').read_text()); path = ROOT/'data/irops_catalog.json.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['sha256']: raise ValueError('Public catalog checksum mismatch')
    instance = load_instance(path)
    if len(instance.parties) != provenance['pnrs'] or len(instance.options) != provenance['options']:
        raise ValueError('Public catalog dimensions differ from provenance')
    return instance, provenance


def load_experiment_catalogs():
    original, provenance = load_public_catalog()
    network = load_network(ROOT/'data/connection_network.json.gz', original, provenance['sha256'])
    expanded, network, audit = expand_connections(original, network)
    retained = tuple(o for o in expanded.options if o.id in original.option_by_id)
    control = Instance(original.parties, retained, dict(original.capacities), 'provided_catalog_policy_filtered_v1')
    return control, expanded, network, provenance, audit


def allocation_table(report):
    rows = []
    fields = ('seed', 'assigned_pnrs', 'assigned_passengers', 'pnr_allocation_percent',
              'passenger_allocation_percent', 'capacity_violations', 'seconds')
    for label, runs in [('Original routes', report['controls']), ('Expanded routes', report['runs'])]:
        rows.extend({'catalog': label, **{k: row[k] for k in fields}} for row in runs)
    return pd.DataFrame(rows).round(4)


def resource_summary(row):
    return {'chunks': len(row['chunks']), 'peak_variables': max((c['variables'] for c in row['chunks']), default=0),
        'peak_couplers': max((c['couplers'] for c in row['chunks']), default=0),
        'maximum_ising_dynamic_range': max((c['ising_dynamic_range'] for c in row['chunks']), default=0),
        'raw_zero_residual_samples': sum(c['samples_with_zero_constraint_residual'] for c in row['chunks']),
        'total_samples': sum(c['reads'] for c in row['chunks']),
        'summed_raw_chunk_overloads': sum(c['raw_best_overloaded_legs'] for c in row['chunks'])}


def execute_variant(instance, network, scenario, seed):
    tick = perf_counter(); assignment, sa = solve_decomposed(instance, scenario, seed)
    assignment, refinement = refine_allocation(instance, assignment, scenario)
    actual = score(instance, assignment, scenario)
    if actual['capacity_violations']: raise ArithmeticError('Independent capacity audit failed')
    if any(route_violations(instance, network, p, instance.option_by_id[oid].legs, CONFIG['connection_generation'])
           for p, oid in assignment.items()): raise ArithmeticError('Final route-policy audit failed')
    row = {**sa, **actual, 'seed': seed, 'after_decomposed_sa': refinement['before'],
        'allocation_refinement': refinement, 'decomposed_sa_seconds': sa['seconds'], 'resources': resource_summary(sa),
        'service_proxies': route_quality(instance, assignment, network),
        'seconds': perf_counter()-tick}
    return assignment, row


def summarize(runs):
    fields = ('pnr_allocation_percent', 'passenger_allocation_percent', 'assigned_pnrs', 'assigned_passengers', 'seconds')
    return {field: {'mean': float(np.mean([r[field] for r in runs])),
                    'sample_std': float(np.std([r[field] for r in runs], ddof=1)),
                    'minimum': float(min(r[field] for r in runs)), 'maximum': float(max(r[field] for r in runs))}
        for field in fields}


def run_experiment(output_dir=None, progress=True):
    tick = perf_counter(); before = source_hashes()
    control, expanded, network, provenance, connection_audit = load_experiment_catalogs()
    control_bounds = verify_public_bounds(control)
    certificate = json.loads((ROOT/'data/expanded_capacity_bounds.json').read_text())
    expanded_hash = instance_fingerprint(expanded)
    capacity_bounds = verify_capacity_bounds(expanded, certificate, expanded_hash)
    scenario = 'S3_capacityStability'; runs = []; controls = []; solutions = {}; control_solutions = {}
    for seed in CONFIG['seeds']:
        control_assignment, control_row = execute_variant(control, network, scenario, seed)
        assignment, row = execute_variant(expanded, network, scenario, seed)
        # Predeclared acceptance: improvements must survive a matched population,
        # strict recovery policy, identical S3 weights and unchanged inventory.
        if row['assigned_pnrs'] < control_row['assigned_pnrs'] or row['assigned_passengers'] < control_row['assigned_passengers']:
            raise ArithmeticError('Expanded-catalog experiment regressed allocation')
        if row['objective'] > control_row['objective']+1e-7: raise ArithmeticError('Expanded-catalog S3 objective regressed')
        controls.append(control_row); runs.append(row)
        solutions[str(seed)] = assignment; control_solutions[str(seed)] = control_assignment
        if progress:
            print(f"seed {seed}: original {control_row['pnr_allocation_percent']:.4f}% PNR/{control_row['passenger_allocation_percent']:.4f}% passengers; expanded {row['pnr_allocation_percent']:.4f}% PNR/{row['passenger_allocation_percent']:.4f}% passengers; zero capacity and route-policy violations", flush=True)
    report = {'schema_version': 2, 'status': 'passed', 'source_hashes': before, 'scenario': scenario,
        'catalog': provenance, 'config': CONFIG, 'connection_audit': connection_audit,
        'expanded_instance_fingerprint': expanded_hash, 'fingerprint_type': 'sha256 of canonical uncompressed instance JSON v1',
        'runs': runs, 'controls': controls, 'summary': summarize(runs), 'control_summary': summarize(controls),
        'capacity_upper_bounds': capacity_bounds, 'control_capacity_upper_bounds': control_bounds,
        'control_bound_scope': 'Original certificate remains valid on the policy-filtered subset with identical PNRs and capacities',
        'comparison': 'Internal matched-catalog ablation, not a client/industry benchmark. Expanded candidates, classical annealing and classical repair gains are separate; no quantum advantage attribution.',
        'client_comparison': {'scope': 'Client performs comparison independently; outside this solver', 'outperformance_claim': False},
        'scope': 'Provided candidate-catalog PNR population; original no-option bookings remain external. Fixed weighted S3 objective with explicit unassignment cost; bounded search does not prove maximum allocation.'}
    if before != source_hashes(): raise RuntimeError('Experiment sources changed during execution')
    if output_dir is not None:
        output = Path(output_dir).resolve()
        if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
            raise ValueError('Generated evidence must stay in artifacts/ or outside source project')
        output.mkdir(parents=True, exist_ok=True)
        (output/'actual_catalog_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        columns = {'PNR': [p.id for p in expanded.parties]}
        for seed in CONFIG['seeds']:
            columns[f'seed_{seed}'] = [solutions[str(seed)].get(p.id) for p in expanded.parties]
            columns[f'control_seed_{seed}'] = [control_solutions[str(seed)].get(p.id) for p in expanded.parties]
        pd.DataFrame(columns).to_csv(output/'assignments.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        plot_experiment(report, output/'allocation_resources.svg')
    report['timing'] = {'end_to_end_seconds': perf_counter()-tick,
        'scope': 'Catalog and network loading, feature verification, connection generation, exact capacity-certificate verification, all fixed-seed control and expanded QUBO/refinement runs, independent final audits, plots and result exports; excludes dependency installation and interpreter/kernel startup',
        'seed_count': len(CONFIG['seeds']), 'variants_per_seed': 2, 'system': platform.system(),
        'architecture': platform.machine(), 'python': platform.python_version(),
        'requested_openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'unspecified'),
        'requested_openmp_threads': os.environ.get('OMP_NUM_THREADS', 'unspecified')}
    if output_dir is not None: (output/'actual_catalog_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def plot_experiment(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    colors = ('#246b84', '#b36526')
    for j, (field, label) in enumerate([('pnr_allocation_percent', 'PNRs'), ('passenger_allocation_percent', 'Passengers')]):
        means = [report[group][field]['mean'] for group in ('control_summary', 'summary')]
        deviations = [report[group][field]['sample_std'] for group in ('control_summary', 'summary')]
        axes[0].bar(np.arange(2)+(j-.5)*.32, means, width=.32, yerr=deviations, capsize=4, label=label, color=colors[j])
    axes[0].set_xticks([0, 1], ['Original routes', 'Expanded routes']); axes[0].set_ylim(0, 100)
    axes[0].axhline(70, linestyle=':', color='grey', label='70% PNR target')
    axes[0].set_ylabel('Feasible allocation (%)'); axes[0].set_title('Same PNRs and inventory; mean ± seed SD'); axes[0].legend(fontsize=8)
    for label, row, color in [('Original', report['controls'][0], colors[0]), ('Expanded', report['runs'][0], colors[1])]:
        axes[1].scatter([c['variables'] for c in row['chunks']], [c['couplers'] for c in row['chunks']], s=14, alpha=.45, label=label, color=color)
    axes[1].axvline(CONFIG['max_bqm_variables'], linestyle='--', color='grey', label='Variable budget')
    axes[1].set_xlabel('Logical BQM variables per chunk'); axes[1].set_ylabel('Nonzero quadratic couplers')
    axes[1].set_title('First-seed QUBO resource profile'); axes[1].legend(fontsize=8)
    fig.suptitle('IROPS QUBO + classical refinement • candidate-coverage ablation'); fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)
    if path.suffix == '.svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/experiment')
    args = parser.parse_args(); print(allocation_table(run_experiment(args.output_dir)).to_string(index=False))
