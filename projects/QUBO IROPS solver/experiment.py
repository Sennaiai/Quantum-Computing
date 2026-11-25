"""Execute the provided de-identified IROPS catalog and record allocation metrics."""
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
from data_contract import ROOT, CONFIG, load_instance
from solver import solve_decomposed, score
from capacity_bounds import verify_public_bounds

NOTEBOOK = 'QUBO_IROPS_Allocation_Experiment.ipynb'


def source_hashes():
    files = list(ROOT.glob('*.py'))+[ROOT/'config.json', ROOT/'requirements.txt']
    files += list((ROOT/'data').glob('*'))
    out = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    if (ROOT/NOTEBOOK).exists():
        nb = json.loads((ROOT/NOTEBOOK).read_text())
        source = [(c['cell_type'], ''.join(c['source']), c.get('metadata', {}).get('tags', [])) for c in nb['cells']]
        out[NOTEBOOK] = hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()
    return dict(sorted(out.items()))


def load_public_catalog():
    provenance = json.loads((ROOT/'data/provenance.json').read_text())
    path = ROOT/'data/irops_catalog.json.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['sha256']:
        raise ValueError('Public catalog checksum mismatch')
    instance = load_instance(path)
    if len(instance.parties) != provenance['pnrs'] or len(instance.options) != provenance['options']:
        raise ValueError('Public catalog dimensions differ from provenance')
    return instance, provenance


def allocation_table(report):
    return pd.DataFrame([{k: row[k] for k in ('seed', 'assigned_pnrs', 'assigned_passengers',
            'pnr_allocation_percent', 'passenger_allocation_percent', 'capacity_violations', 'seconds')}
            for row in report['runs']]).round(4)


def run_experiment(output_dir=None, progress=True):
    tick = perf_counter(); before = source_hashes()
    instance, provenance = load_public_catalog()
    capacity_bounds = verify_public_bounds(instance)
    scenario = 'S3_capacityStability'; runs = []; solutions = {}
    for seed in CONFIG['seeds']:
        assignment, row = solve_decomposed(instance, scenario, seed)
        # A separate, strict scoring pass certifies the returned assignment.
        audit = score(instance, assignment, scenario)
        if audit['capacity_violations']: raise ArithmeticError('Independent capacity audit failed')
        for field, value in audit.items():
            if not np.isclose(row[field], value, rtol=1e-10, atol=1e-8):
                raise ArithmeticError('Final solver report differs from independent rescoring')
        row['seed'] = seed
        row['resources'] = {'chunks': len(row['chunks']),
               'peak_variables': max((c['variables'] for c in row['chunks']), default=0),
               'peak_couplers': max((c['couplers'] for c in row['chunks']), default=0),
               'maximum_ising_dynamic_range': max((c['ising_dynamic_range'] for c in row['chunks']), default=0),
               'raw_zero_residual_samples': sum(c['samples_with_zero_constraint_residual'] for c in row['chunks']),
               'total_samples': sum(c['reads'] for c in row['chunks']),
               'summed_raw_chunk_overloads': sum(c['raw_best_overloaded_legs'] for c in row['chunks'])}
        runs.append(row); solutions[str(seed)] = assignment
        if progress:
            print(f"seed {seed}: PNR {row['pnr_allocation_percent']:.4f}%, passenger {row['passenger_allocation_percent']:.4f}%, capacity violations {row['capacity_violations']}, {row['seconds']:.2f}s", flush=True)
    summary = {}
    for field in ('pnr_allocation_percent', 'passenger_allocation_percent', 'assigned_pnrs', 'seconds'):
        values = np.array([r[field] for r in runs], float)
        summary[field] = {'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)),
                          'minimum': float(values.min()), 'maximum': float(values.max())}
    report = {'status': 'passed', 'source_hashes': before, 'scenario': scenario,
              'catalog': provenance, 'config': CONFIG, 'runs': runs, 'summary': summary,
              'capacity_upper_bounds': capacity_bounds,
              'client_comparison': {'scope': 'Client performs comparison independently; outside this solver',
                                    'outperformance_claim': False},
              'elapsed_seconds': perf_counter()-tick,
              'scope': 'Allocation within provided candidate catalog; fixed weighted S3 objective with explicit unassignment cost, not a proof of maximum allocation'}
    if before != source_hashes(): raise RuntimeError('Experiment sources changed during execution')
    if output_dir is not None:
        output = Path(output_dir).resolve()
        if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
            raise ValueError('Generated evidence must stay in artifacts/ or outside source project')
        output.mkdir(parents=True, exist_ok=True)
        (output/'actual_catalog_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        table = pd.DataFrame({'PNR': [p.id for p in instance.parties],
                    **{f'seed_{seed}': [solutions[str(seed)].get(p.id) for p in instance.parties] for seed in CONFIG['seeds']}})
        table.to_csv(output/'assignments.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        plot_experiment(report, output/'allocation_resources.svg')
    report['timing'] = {'end_to_end_seconds': perf_counter()-tick,
        'scope': 'Function entry through catalog loading, exact capacity-certificate verification, all fixed-seed solver runs, independent final audits, plots and result exports; excludes dependency installation and interpreter/kernel startup',
        'seed_count': len(CONFIG['seeds']), 'system': platform.system(),
        'architecture': platform.machine(), 'python': platform.python_version(),
        'requested_openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'unspecified'),
        'requested_openmp_threads': os.environ.get('OMP_NUM_THREADS', 'unspecified')}
    if output_dir is not None:
        (output/'actual_catalog_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def plot_experiment(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(report['runs']))
    for field, label, color in [('pnr_allocation_percent', 'PNRs', '#246b84'),
                                 ('passenger_allocation_percent', 'Passengers', '#b36526')]:
        axes[0].plot(x, [r[field] for r in report['runs']], 'o-', label=label, color=color)
    axes[0].set_xticks(list(x), [r['seed'] for r in report['runs']]); axes[0].set_ylim(0, 100)
    axes[0].set_xlabel('Fixed annealing seed'); axes[0].set_ylabel('Feasible allocation (%)')
    axes[0].axhline(report['capacity_upper_bounds']['pnrs']['integer_upper_bound_percent'], linestyle='--', color='#246b84', label='PNR ceiling')
    axes[0].axhline(report['capacity_upper_bounds']['passengers']['integer_upper_bound_percent'], linestyle='--', color='#b36526', label='Passenger ceiling')
    axes[0].axhline(70, linestyle=':', color='grey', label='70% target')
    axes[0].legend(fontsize=8, loc='lower left'); axes[0].set_title('Allocation and certified capacity ceilings')
    chunks = report['runs'][0]['chunks']
    axes[1].scatter([c['variables'] for c in chunks], [c['couplers'] for c in chunks], s=14, alpha=.55, color='#246b84')
    axes[1].axvline(CONFIG['max_bqm_variables'], linestyle='--', color='grey', label='Variable budget')
    axes[1].set_xlabel('Logical BQM variables per chunk'); axes[1].set_ylabel('Nonzero quadratic couplers')
    axes[1].set_title('First-seed logical resource profile'); axes[1].legend()
    fig.suptitle('IROPS QUBO experiment • classical simulated annealing')
    fig.tight_layout(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)
    if path.suffix == '.svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/experiment')
    args = parser.parse_args(); report = run_experiment(args.output_dir)
    print(allocation_table(report).to_string(index=False))
