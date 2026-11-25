"""Optional local execution on processed inputs; outputs remain outside public evidence."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_contract import ROOT, CONFIG, load_private_csvs
from solver import solve_decomposed


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
    args = parser.parse_args()
    try: output = private_output(args.output_dir)
    except ValueError as e: parser.error(str(e))
    instance, audit = load_private_csvs(args.input_dir, args.reliable, args.capacity_policy)
    assignment, report = solve_decomposed(instance, args.scenario, args.seed)
    data = {'status': 'passed', 'source': 'local processed inputs; not public benchmark data',
            'input_audit': audit,
            'decomposed_sa': report}
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    pd.DataFrame([{'RECLOC': p.id, 'OPTION_ID': assignment.get(p.id)} for p in instance.parties]).to_csv(output/'solution.csv', index=False)
    print(json.dumps({'status': 'passed', 'pnrs': len(instance.parties),
                      'assigned_pnrs': report['assigned_pnrs'], 'capacity_violations': report['capacity_violations']}))


if __name__ == '__main__': main()
