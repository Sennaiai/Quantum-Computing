"""Validation-selected models, locked heldout test, paired temporal controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_contract import ROOT, SEEDS, load_data, prepare, split_description
from reservoirs import CVReservoir, SpinReservoir, echo_features

NOTEBOOKS = ('CV_Reservoir_Benchmark.ipynb', 'Analogue_Spin_QRC_Benchmark.ipynb')


def source_hashes():
    files = list(ROOT.glob('*.py'))+[ROOT/'requirements.txt', ROOT/'protocol.json']
    files += list((ROOT/'data').glob('*'))
    out = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    for name in NOTEBOOKS:
        if (ROOT/name).exists():
            n = json.loads((ROOT/name).read_text())
            # Execution counts/outputs are evidence, not executable source identity.
            code = [(c['cell_type'], ''.join(c['source'])) for c in n['cells']]
            out[name] = hashlib.sha256(json.dumps(code, ensure_ascii=False).encode()).hexdigest()
    return dict(sorted(out.items()))


def metrics(y, pred, scale):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or y.ndim != 1 or not np.isfinite(pred).all() or scale <= 0:
        raise ValueError('Invalid forecast metrics input')
    return {'MAE_MW': float(np.mean(abs(y-pred))),
            'RMSE_MW': float(np.sqrt(np.mean((y-pred)**2))),
            'MASE': float(np.mean(abs(y-pred))/scale)}


def model_factory(kind, params):
    if kind == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(**params))
    if kind == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=19, n_jobs=1, **params)
    if kind == 'boost':
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=.05,
                    early_stopping=False, random_state=19, **params)
    raise ValueError('Unknown model family')


def select_model(kind, grid, x, data):
    """This function cannot inspect heldout labels: every candidate uses folds()."""
    if x.shape[0] != len(data.y) or not np.isfinite(x).all():
        raise ValueError('Misaligned/non-finite feature matrix')
    residual = data.y-data.persistence
    scale = float(np.mean(abs(np.diff(data.y[:data.initial_train]))))
    candidates = []
    for params in grid:
        truth, pred = [], []
        for train, validation in data.folds():
            if train[-1] >= validation[0] or validation[-1] >= data.test_start:
                raise ValueError('Validation must be chronological and pre-test')
            model = model_factory(kind, params).fit(x[train], residual[train])
            truth.extend(data.y[validation])
            pred.extend(data.persistence[validation]+model.predict(x[validation]))
        candidates.append({'params': params, **metrics(np.array(truth), np.array(pred), scale)})
    best = min(candidates, key=lambda p: p['RMSE_MW'])
    return dict(best['params']), candidates


def fit_evaluate(kind, grid, x, data):
    params, candidates = select_model(kind, grid, x, data)
    start = perf_counter()
    model = model_factory(kind, params).fit(x[:data.test_start],
                        (data.y-data.persistence)[:data.test_start])
    seconds = perf_counter()-start
    p = data.persistence+model.predict(x)
    scale = float(np.mean(abs(np.diff(data.y[:data.initial_train]))))
    row = {'selected_params': params, 'validation_candidates': candidates,
           'validation_RMSE_MW': min(c['RMSE_MW'] for c in candidates),
           'train': metrics(data.y[:data.test_start], p[:data.test_start], scale),
           'test': metrics(data.y[data.test_start:], p[data.test_start:], scale),
           'features': x.shape[1], 'fit_seconds': seconds,
           'test_to_train_MAE_ratio': float(np.mean(abs(data.y[data.test_start:]-p[data.test_start:])) /
                                           max(np.mean(abs(data.y[:data.test_start]-p[:data.test_start])), 1e-12))}
    final = model.steps[-1][1] if hasattr(model, 'steps') else model
    if hasattr(final, 'coef_'): row['trained_readout_coefficients_including_intercept'] = int(final.coef_.size+1)
    if hasattr(final, 'estimators_'): row['tree_nodes'] = int(sum(t.tree_.node_count for t in final.estimators_))
    return row, p[data.test_start:]


def paired_day_bootstrap(y, pred, reference, repeats=1000):
    """Paired 48-step nonoverlapping block means; conditional exploratory interval.

    Fixed 24-hour elapsed blocks, not calendar-day/DST bins. Incomplete final
    block excluded. Does not account for model selection or seasonal dependence.
    """
    differences = abs(y-pred)-abs(y-reference)
    full = len(differences)//48
    if full < 5: raise ValueError('Too few test blocks for temporal bootstrap')
    blocks = differences[:full*48].reshape(full, 48).mean(1)
    rng = np.random.default_rng(812)
    means = blocks[rng.integers(full, size=(repeats, full))].mean(1)
    return {'MAE_difference_MW': float(differences.mean()),
            'block_mean_difference_MW': float(blocks.mean()),
            'conditional_95pct_interval_MW': np.quantile(means, [.025, .975]).tolist(),
            'blocks': full, 'block_steps': 48, 'resamples': repeats,
            'interpretation': 'Exploratory conditional interval; no multiplicity/selection correction or advantage claim'}


RIDGE_GRID = [{'alpha': a} for a in (.1, 10., 1000.)]
CLASSICAL = {
    'raw_ridge': ('ridge', RIDGE_GRID),
    'random_forest': ('rf', [{'max_depth': 12, 'min_samples_leaf': 5},
                           {'max_depth': None, 'min_samples_leaf': 12}]),
    'gradient_boosting': ('boost', [{'max_leaf_nodes': 15, 'l2_regularization': 1.},
                                  {'max_leaf_nodes': 31, 'l2_regularization': 10.}])}


def run_study(study, output_dir=None):
    if study not in ('cv', 'spin'): raise ValueError('Study must be cv or spin')
    data = prepare(load_data()); start = perf_counter()
    rows, predictions, diagnostics = {}, {}, {}
    scale = float(np.mean(abs(np.diff(data.y[:data.initial_train]))))
    val = slice(data.initial_train, data.test_start); test = slice(data.test_start, None)
    for name, p in [('persistence', data.persistence), ('daily', data.daily), ('weekly', data.weekly)]:
        rows[name] = {'validation_RMSE_MW': metrics(data.y[val], p[val], scale)['RMSE_MW'],
                      'test': metrics(data.y[test], p[test], scale), 'features': 1, 'trained_parameters': 0}
        predictions[name] = p[test]
    for name, (kind, grid) in CLASSICAL.items():
        rows[name], predictions[name] = fit_evaluate(kind, grid, data.raw, data)
        print(study, name, rows[name]['test'], flush=True)
    classical_reference = min(rows, key=lambda k: rows[k]['validation_RMSE_MW'])
    families = ('gaussian', 'kerr') if study == 'cv' else ('coupled_spin', 'uncoupled_spin')
    for family in (*families, 'classical_echo'):
        feature_seconds = []
        for seed in SEEDS:
            tick = perf_counter()
            if family in ('gaussian', 'kerr'):
                res = CVReservoir(seed, kerr=0. if family == 'gaussian' else .2)
                h, diag = res.features(data.windows)
                if diag['max_cutoff_boundary_probability'] > 1e-7:
                    raise ArithmeticError('Fock boundary probability exceeds declared acceptance')
                if family == 'gaussian':
                    diag['max_feature_error_vs_coherent_amplitudes'] = float(np.max(abs(h-res.gaussian_reference(data.windows))))
                    if diag['max_feature_error_vs_coherent_amplitudes'] > 1e-6:
                        raise ArithmeticError('Gaussian reference disagreement')
            elif family == 'classical_echo':
                h = echo_features(data.windows, seed); diag = {}
            else:
                h, diag = SpinReservoir(seed, coupled=(family == 'coupled_spin')).features(data.windows)
            elapsed = perf_counter()-tick; feature_seconds.append(elapsed)
            diagnostics[f'{family}_{seed}'] = {**diag, 'feature_seconds': elapsed,
                                              'feature_matrix_bytes': h.nbytes}
            # Both reservoir-only and hybrid use the same causal persistence anchor.
            for variant, x in [('reservoir', h), ('hybrid', np.column_stack([data.raw, h]))]:
                name = f'{family}_{variant}_seed{seed}'
                rows[name], predictions[name] = fit_evaluate('ridge', RIDGE_GRID, x, data)
            print(study, family, seed, rows[f'{family}_hybrid_seed{seed}']['test'], flush=True)
        for variant in ('reservoir', 'hybrid'):
            name = f'{family}_{variant}_ensemble'
            p = np.mean([predictions[f'{family}_{variant}_seed{s}'] for s in SEEDS], axis=0)
            predictions[name] = p
            member_names = [f'{family}_{variant}_seed{s}' for s in SEEDS]
            rows[name] = {'test': metrics(data.y[test], p, scale), 'members': member_names,
                           'seed_test_RMSE_mean_MW': float(np.mean([rows[k]['test']['RMSE_MW'] for k in member_names])),
                           'seed_test_RMSE_std_MW': float(np.std([rows[k]['test']['RMSE_MW'] for k in member_names], ddof=1)),
                           'ensemble_rule': 'Equal weights fixed before test; all three seeds retained'}
            rows[name]['paired_vs_validation_selected_classical'] = paired_day_bootstrap(
                    data.y[test], p, predictions[classical_reference])
    report = {'status': 'passed', 'study': study, 'source_hashes': source_hashes(),
              'protocol': json.loads((ROOT/'protocol.json').read_text()),
              'splits': split_description(data), 'models': rows, 'feature_diagnostics': diagnostics,
              'classical_reference_selected_on_validation': classical_reference,
              'test_targets': len(data.y)-data.test_start, 'elapsed_seconds': perf_counter()-start}
    if output_dir is not None:
        output = Path(output_dir).resolve()
        if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
            raise ValueError('Generated results must be in artifacts/ or outside source project')
        output.mkdir(parents=True, exist_ok=True)
        (output/f'{study}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        pd.DataFrame({'target_utc': data.times[test].astype(str), 'actual_MW': data.y[test], **predictions}).to_csv(
            output/f'{study}_predictions.csv', index=False, float_format='%.12g')
    return report


def summary_table(report):
    names = list(CLASSICAL)+['persistence', 'daily', 'weekly']
    names += [n for n in report['models'] if n.endswith('_ensemble')]
    return pd.DataFrame([{ 'model': n, **report['models'][n]['test']} for n in names]).set_index('model').round(3)


def plot_study(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = summary_table(report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.5, 1]})
    colors = ['#246b84' if 'ensemble' in n else '#7a8791' for n in table.index]
    axes[0].barh(table.index, table.RMSE_MW, color=colors)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Heldout RMSE (MW), lower is better')
    ensembles = [n for n in table.index if n.endswith('hybrid_ensemble')]
    for i, n in enumerate(ensembles):
        r = report['models'][n]['paired_vs_validation_selected_classical']
        lo, hi = r['conditional_95pct_interval_MW']; mean = r['block_mean_difference_MW']
        axes[1].plot([lo, hi], [i, i], color='#246b84', linewidth=3)
        axes[1].plot(mean, i, 'o', color='#163a54')
    axes[1].axvline(0, color='grey', linestyle='--')
    axes[1].set_yticks(range(len(ensembles)), [n.replace('_hybrid_ensemble', '') for n in ensembles])
    axes[1].set_xlabel('MAE difference vs validation-selected\nclassical baseline (MW); conditional 95% interval')
    fig.suptitle(f'{report["study"].upper()} reservoir benchmark • rolling 30-minute hindcast')
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('cv', 'spin'), required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/benchmark')
    args = parser.parse_args()
    report = run_study(args.study, args.output_dir)
    plot_study(report, args.output_dir/f'{args.study}.svg')
    print(summary_table(report).to_string())
