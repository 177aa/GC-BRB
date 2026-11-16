#!/usr/bin/env python3
"""Validation-tuned XGBoost/MLP on the newly exported 37D evidence.

Example: python python/run_same_input_baselines.py --run-dir results/GG_OOF_1_0/pilot/CPDB/run_01
Test labels are read for reporting only. No CV is performed on a fixed matrix.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group_scores(y, score, groups):
    groups = np.asarray(groups)
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    unique, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    yg = np.bincount(inv, weights=y) / counts
    pg = np.bincount(inv, weights=score) / counts
    if not np.isin(yg, [0, 1]).all() or not np.isfinite(pg).all():
        raise ValueError('Conflicting group labels or nonfinite scores')
    return yg.astype(int), pg, unique


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= threshold
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    tn, fn = np.sum(~pred & (y == 0)), np.sum(~pred & (y == 1))
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    return dict(AUPRC=float(average_precision_score(y, p)),
                AUROC=float(roc_auc_score(y, p)), Brier=float(np.mean((p-y)**2)),
                MCC=float((tp*tn-fp*fn)/denom) if denom else 0.,
                F1=float(2*tp/max(2*tp+fp+fn, 1)))


def validation_threshold(y, score):
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[ss[:-1] != ss[1:], True]
    tp = np.cumsum(yy)[ends].astype(float)
    fp = np.cumsum(1-yy)[ends].astype(float)
    fn, tn = y.sum()-tp, len(y)-y.sum()-fp
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    values = np.divide(tp*tn-fp*fn, denominator,
                       out=np.zeros_like(tp), where=denominator>0)
    candidates = np.flatnonzero(np.abs(values-values.max()) <= 1e-12)
    thresholds = ss[ends]
    chosen = candidates[np.argmin(np.abs(thresholds[candidates]-.5))]
    return float(thresholds[chosen])


def load_run(folder: Path):
    meta = json.loads((folder/'baseline_metadata.json').read_text(encoding='utf-8'))
    if meta['Protocol'] != 'GG_OOF_1_0' or meta['Stage'] not in ['smoke', 'pilot', 'formal']:
        raise ValueError('Unsupported protocol/stage')
    feature_file = folder/'baseline_features.csv'
    if digest(feature_file) != meta['FeatureSHA256']:
        raise ValueError('Feature file hash differs from MATLAB export')
    with feature_file.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    columns = meta['FeatureColumns']
    if columns != [f'F{i:02d}' for i in range(1, 38)]:
        raise ValueError('Expected the original ordered 37D feature columns')
    X = np.array([[float(r[c]) for c in columns] for r in rows])
    y = np.array([int(r['Label']) for r in rows])
    groups = np.array([int(r['GroupID']) for r in rows])
    ix = np.array([int(r['GlobalIndex']) for r in rows])
    partitions = np.array([r['Partition'] for r in rows])
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid features or labels')
    if len(ix) != len(set(ix.tolist())):
        raise ValueError('Duplicate record indices')
    masks = {s: partitions == s for s in ['Train', 'Validation', 'Test']}
    if not all(m.any() for m in masks.values()) or sum(m.sum() for m in masks.values()) != len(y):
        raise ValueError('Invalid partitions')
    for a, b in itertools.combinations(masks, 2):
        if set(groups[masks[a]]) & set(groups[masks[b]]):
            raise ValueError('Identity groups cross partitions')
    for mask in masks.values():
        if len(set(y[mask])) != 2:
            raise ValueError('Both classes are required in each subset')
        group_scores(y[mask], np.zeros(mask.sum()), groups[mask])
    return meta, X, y, groups, ix, masks


def candidates(name, seed, smoke=False):
    if name == 'xgb':
        from xgboost import XGBClassifier
        for depth, lr, reg in itertools.product([3, 5], [.03, .1], [1., 10.]):
            params = dict(max_depth=depth, learning_rate=lr, reg_lambda=reg,
                          n_estimators=5 if smoke else 300, subsample=.8,
                          colsample_bytree=1., min_child_weight=1.,
                          tree_method='hist', objective='binary:logistic',
                          eval_metric='logloss', n_jobs=1, random_state=seed)
            yield params, XGBClassifier(**params)
            if smoke:
                break
    elif name == 'mlp':
        for width, alpha in itertools.product([32, 64], [1e-4, 1e-3]):
            params = dict(hidden_layer_sizes=(width,), alpha=alpha,
                          max_iter=5 if smoke else 400, solver='adam',
                          early_stopping=False, random_state=seed, tol=1e-5,
                          n_iter_no_change=30)
            yield params, MLPClassifier(**params)
            if smoke:
                break
    else:
        raise ValueError(name)


def fit_one(name, X, y, groups, masks, seed, smoke=False):
    # Test y is not referenced until model and hyperparameters are fixed.
    tr, va, te = [masks[s] for s in ['Train', 'Validation', 'Test']]
    if name == 'mlp':
        scaler = StandardScaler().fit(X[tr])
        data = scaler.transform(X)
    else:
        data = X
    weights = compute_sample_weight('balanced', y[tr])
    best = None
    trials = []
    for trial, (params, model) in enumerate(candidates(name, seed, smoke)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(data[tr], y[tr], sample_weight=weights)
        p = model.predict_proba(data[va])[:, 1]
        vg_y, vg_p, _ = group_scores(y[va], p, groups[va])
        val = metrics(vg_y, vg_p)
        row = dict(Trial=trial, Params=json.dumps(params),
                   GroupValAUPRC=val['AUPRC'], GroupValBrier=val['Brier'],
                   Warnings='; '.join(str(w.message) for w in caught))
        trials.append(row)
        key = (-val['AUPRC'], val['Brier'], trial)
        if best is None or key < best[0]:
            best = key, model, params, p, vg_y, vg_p
    _, model, params, val_score, vg_y, vg_p = best
    # Freeze validation-selected parameters/threshold before scoring the test.
    threshold = validation_threshold(vg_y, vg_p)
    test_score = model.predict_proba(data[te])[:, 1]
    tg_y, tg_p, tg = group_scores(y[te], test_score, groups[te])
    group_metric = metrics(tg_y, tg_p, threshold)
    entry_metric = metrics(y[te], test_score, threshold)
    return dict(name=name, params=params, threshold=threshold,
                group_metrics=group_metric, entry_metrics=entry_metric,
                trials=trials, test_score=test_score, test_group_id=tg,
                test_group_score=tg_p, test_group_y=tg_y)


def run_folder(folder: Path, models):
    meta, X, y, groups, ix, masks = load_run(folder)
    out = folder/'baselines_python'
    out.mkdir(exist_ok=True)
    import xgboost
    versions = dict(Python=platform.python_version(), NumPy=np.__version__,
                    ScikitLearn=sklearn.__version__, XGBoost=xgboost.__version__)
    fingerprint = hashlib.sha256(json.dumps(dict(metadata=meta, versions=versions,
        script=digest(Path(__file__))), sort_keys=True).encode()).hexdigest()
    summary = []
    for name in models:
        status_path = out/f'{name}_status.json'
        if status_path.exists():
            old = json.loads(status_path.read_text(encoding='utf-8'))
            if old['Fingerprint'] != fingerprint:
                raise ValueError('Baseline environment/config changed; use a new output folder')
            summary.append(old['Summary'])
            continue
        result = fit_one(name, X, y, groups, masks, int(meta['Seed']), meta['Stage']=='smoke')
        write_csv(out/f'{name}_validation_search.csv', result['trials'])
        test_rows = [dict(GlobalIndex=int(i), GroupID=int(g), Label=int(v), Probability=float(p))
                     for i, g, v, p in zip(ix[masks['Test']], groups[masks['Test']],
                                          y[masks['Test']], result['test_score'])]
        write_csv(out/f'{name}_test_predictions.csv', test_rows)
        row = dict(Target=meta['Target'], Run=meta['Run'], Stage=meta['Stage'],
                   Model=name, GroupTestAUPRC=result['group_metrics']['AUPRC'],
                   GroupTestAUROC=result['group_metrics']['AUROC'],
                   GroupTestMCC=result['group_metrics']['MCC'],
                   GroupTestBrier=result['group_metrics']['Brier'],
                   EntryTestAUPRC=result['entry_metrics']['AUPRC'],
                   EntryTestAUROC=result['entry_metrics']['AUROC'],
                   ValidationMCCThreshold=result['threshold'])
        write_json(status_path, dict(Fingerprint=fingerprint, Versions=versions,
            Params=result['params'], Summary=row, Complete=True,
            TestUsedForSelection=False, FeatureSHA256=meta['FeatureSHA256']))
        summary.append(row)
    write_csv(out/'summary.csv', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--results-root', type=Path)
    parser.add_argument('--models', nargs='+', choices=['xgb', 'mlp'], default=['xgb', 'mlp'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise SystemExit('Python 3.11+ is required by the tested dependency versions')
    folders = [args.run_dir] if args.run_dir else sorted(
        p.parent for p in args.results_root.glob('*/run_*/baseline_metadata.json'))
    if not folders:
        raise SystemExit('No completed MATLAB feature exports found')
    rows = []
    for folder in folders:
        load_run(folder)
        if args.check_only:
            print('INPUT CHECK PASSED:', folder)
        else:
            print('RUN:', folder, flush=True)
            rows.extend(run_folder(folder, args.models))
    if args.results_root and rows:
        write_csv(args.results_root/'same_input_baselines_all_runs.csv', rows)


if __name__ == '__main__':
    main()
