#!/usr/bin/env python3
"""Independently check an actual MATLAB run's exported memberships (stdlib only)."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def verify(folder):
    split = read(folder/'split_manifest.csv')
    parts = defaultdict(set)
    record_parts = defaultdict(set)
    labels = {}
    groups = {}
    for r in split:
        i, g = int(r['GlobalIndex']), int(r['GroupID'])
        if i in labels:
            raise ValueError('Duplicate global index in split')
        labels[i], groups[i] = int(r['Label']), g
        parts[r['Partition']].add(g)
        record_parts[r['Partition']].add(i)
    for a, b in [('Train', 'Validation'), ('Train', 'Test'), ('Validation', 'Test')]:
        assert parts[a].isdisjoint(parts[b]), (a, b)
    assert parts['ExcludedTestAlias'] <= parts['Test']
    pipelines = defaultdict(lambda: defaultdict(set))
    for r in read(folder/'pipeline_memberships.csv'):
        i = int(r['GlobalIndex'])
        assert groups[i] == int(r['GroupID']) and labels[i] == int(r['Label'])
        pipelines[r['Pipeline']][r['Role']].add(i)
    all_held = []
    for name, roles in pipelines.items():
        for a, b in [('Fit', 'Stop'), ('Fit', 'Held'), ('Stop', 'Held')]:
            assert {groups[i] for i in roles[a]}.isdisjoint(groups[i] for i in roles[b])
        if name.startswith('outer_'):
            assert roles['Fit'] | roles['Stop'] | roles['Held'] == record_parts['Train']
            all_held.extend(roles['Held'])
        elif name == 'final':
            assert roles['Fit'] == record_parts['Train']
            assert roles['Stop'] == record_parts['Validation']
            assert roles['Held'] == record_parts['Test']
        fold_file = folder/name/'evidence_training_folds.csv'
        if fold_file.exists():
            fold_rows = read(fold_file)
            assert {int(r['GlobalIndex']) for r in fold_rows} == roles['Fit']
            gf = defaultdict(set)
            for r in fold_rows:
                assert groups[int(r['GlobalIndex'])] == int(r['GroupID'])
                gf[int(r['GroupID'])].add(int(r['EvidenceFold']))
            assert all(len(s) == 1 and min(s) > 0 for s in gf.values())
    assert len(all_held) == len(set(all_held))
    assert set(all_held) == record_parts['Train']
    oof = folder/'outer_oof_predictions.csv'
    if oof.exists():
        rows = read(oof)
        assert len(rows) == len(record_parts['Train'])
        assert {int(r['GlobalIndex']) for r in rows} == record_parts['Train']
        group_folds = defaultdict(set)
        for r in rows:
            i = int(r['GlobalIndex'])
            group_folds[groups[i]].add(int(r['OuterFold']))
            for c in ['RuleOOF', 'LROOF']:
                assert math.isfinite(float(r[c])) and 0 <= float(r[c]) <= 1
        assert all(len(s) == 1 for s in group_folds.values())
    return dict(Folder=str(folder), SplitRecords=len(split),
                Pipelines=len(pipelines), GroupIsolationPassed=True,
                CompleteTrainingOOFCoverage=True, ModelPredictionsPresent=oof.exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(verify(args.run_dir))
