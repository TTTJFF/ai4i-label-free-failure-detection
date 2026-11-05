# -*- coding: utf-8 -*-
"""Step 2 — Controlled ablation: does the feature space matter more than the detector?

Protocol
--------
Everything is held fixed except ONE variable: the feature set.

    fixed : the stratified 70/30 split (seed 42, exactly the baseline split)
            the three baseline detectors with identical hyper-parameters
            the contamination prior (3.39%, the same value the baseline used)
            the random seed
    varied: F1_raw / F2_physics / F3_physics_pt

Design decisions worth stating
------------------------------
* The three detectors are exactly the ones in the frozen baseline, with the
  same hyper-parameters, so the F1_raw row doubles as a correctness check on
  this harness: it should reproduce the baseline numbers.
* The threshold is fixed by a contamination prior, NOT tuned on labels.
  Maximising F1 by scanning the threshold would be implicit supervision and
  would invalidate the whole exercise. ROC-AUC and PR-AUC are threshold-free
  and are reported for that reason.
* StandardScaler is fitted on the TRAINING split only. The frozen baseline
  fitted it on the full dataset, so the F1_raw numbers here may differ very
  slightly from the frozen ones; that difference is printed explicitly rather
  than hidden.
* Machine type enters every feature set as the same one-hot block, so the
  comparison is not confounded by the type encoding.

Outputs (written next to this script)
-------------------------------------
    results_step2.csv    one row per (feature set, detector)
    results_step2.json   machine-readable dump, for later merging
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
import features as F  # noqa: E402

SEED = 42
CONTAM = None  # set from the data below (3.39%), same prior as the baseline

# Same hyper-parameters as the frozen baseline (baseline/code/run_experiments.py).
DETECTORS = {
    'Isolation Forest': dict(
        make=lambda c: IsolationForest(n_estimators=300, contamination=c,
                                       random_state=SEED),
        scale=True),
    'One-Class SVM': dict(
        make=lambda c: OneClassSVM(kernel='rbf', nu=c, gamma='auto'),
        scale=True),
    'Local Outlier Factor': dict(
        make=lambda c: LocalOutlierFactor(n_neighbors=20, contamination=c,
                                          novelty=True),
        scale=True),
}

# Frozen baseline (task C), for the reproduction check. From
# baseline/results/results_taskC_anomaly.csv
FROZEN = {
    'Isolation Forest': dict(f1=0.1244, roc_auc=0.7570, pr_auc=0.1049),
    'One-Class SVM': dict(f1=0.2029, roc_auc=0.7637, pr_auc=0.1675),
    'Local Outlier Factor': dict(f1=0.1800, roc_auc=0.8132, pr_auc=0.1497),
}


def scores_of(model, X):
    """Anomaly score, oriented so that larger = more anomalous."""
    if hasattr(model, 'score_samples'):
        return -model.score_samples(X)
    return -model.decision_function(X)


def main():
    df = F.load_data()
    y_all = df[F.LABEL].to_numpy(dtype=int)
    contamination = float(y_all.mean())

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, random_state=SEED, stratify=y_all)

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    frames, y = F.build_frames(df, train_mask)
    y_train, y_test = y[idx_train], y[idx_test]
    mode_test = df.loc[idx_test, F.FAILURE_MODES].to_numpy(dtype=int)
    print(f'train {len(idx_train)} (failures {y_train.sum()}), '
          f'test {len(idx_test)} (failures {y_test.sum()})')
    print(f'contamination prior = {contamination:.4f}')
    print(f'mode counts in test set: '
          + ', '.join(f'{m}={int(mode_test[:, j].sum())}'
                      for j, m in enumerate(F.FAILURE_MODES)) + '\n')

    rows = []
    for set_name, X in frames.items():
        Xtr = X.iloc[idx_train].to_numpy(dtype=float)
        Xte = X.iloc[idx_test].to_numpy(dtype=float)
        for det_name, cfg in DETECTORS.items():
            if cfg['scale']:
                sc = StandardScaler().fit(Xtr)
                A, B = sc.transform(Xtr), sc.transform(Xte)
            else:
                A, B = Xtr, Xte

            model = cfg['make'](contamination)
            t0 = time.time()
            model.fit(A)
            train_time = time.time() - t0

            score = scores_of(model, B)
            pred = (model.predict(B) == -1).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            sc_norm = score[y_test == 0]
            sc_fail = score[y_test == 1]
            gap_z = (sc_fail.mean() - sc_norm.mean()) / sc_norm.std()

            # which failure modes does the detector actually catch at this budget?
            per_mode = {}
            for j, m in enumerate(F.FAILURE_MODES):
                sel = mode_test[:, j] == 1
                per_mode['recall_' + m] = (float(pred[sel].mean())
                                           if sel.sum() else float('nan'))

            rows.append(dict(
                feature_set=set_name, detector=det_name,
                n_features=X.shape[1], flagged=int(tp + fp),
                precision=precision_score(y_test, pred, zero_division=0),
                recall=recall_score(y_test, pred, zero_division=0),
                f1=f1_score(y_test, pred, zero_division=0),
                roc_auc=roc_auc_score(y_test, score),
                pr_auc=average_precision_score(y_test, score),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                score_mean_normal=float(sc_norm.mean()),
                score_mean_failure=float(sc_fail.mean()),
                score_gap_z=float(gap_z),
                train_time_s=round(train_time, 3),
                **per_mode))

            mode_str = ' '.join(f'{m}={per_mode["recall_" + m]:.2f}'
                                for m in F.FAILURE_MODES)
            print(f'{set_name:15s} {det_name:22s} '
                  f'F1={rows[-1]["f1"]:.4f} AUC={rows[-1]["roc_auc"]:.4f} '
                  f'PR-AUC={rows[-1]["pr_auc"]:.4f} '
                  f'TP={tp:3d} FP={fp:4d} gap_z={gap_z:5.2f}  {mode_str}')

    res = pd.DataFrame(rows)
    res.to_csv(HERE / 'results_step2.csv', index=False)
    json.dump(dict(seed=SEED, contamination=contamination,
                   n_train=int(len(idx_train)), n_test=int(len(idx_test)),
                   fail_train=int(y_train.sum()), fail_test=int(y_test.sum()),
                   records=rows, frozen=FROZEN),
              open(HERE / 'results_step2.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    # ---------------- reproduction check ----------------
    print('\n--- reproduction check on F1_raw (harness vs frozen baseline) ---')
    base = res[res.feature_set == 'F1_raw'].set_index('detector')
    for det, fz in FROZEN.items():
        got = base.loc[det]
        print(f'{det:22s} F1 {got.f1:.4f} vs {fz["f1"]:.4f}   '
              f'AUC {got.roc_auc:.4f} vs {fz["roc_auc"]:.4f}   '
              f'PR-AUC {got.pr_auc:.4f} vs {fz["pr_auc"]:.4f}')

    # ---------------- headline comparison ----------------
    print('\n--- headline: F1_raw -> F2_physics ---')
    piv = res.pivot_table(index='detector', columns='feature_set',
                          values=['f1', 'roc_auc', 'pr_auc'])
    print(piv.round(4).to_string())
    print('\nwritten:', HERE / 'results_step2.csv')


if __name__ == '__main__':
    main()
