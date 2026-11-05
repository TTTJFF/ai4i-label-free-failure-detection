# -*- coding: utf-8 -*-
"""Step 3 — The pure-statistics control arm (decision Q3).

Why this step exists
--------------------
Step 2 showed that adding physics-derived features improves every detector.
But it left one hole open:

    Was the gain caused by *choosing the physically correct quantities*,
    or would *any* nonlinear expansion of the raw sensors do just as well?

That distinction decides what the report is allowed to claim. If any generic
expansion works, then "domain knowledge" is not the driver and the Q1 framing
('inductive bias via physics features') has to be weakened to
'any interaction terms will do'.

Arms compared (everything label-free, nothing reads Machine failure)
-------------------------------------------------------------------
    A_physics     the domain-knowledge arm, identical to step 2's F2:
                  5 sensors + Power + Temp diff + Overstrain + type
    B0_raw        no expansion at all: 5 sensors + type  (= F1_raw)
    B1_poly2      generic degree-2 expansion: 5 sensors, their 5 squares and
                  all 10 pairwise products + type. No physics reasoning used.
    B2_nystroem   5 sensors + 10 RBF kernel features from Nystroem (a generic
                  unsupervised nonlinear representation) + type

Note on B1_poly2 — it is a *strong* control, not a straw man. It contains
    Torque x Speed     = Power x (60/2pi)      (up to a scalar multiple)
    Torque x Tool wear = Overstrain            (exactly)
so two of the three physics features are inside the generic expansion. It does
NOT contain Process temp - Air temp, because a difference is a degree-1
linear combination, not a monomial.

Everything else is held fixed exactly as in step 2: same 70/30 stratified
split (seed 42), same three detectors, same contamination prior (3.39%),
scaler fitted on the training split only.

Outputs: results_step3.csv, results_step3.json, fig_step3_control.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
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
NYS_COMPONENTS = 10

DETECTORS = {
    'Isolation Forest': lambda c: IsolationForest(
        n_estimators=300, contamination=c, random_state=SEED),
    'One-Class SVM': lambda c: OneClassSVM(kernel='rbf', nu=c, gamma='auto'),
    'Local Outlier Factor': lambda c: LocalOutlierFactor(
        n_neighbors=20, contamination=c, novelty=True),
}


def sensors_of(df):
    return df[F.RAW_SENSORS].to_numpy(dtype=float)


def build_poly2(df, train_mask):
    """All degree-2 monomials of the 5 sensors. Generic, no physics."""
    S = sensors_of(df)
    names, cols = [], []
    for i in range(S.shape[1]):
        for j in range(i, S.shape[1]):
            names.append(f'{F.RAW_SENSORS[i]}*{F.RAW_SENSORS[j]}')
            cols.append(S[:, i] * S[:, j])
    P = np.column_stack(cols)
    return names, P


def build_nystroem(df, train_mask, idx_train):
    """Generic unsupervised nonlinear representation: RBF kernel features.

    Fitted on the training rows only (Nystroem picks landmark points), then
    applied to every row.
    """
    S = sensors_of(df)
    sc = StandardScaler().fit(S[train_mask])
    ks = Nystroem(kernel='rbf', n_components=NYS_COMPONENTS,
                  random_state=SEED).fit(sc.transform(S[train_mask]))
    Z = ks.transform(sc.transform(S))
    return [f'nys_{i}' for i in range(Z.shape[1])], Z


def type_onehot(df):
    return np.column_stack([(df['Type'] == t).astype(float)
                            for t in F.TYPES[1:]])


def build_naive3(df):
    """Three pairwise products, deliberately NOT the physics ones.

    Chosen a priori as plausible-looking but physically uninformative
    interactions, so that this arm spends exactly the same feature budget as
    A_physics (3 added features) without using any domain knowledge.
    Excluded on purpose: Torque x Speed (= Power) and Torque x Tool wear
    (= Overstrain).
    """
    pairs = [('Torque [Nm]', 'Air temperature [K]'),
             ('Rotational speed [rpm]', 'Process temperature [K]'),
             ('Tool wear [min]', 'Air temperature [K]')]
    names, cols = [], []
    for a, b in pairs:
        names.append(f'{a}*{b}')
        cols.append(df[a].to_numpy(dtype=float) * df[b].to_numpy(dtype=float))
    return names, np.column_stack(cols)


def build_arms(df, train_mask):
    """Return {arm_name: (DataFrame or ndarray of features, column names)}."""
    frames, y = F.build_frames(df, train_mask)
    T = type_onehot(df)
    type_names = ['Type=' + t for t in F.TYPES[1:]]

    arms = {}
    arms['A_physics'] = (frames['F2_physics'].to_numpy(dtype=float),
                         list(frames['F2_physics'].columns))
    arms['B0_raw'] = (frames['F1_raw'].to_numpy(dtype=float),
                      list(frames['F1_raw'].columns))

    n3_names, P3 = build_naive3(df)
    arms['B1a_naive3'] = (np.column_stack([sensors_of(df), P3, T]),
                          F.RAW_SENSORS + n3_names + type_names)

    p_names, P = build_poly2(df, train_mask)
    X1 = np.column_stack([sensors_of(df), P, T])
    arms['B1b_poly2_all'] = (X1, F.RAW_SENSORS + p_names + type_names)

    n_names, Z = build_nystroem(df, train_mask, None)
    X2 = np.column_stack([sensors_of(df), Z, T])
    arms['B2_nystroem'] = (X2, F.RAW_SENSORS + n_names + type_names)
    return arms, y


def scores_of(model, X):
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

    arms, y = build_arms(df, train_mask)
    y_test = y[idx_test]
    mode_test = df.loc[idx_test, F.FAILURE_MODES].to_numpy(dtype=int)

    print(f'test {len(idx_test)} (failures {y_test.sum()}), '
          f'contamination {contamination:.4f}\n')
    print('arms:')
    for k, (X, names) in arms.items():
        print(f'  {k:14s} dim={X.shape[1]:3d}')
    print()

    rows = []
    for arm, (X, names) in arms.items():
        Xtr, Xte = X[idx_train], X[idx_test]
        for det_name, make in DETECTORS.items():
            sc = StandardScaler().fit(Xtr)
            A, B = sc.transform(Xtr), sc.transform(Xte)
            model = make(contamination)
            t0 = time.time()
            model.fit(A)
            tt = time.time() - t0
            score = scores_of(model, B)
            pred = (model.predict(B) == -1).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            per_mode = {}
            for j, m in enumerate(F.FAILURE_MODES):
                sel = mode_test[:, j] == 1
                per_mode['recall_' + m] = (float(pred[sel].mean())
                                           if sel.sum() else float('nan'))
            rows.append(dict(
                arm=arm, arm_kind='physics' if arm.startswith('A_') else 'control',
                detector=det_name, n_features=X.shape[1],
                precision=precision_score(y_test, pred, zero_division=0),
                recall=recall_score(y_test, pred, zero_division=0),
                f1=f1_score(y_test, pred, zero_division=0),
                roc_auc=roc_auc_score(y_test, score),
                pr_auc=average_precision_score(y_test, score),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                train_time_s=round(tt, 3), **per_mode))
            r = rows[-1]
            print(f'{arm:14s} {det_name:22s} F1={r["f1"]:.4f} '
                  f'AUC={r["roc_auc"]:.4f} PR-AUC={r["pr_auc"]:.4f} '
                  f'TP={tp:3d} FP={fp:4d}')

    res = pd.DataFrame(rows)
    res.to_csv(HERE / 'results_step3.csv', index=False)
    json.dump(dict(seed=SEED, contamination=contamination,
                   nystroem_components=NYS_COMPONENTS,
                   records=rows),
              open(HERE / 'results_step3.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print('\n--- arm x detector (F1) ---')
    print(res.pivot_table(index='detector', columns='arm',
                          values='f1').round(4).to_string())
    print('\n--- arm mean over detectors ---')
    agg = res.groupby('arm')[['f1', 'roc_auc', 'pr_auc']].mean().round(4)
    agg['kind'] = [('physics' if a.startswith('A_') else 'control')
                   for a in agg.index]
    print(agg.to_string())
    print('\nwritten:', HERE / 'results_step3.csv')


if __name__ == '__main__':
    main()
