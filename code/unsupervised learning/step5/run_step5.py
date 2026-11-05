# -*- coding: utf-8 -*-
"""Step 5 — full ablation matrix and the L0 / L1 / L2 label-budget layers.

Two questions
-------------
(1) Is step 4's conclusion an artefact of the three baseline detectors? The
    matrix crosses 8 detectors with the 3 feature sets from step 1, so the
    boundary-violation scorer can be compared against GMM, KDE, Mahalanobis
    and a histogram detector as well.

(2) How much do labels buy, and at what point? Four threshold layers:
        L0   no contamination knowledge: POT return level, once in 1,000
        L1   a weak, deliberately mis-specified prior: contamination = 1%
             (the true rate is 3.39%, so this layer tests a *wrong* prior)
        Lref the true rate, 3.39% -- the protocol used in steps 2-4, reported
             as a reference because the number is label-derived
        L2   a small labelled calibration set (N failures + N normals) used
             ONLY to pick the threshold, never to fit or select the model

Every score is produced by a label-free `fit`. AUC and PR-AUC are
threshold-free, so they are identical across the L layers for a given
(feature set, detector) pair; only the operating point changes.

Outputs: results_step5.csv / .json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE.parent / 'step4'))
sys.path.insert(0, str(HERE))
import features as F            # noqa: E402
import detectors as D           # noqa: E402

SEED = 42
TRUE_CONTAM = None              # filled from the data
L2_SIZES = [10, 20, 50, 100]    # labelled FAILURES per calibration set (L2a)
L2_RATE_SIZES = [100, 300, 1000]  # size of the labelled random sample (L2b)


def pot_threshold(scores, target_rate=0.01, q=0.95):
    """Peaks-over-threshold level whose exceedance probability is `target_rate`.

    `target_rate` is the only assumption; it is NOT the true contamination
    rate, and no label is involved. The level comes from a generalised Pareto
    fit to the upper tail, so this layer tests "a parametric tail model" as
    opposed to the empirical quantile used by the L1 layer.
    """
    s = np.asarray(scores, dtype=float)
    u = np.quantile(s, q)
    exc = s[s > u] - u
    if len(exc) < 20 or target_rate >= (1.0 - q):
        return float(np.quantile(s, 1.0 - target_rate))
    try:
        c, loc, scale = stats.genpareto.fit(exc, floc=0.0)
        p_excess = target_rate / (1.0 - q)
        return float(u + stats.genpareto.ppf(1.0 - p_excess, c,
                                             loc=0.0, scale=scale))
    except Exception:
        return float(np.quantile(s, 1.0 - target_rate))


def f1_at(y, score, thr):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                n_flagged=int(tp + fp))


def calibrate_threshold(score_cal, y_cal):
    """Pick the threshold maximising F1 on a small labelled calibration set."""
    cands = np.unique(np.quantile(score_cal, np.linspace(0.50, 0.9999, 400)))
    best, best_f1 = cands[0], -1.0
    for t in cands:
        pred = (score_cal >= t).astype(int)
        if pred.sum() == 0:
            continue
        v = f1_score(y_cal, pred, zero_division=0)
        if v > best_f1:
            best_f1, best = v, t
    return float(best)


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

    # labelled calibration pools drawn from the TRAIN split only
    tr_fail = idx_train[y_train == 1]
    tr_norm = idx_train[y_train == 0]
    pos_tr = {int(g): i for i, g in enumerate(idx_train)}
    rng = np.random.RandomState(SEED)

    print(f'train {len(idx_train)} (failures {y_train.sum()}), '
          f'test {len(idx_test)} (failures {y_test.sum()})')
    print(f'contamination -> Lref uses the true rate {contamination:.4f}; '
          f'L1 deliberately uses 0.01\n')

    rows = []
    for set_name, Xdf in frames.items():
        cols = list(Xdf.columns)
        Xtr_raw = Xdf.iloc[idx_train].to_numpy(dtype=float)
        Xte_raw = Xdf.iloc[idx_test].to_numpy(dtype=float)

        pool = D.default_pool()
        pool['MSV (boundary violation)'] = (
            lambda: D.MsvDetector(D.msv_pool_for(cols), cols, k=5))
        n_sub = {k: (len(D.msv_pool_for(cols))
                     if k.startswith('MSV') else 0) for k in pool}

        for det_name, make in pool.items():
            det = make()
            use_msv = det_name.startswith('MSV')
            # every detector sees standardised features; the MSV scorer
            # standardises internally, so it is handed the raw matrix
            if use_msv:
                A, B = Xtr_raw, Xte_raw
            else:
                sc = StandardScaler().fit(Xtr_raw)
                A, B = sc.transform(Xtr_raw), sc.transform(Xte_raw)

            t0 = time.time()
            det.fit(A)
            fit_s = time.time() - t0
            if use_msv:
                s_tr, s_te = det.score(A), det.score(B)
            else:
                s_tr, s_te = det.score(A), det.score(B)

            auc = roc_auc_score(y_test, s_te)
            pr = average_precision_score(y_test, s_te)

            layers = {
                'L0 POT target 1%': pot_threshold(s_tr, 0.01),
                'L1 ctm=1%': float(np.quantile(s_tr, 1.0 - 0.01)),
                'Lref ctm=3.39%': float(np.quantile(s_tr, 1.0 - contamination)),
            }
            # L2a: naive calibration -- maximise F1 on N labelled failures and
            # N labelled normals. Kept in the matrix because it is a realistic
            # thing to try, and because it fails in an instructive way.
            for n in L2_SIZES:
                f = rng.choice(tr_fail, size=n, replace=False)
                nn = rng.choice(tr_norm, size=n, replace=False)
                pos = [pos_tr[int(g)] for g in np.concatenate([f, nn])]
                s_cal = s_tr[np.array(pos)]
                y_cal = y[np.concatenate([f, nn])]
                layers[f'L2a F1max N={n}'] = calibrate_threshold(s_cal, y_cal)
            # L2b: sensible calibration -- draw a labelled RANDOM sample of the
            # training distribution, estimate the failure rate from it, and set
            # the threshold at the matching score quantile. Uses labels only to
            # estimate a rate, never to rank or fit.
            for m in L2_RATE_SIZES:
                samp = rng.choice(idx_train, size=m, replace=False)
                rate_hat = float(y[samp].mean())
                rate_hat = min(max(rate_hat, 1.0 / m), 0.5)
                layers[f'L2b rate from M={m}'] = float(
                    np.quantile(s_tr, 1.0 - rate_hat))

            for layer, thr in layers.items():
                r = f1_at(y_test, s_te, thr)
                r.update(dict(feature_set=set_name, detector=det_name,
                              layer=layer, threshold=thr,
                              roc_auc=auc, pr_auc=pr,
                              n_features=Xdf.shape[1],
                              n_subspaces=n_sub.get(det_name, 0),
                              fit_time_s=round(fit_s, 3)))
                for j, m in enumerate(F.FAILURE_MODES):
                    sel = mode_test[:, j] == 1
                    pred = (s_te >= thr).astype(int)
                    r['recall_' + m] = (float(pred[sel].mean())
                                        if sel.sum() else float('nan'))
                rows.append(r)

            got = {r['layer']: r['f1'] for r in rows[-len(layers):]}
            print(f'{set_name:15s} {det_name:24s} '
                  f'AUC={auc:.4f} PR-AUC={pr:.4f}  F1 | '
                  f'L0={got["L0 POT target 1%"]:.4f} '
                  f'L1={got["L1 ctm=1%"]:.4f} '
                  f'Lref={got["Lref ctm=3.39%"]:.4f} '
                  f'L2a(100)={got["L2a F1max N=100"]:.4f} '
                  f'L2b(1000)={got["L2b rate from M=1000"]:.4f}')

    res = pd.DataFrame(rows)
    res.to_csv(HERE / 'results_step5.csv', index=False)
    json.dump(dict(seed=SEED, contamination=contamination,
                   l2_sizes=L2_SIZES, l2_rate_sizes=L2_RATE_SIZES, n_train=int(len(idx_train)),
                   n_test=int(len(idx_test)), fail_test=int(y_test.sum()),
                   records=rows),
              open(HERE / 'results_step5.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print('\n=== best detector per (feature set, layer), by F1 ===')
    best = (res.loc[res.groupby(['feature_set', 'layer'])['f1'].idxmax()]
            [['feature_set', 'layer', 'detector', 'f1', 'precision', 'recall',
              'n_flagged']])
    print(best.sort_values(['layer', 'f1'], ascending=[True, False])
          .to_string(index=False))

    print('\n=== F1 by detector x feature set, at the Lref layer ===')
    piv = res[res.layer == 'Lref ctm=3.39%'].pivot_table(
        index='detector', columns='feature_set', values='f1')
    print(piv.round(4).to_string())

    print('\n=== PR-AUC by detector x feature set (threshold-free) ===')
    piv2 = res[res.layer == 'L0 POT target 1%'].pivot_table(
        index='detector', columns='feature_set', values='pr_auc')
    print(piv2.round(4).to_string())

    print('\n=== label value: best F1 per L layer (any detector / feature set) ===')
    lab = res.groupby('layer')[['f1', 'precision', 'recall']].max().round(4)
    print(lab.to_string())
    print('\nwritten:', HERE / 'results_step5.csv')


if __name__ == '__main__':
    main()
