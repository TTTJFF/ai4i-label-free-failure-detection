# -*- coding: utf-8 -*-
"""Step 4 — label-free threshold learning and boundary-violation detection.

Three questions this step answers
---------------------------------
Q1 (Part A)  Can the support boundaries be recovered from data alone, with no
             labels and no reference to the dataset documentation? Three
             label-free strategies are run and compared against the true
             generator constants, which the methods never see.

Q2 (Part B)  If the score measures *boundary violation* instead of *rarity*,
             how much performance do we recover? The scanner is a
             union-of-subspaces support estimator (see support.py).

Q3 (Part C)  How much is lost by having to *learn* the boundaries rather than
             being told them? This is quantified by building the explicit
             rule detector twice: once with the true constants (oracle) and
             once with the learned ones.

Everything is fitted on the training split only, on the same split as steps
2 and 3 (stratified 70/30, seed 42). No label is used for fitting, model
selection or thresholding; labels are used only for the final scoring.

Outputs: results_step4_*.csv/json, fig_step4_*.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE))
import features as F            # noqa: E402
import support as S             # noqa: E402

SEED = 42
TRUE = {                        # generator constants, used ONLY for validation
    'Power [W]': (3500.0, 9000.0),
    'Temp diff [K]': (8.6, None),
    'Rotational speed [rpm]': (None, 1380.0),
    'Overstrain (L)': (None, 11000.0),
    'Overstrain (M)': (None, 12000.0),
    'Overstrain (H)': (None, 13000.0),
}


def metric_row(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    d = dict(precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0),
             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             n_flagged=int(tp + fp))
    if score is not None:
        d['roc_auc'] = roc_auc_score(y, score)
        d['pr_auc'] = average_precision_score(y, score)
    return d


# ------------------------------------------------------------------ Part A
def part_a(train, test):
    rows = []
    for name, col, sub in [
            ('Power [W]', 'Power [W]', None),
            ('Temp diff [K]', 'Temp diff [K]', None),
            ('Rotational speed [rpm]', 'Rotational speed [rpm]', None),
            ('Overstrain (L)', 'Overstrain [Nm*min]', 'L'),
            ('Overstrain (M)', 'Overstrain [Nm*min]', 'M'),
            ('Overstrain (H)', 'Overstrain [Nm*min]', 'H')]:
        tr = train if sub is None else train[train['Type'] == sub]
        x = tr[col].to_numpy(dtype=float)
        lo_q, hi_q = S.quantile_boundary(x, 0.005)
        lo_c, hi_c = S.cliff_boundary(x)
        lo_e, hi_e = S.evt_boundary(x, q=0.95, n_return=len(x))
        t_lo, t_hi = TRUE[name]
        rows.append(dict(
            feature=name,
            true_low=t_lo, true_high=t_hi,
            quantile_low=lo_q, quantile_high=hi_q,
            cliff_low=lo_c, cliff_high=hi_c,
            evt_low=lo_e, evt_high=hi_e,
            train_min=float(x.min()), train_max=float(x.max())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part B/C
def hdf_diagnostic(train):
    """Why a label-free detector cannot see HDF.

    HDF fires exactly in the corner {Temp diff < 8.6 AND Speed < 1380}. The
    question is whether that corner is a *hole* in the training distribution
    (a support violation, which would be detectable) or merely a sparse
    region that is nonetheless populated (which would not be).

    This block answers it with counts only.
    """
    corner = ((train['Temp diff [K]'] < 8.6)
              & (train['Rotational speed [rpm]'] < 1380))
    normal = train[F.LABEL] == 0
    p_dt = float((train['Temp diff [K]'] < 8.6).mean())
    p_sp = float((train['Rotational speed [rpm]'] < 1380).mean())
    n = len(train)
    return dict(
        p_tempdiff_low=p_dt, p_speed_low=p_sp,
        corner_observed=int(corner.sum()),
        corner_expected_if_independent=p_dt * p_sp * n,
        corner_observed_all_are_HDF=bool(
            train.loc[corner, F.LABEL].sum() == corner.sum()),
        corner_observed_HDF=int(train.loc[corner, 'HDF'].sum()),
        corner_observed_normal=int((corner & normal).sum()),
        corner_expected_normal_if_independent=p_dt * p_sp * int(normal.sum()),
        marginal_tempdiff_min=float(train['Temp diff [K]'].min()),
        marginal_speed_min=float(train['Rotational speed [rpm]'].min()),
    )


def is_pwf(x, lo, hi):
    return (x['Power [W]'] < lo) | (x['Power [W]'] > hi)


def is_hdf(x, dt_lo, sp_lo):
    return (x['Temp diff [K]'] < dt_lo) & (x['Rotational speed [rpm]'] < sp_lo)


def is_osf(x, thr_by_type):
    thr = x['Type'].map(thr_by_type).astype(float)
    return x['Overstrain [Nm*min]'] > thr


def main():
    df = F.load_data()
    d = F.add_physics(df)
    y_all = d[F.LABEL].to_numpy(dtype=int)
    contamination = float(y_all.mean())

    idx = np.arange(len(d))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, random_state=SEED, stratify=y_all)

    train = d.iloc[idx_train].reset_index(drop=True)
    test = d.iloc[idx_test].reset_index(drop=True)
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]
    mode_test = test[F.FAILURE_MODES].to_numpy(dtype=int)

    print(f'train {len(train)} (failures {y_train.sum()}), '
          f'test {len(test)} (failures {y_test.sum()})\n')

    # ---------------- Part A: can the boundaries be learned? ----------------
    pa = part_a(train, test)
    pa.to_csv(HERE / 'results_step4_boundaries.csv', index=False)
    print('--- Part A: label-free boundary recovery (vs generator constants) ---')
    for _, r in pa.iterrows():
        def fmt(v):
            return '  n/a ' if pd.isna(v) else f'{v:9.1f}'
        print(f'{r["feature"]:24s} true=[{fmt(r["true_low"])},{fmt(r["true_high"])}]'
              f'  cliff=[{fmt(r["cliff_low"])},{fmt(r["cliff_high"])}]'
              f'  evt=[{fmt(r["evt_low"])},{fmt(r["evt_high"])}]'
              f'  quantile=[{fmt(r["quantile_low"])},{fmt(r["quantile_high"])}]')

    # ---------------- Part A2: is the HDF corner a support hole? -----------
    diag = hdf_diagnostic(train)
    print('\n--- Part A2: is the HDF corner a support hole or a populated '
          'region? ---')
    print(f'  P(Temp diff < 8.6) = {diag["p_tempdiff_low"] * 100:.2f}%   '
          f'P(Speed < 1380) = {diag["p_speed_low"] * 100:.2f}%')
    print(f'  corner observed = {diag["corner_observed"]} training rows   '
          f'expected under independence = '
          f'{diag["corner_expected_if_independent"]:.0f}')
    print(f'  every occupant is a failure: '
          f'{diag["corner_observed_all_are_HDF"]}')
    print(f'  restricted to NORMAL rows: observed = '
          f'{diag["corner_observed_normal"]}   expected = '
          f'{diag["corner_expected_normal_if_independent"]:.0f}')
    print(f'  marginal support edges: Temp diff min = '
          f'{diag["marginal_tempdiff_min"]:.2f} K (rule constant 8.6); '
          f'Speed min = {diag["marginal_speed_min"]:.0f} rpm '
          f'(rule constant 1380)')

    # ---------------- Part B: violation scanner ----------------------------
    coords = S.CONSTRAINT_COORDS + S.CONJUNCTION_PARTNERS
    rows = []

    configs = {
        'MSV_full': S.default_subspaces(False),
        'MSV_singletons_only': [(c,) for c in S.CONSTRAINT_COORDS],
        'MSV_with_speed_singleton': S.default_subspaces(True),
    }
    scores = {}
    for name, subs in configs.items():
        sc = S.MarginalSupportScorer(subs, k=5).fit(train[coords].to_numpy(),
                                                     coords)
        t0 = time.time()
        s_te = sc.score(test[coords].to_numpy())
        s_tr = sc.score(train[coords].to_numpy())
        fit_time = time.time() - t0
        scores[name] = s_te

        # label-free operating point: top-contamination of the TRAIN scores
        thr = float(np.quantile(s_tr, 1.0 - contamination))
        pred_nat = (s_te >= thr).astype(int)
        # fixed-budget point, comparable with steps 2 and 3
        thr_b = float(np.quantile(s_te, 1.0 - contamination))
        pred_bud = (s_te >= thr_b).astype(int)

        r = metric_row(y_test, pred_nat, s_te)
        r.update(dict(method=name, n_subspaces=len(subs),
                      fit_and_score_s=round(fit_time, 2),
                      decision='train-quantile threshold',
                      train_threshold=thr))
        for j, m in enumerate(F.FAILURE_MODES):
            sel = mode_test[:, j] == 1
            r['recall_' + m] = (float(pred_nat[sel].mean()) if sel.sum()
                                else float('nan'))
        rows.append(r)
        r2 = metric_row(y_test, pred_bud, s_te)
        r2.update(dict(method=name, n_subspaces=len(subs),
                       fit_and_score_s=round(fit_time, 2),
                       decision='fixed 3.39% budget (test)',
                       train_threshold=thr))
        rows.append(r2)
        print(f'\n{name:26s} subspaces={len(subs)}')
        for rr in (r, r2):
            print(f'   {rr["decision"]:34s} P={rr["precision"]:.3f} '
                  f'R={rr["recall"]:.3f} F1={rr["f1"]:.4f} '
                  f'AUC={rr["roc_auc"]:.4f} PR-AUC={rr["pr_auc"]:.4f} '
                  f'flagged={rr["n_flagged"]}')

    # ---- variant: support model fitted after trimming the top 10% --------
    # Tests whether the low recall on HDF is caused by the training failures
    # themselves "supporting" their own region.
    subs = configs['MSV_full']
    keep = np.arange(len(train))
    Xtr_all = train[coords].to_numpy()
    for _ in range(3):
        sc_t = S.MarginalSupportScorer(subs, k=5).fit(Xtr_all[keep], coords)
        st = sc_t.score(Xtr_all[keep])
        keep = keep[st < np.quantile(st, 0.90)]
    sc_t = S.MarginalSupportScorer(subs, k=5).fit(Xtr_all[keep], coords)
    s_te_t = sc_t.score(test[coords].to_numpy())
    s_tr_t = sc_t.score(Xtr_all[keep])
    thr_t = float(np.quantile(s_tr_t, 1.0 - contamination))
    pred_t = (s_te_t >= thr_t).astype(int)
    rt = metric_row(y_test, pred_t, s_te_t)
    rt.update(dict(method='MSV_full_trimmed10', n_subspaces=len(subs),
                   fit_and_score_s='', decision='train-quantile threshold',
                   train_threshold=thr_t))
    rows.append(rt)
    print(f'\nMSV_full_trimmed10        kept={len(keep)} training rows')
    print(f'   {"train-quantile threshold":34s} P={rt["precision"]:.3f} '
          f'R={rt["recall"]:.3f} F1={rt["f1"]:.4f} AUC={rt["roc_auc"]:.4f} '
          f'PR-AUC={rt["pr_auc"]:.4f} flagged={rt["n_flagged"]}')
    per_mode_t = {}
    for j, m in enumerate(F.FAILURE_MODES):
        sel = mode_test[:, j] == 1
        per_mode_t[m] = (float(pred_t[sel].mean()) if sel.sum()
                         else float('nan'))
    print('   per-mode recall:', {k: round(v, 3)
                                  for k, v in per_mode_t.items()})

    # per-mode recall at the natural operating point for the main config
    thr = float(np.quantile(scores['MSV_full'], 1.0 - contamination))
    pred_main = (scores['MSV_full'] >= thr).astype(int)
    per_mode = {}
    for j, m in enumerate(F.FAILURE_MODES):
        sel = mode_test[:, j] == 1
        per_mode[m] = float(pred_main[sel].mean()) if sel.sum() else float('nan')

    # ---------------- Part C: learned rules vs oracle rules -----------------
    # (1) oracle rules: the true generator constants
    oracle = (is_pwf(d, 3500.0, 9000.0)
              | is_hdf(d, 8.6, 1380.0)
              | is_osf(d, {'L': 11000.0, 'M': 12000.0, 'H': 13000.0})
              ).to_numpy().astype(int)

    # (2) learned rules: boundaries recovered from the training split only
    lo_c, hi_c = S.cliff_boundary(train['Power [W]'].to_numpy(float))
    dt_lo, _ = S.cliff_boundary(train['Temp diff [K]'].to_numpy(float))
    sp_lo, _ = S.cliff_boundary(train['Rotational speed [rpm]'].to_numpy(float))
    osf_thr = {}
    for t in F.TYPES:
        _, hi_t = S.cliff_boundary(train.loc[train.Type == t,
                                             'Overstrain [Nm*min]'].to_numpy(float))
        osf_thr[t] = hi_t
    learned = (is_pwf(d, lo_c, hi_c)
               | is_hdf(d, dt_lo, sp_lo)
               | is_osf(d, osf_thr)).to_numpy().astype(int)

    # (3) learned rules using the quantile learner instead of the cliff learner
    p_lo, p_hi = S.quantile_boundary(train['Power [W]'].to_numpy(float), 0.005)
    q_dt, _ = S.quantile_boundary(train['Temp diff [K]'].to_numpy(float), 0.005)
    q_sp, _ = S.quantile_boundary(
        train['Rotational speed [rpm]'].to_numpy(float), 0.005)
    q_osf = {}
    for t in F.TYPES:
        _, q_hi = S.quantile_boundary(
            train.loc[train.Type == t, 'Overstrain [Nm*min]'].to_numpy(float),
            0.005)
        q_osf[t] = q_hi
    learned_q = (is_pwf(d, p_lo, p_hi)
                 | is_hdf(d, q_dt, q_sp)
                 | is_osf(d, q_osf)).to_numpy().astype(int)

    rules = []
    for tag, rule in [('oracle rules (true constants)', oracle),
                      ('learned rules (cliff boundaries)', learned),
                      ('learned rules (quantile boundaries)', learned_q)]:
        r = metric_row(y_test, rule[idx_test])
        r['method'] = tag
        r['decision'] = 'rule OR'
        r['n_subspaces'] = 0
        rules.append(r)
        print(f'\n{tag:30s} P={r["precision"]:.4f} R={r["recall"]:.4f} '
              f'F1={r["f1"]:.4f} flagged={r["n_flagged"]} '
              f'(TP={r["tp"]} FP={r["fp"]})')
    print('learned boundaries: power=[%.0f, %.0f] tempdiff<%.2f speed<%.0f '
          'overstrain=%s' % (lo_c, hi_c, dt_lo, sp_lo,
                             {k: round(v) for k, v in osf_thr.items()}))

    # ---------------- reference rows from earlier steps --------------------
    ref = []
    s2 = pd.read_csv(HERE.parent / 'step2' / 'results_step2.csv')
    best2 = s2.sort_values('f1', ascending=False).iloc[0]
    ref.append(dict(method=f'step2 best: {best2.detector} @ {best2.feature_set}',
                    f1=float(best2.f1), roc_auc=float(best2.roc_auc),
                    pr_auc=float(best2.pr_auc), precision=float(best2.precision),
                    recall=float(best2.recall), n_flagged=int(best2.flagged)))
    base = pd.read_csv(HERE.parents[1] / 'baseline' / 'results' /
                       'results_taskC_anomaly.csv')
    bb = base.sort_values('F1', ascending=False).iloc[0]
    ref.append(dict(method=f'frozen baseline best: {bb["Model"]}',
                    f1=float(bb['F1']), roc_auc=float(bb['ROC-AUC']),
                    pr_auc=float(bb['PR-AUC']), precision=float(bb['Precision']),
                    recall=float(bb['Recall']),
                    n_flagged=int(bb['TP'] + bb['FP'])))

    res = pd.DataFrame(rows + rules)
    res.to_csv(HERE / 'results_step4_scanner.csv', index=False)
    pd.DataFrame(ref).to_csv(HERE / 'results_step4_reference.csv', index=False)
    json.dump(dict(seed=SEED, contamination=contamination,
                   n_train=int(len(train)), n_test=int(len(test)),
                   fail_test=int(y_test.sum()),
                   learned_boundaries=dict(
                       power=[lo_c, hi_c], tempdiff_low=dt_lo,
                       speed_low=sp_lo, overstrain=osf_thr),
                   hdf_diagnostic=diag,
                   per_mode_recall_main_config=per_mode,
                   per_mode_recall_trimmed=per_mode_t,
                   scanner=rows, rules=rules, reference=ref),
              open(HERE / 'results_step4.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print('\n--- per-mode recall, MSV_full at the label-free operating point ---')
    for m, v in per_mode.items():
        print(f'   {m}: {v:.3f}')
    print('\n--- reference points ---')
    for r in ref:
        print(f'   {r["method"]:44s} F1={r["f1"]:.4f}')
    print('\nwritten:', HERE / 'results_step4_scanner.csv')


if __name__ == '__main__':
    main()
