# -*- coding: utf-8 -*-
"""Step 6 — evaluate the end-to-end pipeline on the test split.

Reports three things that matter for a decision system, not just a score:

1. Alarm quality at the cost-optimal operating point (P / R / F1 / AUC / PR-AUC).
2. Expected maintenance cost (C_FN = 1000, C_FP = 100) against reference
   policies, including the supervised baseline's own cost-optimal point from
   the frozen baseline (same cost model, same test set -- directly comparable).
3. Reason-code attribution quality: when the system alarms on a true failure,
   does the attributed constraint match the actual failure mode?

Outputs: results_step6.csv / .json, fig_step6_cost_and_reason.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE.parent / 'step4'))
sys.path.insert(0, str(HERE))
import features as F                     # noqa: E402
import pipeline as P                     # noqa: E402

MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def cost_of(y, pred, c_fn=P.C_FN, c_fp=P.C_FP):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return c_fn * fn + c_fp * fp, dict(tp=int(tp), fp=int(fp), fn=int(fn),
                                       tn=int(tn))


def main():
    df = F.load_data()
    y_all = df[F.LABEL].to_numpy(dtype=int)
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42,
                                      stratify=y_all)
    train, test = df.iloc[idx_tr], df.iloc[idx_te]
    y_te = y_all[idx_te]
    modes_te = test[MODES].to_numpy(dtype=int)

    pipe = P.MaintenancePipeline().fit(train)
    info = pipe.explain()
    print('--- fitted pipeline ---')
    for k, v in info.items():
        print(f'  {k}: {v}')

    out = pipe.predict(test)
    score, alarm = out['score'].to_numpy(), out['alarm'].to_numpy()

    # ---------------- 1. alarm quality ----------------
    rows = []
    c, cm = cost_of(y_te, alarm)
    rows.append(dict(protocol='cost-optimal threshold (L2b, 300 rows)',
                     precision=precision_score(y_te, alarm, zero_division=0),
                     recall=recall_score(y_te, alarm, zero_division=0),
                     f1=f1_score(y_te, alarm, zero_division=0),
                     expected_cost=c, n_alarms=int(alarm.sum()), **cm,
                     roc_auc=roc_auc_score(y_te, score),
                     pr_auc=average_precision_score(y_te, score)))

    # L0-style threshold (POT, 1% target) for contrast
    sys.path.insert(0, str(HERE.parent / 'step5'))
    from run_step5 import pot_threshold
    s_tr = pipe.scorer.score(F.add_physics(train)[pipe.cols].to_numpy(float))
    for tag, thr in [('L0 POT target 1%', pot_threshold(s_tr, 0.01)),
                     ('watch-list threshold (3x rate)',
                      pipe.watch_threshold),
                     ('never alarm', np.inf),
                     ('always alarm', -np.inf)]:
        pred = (score >= thr).astype(int)
        c, cm = cost_of(y_te, pred)
        rows.append(dict(protocol=tag,
                         precision=precision_score(y_te, pred, zero_division=0),
                         recall=recall_score(y_te, pred, zero_division=0),
                         f1=f1_score(y_te, pred, zero_division=0),
                         expected_cost=c, n_alarms=int(pred.sum()), **cm,
                         roc_auc=roc_auc_score(y_te, score),
                         pr_auc=average_precision_score(y_te, score)))

    res = pd.DataFrame(rows)
    print('\n--- operating points (test set, C_FN=1000, C_FP=100) ---')
    print(res[['protocol', 'precision', 'recall', 'f1', 'expected_cost',
               'n_alarms']].round(4).to_string(index=False))

    # supervised reference from the frozen baseline (same cost model)
    base = json.load(open(HERE.parents[1] / 'baseline' / 'results' /
                          'summary.json', encoding='utf-8'))['threshold']
    print(f"\nsupervised MLP reference (frozen baseline, same cost model): "
          f"cost_at_default={base['cost_at_default']:.0f}, "
          f"cost_at_optimal={base['cost_at_optimal']:.0f}, "
          f"threshold={base['min_cost_threshold']}")
    print(f"this pipeline: {c if False else res.loc[0, 'expected_cost']:.0f} "
          f"at threshold {info['threshold']:.4f}")

    # ---------------- 2. reason attribution ----------------
    reason = out['reason'].to_numpy()
    alarmed = alarm == 1
    true_fail = y_te == 1
    hit = alarmed & true_fail
    print(f'\n--- reason attribution (on the {int(hit.sum())} alarmed true '
          f'failures) ---')
    tab = []
    for code in ['PWF', 'OSF', 'HDF', 'unknown']:
        sel = hit & (reason == code)
        if sel.sum() == 0:
            tab.append(dict(reason=code, n=0, agrees_with_mode=0, agree_rate=0.0,
                            mode_breakdown=''))
            continue
        brk = {m: int((modes_te[sel, j] == 1).sum())
               for j, m in enumerate(MODES)}
        agree = int(brk.get(code, 0))
        tab.append(dict(reason=code, n=int(sel.sum()),
                        agrees_with_mode=agree,
                        agree_rate=agree / float(sel.sum()),
                        mode_breakdown=str(brk)))
    tab = pd.DataFrame(tab)
    print(tab.to_string(index=False))

    # how much does each reason code contribute, in true detections vs false alarms?
    print('\n--- contribution of each reason code (at the cost-optimal point) ---')
    contrib = []
    for code in ['PWF', 'OSF', 'HDF', 'unknown']:
        sel = alarmed & (reason == code)
        tp_ = int((sel & true_fail).sum())
        fp_ = int((sel & ~true_fail).sum())
        contrib.append(dict(reason=code, true_detections=tp_, false_alarms=fp_,
                            precision=(tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0)))
    contrib = pd.DataFrame(contrib)
    print(contrib.round(4).to_string(index=False))

    # what reason did the system give for HDF failures, and why did it miss?
    hdf = modes_te[:, MODES.index('HDF')] == 1
    print(f'\n  HDF failures in test: {int(hdf.sum())}, '
          f'alarmed: {int((hdf & alarmed).sum())}, '
          f'attributed to HDF: {int((hdf & (reason == "HDF")).sum())}')
    pwf = modes_te[:, MODES.index('PWF')] == 1
    osf = modes_te[:, MODES.index('OSF')] == 1
    print(f'  PWF: {int(pwf.sum())} / alarmed {int((pwf & alarmed).sum())} / '
          f'attributed PWF {int((pwf & (reason == "PWF")).sum())}')
    print(f'  OSF: {int(osf.sum())} / alarmed {int((osf & alarmed).sum())} / '
          f'attributed OSF {int((osf & (reason == "OSF")).sum())}')

    res.to_csv(HERE / 'results_step6.csv', index=False)
    tab.to_csv(HERE / 'results_step6_reason.csv', index=False)
    contrib.to_csv(HERE / 'results_step6_contrib.csv', index=False)
    json.dump(dict(pipeline=info, operating_points=rows,
                   reason_attribution=tab.to_dict('records'),
                   reason_contribution=contrib.to_dict('records'),
                   supervised_reference=base),
              open(HERE / 'results_step6.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwritten:', HERE / 'results_step6.csv')


if __name__ == '__main__':
    main()
