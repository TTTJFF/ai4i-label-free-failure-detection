# -*- coding: utf-8 -*-
"""Step 6-after — five targeted improvements on top of the step 1-6 baseline.

Nothing here changes the research question; it tests whether the *remaining*
head-room identified at the end of step 6 is reachable, and how hard the
existing conclusions are. Five parts:

    A1  multi-seed repetition          -> confidence intervals + paired tests
    A2  three-tier action policy       -> inspect / monitor / ignore decision rule
    A3  subspace fusion variants       -> consensus instead of max, drop the HDF arm
    A4  per-machine-type quantiles     -> fixes the +9% / -11% boundary errors
    A5  label-free contamination rate  -> can the 0.4348 result be reached at L0?

Protocol is unchanged from steps 2-6: stratified 70/30, the same detectors, the
same contamination prior (3.39%), preprocessing fitted on the training split
only, and every threshold derived without touching the test labels (except in
the explicitly labelled "oracle" reference rows).

Run:  python run_step6after.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'step1'))
sys.path.insert(0, os.path.join(HERE, '..', 'step4'))
import features as F                                    # noqa: E402
import support as S                                     # noqa: E402

SEED = 42                       # canonical split (steps 2-6)
SEEDS = [42, 7, 13, 21, 101, 202, 303, 404, 505, 606]
CONTAM = 0.0339                 # true failure rate; used as the prior
C_FN, C_INSP = 1000.0, 100.0    # cost model inherited from step 6
C_MON_DEFAULT = 20.0            # cost of a monitoring action (new, step 6-after)
N_CAL_DRAW = 20                 # calibration draws for the decision-rule parts
CAL_N = 300                     # labelled random rows per calibration draw
J = os.path.join

np.set_printoptions(suppress=True)


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def metrics(y, pred, score):
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, score),
                pr_auc=average_precision_score(y, score),
                n_flagged=int(pred.sum()))


def prepare(seed):
    """Return train/test frames (with derived columns) and everything needed."""
    df = F.load_data()
    y = df[F.LABEL].to_numpy(dtype=int)
    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.3, random_state=seed,
                                stratify=y)
    mask = np.zeros(len(df), dtype=bool)
    mask[itr] = True
    frames, _ = F.build_frames(df, mask)
    d = F.add_physics(df)
    return dict(df=df, d=d, y=y, itr=itr, ite=ite,
                frames=frames, y_tr=y[itr], y_te=y[ite])


# ----------------------------------------------------------------- detectors
def base_detector(name):
    if name == 'Isolation Forest':
        return IsolationForest(n_estimators=300, contamination=CONTAM,
                               random_state=SEED), False
    if name == 'Local Outlier Factor':
        return LocalOutlierFactor(n_neighbors=20, contamination=CONTAM,
                                  novelty=True), True
    if name == 'One-Class SVM':
        return OneClassSVM(kernel='rbf', nu=CONTAM, gamma='auto'), True
    raise KeyError(name)


def scores_of(name, Xtr, Xte):
    """Return (train_scores, test_scores) oriented so that higher = more unusual."""
    mdl, needs_scale = base_detector(name)
    if needs_scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    mdl.fit(Xtr)
    return -mdl.decision_function(Xtr), -mdl.decision_function(Xte)


# ============================================================== Part A1
def part_a1():
    hdr('A1  multi-seed repetition (10 seeds): confidence intervals')
    rows = []
    for seed in SEEDS:
        p = prepare(seed)
        itr, ite = p['itr'], p['ite']
        y_tr, y_te = p['y_tr'], p['y_te']
        for fs in ['F1_raw', 'F2_physics']:
            X = p['frames'][fs]
            Xtr = X.iloc[itr].to_numpy(dtype=float)
            Xte = X.iloc[ite].to_numpy(dtype=float)
            for det in ['Isolation Forest', 'Local Outlier Factor',
                        'One-Class SVM']:
                s_tr, s_te = scores_of(det, Xtr, Xte)
                thr = np.quantile(s_tr, 1.0 - CONTAM)
                m = metrics(y_te, (s_te >= thr).astype(int), s_te)
                rows.append(dict(seed=seed, feature_set=fs, detector=det, **m))

        # MSV variants: fit once per seed on the 9-subspace pool
        coords = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]',
                  'Rotational speed [rpm]']
        d = p['d']
        subs = S.default_subspaces(False)
        scorer = S.MarginalSupportScorer(subs, k=5)
        scorer.fit(d.iloc[itr][coords].to_numpy(dtype=float), coords)
        sd_tr = scorer.subspace_distances(d.iloc[itr][coords].to_numpy(float))
        sd_te = scorer.subspace_distances(d.iloc[ite][coords].to_numpy(float))
        for tag, keys in [('MSV_full', list(scorer._models.keys())),
                          ('MSV_singletons', [(c,) for c in
                                              S.CONSTRAINT_COORDS]),
                          ('MSV_no_hdf',
                           [k for k in scorer._models.keys()
                            if 'Temp diff [K]' not in k])]:
            s_tr = np.max(np.vstack([sd_tr[k] for k in keys]), axis=0)
            s_te = np.max(np.vstack([sd_te[k] for k in keys]), axis=0)
            thr = np.quantile(s_tr, 1.0 - CONTAM)
            m = metrics(y_te, (s_te >= thr).astype(int), s_te)
            rows.append(dict(seed=seed, feature_set='F2_physics',
                             detector=tag, **m))
        print(f'  seed {seed:4d}: done '
              f'(train failures {int(y_tr.sum())}, test failures {int(y_te.sum())})')

    res = pd.DataFrame(rows)
    res.to_csv(J(HERE, 'results_a1_multiseed.csv'), index=False)

    agg = (res.groupby(['feature_set', 'detector'])
              .agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                   f1_min=('f1', 'min'), f1_max=('f1', 'max'),
                   pr_mean=('pr_auc', 'mean'), pr_std=('pr_auc', 'std'),
                   auc_mean=('roc_auc', 'mean'), auc_std=('roc_auc', 'std'))
              .reset_index().sort_values('f1_mean', ascending=False))
    agg.to_csv(J(HERE, 'results_a1_summary.csv'), index=False)
    print('\nmean F1 +/- sd over 10 seeds (sorted):')
    print(agg[['feature_set', 'detector', 'f1_mean', 'f1_std',
               'pr_mean', 'pr_std']].round(4).to_string(index=False))

    # paired tests on the comparisons that were previously undecidable
    pairs = [('F2_physics', 'MSV_no_hdf', 'F2_physics', 'MSV_full'),
             ('F2_physics', 'MSV_no_hdf', 'F2_physics', 'MSV_singletons'),
             ('F2_physics', 'MSV_singletons', 'F2_physics', 'MSV_full'),
             ('F2_physics', 'MSV_no_hdf', 'F2_physics', 'Isolation Forest'),
             ('F2_physics', 'Isolation Forest', 'F1_raw', 'Isolation Forest')]
    prows = []
    for fs_a, d_a, fs_b, d_b in pairs:
        a = res[(res.feature_set == fs_a) & (res.detector == d_a)] \
            .sort_values('seed').set_index('seed')
        b = res[(res.feature_set == fs_b) & (res.detector == d_b)] \
            .sort_values('seed').set_index('seed')
        for metric in ['f1', 'pr_auc']:
            va, vb = a[metric].to_numpy(), b[metric].to_numpy()
            diff = va - vb
            try:
                p_val = stats.wilcoxon(va, vb).pvalue
            except Exception:
                p_val = float('nan')
            # 95% CI of the mean difference via the t distribution
            n = len(diff)
            half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
            prows.append(dict(comparison=f'{d_a} @{fs_a}  vs  {d_b} @{fs_b}',
                              metric=metric,
                              mean_diff=float(diff.mean()),
                              ci_lo=float(diff.mean() - half),
                              ci_hi=float(diff.mean() + half),
                              p_wilcoxon=float(p_val),
                              n_seeds=n))
    pr = pd.DataFrame(prows)
    pr.to_csv(J(HERE, 'results_a1_paired.csv'), index=False)
    print('\npaired comparisons (10 seeds):')
    print(pr.round(4).to_string(index=False))
    return res, agg, pr


# ============================================================== Part A4
def part_a4():
    hdr('A4  per-machine-type quantile boundaries')
    p = prepare(SEED)
    d = p['d']
    tr = d.iloc[p['itr']]
    rows = []
    # (feature, true constant per type or global, scope)
    specs = [('Overstrain [Nm*min]', {'L': 11000.0, 'M': 12000.0, 'H': 13000.0},
              'upper', 'OSF'),
             ('Power [W]', {'L': 9000.0, 'M': 9000.0, 'H': 9000.0},
              'upper', 'PWF'),
             ('Power [W]', {'L': 3500.0, 'M': 3500.0, 'H': 3500.0},
              'lower', 'PWF')]

    glo = {}
    for feat, truth, side, tag in specs:
        if side == 'upper':
            lo_g, hi_g = S.quantile_boundary(tr[feat].to_numpy(), 0.005)
            glo[(feat, side)] = hi_g
        else:
            lo_g, hi_g = S.quantile_boundary(tr[feat].to_numpy(), 0.005)
            glo[(feat, side)] = lo_g
        for t in ['L', 'M', 'H']:
            sub = tr[tr['Type'] == t][feat].to_numpy()
            lo_t, hi_t = S.quantile_boundary(sub, 0.005)
            learned = hi_t if side == 'upper' else lo_t
            global_learned = glo[(feat, side)]
            rows.append(dict(
                feature=feat, side=side, mode=tag, machine_type=t,
                n_train=len(sub),
                true_constant=truth[t],
                learned_global=round(global_learned, 1),
                learned_per_type=round(learned, 1),
                err_global_pct=round(
                    100.0 * (global_learned - truth[t]) / truth[t], 2),
                err_per_type_pct=round(
                    100.0 * (learned - truth[t]) / truth[t], 2)))
    res = pd.DataFrame(rows)
    res.to_csv(J(HERE, 'results_a4_grouped_boundaries.csv'), index=False)
    print(res.to_string(index=False))
    print('\nmean |error| by scope:')
    print(res.groupby('side')[['err_global_pct', 'err_per_type_pct']]
             .apply(lambda s: s.abs().mean()).round(2).to_string())

    # ---- does the finer boundary actually buy anything? -------------------
    # The boundary-violation *rule* detector from step 4 is the cleanest place
    # to test this: it consumes the cut-offs directly. The HDF arm is excluded,
    # because its conjunction cannot be recovered from the margins (proved in
    # step 4 with the 73-vs-67 density check).
    tr, te = d.iloc[p['itr']], d.iloc[p['ite']]
    y_te = p['y_te']

    p_lo_g, p_hi_g = S.quantile_boundary(tr['Power [W]'].to_numpy(), 0.005)
    osf_g = S.quantile_boundary(tr['Overstrain [Nm*min]'].to_numpy(), 0.005)[1]
    lo_t, hi_t, osf_t = {}, {}, {}
    for t in ['L', 'M', 'H']:
        sub = tr[tr['Type'] == t]
        lo_t[t] = S.quantile_boundary(sub['Power [W]'].to_numpy(), 0.005)[0]
        hi_t[t] = S.quantile_boundary(sub['Power [W]'].to_numpy(), 0.005)[1]
        osf_t[t] = S.quantile_boundary(
            sub['Overstrain [Nm*min]'].to_numpy(), 0.005)[1]

    pv = te['Power [W]'].to_numpy()
    ov = te['Overstrain [Nm*min]'].to_numpy()
    tv = te['Type'].to_numpy()

    def rule(lo, hi, osf):
        plo = np.array([lo[t] for t in tv]) if isinstance(lo, dict) else lo
        phi = np.array([hi[t] for t in tv]) if isinstance(hi, dict) else hi
        oth = np.array([osf[t] for t in tv]) if isinstance(osf, dict) else osf
        return (((pv < plo) | (pv > phi)) | (ov > oth)).astype(int)

    true_thr = np.array([{'L': 11000.0, 'M': 12000.0, 'H': 13000.0}[t]
                         for t in tv])
    out = []
    for tag, pred in [
            ('learned, global quantile', rule(p_lo_g, p_hi_g, osf_g)),
            ('learned, per-machine-type quantile', rule(lo_t, hi_t, osf_t)),
            ('true document constants (reference)',
             (((pv < 3500.0) | (pv > 9000.0)) | (ov > true_thr)).astype(int))]:
        m = metrics(y_te, pred, pred.astype(float))
        m.pop('roc_auc')
        m.pop('pr_auc')
        out.append(dict(variant=tag, **m))
    res2 = pd.DataFrame(out)
    res2.to_csv(J(HERE, 'results_a4_effect.csv'), index=False)
    print('\nPower + Overstrain rule detector, zero labels (HDF arm excluded):')
    print(res2.round(4).to_string(index=False))
    return res


# ============================================================== Part A5
def rate_kneedle(scores):
    """Largest deviation from the chord on the descending-score curve."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    x = np.arange(n) / (n - 1)
    rng = s[0] - s[-1]
    if rng <= 0:
        return float('nan')
    y = (s - s[-1]) / rng
    idx = int(np.argmax(y - x))
    return float((idx + 1) / n)


def rate_gmm2(scores):
    """Weight of the high component of a 2-component mixture on log-scores."""
    s = np.asarray(scores, dtype=float)
    z = np.log(np.clip(s, 1e-9, None)).reshape(-1, 1)
    try:
        g = GaussianMixture(n_components=2, covariance_type='full',
                            random_state=SEED, n_init=4).fit(z)
        mu = g.means_.ravel()
        hi = int(np.argmax(mu))
        return float(g.weights_[hi])
    except Exception:
        return float('nan')


def rate_gap(scores, window=0.2):
    """Largest multiplicative jump between consecutive order statistics."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    m = max(3, int(window * n))
    seg = np.clip(s[:m], 1e-9, None)
    ratio = seg[:-1] / seg[1:]
    k = int(np.argmax(ratio))
    return float((k + 2) / n)


def rate_hill(scores, k_max_frac=0.10, k_min=20):
    """Hill estimator of the tail index; report the rate at its plateau."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    ks = np.arange(k_min, max(k_min + 5, int(k_max_frac * n)))
    gamma = []
    for k in ks:
        top = np.clip(s[:k], 1e-9, None)
        gamma.append(np.mean(np.log(top) - np.log(max(s[k], 1e-9))))
    gamma = np.asarray(gamma)
    if len(gamma) < 5:
        return float('nan')
    d = np.abs(np.diff(gamma))
    stable = int(np.argmin(np.convolve(d, np.ones(5) / 5, mode='same')))
    return float((ks[stable] + 1) / n)


def part_a5(res_a1):
    hdr('A5  label-free estimation of the contamination rate')
    p = prepare(SEED)
    coords = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]']
    sc = S.MarginalSupportScorer([(c,) for c in coords], k=5)
    sc.fit(p['d'].iloc[p['itr']][coords].to_numpy(float), coords)
    s_tr = sc.score(p['d'].iloc[p['itr']][coords].to_numpy(float))
    s_te = sc.score(p['d'].iloc[p['ite']][coords].to_numpy(float))

    est = {'kneedle': rate_kneedle(s_tr),
           'gmm2_log': rate_gmm2(s_tr),
           'largest gap': rate_gap(s_tr),
           'hill plateau': rate_hill(s_tr)}
    rows = []
    for name, r in est.items():
        if not np.isfinite(r) or r <= 0 or r >= 0.5:
            rows.append(dict(method=name, est_rate=float(r), f1=float('nan'),
                             precision=float('nan'), recall=float('nan'),
                             ratio_to_truth=float('nan')))
            continue
        thr = float(np.quantile(s_tr, 1.0 - r))
        m = metrics(p['y_te'], (s_te >= thr).astype(int), s_te)
        rows.append(dict(method=name, est_rate=float(r),
                         ratio_to_truth=float(r / CONTAM), **m))
    rows.append(dict(method='[reference] true rate 3.39%', est_rate=CONTAM,
                     ratio_to_truth=1.0,
                     **metrics(p['y_te'],
                               (s_te >= np.quantile(s_tr, 1 - CONTAM)).astype(int),
                               s_te)))
    res = pd.DataFrame(rows)
    res.to_csv(J(HERE, 'results_a5_rate_estimates.csv'), index=False)
    print(res.round(4).to_string(index=False))
    ok = res[(res.ratio_to_truth.notna())
             & (res.method.str.startswith('[', na=False) == False)]
    finite = ok[np.isfinite(ok.ratio_to_truth)]
    if len(finite):
        print(f'\nlabel-free rate estimates span '
              f'{finite.ratio_to_truth.min():.2f}x - '
              f'{finite.ratio_to_truth.max():.2f}x of the true rate')
    return res


# ============================================================== Part A3
def part_a3():
    hdr('A3  subspace fusion variants')
    p = prepare(SEED)
    d = p['d']
    coords = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]',
              'Rotational speed [rpm]']
    subs = S.default_subspaces(False)
    sc = S.MarginalSupportScorer(subs, k=5)
    sc.fit(d.iloc[p['itr']][coords].to_numpy(float), coords)
    sd_tr = sc.subspace_distances(d.iloc[p['itr']][coords].to_numpy(float))
    sd_te = sc.subspace_distances(d.iloc[p['ite']][coords].to_numpy(float))
    keys = list(sc._models.keys())
    tri = [(c,) for c in S.CONSTRAINT_COORDS]
    no_hdf = [k for k in keys if 'Temp diff [K]' not in k]
    pow_osf = [('Power [W]',), ('Overstrain [Nm*min]',)]

    def stack(sd, ks):
        return np.vstack([sd[k] for k in ks])

    def top2(m):
        return np.sort(m, axis=0)[-2:]

    variants = {}
    variants['max (steps 4-6)'] = (stack(sd_tr, keys), stack(sd_te, keys), 'max')
    variants['2nd largest'] = (stack(sd_tr, keys), stack(sd_te, keys), 'second')
    variants['mean of top-2'] = (stack(sd_tr, keys), stack(sd_te, keys), 'top2')
    variants['singletons only (3)'] = (stack(sd_tr, tri), stack(sd_te, tri), 'max')
    variants['no-HDF subspaces'] = (stack(sd_tr, no_hdf), stack(sd_te, no_hdf), 'max')
    variants['Power+Overstrain only'] = (stack(sd_tr, pow_osf),
                                         stack(sd_te, pow_osf), 'max')

    rows = []
    for name, (m_tr, m_te, how) in variants.items():
        if how == 'max':
            a_tr, a_te = m_tr.max(axis=0), m_te.max(axis=0)
        elif how == 'second':
            a_tr, a_te = np.sort(m_tr, axis=0)[-2], np.sort(m_te, axis=0)[-2]
        else:
            a_tr, a_te = top2(m_tr).mean(axis=0), top2(m_te).mean(axis=0)
        thr = float(np.quantile(a_tr, 1.0 - CONTAM))
        m = metrics(p['y_te'], (a_te >= thr).astype(int), a_te)
        rows.append(dict(variant=name, n_subspaces=m_tr.shape[0],
                         threshold=round(thr, 3), **m))

    # k-of-3 vote on the constraint singletons
    cut = {k: float(np.quantile(sd_tr[k], 1.0 - CONTAM)) for k in tri}
    v_tr = np.vstack([(sd_tr[k] >= cut[k]).astype(float) for k in tri])
    v_te = np.vstack([(sd_te[k] >= cut[k]).astype(float) for k in tri])
    tie_tr = stack(sd_tr, tri).max(axis=0) * 1e-6
    tie_te = stack(sd_te, tri).max(axis=0) * 1e-6
    for k_req in [1, 2, 3]:
        a_tr = (v_tr.sum(axis=0) >= k_req) * 1.0 + tie_tr
        a_te = (v_te.sum(axis=0) >= k_req) * 1.0 + tie_te
        thr = float(np.quantile(a_tr, 1.0 - CONTAM))
        m = metrics(p['y_te'], (a_te >= thr).astype(int), a_te)
        rows.append(dict(variant=f'vote >= {k_req} of 3 singletons',
                         n_subspaces=3, threshold=round(thr, 3), **m))

    res = pd.DataFrame(rows)
    res.to_csv(J(HERE, 'results_a3_fusion.csv'), index=False)
    print(res.round(4).to_string(index=False))
    return res


# ============================================================== Part A2
def tilt_two(s_tr, y_tr, s_te, y_te, c_mon, alpha, cal_idx, method):
    """Calibrate P(failure | score) on the labelled rows, derive two thresholds,
    then price the resulting three-tier policy on the TEST set."""
    x = np.log(np.clip(s_tr[cal_idx], 1e-9, None)).reshape(-1, 1)
    yc = y_tr[cal_idx]
    if yc.sum() == 0:
        return None
    if method == 'isotonic':
        cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        cal.fit(x.ravel(), yc)
        p_hat = lambda z: cal.predict(z)
    else:
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(x, yc)
        p_hat = lambda z: lr.predict_proba(z.reshape(-1, 1))[:, 1]

    z = np.log(np.clip(s_tr, 1e-9, None))
    order = np.argsort(z)
    zs = z[order]
    ps = p_hat(zs)
    r_alarm = C_INSP / (C_FN + C_INSP)
    r_mon = c_mon / (C_FN + c_mon)

    def thr_for(target):
        hit = np.where(ps >= target)[0]
        return float(np.exp(zs[hit[0]])) if len(hit) else float(np.inf)

    t_alarm, t_mon = thr_for(r_alarm), thr_for(r_mon)
    if not np.isfinite(t_alarm) or t_alarm > s_tr.max():
        t_alarm = float(np.quantile(s_tr, 1 - CONTAM))
    if not np.isfinite(t_mon) or t_mon > s_tr.max():
        t_mon = float(np.quantile(s_tr, 1 - 2 * CONTAM))
    if t_mon > t_alarm:
        t_mon = min(float(np.quantile(s_tr, 1 - 2 * CONTAM)), t_alarm)

    fail, norm = y_te == 1, y_te == 0
    missed = int((fail & (s_te < t_mon)).sum())
    f_band = int((fail & (s_te >= t_mon) & (s_te < t_alarm)).sum())
    n_band = int((norm & (s_te >= t_mon) & (s_te < t_alarm)).sum())
    n_alarm = int((norm & (s_te >= t_alarm)).sum())
    cost = (C_FN * missed
            + ((1 - alpha) * C_FN + c_mon) * f_band
            + c_mon * n_band
            + C_INSP * n_alarm)
    return dict(cost=cost, t_alarm=t_alarm, t_mon=t_mon,
                missed=missed, monitored_failures=f_band,
                monitored_normals=n_band, inspected_normals=n_alarm,
                alarms_any=int(((s_te >= t_mon)).sum()))


def part_a2():
    hdr('A2  three-tier action policy (inspect / monitor / ignore)')
    p = prepare(SEED)
    d = p['d']
    coords = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]']
    sc = S.MarginalSupportScorer([(c,) for c in coords], k=5)
    sc.fit(d.iloc[p['itr']][coords].to_numpy(float), coords)
    s_tr = sc.score(d.iloc[p['itr']][coords].to_numpy(float))
    s_te = sc.score(d.iloc[p['ite']][coords].to_numpy(float))
    y_tr, y_te = p['y_tr'], p['y_te']

    # ------- references under the same cost model -------------------------
    def cost2(t):
        return (C_FN * int(((y_te == 1) & (s_te < t)).sum())
                + C_INSP * int(((y_te == 0) & (s_te >= t)).sum()))

    grid = np.unique(np.quantile(s_te, np.linspace(0, 1, 2000)))
    ref = []
    for t in grid:
        c = cost2(t)
        ref.append((c, t))
    c2_best, t2_best = min(ref)
    ref_rows = [
        dict(rule='never alarm', cost=C_FN * int((y_te == 1).sum()),
             alarms=0, note='reference'),
        dict(rule='always inspect', cost=C_INSP * int((y_te == 0).sum()),
             alarms=3000, note='reference'),
        dict(rule='oracle 2-tier (test-tuned)',
             cost=c2_best, alarms=int((s_te >= t2_best).sum()),
             note='oracle, uses test labels'),
        dict(rule='step 6 published pipeline (L2b, 300 rows)',
             cost=67500.0, alarms=95, note='from step 6'),
    ]

    # ------- single-threshold watch-list, priced as MONITORING -----------
    # This is the decisive comparison. Under the 2-tier model every flagged
    # point costs C_INSP, which is why step 6 found the low-threshold watch-list
    # (523 flags, recall 0.75) unattractive. Priced as triage instead, the same
    # policy costs C_MON per flag. Thresholds come from the TRAIN score
    # distribution, so this block uses no labels at all.
    wl = []
    for q in [0.01, 0.02, 0.0339, 0.05, 0.10, 0.20]:
        t = float(np.quantile(s_tr, 1.0 - q))
        pred = s_te >= t
        missed = int(((y_te == 1) & ~pred).sum())
        recall = 1.0 - missed / int((y_te == 1).sum())
        n = int(pred.sum())
        for cm in [10.0, 20.0, 50.0, 100.0]:
            wl.append(dict(quantile=q, threshold=round(t, 3), n_flagged=n,
                           recall=round(recall, 4),
                           missed_failures=missed, c_monitor=cm,
                           cost=C_FN * missed + cm * n))
    wl_res = pd.DataFrame(wl)
    wl_res.to_csv(J(HERE, 'results_a2_watchlist.csv'), index=False)

    # Break-even: how good must triage be before the watch-list beats the
    # published step-6 pipeline (67,500)? Monitoring a failure reduces its cost
    # from C_FN to (1-alpha)*C_FN, so
    #   cost(alpha) = C_FN*missed + C_mon*n_flagged + (1-alpha)*C_FN*TP
    # and we solve cost(alpha) = 67,500 for alpha.
    be = []
    n_fail_te = int((y_te == 1).sum())
    for _, r in wl_res.iterrows():
        tp = int(round(r['recall'] * n_fail_te))
        base = C_FN * r['missed_failures'] + r['c_monitor'] * r['n_flagged']
        if tp == 0:
            alpha_be = float('nan')
        else:
            alpha_be = 1.0 - (67500.0 - base) / (C_FN * tp)
        be.append(dict(quantile=r['quantile'], c_monitor=r['c_monitor'],
                       n_flagged=int(r['n_flagged']), tp=tp,
                       cost_alpha1=base,
                       alpha_breakeven_vs_67500=round(alpha_be, 3)))
    be_res = pd.DataFrame(be)
    be_res.to_csv(J(HERE, 'results_a2_breakeven.csv'), index=False)
    print('\nbreak-even triage quality (alpha = P(a monitored failure is caught)):')
    print(be_res[be_res.c_monitor == 20.0]
          [['quantile', 'n_flagged', 'tp', 'cost_alpha1',
            'alpha_breakeven_vs_67500']].to_string(index=False))

    print('\nsingle-threshold watch-list priced as monitoring '
          '(no labels used):')
    print(wl_res[wl_res.c_monitor.isin([10.0, 20.0])]
          [['quantile', 'n_flagged', 'recall', 'missed_failures',
            'c_monitor', 'cost']].to_string(index=False))

    # ------- three-tier, calibrated on labelled rows only ----------------
    rng = np.random.RandomState(777)
    rows = []
    for alpha in [1.0, 0.5]:
        for method in ['isotonic', 'logistic']:
            for c_mon in [5.0, 10.0, 20.0, 50.0, 100.0]:
                draws = []
                for _ in range(N_CAL_DRAW):
                    cal_idx = rng.choice(len(y_tr), size=CAL_N, replace=False)
                    r = tilt_two(s_tr, y_tr, s_te, y_te, c_mon, alpha,
                                 cal_idx, method)
                    if r:
                        draws.append(r)
                if not draws:
                    continue
                cost = float(np.mean([x['cost'] for x in draws]))
                rows.append(dict(alpha=alpha, calibration=method,
                                 c_monitor=c_mon,
                                 cost_mean=cost,
                                 cost_sd=float(np.std([x['cost'] for x in draws])),
                                 t_alarm=float(np.mean([x['t_alarm'] for x in draws])),
                                 t_monitor=float(np.mean([x['t_mon'] for x in draws])),
                                 monitored_failures=float(np.mean([x['monitored_failures'] for x in draws])),
                                 monitored_normals=float(np.mean([x['monitored_normals'] for x in draws])),
                                 inspected_normals=float(np.mean([x['inspected_normals'] for x in draws])),
                                 alarms_any=float(np.mean([x['alarms_any'] for x in draws]))))
    res = pd.DataFrame(rows)
    res.to_csv(J(HERE, 'results_a2_three_tier.csv'), index=False)
    refs = pd.DataFrame(ref_rows)
    refs.to_csv(J(HERE, 'results_a2_reference.csv'), index=False)

    print('references (same cost model, C_FN=1000, C_INSP=100):')
    print(refs.to_string(index=False))
    print('\nthree-tier policy, calibrated on 300 labelled rows '
          f'({N_CAL_DRAW} draws each):')
    show = res[res.c_monitor.isin([10.0, 20.0, 50.0])]
    print(show[['alpha', 'calibration', 'c_monitor', 'cost_mean', 'cost_sd',
                't_monitor', 't_alarm']].round(2).to_string(index=False))

    best2 = res[(res.c_monitor == 100.0)].groupby('alpha').cost_mean.min()
    best_wl = wl_res.groupby('c_monitor').cost.min()
    print('\nbest watch-list cost by monitor price:',
          {int(k): int(v) for k, v in best_wl.items()})
    print(f'\nbest three-tier cost at C_mon=20: '
          f'{res[res.c_monitor == 20.0].cost_mean.min():,.0f}  '
          f'vs step 6 pipeline 67,500  vs oracle 2-tier {c2_best:,.0f}')
    return res, refs, wl_res


# ============================================================== main
def main():
    t0 = time.time()
    a1_res, a1_agg, a1_pr = part_a1()
    a3 = part_a3()
    a4 = part_a4()
    a5 = part_a5(a1_res)
    a2, a2_ref, a2_wl = part_a2()

    summary = dict(
        protocol=dict(seeds=SEEDS, contamination=CONTAM, c_fn=C_FN,
                      c_insp=C_INSP, cal_n=CAL_N, n_cal_draw=N_CAL_DRAW),
        a1_top=json.loads(a1_agg.head(8).to_json(orient='records')),
        a1_paired=json.loads(a1_pr.to_json(orient='records')),
        a3_fusion=json.loads(a3.to_json(orient='records')),
        a4_boundaries=json.loads(a4.to_json(orient='records')),
        a5_estimates=json.loads(a5.to_json(orient='records')),
        a2_reference=json.loads(a2_ref.to_json(orient='records')),
        a2_watchlist_best={int(k): float(v) for k, v in
                           a2_wl.groupby('c_monitor').cost.min().items()},
        a2_best_three_tier=float(a2[a2.c_monitor == 20.0].cost_mean.min()),
        runtime_s=round(time.time() - t0, 1))
    with open(J(HERE, 'summary_step6after.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    hdr(f'done in {summary["runtime_s"]} s')
    print('files written to', HERE)


if __name__ == '__main__':
    main()
