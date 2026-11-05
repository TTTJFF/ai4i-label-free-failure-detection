# -*- coding: utf-8 -*-
"""Step 6 — the end-to-end decision pipeline.

    raw sensor reading (5 values + machine type)
        -> feature layer            (power, temp diff, overstrain; label-free)
        -> violation scorer         (MSV, step 4; label-free)
        -> cost-sensitive decision  (alarm threshold from 1000:100 cost model)
        -> reason code + action     (which constraint was violated)

The pipeline object is fitted once on the training split and then used by both
`run_step6.py` (evaluation) and `demo.py` (single-reading demonstration).

What is label-free and what is not
----------------------------------
* The feature layer and the scorer never see a label.
* The alarm threshold comes from the business cost model (C_FN = 1000,
  C_FP = 100) applied to a SMALL LABELLED RANDOM SAMPLE of the training data
  (the L2b protocol validated in step 5). The cost model is domain knowledge;
  the sample supplies the failure rate, nothing else.
* The reason code is a *violation attribution*, not a trained failure-mode
  classifier. It is mapped to a maintenance action through a small, explicit
  action table (again domain knowledge).

Everything the pipeline needs at inference time is five sensor values and the
machine type. No label, no history, no network.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE.parent / 'step4'))
import features as F                        # noqa: E402
import support as S                          # noqa: E402

C_FN = 1000.0        # cost of a missed failure
C_FP = 100.0         # cost of a false alarm
CAL_SAMPLE = 300     # labelled random rows used only to estimate the rate
SEED = 42

# Violation attribution -> failure mode to investigate.
# Rule: the constraint coordinate whose ONE-DIMENSIONAL support violation is
# largest at that point. Using the per-coordinate distance (rather than a
# priority lookup on the winning subspace) avoids mislabelling a mixed pair:
# e.g. the subspace (Power, Temp diff) used to be reported as PWF even when the
# violation was entirely in the temperature direction.
REASON_COORDS = [('Power [W]', 'PWF'), ('Temp diff [K]', 'HDF'),
                 ('Overstrain [Nm*min]', 'OSF')]
COORD_CODE = {c: code for c, code in REASON_COORDS}

REASON_TEXT = {
    'PWF': 'power out of the operating band',
    'OSF': 'accumulated overstrain above the limit',
    'HDF': 'heat-dissipation margin too small at low speed',
    'unknown': 'no single constraint dominates',
}

# Business action table (domain knowledge, not learned).
ACTION_TABLE = {
    'PWF': 'Check the drive / power electronics and verify the load and speed setpoint.',
    'OSF': 'Replace the tool and re-check the torque setpoint; avoid sustained overload.',
    'HDF': 'Check cooling and heat dissipation; verify ambient and process temperature.',
    'unknown': 'Schedule an inspection.',
}


def reason_of(subspace):
    for col, code in REASON_ORDER:
        if col in subspace:
            return code
    return 'unknown'


def cost_threshold(score, y, c_fn=C_FN, c_fp=C_FP):
    """Threshold minimising c_fn*FN + c_fp*FP on a labelled sample."""
    cands = np.unique(np.quantile(score, np.linspace(0.0, 0.9999, 600)))
    best_t, best_c = float(cands[0]), np.inf
    for t in cands:
        pred = (score >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        c = c_fn * fn + c_fp * fp
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


class MaintenancePipeline:
    """Fit on a dataframe with the raw AI4I columns and the `Machine failure`
    label (the label is used only for the calibration sample)."""

    def __init__(self, c_fn=C_FN, c_fp=C_FP, cal_sample=CAL_SAMPLE, seed=SEED):
        self.c_fn, self.c_fp = c_fn, c_fp
        self.cal_sample, self.seed = cal_sample, seed

    # ------------------------------------------------------------------ fit
    def fit(self, df_train):
        cols = S.CONSTRAINT_COORDS + S.CONJUNCTION_PARTNERS
        d = F.add_physics(df_train)
        self.cols = cols
        self.scorer = S.MarginalSupportScorer(S.default_subspaces(False), k=5)
        self.scorer.fit(d[cols].to_numpy(dtype=float), cols)
        self.reason_scorer = S.MarginalSupportScorer(
            [(c,) for c, _ in REASON_COORDS], k=5)
        self.reason_scorer.fit(d[cols].to_numpy(dtype=float), cols)

        s_tr = self.scorer.score(d[cols].to_numpy(dtype=float))
        y_tr = d[F.LABEL].to_numpy(dtype=int)

        rng = np.random.RandomState(self.seed)
        samp = rng.choice(len(d), size=min(self.cal_sample, len(d)),
                          replace=False)
        self.calibration_rate = float(y_tr[samp].mean())
        self.threshold = cost_threshold(s_tr[samp], y_tr[samp],
                                        self.c_fn, self.c_fp)
        self.inspection_threshold_ = None
        # a second, softer operating point for "watch list" reporting
        self.watch_threshold = float(np.quantile(s_tr, 1.0 - min(
            3.0 * max(self.calibration_rate, 1e-3), 0.2)))
        return self

    # -------------------------------------------------------------- predict
    def predict(self, df):
        """Return a verdict frame: alarm, score, reason code, action."""
        d = F.add_physics(df)
        X = d[self.cols].to_numpy(dtype=float)
        score = self.scorer.score(X)

        # Attribution: take the subspace that actually produced the score. If it
        # is a singleton constraint coordinate, that is the reason. If it is a
        # pair, resolve it to the member whose one-dimensional violation is
        # larger -- so a point sitting in the (Temp diff, Speed) corner is
        # reported as HDF rather than as whatever priority order says.
        sd = self.scorer.subspace_distances(X)
        names = list(sd.keys())
        mat = np.vstack([sd[S] for S in names])
        win = np.argmax(mat, axis=0)
        sing = self.reason_scorer.subspace_distances(X)
        reason = []
        for j, w in enumerate(win):
            S = names[w]
            members = [c for c in S if c in COORD_CODE]
            if not members:
                reason.append('unknown')
            elif len(members) == 1:
                reason.append(COORD_CODE[members[0]])
            else:
                jj = int(np.argmax([sing[(c,)][j] for c in members]))
                reason.append(COORD_CODE[members[jj]])
        alarm = (score >= self.threshold).astype(int)
        watch = (score >= self.watch_threshold).astype(int)
        return pd.DataFrame({
            'score': score,
            'alarm': alarm,
            'watch': watch,
            'reason': reason,
            'reason_text': [REASON_TEXT[r] for r in reason],
            'action': [ACTION_TABLE[r] for r in reason],
        })

    def explain(self):
        return dict(threshold=self.threshold,
                    calibration_rate=self.calibration_rate,
                    watch_threshold=self.watch_threshold,
                    cost_model=f'C(FN)={self.c_fn:.0f}, C(FP)={self.c_fp:.0f}',
                    subspaces=len(self.scorer.subspaces),
                    coords=self.cols)


def fit_from_project(n_train_rows=None, **kw):
    """Convenience: fit the pipeline on the project's own training split."""
    from sklearn.model_selection import train_test_split
    df = F.load_data()
    y = df[F.LABEL].to_numpy(dtype=int)
    idx = np.arange(len(df))
    idx_tr, _ = train_test_split(idx, test_size=0.3, random_state=SEED,
                                 stratify=y)
    if n_train_rows:
        idx_tr = idx_tr[:n_train_rows]
    return MaintenancePipeline(**kw).fit(df.iloc[idx_tr]), df, idx_tr
