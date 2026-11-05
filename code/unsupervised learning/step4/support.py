# -*- coding: utf-8 -*-
"""Step 4 core: label-free support-boundary learning and violation scoring.

Two things live here.

(A) Support-boundary learners (1-D)
    Three label-free strategies for locating the edge of a distribution's
    support. None of them sees a label; all of them are estimated on the
    training split only.

        quantile_boundary : extreme empirical quantiles
        cliff_boundary    : walk outward from the mode bin until the density
                            falls below a fraction of the peak ("density cliff")
        evt_boundary      : peaks-over-threshold, generalised Pareto fit to the
                            tail, boundary at a chosen return period

(B) MarginalSupportScorer — the boundary-violation detector
    The step 2/3 diagnosis was that standard detectors score *rarity* while the
    failures are *support violations*: in the correct coordinates the failure
    region is literally empty of normal samples (Power outside [3500, 9000],
    the corner TempDiff < 8.6 AND Speed < 1380, Overstrain above a threshold).

    So instead of estimating density in the full space, this scorer estimates
    the support in every low-dimensional marginal and its 2-D projections:

        for every subspace S (1-D constraint coordinate, and every pair)
            d_S(x) = distance from x to its k-th nearest TRAINING neighbour
                     inside S, divided by the median such distance over the
                     training set
        score(x) = max over S of d_S(x)

    A point that is far from the training cloud in *any* low-dimensional
    projection is outside the support of that projection. Maximising over
    projections is what lets a conjunction (TempDiff AND Speed) be caught:
    the corner is empty in the (TempDiff, Speed) plane even though neither
    marginal alone is unusual there.

    This is a union-of-subspaces support estimator. Every quantity it uses is
    computed from unlabelled training data only.
"""
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------- (A)
def quantile_boundary(x, p=0.005):
    """Boundary at the p and 1-p empirical quantiles. Returns (lo, hi)."""
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, p)), float(np.quantile(x, 1.0 - p))


def cliff_boundary(x, bins=None, frac=0.005, pad=0.05, smooth=5):
    """Walk outward from the modal bin until the (smoothed) bin count drops
    below frac * peak count; the boundary is the outer edge of the last bin
    that still satisfies the condition.

    This is the "density cliff" rule: for a feature whose support has a hard
    edge, the histogram simply stops. The bin count adapts to the sample size
    so that small subsets (e.g. one machine type) are not shredded into empty
    bins, and a moving average is applied so that the walk is not terminated
    by a single empty bin inside an otherwise populated tail.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if bins is None:
        bins = int(np.clip(round(n / 30.0), 16, 200))
    lo, hi = x.min(), x.max()
    span = hi - lo if hi > lo else 1.0
    edges = np.linspace(lo - pad * span, hi + pad * span, bins + 1)
    cnt, _ = np.histogram(x, bins=edges)

    if smooth > 1 and len(cnt) > smooth:
        k = np.ones(smooth) / smooth
        sm = np.convolve(cnt, k, mode='same')
    else:
        sm = cnt.astype(float)

    peak = sm.max()
    need = max(frac * peak, 1e-12)
    mode = int(np.argmax(sm))

    left = mode
    while left - 1 >= 0 and sm[left - 1] >= need:
        left -= 1
    right = mode
    while right + 1 < len(sm) and sm[right + 1] >= need:
        right += 1
    return float(edges[left]), float(edges[right + 1])


def evt_boundary(x, q=0.95, n_return=None, side='upper'):
    """Peaks-over-threshold boundary.

    Fits a generalised Pareto distribution to the exceedances above the
    q-quantile (or below the 1-q quantile) and returns the level that is
    exceeded once per `n_return` observations. Falls back to the empirical
    quantile if the fit fails.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_return = n_return or n
    p_exceed = (1.0 - q) / n_return
    try:
        if side == 'upper':
            u = np.quantile(x, q)
            exc = x[x > u] - u
            if len(exc) < 20:
                raise ValueError('too few exceedances')
            c, loc, scale = stats.genpareto.fit(exc, floc=0.0)
            level = u + stats.genpareto.ppf(1.0 - p_exceed / (1.0 - q), c,
                                            loc=0.0, scale=scale)
            return float(np.quantile(x, 1.0 - p_exceed)), float(level)
        u = np.quantile(x, 1.0 - q)
        exc = u - x[x < u]
        if len(exc) < 20:
            raise ValueError('too few exceedances')
        c, loc, scale = stats.genpareto.fit(exc, floc=0.0)
        level = u - stats.genpareto.ppf(1.0 - p_exceed / (1.0 - q), c,
                                        loc=0.0, scale=scale)
        return float(level), float(np.quantile(x, p_exceed))
    except Exception:
        return (float(np.quantile(x, p_exceed)),
                float(np.quantile(x, 1.0 - p_exceed)))


# ----------------------------------------------------------------- (B)
class MarginalSupportScorer:
    """Union-of-subspaces support violation score.

    Parameters
    ----------
    subspaces : list of tuple of column names
    k : neighbour rank used as the distance statistic
    """

    def __init__(self, subspaces, k=5):
        self.subspaces = [tuple(s) for s in subspaces]
        self.k = k
        self._models = {}
        self._median = {}
        self._scale = None

    def fit(self, X, cols):
        X = np.asarray(X, dtype=float)
        self.cols = list(cols)
        self._scale = X.mean(axis=0), X.std(axis=0) + 1e-12
        Z = self._standardise(X)
        for S in self.subspaces:
            idx = [self.cols.index(c) for c in S]
            sub = Z[:, idx]
            nn = NearestNeighbors(n_neighbors=self.k).fit(sub)
            d, _ = nn.kneighbors(sub)          # column 0 is self
            dk = d[:, -1]
            self._models[S] = nn
            self._median[S] = float(np.median(dk)) + 1e-12
        return self

    def _standardise(self, X):
        mu, sd = self._scale
        return (np.asarray(X, dtype=float) - mu) / sd

    def score(self, X):
        Z = self._standardise(X)
        out = np.full(len(Z), -np.inf)
        for S, nn in self._models.items():
            idx = [self.cols.index(c) for c in S]
            d, _ = nn.kneighbors(Z[:, idx])
            out = np.maximum(out, d[:, -1] / self._median[S])
        return out

    def subspace_distances(self, X):
        """Normalised k-NN distance inside every fitted subspace, as a dict."""
        Z = self._standardise(X)
        out = {}
        for S, nn in self._models.items():
            idx = [self.cols.index(c) for c in S]
            d, _ = nn.kneighbors(Z[:, idx])
            out[S] = d[:, -1] / self._median[S]
        return out

    def score_with_reason(self, X):
        """Same score, plus which subspace produced it.

        Attribution is what lets an alarm carry a reason code ("power out of
        range", "overstrain", "heat dissipation") instead of just a number.
        It is an *attribution of the violation*, not a certified failure-mode
        label; run_step6.py measures how often it agrees with the true mode.
        """
        Z = self._standardise(X)
        best = np.full(len(Z), -np.inf)
        who = np.zeros(len(Z), dtype=int)
        names = list(self._models.keys())
        for i, (S, nn) in enumerate(self._models.items()):
            idx = [self.cols.index(c) for c in S]
            d, _ = nn.kneighbors(Z[:, idx])
            r = d[:, -1] / self._median[S]
            upd = r > best
            best = np.where(upd, r, best)
            who = np.where(upd, i, who)
        return best, [names[i] for i in who]


# The subspace pool used by the main configuration.
# Constraint coordinates (singletons) plus every pair that can express a
# conjunction. The Speed singleton is deliberately excluded: its extreme
# values are not failures, and including it only adds false alarms (this is a
# documented design choice; the sensitivity variant in run_step4.py reports
# the variant that includes it).
CONSTRAINT_COORDS = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]']
CONJUNCTION_PARTNERS = ['Rotational speed [rpm]']


def default_subspaces(include_speed_singleton=False):
    tri = list(CONSTRAINT_COORDS)
    pool = tri + CONJUNCTION_PARTNERS
    subs = [(c,) for c in tri]
    if include_speed_singleton:
        subs += [(c,) for c in CONJUNCTION_PARTNERS]
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            subs.append((a, b))
    return subs
