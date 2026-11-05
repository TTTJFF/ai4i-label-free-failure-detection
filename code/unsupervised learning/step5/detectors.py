# -*- coding: utf-8 -*-
"""Step 5 — detector pool.

Every detector exposes the same two calls:

    fit(X_train)          -> self
    score(X_test)         -> np.ndarray, larger = more anomalous

so the ablation matrix can treat them interchangeably. All of them are
label-free: `fit` sees only feature vectors.

The three baseline detectors (Isolation Forest, One-Class SVM, LOF) are
imported from scikit-learn with the same hyper-parameters the frozen baseline
used. The four additional ones widen the pool so that step 4's conclusion can
be checked against more than the three original algorithms:

    Gaussian mixture      negative log-likelihood of a 5-component GMM
    Kernel density        negative log-likelihood of a Gaussian KDE
    Mahalanobis distance  distance under the empirical covariance
    HBOS-style histogram  sum over features of -log(bin probability)

Plus `MsvDetector`, the boundary-violation scorer introduced in step 4.

Honest labelling: the histogram detector is our own implementation of the
HBOS idea ("histogram-based outlier score": fit one histogram per feature and
sum the negative log bin probabilities). It is not the pyod implementation,
and it is reported under the name HBOS-style rather than HBOS.
"""
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.svm import OneClassSVM

import support as S


class SklearnDetector:
    """Thin wrapper with a uniform interface."""

    def __init__(self, make, uses_score_samples=False):
        self.make = make
        self.uses_score_samples = uses_score_samples
        self.model = None

    def fit(self, X):
        self.model = self.make()
        self.model.fit(X)
        return self

    def score(self, X):
        if self.uses_score_samples:
            return -self.model.score_samples(X)
        return -self.model.decision_function(X)


class GaussianMixtureDetector:
    def __init__(self, n_components=5, seed=42):
        self.k = n_components
        self.seed = seed

    def fit(self, X):
        self.model = GaussianMixture(n_components=self.k, covariance_type='full',
                                     random_state=self.seed, reg_covar=1e-6,
                                     max_iter=300).fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class KdeDetector:
    """Gaussian kernel density with Scott's rule bandwidth."""

    def __init__(self):
        self.bw = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.bw = n ** (-1.0 / (d + 4.0))
        self.model = KernelDensity(kernel='gaussian', bandwidth=self.bw).fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class MahalanobisDetector:
    def fit(self, X):
        self.model = EmpiricalCovariance().fit(X)
        return self

    def score(self, X):
        return self.model.mahalanobis(X)


class HbosStyleDetector:
    """Per-feature histograms; score = sum of -log(bin probability).

    Bin count adapts to sample size (about 30 observations per bin) so that
    every feature is treated the same way regardless of scale.
    """

    def __init__(self, bins=None):
        self.bins = bins

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        b = self.bins or int(np.clip(round(n / 30.0), 16, 120))
        self.bins_ = b
        self.edges_ = []
        self.logp_ = []
        for j in range(d):
            lo, hi = X[:, j].min(), X[:, j].max()
            span = hi - lo if hi > lo else 1.0
            e = np.linspace(lo - 0.05 * span, hi + 0.05 * span, b + 1)
            cnt, _ = np.histogram(X[:, j], bins=e)
            p = (cnt + 1.0) / (n + b)
            self.edges_.append(e)
            self.logp_.append(-np.log(p))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X))
        for j in range(X.shape[1]):
            idx = np.clip(np.digitize(X[:, j], self.edges_[j]) - 1,
                          0, self.bins_ - 1)
            out += self.logp_[j][idx]
        return out


class MsvDetector:
    """Boundary-violation scorer from step 4 (union of low-dimensional
    marginal supports)."""

    def __init__(self, subspaces, cols, k=5):
        self.subspaces = subspaces
        self.cols = list(cols)
        self.k = k

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.model = S.MarginalSupportScorer(self.subspaces, k=self.k)
        self.model.fit(X, self.cols)
        return self

    def score(self, X):
        return self.model.score(np.asarray(X, dtype=float))


def default_pool():
    """Name -> zero-argument factory. Hyper-parameters mirror the frozen
    baseline wherever a baseline detector is used."""
    return {
        'Isolation Forest': lambda: SklearnDetector(
            lambda: IsolationForest(n_estimators=300, contamination='auto',
                                    random_state=42)),
        'One-Class SVM': lambda: SklearnDetector(
            lambda: OneClassSVM(kernel='rbf', nu=0.0339, gamma='auto')),
        'Local Outlier Factor': lambda: SklearnDetector(
            lambda: LocalOutlierFactor(n_neighbors=20, novelty=True)),
        'Gaussian Mixture': lambda: GaussianMixtureDetector(5, 42),
        'Kernel Density': lambda: KdeDetector(),
        'Mahalanobis': lambda: MahalanobisDetector(),
        'HBOS-style': lambda: HbosStyleDetector(),
    }


# Subspace pools for the MSV detector.
# F2 / F3 contain the derived constraint coordinates, so the pool is the same
# one validated in step 4 (three constraint coordinates + every pair, with the
# speed singleton excluded -- see step 4 for the sensitivity check).
# F1_raw has no derived coordinates, so the pool falls back to the raw sensors.
MSV_POOL_DERIVED = S.default_subspaces(False)


def msv_pool_for(feature_names):
    """Pick the MSV subspace pool appropriate for a feature set."""
    numeric = [c for c in feature_names if not c.startswith('Type=')]
    if all(c in numeric for c in S.CONSTRAINT_COORDS):
        pool = list(MSV_POOL_DERIVED)
        if 'Rotational speed [rpm]' in numeric:
            return pool
    pool = [(c,) for c in numeric]
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            pool.append((a, b))
    return pool
