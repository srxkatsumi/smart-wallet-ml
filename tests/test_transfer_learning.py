import numpy as np
import pytest
from evaluation.transfer_learning import (
    maximum_mean_discrepancy,
    coral_align,
    cross_domain_evaluation,
)


def _rng(seed=42):
    return np.random.default_rng(seed)


def _make_domain(n: int, n_features: int, shift: float = 0.0, seed: int = 42):
    rng = _rng(seed)
    X   = rng.standard_normal((n, n_features)) + shift
    y   = rng.integers(0, 2, n).astype(float)
    return X, y


# ── MMD ───────────────────────────────────────────────────────────────────

def test_mmd_same_distribution_near_zero():
    rng    = _rng()
    X      = rng.standard_normal((100, 10))
    mmd    = maximum_mean_discrepancy(X, X)
    assert mmd < 0.05


def test_mmd_different_distributions_positive():
    rng = _rng()
    Xs  = rng.standard_normal((100, 10))
    Xt  = rng.standard_normal((100, 10)) + 3.0
    mmd = maximum_mean_discrepancy(Xs, Xt)
    assert mmd > 0.0


def test_mmd_non_negative():
    rng = _rng()
    Xs  = rng.standard_normal((80, 5))
    Xt  = rng.standard_normal((80, 5)) * 2
    assert maximum_mean_discrepancy(Xs, Xt) >= 0.0


# ── CORAL ─────────────────────────────────────────────────────────────────

def test_coral_output_shape():
    rng = _rng()
    Xs  = rng.standard_normal((100, 10))
    Xt  = rng.standard_normal((80,  10)) + 2.0
    Xa  = coral_align(Xs, Xt)
    assert Xa.shape == Xs.shape


def test_coral_reduces_mmd():
    rng  = _rng()
    Xs   = rng.standard_normal((150, 10))
    Xt   = rng.standard_normal((150, 10)) + 2.0
    mmd_before = maximum_mean_discrepancy(Xs, Xt)
    Xa   = coral_align(Xs, Xt)
    mmd_after  = maximum_mean_discrepancy(Xa, Xt)
    assert mmd_after <= mmd_before + 0.5


# ── Cross-domain Evaluation ───────────────────────────────────────────────

def test_cross_domain_returns_keys():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler

    Xs, ys = _make_domain(100, 10, shift=0.0, seed=1)
    Xt, yt = _make_domain(80,  10, shift=1.0, seed=2)

    sc  = RobustScaler()
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(sc.fit_transform(Xs), ys.astype(int))

    def predict_fn(X):
        return clf.predict_proba(X)[:, 1]

    r = cross_domain_evaluation(predict_fn, Xs, ys, Xt, yt, use_coral=True)
    for k in ["mmd", "acc_source", "acc_transfer", "transfer_gap",
              "acc_coral", "coral_improvement"]:
        assert k in r


def test_cross_domain_acc_in_bounds():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler

    Xs, ys = _make_domain(100, 10, seed=10)
    Xt, yt = _make_domain(80,  10, seed=20)

    sc  = RobustScaler()
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(sc.fit_transform(Xs), ys.astype(int))

    def predict_fn(X):
        return clf.predict_proba(X)[:, 1]

    r = cross_domain_evaluation(predict_fn, Xs, ys, Xt, yt, use_coral=False)
    assert 0.0 <= r["acc_source"]   <= 1.0
    assert 0.0 <= r["acc_transfer"] <= 1.0
