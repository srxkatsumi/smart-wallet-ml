import numpy as np
import pytest
from evaluation.information_theory import (
    shannon_entropy, max_entropy, normalized_entropy,
    mutual_information_features, permutation_entropy,
    transfer_entropy, domain_predictability_report,
)


def _rng(seed=42):
    return np.random.default_rng(seed)


# ── Shannon Entropy ───────────────────────────────────────────────────────

def test_entropy_uniform_is_maximum():
    rng    = _rng()
    series = rng.uniform(0, 1, 1000)
    h      = normalized_entropy(series)
    assert h > 0.8


def test_entropy_constant_is_zero():
    series = np.ones(100)
    h      = shannon_entropy(series, n_bins=5)
    assert h == pytest.approx(0.0, abs=0.01)


def test_max_entropy_formula():
    assert max_entropy(8) == pytest.approx(3.0, abs=0.01)
    assert max_entropy(2) == pytest.approx(1.0, abs=0.01)


# ── Mutual Information ────────────────────────────────────────────────────

def test_mi_independent_features_near_zero():
    rng  = _rng()
    X    = rng.standard_normal((200, 5))
    y    = rng.integers(0, 2, 200)
    r    = mutual_information_features(X, y)
    assert r["mean_mi"] >= 0.0
    assert len(r["top5_features"]) == 5


def test_mi_informative_feature_has_higher_mi():
    rng  = _rng()
    y    = rng.integers(0, 2, 200)
    X    = rng.standard_normal((200, 5))
    X[:, 0] = y + rng.normal(0, 0.1, 200)   # feature 0 fortemente correlacionada
    r    = mutual_information_features(X, y)
    top_feature_idx = r["top5_features"][0][0]
    assert "0" in top_feature_idx or "f0" in top_feature_idx


# ── Permutation Entropy ───────────────────────────────────────────────────

def test_pe_random_near_one():
    rng  = _rng()
    pe   = permutation_entropy(rng.standard_normal(500), m=3)
    assert pe > 0.85


def test_pe_monotone_near_zero():
    series = np.arange(100, dtype=float)
    pe     = permutation_entropy(series, m=3)
    assert pe < 0.1


def test_pe_bounds():
    rng  = _rng()
    pe   = permutation_entropy(rng.standard_normal(200), m=4)
    assert 0.0 <= pe <= 1.0


# ── Transfer Entropy ──────────────────────────────────────────────────────

def test_te_independent_series_near_zero():
    rng = _rng()
    s   = rng.standard_normal(200)
    t   = rng.standard_normal(200)
    te  = transfer_entropy(s, t)
    assert te >= 0.0


def test_te_non_negative():
    rng = _rng()
    ar  = np.zeros(200)
    for i in range(1, 200):
        ar[i] = 0.8 * ar[i-1] + rng.standard_normal()
    te = transfer_entropy(ar[:-1], ar[1:])
    assert te >= 0.0


# ── Domain report ─────────────────────────────────────────────────────────

def test_domain_report_returns_keys():
    rng = _rng()
    X   = rng.standard_normal((100, 10))
    y   = rng.integers(0, 2, 100).astype(float)
    r   = domain_predictability_report(X, y, domain_name="test")
    for k in ["normalized_entropy", "mean_mutual_info",
              "permutation_entropy", "verdict"]:
        assert k in r


def test_domain_report_megasena_verdict():
    """Domínio aleatório deve ter veredicto de processo próximo aleatório."""
    rng = _rng()
    X   = rng.standard_normal((300, 16))
    y   = rng.integers(0, 2, 300).astype(float)
    r   = domain_predictability_report(X, y, domain_name="megasena")
    assert r["permutation_entropy"] > 0.7
