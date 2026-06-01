import numpy as np
import pytest
from evaluation.statistical_tests import (
    diebold_mariano,
    mcnemar_test,
    ljung_box_test,
    metrics_carteira,
    metrics_megasena,
    full_report,
)


def _rng(seed=42):
    return np.random.default_rng(seed)


# ── Diebold-Mariano ───────────────────────────────────────────────────────

def test_dm_equal_models():
    rng = _rng()
    e = rng.standard_normal(100)
    r = diebold_mariano(e, e)
    assert r["statistic"] == 0.0 or abs(r["p_value"] - 1.0) < 0.01


def test_dm_returns_required_keys():
    rng = _rng()
    e1, e2 = rng.standard_normal(100), rng.standard_normal(100)
    r = diebold_mariano(e1, e2)
    assert "statistic" in r
    assert "p_value"   in r
    assert "conclusion" in r
    assert 0.0 <= r["p_value"] <= 1.0


def test_dm_significantly_different():
    rng = _rng()
    e1 = rng.standard_normal(200) * 2   # modelo fraco
    e2 = rng.standard_normal(200) * 0.1 # modelo forte
    r  = diebold_mariano(e1, e2)
    assert r["p_value"] < 0.05


# ── McNemar ───────────────────────────────────────────────────────────────

def test_mcnemar_identical_classifiers():
    c = np.array([True, False, True, True, False])
    r = mcnemar_test(c, c)
    assert r["p_value"] >= 0.05


def test_mcnemar_returns_keys():
    rng = _rng()
    c1 = rng.integers(0, 2, 100).astype(bool)
    c2 = rng.integers(0, 2, 100).astype(bool)
    r  = mcnemar_test(c1, c2)
    for k in ["n01", "n10", "statistic", "p_value", "conclusion"]:
        assert k in r


# ── Ljung-Box ─────────────────────────────────────────────────────────────

def test_ljungbox_white_noise():
    rng = _rng()
    r   = ljung_box_test(rng.standard_normal(200), lags=10)
    assert "has_autocorrelation" in r
    assert 0.0 <= r["p_value"] <= 1.0


def test_ljungbox_autocorrelated():
    ar = np.zeros(200)
    for t in range(1, 200):
        ar[t] = 0.8 * ar[t - 1] + np.random.randn()
    r = ljung_box_test(ar)
    assert r["has_autocorrelation"] is True


# ── Métricas Carteira ─────────────────────────────────────────────────────

def test_metrics_carteira_keys():
    rng    = _rng()
    y_true = rng.integers(0, 2, 100).astype(float)
    y_prob = rng.uniform(0, 1, 100)
    r      = metrics_carteira(y_true, y_prob)
    for k in ["accuracy", "f1", "brier_score", "auc_roc"]:
        assert k in r
    assert 0.0 <= r["accuracy"] <= 1.0


def test_metrics_perfect_classifier():
    y = np.array([1, 0, 1, 0, 1], dtype=float)
    r = metrics_carteira(y, y)
    assert r["accuracy"] == 1.0
    assert r["brier_score"] == 0.0


# ── Métricas Mega Sena ────────────────────────────────────────────────────

def test_metrics_megasena():
    probs  = np.zeros(60)
    probs[:6] = 0.9
    actual = [1, 2, 3, 4, 5, 6]
    r = metrics_megasena(probs, actual)
    assert r["matches"] == 6
    assert r["accuracy_at_k"] == 1.0


def test_metrics_megasena_random_baseline():
    rng   = _rng()
    probs = rng.uniform(0, 1, 60)
    r     = metrics_megasena(probs, [1, 2, 3, 4, 5, 6])
    assert 0.0 <= r["accuracy_at_k"] <= 1.0
    assert r["baseline_random"] == pytest.approx(6 / 60)


# ── Full report ───────────────────────────────────────────────────────────

def test_full_report_with_baseline():
    rng      = _rng()
    y_true   = rng.integers(0, 2, 100).astype(float)
    y_model  = rng.uniform(0, 1, 100)
    y_base   = np.full(100, 0.5)
    r = full_report("test_model", y_true, y_model, y_base)
    assert "metrics"          in r
    assert "ljung_box"        in r
    assert "diebold_mariano"  in r
    assert "mcnemar"          in r
