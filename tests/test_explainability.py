import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from evaluation.explainability import (
    shap_tree,
    lime_explain,
    feature_importance_report,
)


def _make_data(n: int = 100, n_features: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, n_features))
    y   = rng.integers(0, 2, n)
    return X, y


def _train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


# ── SHAP tree ─────────────────────────────────────────────────────────────

def test_shap_tree_returns_keys():
    X, y = _make_data()
    clf  = _train_rf(X, y)
    r    = shap_tree(clf, X)
    for k in ["shap_values", "expected_value", "mean_abs_shap", "top5_features"]:
        assert k in r


def test_shap_tree_shape():
    X, y = _make_data()
    clf  = _train_rf(X, y)
    r    = shap_tree(clf, X)
    assert r["shap_values"].shape == X.shape


def test_shap_tree_top5_length():
    X, y = _make_data()
    clf  = _train_rf(X, y)
    names = [f"feat_{i}" for i in range(X.shape[1])]
    r     = shap_tree(clf, X, feature_names=names)
    assert len(r["top5_features"]) == 5
    assert all(isinstance(name, str) for name, _ in r["top5_features"])


# ── LIME ─────────────────────────────────────────────────────────────────

def test_lime_returns_keys():
    X, y  = _make_data(n=30)
    clf   = _train_rf(X, y)
    r     = lime_explain(clf.predict_proba, X[:5], n_samples=20)
    for k in ["explanations", "global_importance", "top5_features"]:
        assert k in r


def test_lime_global_importance_shape():
    X, y = _make_data(n=30)
    clf  = _train_rf(X, y)
    r    = lime_explain(clf.predict_proba, X[:5], n_samples=20)
    assert r["global_importance"].shape == (X.shape[1],)


def test_lime_top5_names():
    X, y  = _make_data(n=30)
    clf   = _train_rf(X, y)
    names = [f"feat_{i}" for i in range(X.shape[1])]
    r     = lime_explain(clf.predict_proba, X[:5],
                         feature_names=names, n_samples=20)
    assert len(r["top5_features"]) == 5


# ── feature_importance_report ─────────────────────────────────────────────

def test_report_tree_type():
    X, y = _make_data()
    clf  = _train_rf(X, y)
    r    = feature_importance_report("tree", clf, X)
    assert "top5_features" in r


def test_report_unknown_type_raises():
    X, y = _make_data()
    with pytest.raises(ValueError):
        feature_importance_report("unknown", None, X)
