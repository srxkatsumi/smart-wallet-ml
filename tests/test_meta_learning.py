import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from evaluation.meta_learning import (
    train_stacking, predict_stacking,
    optimize_rf, optimize_xgb, optimize_lgbm,
)


def _make_data(n: int = 150, n_features: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, n_features))
    y   = rng.integers(0, 2, n).astype(float)
    return X, y


def _make_predict_fn(seed: int):
    """Retorna uma predict_fn que treina RF internamente (para stacking)."""
    def fn(X_tr, y_tr, X_val):
        sc  = RobustScaler()
        clf = RandomForestClassifier(n_estimators=10, random_state=seed)
        clf.fit(sc.fit_transform(X_tr), y_tr.astype(int))
        return clf.predict_proba(sc.transform(X_val))[:, 1]
    return fn


def _make_inference_fn(X_tr, y_tr, seed: int):
    """Retorna uma predict_fn(X) para predict_stacking."""
    sc  = RobustScaler()
    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    clf.fit(sc.fit_transform(X_tr), y_tr.astype(int))
    def fn(X):
        return clf.predict_proba(sc.transform(X))[:, 1]
    return fn


# ── Stacking ──────────────────────────────────────────────────────────────

def test_stacking_returns_keys():
    X, y   = _make_data()
    fns    = [_make_predict_fn(s) for s in [1, 2, 3]]
    meta   = train_stacking(fns, X, y, n_splits=3)
    for k in ["meta_model", "scaler", "n_base_models"]:
        assert k in meta


def test_stacking_n_base_models():
    X, y = _make_data()
    fns  = [_make_predict_fn(s) for s in [1, 2]]
    meta = train_stacking(fns, X, y, n_splits=3)
    assert meta["n_base_models"] == 2


def test_stacking_predict_bounds():
    X, y   = _make_data()
    fns_tr = [_make_predict_fn(s) for s in [1, 2, 3]]
    meta   = train_stacking(fns_tr, X, y, n_splits=3)
    fns_inf = [_make_inference_fn(X, y, s) for s in [1, 2, 3]]
    probs   = predict_stacking(meta, fns_inf, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Optuna ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("optimize_fn", [optimize_rf, optimize_xgb, optimize_lgbm])
def test_optuna_returns_best_params(optimize_fn):
    X, y = _make_data(n=80)
    r    = optimize_fn(X, y, n_trials=3)
    assert "best_params"   in r
    assert "best_accuracy" in r
    assert 0.0 <= r["best_accuracy"] <= 1.0
    assert r["n_trials"] == 3
