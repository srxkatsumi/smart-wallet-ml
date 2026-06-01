import numpy as np
import pytest
from models.classical import (
    train, predict,
    train_xgb, predict_xgb,
    train_lgbm, predict_lgbm,
    train_cat, predict_cat,
    train_svm, predict_svm,
)
from sklearn.preprocessing import RobustScaler


def _make_data(n: int = 200, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


def _scaler(X):
    sc = RobustScaler()
    sc.fit(X)
    return sc


# ── Modelos individuais ───────────────────────────────────────────────────

@pytest.mark.parametrize("train_fn,predict_fn", [
    (train_xgb, predict_xgb),
    (train_lgbm, predict_lgbm),
    (train_cat, predict_cat),
    (train_svm, predict_svm),
])
def test_individual_bounds(train_fn, predict_fn):
    X, y = _make_data()
    sc = _scaler(X)
    model = train_fn(X, y, sc)
    probs = predict_fn(model, X, sc)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_returns_all_models():
    X, y = _make_data()
    model_dict = train(X, y)
    for key in ["scaler", "xgb", "lgbm", "cat", "svm"]:
        assert key in model_dict


def test_predict_bounds_unified():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_is_average_of_four():
    X, y = _make_data(n=100)
    model_dict = train(X, y)
    sc = model_dict["scaler"]
    p_xgb  = predict_xgb(model_dict["xgb"],  X, sc)
    p_lgbm = predict_lgbm(model_dict["lgbm"], X, sc)
    p_cat  = predict_cat(model_dict["cat"],  X, sc)
    p_svm  = predict_svm(model_dict["svm"],  X, sc)
    expected = (p_xgb + p_lgbm + p_cat + p_svm) / 4.0
    np.testing.assert_allclose(predict(model_dict, X), expected, atol=1e-9)


def test_small_dataset_does_not_crash():
    X, y = _make_data(n=30, n_features=20)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (30,)
