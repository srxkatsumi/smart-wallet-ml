import numpy as np
import pytest
from models.timeseries import (
    train, predict,
    train_arima, predict_arima,
    train_sarima, predict_sarima,
    train_ets, predict_ets,
    train_holtwinters, predict_holtwinters,
    train_prophet, predict_prophet,
)


def _make_series(n: int = 80, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


# ── Modelos individuais ───────────────────────────────────────────────────

@pytest.mark.parametrize("train_fn,predict_fn", [
    (train_arima,      predict_arima),
    (train_sarima,     predict_sarima),
    (train_ets,        predict_ets),
    (train_holtwinters, predict_holtwinters),
    (train_prophet,    predict_prophet),
])
def test_individual_bounds(train_fn, predict_fn):
    _, y = _make_series()
    model = train_fn(y)
    probs = predict_fn(model, len(y))
    assert probs.shape == (len(y),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_pad_shorter_output():
    _, y = _make_series(n=80)
    model = train_arima(y)
    probs = predict_arima(model, 100)   # pede mais do que treinou
    assert probs.shape == (100,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_returns_all_models():
    X, y = _make_series()
    model_dict = train(X, y)
    for key in ["arima", "sarima", "ets", "holtwinters", "prophet"]:
        assert key in model_dict


def test_predict_bounds_unified():
    X, y = _make_series()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_different_length():
    X, y = _make_series(n=80)
    model_dict = train(X, y)
    X_new = np.random.default_rng(99).standard_normal((10, 20))
    probs = predict(model_dict, X_new)
    assert probs.shape == (10,)
