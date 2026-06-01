import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models.timeseries import train, predict, train_arima, predict_arima
from config import N_BALLS, BALLS_PER_DRAW


def _make_series(n_draws: int = 60, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_draws * N_BALLS, 16))
    y = np.zeros(n_draws * N_BALLS, dtype=float)
    for i in range(n_draws):
        chosen = rng.choice(N_BALLS, size=BALLS_PER_DRAW, replace=False)
        y[i * N_BALLS + chosen] = 1.0
    return X, y


def test_train_returns_all_keys():
    X, y = _make_series()
    model_dict = train(X, y)
    for key in ["arima", "sarima", "ets", "holtwinters", "prophet"]:
        assert key in model_dict


def test_predict_bounds():
    X, y = _make_series()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_arima_individual():
    _, y = _make_series()
    model = train_arima(y)
    probs = predict_arima(model, len(y))
    assert probs.shape == (len(y),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_baseline_near_random():
    """Num processo aleatório, média das probabilidades deve ser ≈ 6/60 = 0.10."""
    X, y = _make_series(n_draws=100)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert abs(probs.mean() - BALLS_PER_DRAW / N_BALLS) < 0.15
