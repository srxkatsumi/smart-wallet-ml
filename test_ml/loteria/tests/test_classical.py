import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models.classical import train, predict, train_xgb, predict_xgb
from sklearn.preprocessing import RobustScaler
from config import N_BALLS, BALLS_PER_DRAW


def _make_data(n_draws: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_features = 16
    X = rng.standard_normal((n_draws * N_BALLS, n_features))
    X[:, 0] = rng.uniform(0, 0.5, n_draws * N_BALLS)
    y = np.zeros(n_draws * N_BALLS, dtype=float)
    for i in range(n_draws):
        chosen = rng.choice(N_BALLS, size=BALLS_PER_DRAW, replace=False)
        y[i * N_BALLS + chosen] = 1.0
    return X, y


def _scaler(X):
    sc = RobustScaler()
    sc.fit(X)
    return sc


def test_train_returns_all_models():
    X, y = _make_data()
    model_dict = train(X, y)
    for key in ["scaler", "xgb", "lgbm", "cat", "svm"]:
        assert key in model_dict


def test_predict_bounds():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_xgb_individual():
    X, y = _make_data()
    sc = _scaler(X)
    model = train_xgb(X, y, sc)
    probs = predict_xgb(model, X, sc)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_baseline_near_random():
    """Num processo aleatório, média das probabilidades deve ser ≈ 6/60 = 0.10."""
    X, y = _make_data(n_draws=200)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert abs(probs.mean() - BALLS_PER_DRAW / N_BALLS) < 0.15
