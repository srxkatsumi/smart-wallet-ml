import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import models.transformer as tf_mod
from models.transformer import train, predict
from config import N_BALLS, BALLS_PER_DRAW


def _make_data(n_draws: int = 40, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_features = 16
    X = rng.standard_normal((n_draws * N_BALLS, n_features))
    y = np.zeros(n_draws * N_BALLS, dtype=float)
    for i in range(n_draws):
        chosen = rng.choice(N_BALLS, size=BALLS_PER_DRAW, replace=False)
        y[i * N_BALLS + chosen] = 1.0
    return X, y


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    monkeypatch.setattr(tf_mod, "EPOCHS",  2)
    monkeypatch.setattr(tf_mod, "D_MODEL", 8)
    monkeypatch.setattr(tf_mod, "HIDDEN",  8)
    monkeypatch.setattr(tf_mod, "NHEAD",   2)


def test_train_returns_all_models():
    X, y = _make_data()
    md = train(X, y)
    for key in ["transformer", "tft", "nbeats"]:
        assert key in md
        assert md[key] is not None


def test_predict_bounds():
    X, y = _make_data()
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_different_length():
    X, y = _make_data()
    md = train(X, y)
    X_new = np.random.default_rng(9).standard_normal((30, 16))
    probs = predict(md, X_new)
    assert probs.shape == (30,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
