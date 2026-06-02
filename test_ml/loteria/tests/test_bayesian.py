import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import models.bayesian as bay_mod
from models.bayesian import train, predict
from config import N_BALLS, BALLS_PER_DRAW


def _make_data(n_draws: int = 20, seed: int = 42):
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
    monkeypatch.setattr(bay_mod, "EPOCHS",         2)
    monkeypatch.setattr(bay_mod, "HIDDEN",         8)
    monkeypatch.setattr(bay_mod, "MC_SAMPLES",     3)
    monkeypatch.setattr(bay_mod, "GP_MAX_SAMPLES", 100)


def test_train_returns_models():
    X, y = _make_data()
    md = train(X, y)
    assert "gp"  in md
    assert "bnn" in md


def test_predict_bounds():
    X, y = _make_data()
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_gp_handles_class_imbalance():
    """GP deve funcionar mesmo com ~10% de positivos (6/60)."""
    X, y = _make_data(n_draws=30)
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
