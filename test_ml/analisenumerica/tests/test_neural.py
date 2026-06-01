import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import models.neural as neural_mod
from models.neural import train, predict
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
    monkeypatch.setattr(neural_mod, "EPOCHS", 2)
    monkeypatch.setattr(neural_mod, "HIDDEN", 8)


def test_train_returns_models():
    X, y = _make_data()
    model_dict = train(X, y)
    assert model_dict["lstm"] is not None
    assert model_dict["gru"]  is not None


def test_predict_bounds():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_output_valid_probabilities():
    """Saída deve ser sempre probabilidades válidas em [0, 1]."""
    X, y = _make_data(n_draws=60)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
