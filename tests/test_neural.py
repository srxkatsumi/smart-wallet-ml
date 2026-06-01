import numpy as np
import pytest
import models.neural as neural_mod
from models.neural import train, predict


def _make_data(n: int = 150, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    monkeypatch.setattr(neural_mod, "EPOCHS", 2)
    monkeypatch.setattr(neural_mod, "HIDDEN", 8)


def test_train_returns_lstm_and_gru():
    X, y = _make_data()
    model_dict = train(X, y)
    assert "lstm" in model_dict
    assert "gru"  in model_dict
    assert model_dict["lstm"] is not None
    assert model_dict["gru"]  is not None


def test_predict_bounds():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_first_positions_use_mean():
    X, y = _make_data(n=100)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    seq_len = neural_mod.SEQ_LEN
    expected_mean = float(y.mean())
    np.testing.assert_allclose(probs[:seq_len], expected_mean, atol=1e-6)


def test_small_dataset_does_not_crash():
    X, y = _make_data(n=neural_mod.SEQ_LEN + 5)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)


def test_predict_different_length():
    X, y = _make_data(n=150)
    model_dict = train(X, y)
    X_new = np.random.default_rng(99).standard_normal((50, 20))
    probs = predict(model_dict, X_new)
    assert probs.shape == (50,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
