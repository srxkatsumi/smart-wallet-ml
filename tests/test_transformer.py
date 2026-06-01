import numpy as np
import pytest
import models.transformer as tf_mod
from models.transformer import train, predict


def _make_data(n: int = 150, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
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


def test_first_positions_use_mean():
    X, y = _make_data(n=100)
    md = train(X, y)
    probs = predict(md, X)
    seq_len = tf_mod.SEQ_LEN
    np.testing.assert_allclose(probs[:seq_len], float(y.mean()), atol=1e-6)


def test_predict_different_length():
    X, y = _make_data(n=150)
    md = train(X, y)
    X_new = np.random.default_rng(7).standard_normal((40, 20))
    probs = predict(md, X_new)
    assert probs.shape == (40,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_small_dataset():
    X, y = _make_data(n=tf_mod.SEQ_LEN + 5)
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
