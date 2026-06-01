import numpy as np
import pytest
from models.markov import train, predict, train_markov, predict_markov, train_hmm, predict_hmm

_RNG = np.random.default_rng(42)


def _make_data(n: int = 200, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


# ── Cadeia de Markov ──────────────────────────────────────────────────────

def test_markov_train_returns_matrix():
    X, y = _make_data()
    model = train_markov(X, y)
    T = model["transition_matrix"]
    assert T.shape == (2, 2)


def test_markov_rows_sum_to_one():
    X, y = _make_data()
    model = train_markov(X, y)
    T = model["transition_matrix"]
    np.testing.assert_allclose(T.sum(axis=1), [1.0, 1.0], atol=1e-9)


def test_markov_predict_bounds():
    X, y = _make_data()
    model = train_markov(X, y)
    probs = predict_markov(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── HMM ──────────────────────────────────────────────────────────────────

def test_hmm_train_runs():
    X, y = _make_data()
    model = train_hmm(X, y)
    assert model["model"] is not None
    assert model["bull_state"] in [0, 1]


def test_hmm_predict_bounds():
    X, y = _make_data()
    model = train_hmm(X, y)
    probs = predict_hmm(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_returns_both_models():
    X, y = _make_data()
    model_dict = train(X, y)
    assert "markov" in model_dict
    assert "hmm" in model_dict


def test_predict_bounds_unified():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_train_without_hmm():
    X, y = _make_data()
    model_dict = train(X, y, use_hmm=False)
    assert model_dict["hmm"] is None
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
