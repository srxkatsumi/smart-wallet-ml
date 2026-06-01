import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models.markov import train, predict, train_markov, predict_markov, train_hmm, predict_hmm
from config import N_BALLS, BALLS_PER_DRAW


def _make_data(n_draws: int = 200, seed: int = 42):
    """Simula (n_draws * N_BALLS) amostras com features e labels."""
    rng = np.random.default_rng(seed)
    n_features = 16  # len(FEATURE_COLS) na Mega Sena
    X = rng.standard_normal((n_draws * N_BALLS, n_features))
    # freq_5d (índice 0) entre 0 e 1
    X[:, 0] = rng.uniform(0, 0.5, n_draws * N_BALLS)
    # labels: exactamente 6 bolas por sorteio marcadas como 1
    y = np.zeros(n_draws * N_BALLS, dtype=float)
    for i in range(n_draws):
        chosen = rng.choice(N_BALLS, size=BALLS_PER_DRAW, replace=False)
        y[i * N_BALLS + chosen] = 1.0
    return X, y


# ── Cadeia de Markov ──────────────────────────────────────────────────────

def test_markov_train_returns_matrix():
    X, y = _make_data()
    model = train_markov(X, y)
    T = model["transition_matrix"]
    assert T.shape == (2, 2)


def test_markov_rows_sum_to_one():
    X, y = _make_data()
    T = train_markov(X, y)["transition_matrix"]
    np.testing.assert_allclose(T.sum(axis=1), [1.0, 1.0], atol=1e-9)


def test_markov_predict_bounds():
    X, y = _make_data()
    model = train_markov(X, y)
    probs = predict_markov(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_markov_baseline_near_random():
    """P(aparece) deve ser próximo de 6/60 = 0.10 — confirma comportamento aleatório."""
    X, y = _make_data(n_draws=500)
    model = train_markov(X, y)
    probs = predict_markov(model, X)
    assert abs(probs.mean() - BALLS_PER_DRAW / N_BALLS) < 0.15


# ── HMM ──────────────────────────────────────────────────────────────────

def test_hmm_train_runs():
    X, y = _make_data()
    model = train_hmm(X, y)
    assert model["model"] is not None
    assert model["active_state"] in [0, 1]


def test_hmm_predict_bounds():
    X, y = _make_data()
    model = train_hmm(X, y)
    probs = predict_hmm(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_predict_pipeline():
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
