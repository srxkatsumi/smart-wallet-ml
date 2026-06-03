import numpy as np
import pytest
from models.contrarian import (
    train, predict,
    train_cb, predict_cb,
    train_ewi, predict_ewi,
    train_pel, predict_pel,
)


def _make_data(n: int = 100, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


# ── Modelos individuais ───────────────────────────────────────────────────

@pytest.mark.parametrize("train_fn,predict_fn", [
    (train_cb,  predict_cb),
    (train_ewi, predict_ewi),
    (train_pel, predict_pel),
])
def test_individual_bounds(train_fn, predict_fn):
    X, y = _make_data()
    model = train_fn(X, y)
    probs = predict_fn(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_returns_all_models():
    X, y = _make_data()
    model_dict = train(X, y)
    for key in ["cb", "ewi", "pel"]:
        assert key in model_dict


def test_predict_bounds_unified():
    X, y = _make_data()
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ── CB inverte as probabilidades ──────────────────────────────────────────

def test_cb_is_inverse_of_base():
    X, y = _make_data()
    model_dict = train(X, y)
    p_cb  = predict_cb(model_dict["cb"], X)
    p_base = model_dict["cb"]["base"].predict_proba(
        model_dict["cb"]["scaler"].transform(X)
    )[:, 1]
    np.testing.assert_allclose(p_cb, 1.0 - p_base, atol=1e-6)


# ── EWI expõe estado de inversão ──────────────────────────────────────────

def test_ewi_has_inverted_flag():
    X, y = _make_data()
    model = train_ewi(X, y)
    assert "inverted" in model
    assert isinstance(model["inverted"], bool)


# ── Dados pequenos não crasham ────────────────────────────────────────────

def test_small_dataset_does_not_crash():
    X, y = _make_data(n=60)
    model_dict = train(X, y)
    probs = predict(model_dict, X)
    assert probs.shape == (len(X),)
