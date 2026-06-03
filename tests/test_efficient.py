import numpy as np
import pytest
from models.efficient import train, predict, _TCN, _DLinear, _NLinear, _PatchTST
import torch

SEQ_LEN = 20


def _make_data(n: int = 120, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


# ── Interface unificada ───────────────────────────────────────────────────

def test_train_returns_all_keys():
    X, y = _make_data()
    m = train(X, y)
    for key in ["scaler", "y_mean", "tcn", "dlinear", "nlinear", "patchtst"]:
        assert key in m


def test_predict_bounds():
    X, y = _make_data()
    m = train(X, y)
    probs = predict(m, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_shape_matches_input():
    X, y = _make_data(n=150)
    m = train(X, y)
    probs = predict(m, X)
    assert probs.shape == (150,)


# ── Módulos individuais (forward pass) ───────────────────────────────────

def _batch(n_feat: int, batch: int = 8):
    return torch.randn(batch, SEQ_LEN, n_feat)


def test_tcn_output_shape():
    import torch
    model = _TCN(n_feat=10, channels=32, kernel=3, dilations=[1, 2, 4, 8])
    out = model(_batch(10))
    assert out.shape == (8, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_dlinear_output_shape():
    model = _DLinear(seq_len=SEQ_LEN, n_feat=10)
    out = model(_batch(10))
    assert out.shape == (8, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_nlinear_output_shape():
    model = _NLinear(seq_len=SEQ_LEN, n_feat=10)
    out = model(_batch(10))
    assert out.shape == (8, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_patchtst_output_shape():
    model = _PatchTST(seq_len=SEQ_LEN, n_feat=10,
                      patch_size=4, stride=2, d_model=32, nhead=4)
    out = model(_batch(10))
    assert out.shape == (8, 1)
    assert (out >= 0).all() and (out <= 1).all()


# ── Dataset pequeno não crasha ────────────────────────────────────────────

def test_small_dataset():
    X, y = _make_data(n=60)
    m = train(X, y)
    probs = predict(m, X)
    assert probs.shape == (60,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
