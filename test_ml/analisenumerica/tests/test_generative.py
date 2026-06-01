import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import models.generative as gen_mod
from models.generative import train, predict
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
    monkeypatch.setattr(gen_mod, "EPOCHS_VAE", 2)
    monkeypatch.setattr(gen_mod, "EPOCHS_GAN", 2)
    monkeypatch.setattr(gen_mod, "HIDDEN",     8)
    monkeypatch.setattr(gen_mod, "LATENT_DIM", 4)
    monkeypatch.setattr(gen_mod, "NOISE_DIM",  8)


def test_train_returns_components():
    X, y = _make_data()
    md = train(X, y)
    for key in ["scaler", "vae", "clf", "D"]:
        assert key in md


def test_predict_bounds():
    X, y = _make_data()
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_vae_latent_shape():
    import torch
    X, y = _make_data()
    md = train(X, y)
    vae    = md["vae"]
    scaler = md["scaler"]
    X_sc   = scaler.transform(X).astype(np.float32)
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(torch.from_numpy(X_sc))
    assert mu.shape == (len(X), gen_mod.LATENT_DIM)
