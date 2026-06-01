import numpy as np
import pytest
import models.generative as gen_mod
from models.generative import train, predict


def _make_data(n: int = 120, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    monkeypatch.setattr(gen_mod, "EPOCHS_VAE", 2)
    monkeypatch.setattr(gen_mod, "EPOCHS_GAN", 2)
    monkeypatch.setattr(gen_mod, "HIDDEN",     8)
    monkeypatch.setattr(gen_mod, "LATENT_DIM", 4)
    monkeypatch.setattr(gen_mod, "NOISE_DIM",  8)


def test_train_returns_all_components():
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


def test_vae_latent_space():
    """VAE deve produzir representações latentes de dimensão correcta."""
    import torch
    X, y = _make_data()
    md = train(X, y)
    vae = md["vae"]
    scaler = md["scaler"]
    X_sc = scaler.transform(X).astype(np.float32)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(torch.from_numpy(X_sc))
    assert mu.shape == (len(X), gen_mod.LATENT_DIM)
    assert logvar.shape == (len(X), gen_mod.LATENT_DIM)


def test_gan_discriminator_output():
    """Discriminador deve produzir probabilidades em [0, 1]."""
    import torch
    X, y = _make_data()
    md = train(X, y)
    D = md["D"]
    scaler = md["scaler"]
    X_sc = scaler.transform(X).astype(np.float32)
    D.eval()
    with torch.no_grad():
        out = D(torch.from_numpy(X_sc)).squeeze().numpy()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
