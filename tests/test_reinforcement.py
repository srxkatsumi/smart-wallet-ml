import numpy as np
import pytest
import models.reinforcement as rl_mod
from models.reinforcement import train, predict


def _make_data(n: int = 100, n_features: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)
    return X, y


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    monkeypatch.setattr(rl_mod, "DQN_EPOCHS",    2)
    monkeypatch.setattr(rl_mod, "DQN_HIDDEN",    8)
    monkeypatch.setattr(rl_mod, "DQN_BATCH",     16)
    monkeypatch.setattr(rl_mod, "PPO_EPISODES",  2)
    monkeypatch.setattr(rl_mod, "PPO_HIDDEN",    8)
    monkeypatch.setattr(rl_mod, "PPO_STEPS",     32)
    monkeypatch.setattr(rl_mod, "PPO_K_EPOCHS",  2)


def test_train_returns_dqn_and_ppo():
    X, y = _make_data()
    md = train(X, y)
    assert "dqn" in md
    assert "ppo" in md


def test_predict_bounds():
    X, y = _make_data()
    md = train(X, y)
    probs = predict(md, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_dqn_q_values_shape():
    """Q-network deve produzir 2 Q-values por estado (UP e DOWN)."""
    import torch
    X, y = _make_data()
    md = train(X, y)
    q_net  = md["dqn"]
    scaler = md["scaler"]
    X_sc   = scaler.transform(X).astype(np.float32)
    q_net.eval()
    with torch.no_grad():
        q_vals = q_net(torch.from_numpy(X_sc))
    assert q_vals.shape == (len(X), 2)


def test_ppo_policy_sums_to_one():
    """Política do actor deve somar 1 por amostra."""
    import torch
    X, y = _make_data()
    md = train(X, y)
    ac     = md["ppo"]
    scaler = md["scaler"]
    X_sc   = scaler.transform(X).astype(np.float32)
    ac.eval()
    with torch.no_grad():
        probs, _ = ac(torch.from_numpy(X_sc))
    np.testing.assert_allclose(probs.sum(dim=1).numpy(), 1.0, atol=1e-5)


def test_predict_different_length():
    X, y = _make_data(n=100)
    md = train(X, y)
    X_new = np.random.default_rng(7).standard_normal((30, 20))
    probs = predict(md, X_new)
    assert probs.shape == (30,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
