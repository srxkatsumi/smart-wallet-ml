"""
Família: Reinforcement Learning
Modelos: DQN (Deep Q-Network), PPO (Proximal Policy Optimization)
Projecto: Carteira Inteligente (não aplicável à Mega Sena)

DQN (Mnih et al., 2015 — "Human-level control through deep reinforcement
learning", Nature):
  Rede neural que aprende Q(s, a) — o valor esperado de tomar a acção a
  no estado s. O estado é o vector de features X[t]; as acções são {0=DOWN,
  1=UP}; a recompensa é +1 para acção correcta, -1 para incorrecta.
  Usa experience replay e target network para estabilidade do treino.

PPO (Schulman et al., 2017 — "Proximal Policy Optimization Algorithms",
arXiv 1707.06347):
  Método Actor-Critic com objectivo surrogate clipped. O actor aprende
  π(a|s) — a política; o crítico aprende V(s) — a função de valor.
  O clipping ε evita actualizações de política demasiado grandes,
  tornando o treino mais estável que TRPO com menor complexidade.

Diferença conceptual face às outras famílias:
  Os modelos supervisionados minimizam uma loss pontual sobre (x, y).
  Os modelos de RL optimizam a recompensa acumulada ao longo de uma
  sequência — tratam a previsão como uma política de decisão sequencial.
  Esta distinção é o argumento central do Capítulo 8 da tese.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades P(UP) em [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from sklearn.preprocessing import RobustScaler

# ── Hiperparâmetros ───────────────────────────────────────────────────────
DQN_HIDDEN       = 64
DQN_EPOCHS       = 30
DQN_LR           = 1e-3
DQN_BATCH        = 64
DQN_BUFFER_SIZE  = 2000
DQN_TARGET_UPD   = 20    # steps entre actualizações da target network
DQN_GAMMA        = 0.95

PPO_HIDDEN       = 64
PPO_LR           = 3e-4
PPO_EPISODES     = 20
PPO_STEPS        = 128   # steps colectados por episódio
PPO_K_EPOCHS     = 4     # optimizações por episódio
PPO_CLIP         = 0.2
PPO_GAMMA        = 0.99
PPO_GAE_LAMBDA   = 0.95
PPO_ENT_COEF     = 0.01
PPO_VF_COEF      = 0.5


# ── DQN ───────────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    def __init__(self, state_size: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def _train_dqn(X: np.ndarray, y: np.ndarray) -> _QNetwork:
    n = len(X)
    q_net    = _QNetwork(X.shape[1], DQN_HIDDEN)
    q_target = _QNetwork(X.shape[1], DQN_HIDDEN)
    q_target.load_state_dict(q_net.state_dict())
    opt     = torch.optim.Adam(q_net.parameters(), lr=DQN_LR)
    loss_fn = nn.MSELoss()
    buffer  = deque(maxlen=DQN_BUFFER_SIZE)
    step    = 0

    for _ in range(DQN_EPOCHS):
        for t in range(n - 1):
            s  = torch.from_numpy(X[t].astype(np.float32))
            a  = int(y[t])
            r  = 1.0 if a == int(round(q_net(s.unsqueeze(0)).argmax().item())) else -1.0
            s2 = torch.from_numpy(X[t + 1].astype(np.float32))
            buffer.append((s, a, r, s2))
            step += 1

            if len(buffer) < DQN_BATCH:
                continue

            idx     = np.random.choice(len(buffer), DQN_BATCH, replace=False)
            batch   = [buffer[i] for i in idx]
            sb      = torch.stack([b[0] for b in batch])
            ab      = torch.tensor([b[1] for b in batch])
            rb      = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            sb2     = torch.stack([b[3] for b in batch])

            with torch.no_grad():
                q_next = q_target(sb2).max(1).values
            q_target_val = rb + DQN_GAMMA * q_next

            q_pred = q_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            loss   = loss_fn(q_pred, q_target_val)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % DQN_TARGET_UPD == 0:
                q_target.load_state_dict(q_net.state_dict())

    return q_net


def _predict_dqn(q_net: _QNetwork, X: np.ndarray) -> np.ndarray:
    q_net.eval()
    with torch.no_grad():
        q_vals = q_net(torch.from_numpy(X.astype(np.float32)))
    return torch.softmax(q_vals, dim=1)[:, 1].numpy()


# ── PPO ───────────────────────────────────────────────────────────────────

class _ActorCritic(nn.Module):
    def __init__(self, state_size: int, hidden: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 2),          nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def act(self, x):
        probs = self.actor(x)
        dist  = torch.distributions.Categorical(probs)
        a     = dist.sample()
        return a, dist.log_prob(a), dist.entropy()


def _train_ppo(X: np.ndarray, y: np.ndarray) -> _ActorCritic:
    n    = len(X)
    ac   = _ActorCritic(X.shape[1], PPO_HIDDEN)
    opt  = torch.optim.Adam(ac.parameters(), lr=PPO_LR)
    X_t  = torch.from_numpy(X.astype(np.float32))
    y_t  = torch.from_numpy(y.astype(np.float32))

    for _ in range(PPO_EPISODES):
        # colectar trajectória
        states, actions, log_probs_old, rewards, values = [], [], [], [], []
        t0 = np.random.randint(0, max(1, n - PPO_STEPS))
        for t in range(t0, min(t0 + PPO_STEPS, n)):
            s   = X_t[t].unsqueeze(0)
            a, lp, _ = ac.act(s)
            v   = ac.critic(s).squeeze()
            r   = torch.tensor(1.0 if a.item() == int(y_t[t].item()) else -1.0)
            states.append(s.squeeze(0))
            actions.append(a)
            log_probs_old.append(lp.detach())
            rewards.append(r)
            values.append(v.detach())

        # GAE advantages
        returns, adv = [], []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            nxt = values[i + 1].item() if i + 1 < len(values) else 0.0
            delta = rewards[i].item() + PPO_GAMMA * nxt - values[i].item()
            gae   = delta + PPO_GAMMA * PPO_GAE_LAMBDA * gae
            adv.insert(0, gae)
            returns.insert(0, gae + values[i].item())

        states_t   = torch.stack(states)
        actions_t  = torch.stack(actions)
        lp_old_t   = torch.stack(log_probs_old)
        returns_t  = torch.tensor(returns, dtype=torch.float32)
        adv_t      = torch.tensor(adv, dtype=torch.float32)
        adv_t      = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # optimizar K epochs
        for _ in range(PPO_K_EPOCHS):
            probs, vals = ac(states_t)
            dist        = torch.distributions.Categorical(probs)
            lp_new      = dist.log_prob(actions_t)
            entropy     = dist.entropy().mean()
            ratio       = torch.exp(lp_new - lp_old_t)
            surr1       = ratio * adv_t
            surr2       = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(vals.squeeze(), returns_t)
            loss        = actor_loss + PPO_VF_COEF * critic_loss - PPO_ENT_COEF * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()

    return ac


def _predict_ppo(ac: _ActorCritic, X: np.ndarray) -> np.ndarray:
    ac.eval()
    with torch.no_grad():
        probs, _ = ac(torch.from_numpy(X.astype(np.float32)))
    return probs[:, 1].numpy()


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    X_sc = scaler.transform(X)
    return {
        "scaler": scaler,
        "dqn":   _train_dqn(X_sc, y),
        "ppo":   _train_ppo(X_sc, y),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    X_sc  = model_dict["scaler"].transform(X)
    p_dqn = _predict_dqn(model_dict["dqn"], X_sc)
    p_ppo = _predict_ppo(model_dict["ppo"], X_sc)
    return (p_dqn + p_ppo) / 2.0
