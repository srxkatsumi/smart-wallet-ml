"""
Família: Estado Oculto
Modelos: Cadeia de Markov + HMM (Hidden Markov Model)
Projecto: Mega Sena ML Experiment

Cadeia de Markov:
  Para cada uma das 60 bolas, estima:
    P(bola aparece | apareceu no sorteio anterior)
    P(bola aparece | NÃO apareceu no sorteio anterior)
  Usa o sorteio anterior (coluna "prev_appeared" implícita em X) para
  determinar o estado actual de cada bola.

HMM (hmmlearn.GaussianHMM):
  Descobre estados ocultos nos padrões dos sorteios a partir das features
  de cada bola. O estado "activo" é identificado como aquele com maior
  frequência média (feature freq_5d, índice 0). Retorna a probabilidade
  posterior do estado activo para cada bola.

Interface (compatível com ensemble.py da Mega Sena):
  train(X, y)  -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1], shape (60,)

Nota académica:
  Numa loteria verdadeiramente aleatória, a cadeia de Markov deve convergir
  para P(aparece) ≈ 6/60 = 0.10 independentemente do estado anterior.
  Qualquer desvio persistente é artefacto de amostragem, não padrão real.
"""

import numpy as np
from hmmlearn import hmm

# índice de freq_5d em FEATURE_COLS (posição 0)
_FREQ_5D_IDX = 0


# ── Cadeia de Markov ──────────────────────────────────────────────────────

def train_markov(X: np.ndarray, y: np.ndarray) -> dict:
    """
    X: (N_samples, N_features) — uma linha por bola por sorteio
    y: (N_samples,) — 1 se a bola foi sorteada, 0 se não foi

    Estima, para cada bola (60 ao todo), as probabilidades de transição
    agrupando por estado anterior (apareceu / não apareceu).
    Como X não inclui o estado anterior directamente, usamos freq_5d
    (frequência nos últimos 5 sorteios) > 0 como proxy de "apareceu recentemente".
    """
    appeared = (X[:, _FREQ_5D_IDX] > 0).astype(int)   # proxy estado anterior
    counts = np.zeros((2, 2), dtype=float)
    for s, label in zip(appeared, y):
        counts[int(s), int(label)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    T = np.where(row_sums > 0, counts / safe_sums, 0.5)
    return {"type": "markov", "transition_matrix": T}


def predict_markov(model: dict, X: np.ndarray) -> np.ndarray:
    T = model["transition_matrix"]
    appeared = (X[:, _FREQ_5D_IDX] > 0).astype(int)
    return T[appeared, 1]


# ── HMM ──────────────────────────────────────────────────────────────────

def train_hmm(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> dict:
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X)
    # estado "activo" = maior frequência média (freq_5d)
    active_state = int(np.argmax(model.means_[:, _FREQ_5D_IDX]))
    return {"type": "hmm", "model": model, "active_state": active_state}


def predict_hmm(model_dict: dict, X: np.ndarray) -> np.ndarray:
    model        = model_dict["model"]
    active_state = model_dict["active_state"]
    log_proba    = model.predict_proba(X)
    return log_proba[:, active_state]


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, use_hmm: bool = True) -> dict:
    markov = train_markov(X, y)
    if use_hmm:
        try:
            hidden = train_hmm(X, y)
        except Exception:
            hidden = None
    else:
        hidden = None
    return {"markov": markov, "hmm": hidden}


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    p_markov = predict_markov(model_dict["markov"], X)
    if model_dict["hmm"] is not None:
        p_hmm = predict_hmm(model_dict["hmm"], X)
        return (p_markov + p_hmm) / 2.0
    return p_markov
