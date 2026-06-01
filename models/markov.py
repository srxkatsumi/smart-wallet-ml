"""
Família: Estado Oculto
Modelos: Cadeia de Markov + HMM (Hidden Markov Model)

Cadeia de Markov:
  Estima a matriz de transição entre estados UP/DOWN a partir da sequência
  histórica de labels. Para cada observação, usa o sinal de ret_1d como
  proxy do estado actual e retorna P(UP | estado_actual).

HMM (hmmlearn.GaussianHMM):
  Descobre estados ocultos (ex: bull/bear) a partir das features sem usar
  os labels no treino. O estado "bull" é identificado como aquele com maior
  média em ret_1d. Retorna a probabilidade posterior de estar no estado bull.

Interface (compatível com ensemble.py):
  train(X, y)  -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import numpy as np
from hmmlearn import hmm

# índice de ret_1d em FEATURE_COLS (posição 10)
_RET_1D_IDX = 10


# ── Cadeia de Markov ──────────────────────────────────────────────────────

def train_markov(X: np.ndarray, y: np.ndarray) -> dict:
    n_states = 2
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(y) - 1):
        s_now  = int(y[t])
        s_next = int(y[t + 1])
        counts[s_now, s_next] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    T = np.where(row_sums > 0, counts / safe_sums, 0.5)
    return {"type": "markov", "transition_matrix": T}


def predict_markov(model: dict, X: np.ndarray) -> np.ndarray:
    T = model["transition_matrix"]
    current_states = (X[:, _RET_1D_IDX] >= 0).astype(int)
    return T[current_states, 1]


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

    # identifica qual estado é "bull" (maior média em ret_1d)
    bull_state = int(np.argmax(model.means_[:, _RET_1D_IDX]))
    return {"type": "hmm", "model": model, "bull_state": bull_state}


def predict_hmm(model_dict: dict, X: np.ndarray) -> np.ndarray:
    model      = model_dict["model"]
    bull_state = model_dict["bull_state"]
    log_proba  = model.predict_proba(X)         # shape (N, n_components)
    return log_proba[:, bull_state]


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
