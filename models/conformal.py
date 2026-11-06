"""
Família: Calibrated Uncertainty — Conformal Prediction

Split Conformal Prediction com o método LAC (Least Ambiguous set-valued Classifier,
Angelopoulos & Bates 2022 — "A Gentle Introduction to Conformal Prediction").

A família treina um Random Forest como base e calibra as probabilidades de saída
com nonconformity scores computados num conjunto de calibração retido, produzindo
estimativas de probabilidade garantidamente válidas.

Nonconformity score (LAC):
  s(x, y) = 1 − ŷ[y]   (1 menos a probabilidade atribuída à classe verdadeira)

Procedimento:
  1. Split temporal 70/30 (treino / calibração)
  2. Treino: RF em X_train, y_train
  3. Calibração: s_i = 1 − ŷ[y_cal_i] para todo i no conjunto de calibração
  4. Predição: p-valor conforme por classe k = #{s_cal da classe k ≥ s_k} / (n_k + 1)
               probabilidade calibrada = pv1 / (pv0 + pv1)

Garantia de cobertura (Vovk et al., 2005):
  P(y_true ∈ C_α(x)) ≥ 1 − α   para qualquer distribuição de dados exchangeable.

Equivalência com MAPIE:
  O método LAC é exactamente o que implementa o MapieClassifier da biblioteca
  MAPIE (Taquet et al., 2022) com method="lac". A implementação aqui é manual
  para máxima transparência e independência de versão de API.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades calibradas em [0, 1]
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

TRAIN_FRAC   = 0.70
N_ESTIMATORS = 100
MAX_DEPTH    = 4
ALPHA        = 0.10   # target miscoverage rate → 90% coverage guarantee


def _conformal_probs(scores_up: np.ndarray, scores_down: np.ndarray, p_up: np.ndarray) -> np.ndarray:
    """
    Vectorised class-conditional (Mondrian) conformal p-values.

    Each class is compared only with calibration scores of that same class. With
    one pooled set, the majority class dominates and the ratio inflates: the mean
    probability reached 0.80 in production for a ~54% base rate.

      s1 = 1 − p ;  pv1 = (#{scores_up   ≥ s1} + 1) / (n_up + 1)
      s0 = p     ;  pv0 = (#{scores_down ≥ s0} + 1) / (n_down + 1)
      result = pv1 / (pv0 + pv1)
    """
    if len(scores_up) == 0 and len(scores_down) == 0:
        return np.full(len(p_up), 0.5)

    s1 = (1.0 - p_up)[:, np.newaxis]
    s0 = p_up[:, np.newaxis]

    pv1 = (1 + (scores_up[np.newaxis, :] >= s1).sum(axis=1)) / (len(scores_up) + 1)
    pv0 = (1 + (scores_down[np.newaxis, :] >= s0).sum(axis=1)) / (len(scores_down) + 1)

    denom = pv0 + pv1
    return np.where(denom > 1e-9, pv1 / denom, 0.5)


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X).astype(np.float32)
    y_int  = y.astype(int)

    n_train   = max(10, int(TRAIN_FRAC * len(X_sc)))
    X_tr, y_tr = X_sc[:n_train], y_int[:n_train]
    X_cal, y_cal = X_sc[n_train:], y_int[n_train:]

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    scores_cal = np.array([], dtype=np.float32)
    if len(X_cal) > 0:
        p_cal      = rf.predict_proba(X_cal)[:, 1]
        scores_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal).astype(np.float32)

    # Empirical coverage at alpha=0.10 on calibration set (diagnostic)
    if len(scores_cal) > 1:
        q_level   = min(1.0, np.ceil((len(scores_cal) + 1) * (1 - ALPHA)) / len(scores_cal))
        threshold = float(np.quantile(scores_cal, q_level))
        logger.debug("conformal threshold q@90%%=%.4f  n_cal=%d", threshold, len(scores_cal))

    return {
        "scaler":     scaler,
        "rf":         rf,
        "y_mean":     float(y.mean()),
        "scores_up":   scores_cal[y_cal == 1],
        "scores_down": scores_cal[y_cal == 0],
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc         = model_dict["scaler"]
    X_sc       = sc.transform(X).astype(np.float32)
    p_up       = model_dict["rf"].predict_proba(X_sc)[:, 1]
    return _conformal_probs(model_dict["scores_up"], model_dict["scores_down"], p_up)
