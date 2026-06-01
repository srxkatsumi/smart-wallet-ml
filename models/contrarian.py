"""
Família: Contrarian / Error Learning
Modelos: Contrarian Baseline (CB), Error-Weighted Inverter (EWI),
         Predictive Error Learning (PEL)

Esta família é a única completamente desenvolvida para esta tese.
Enquanto as 7 famílias anteriores aprendem a prever NA MESMA DIRECÇÃO
que o sinal histórico, estes 3 modelos aprendem a partir dos ERROS
do modelo base — uma abordagem oposta e complementar.

Contrarian Baseline (CB):
  Sempre inverte a previsão do modelo base. Se o ensemble prevê 70% UP,
  CB prevê 30% UP. Academicamente: se CB superar o ensemble, o ensemble
  é pior que inútil (pior que acaso ajustado). Funciona como um "null
  hypothesis test" sobre a qualidade do ensemble principal.

Error-Weighted Inverter (EWI):
  Monitora a taxa de erro recente do modelo base numa janela deslizante.
  Quando a taxa de erro excede um threshold (default 60%), inverte a
  previsão. Quando cai abaixo de 40%, volta ao normal. Baseado no princípio
  de anti-persistência: se um modelo erra sistematicamente, a inversão
  temporária pode recuperar valor.

Predictive Error Learning (PEL):
  Testa se a sequência de erros do modelo base tem autocorrelação
  (Ljung-Box, 1978). Se sim, ajusta um modelo AR(p) sobre os erros.
  A previsão final é: pred_base + AR_forecast(erros_recentes).
  Contribuição original: aplica o conceito de autocorrelação de resíduos
  (geralmente usado apenas para diagnóstico) como mecanismo de correcção
  activa das previsões.

  Referência para autocorrelação de erros:
  Box, G.E.P. & Jenkins, G.M. (1976) — "Time Series Analysis: Forecasting
  and Control". Holden-Day.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ── Parâmetros ────────────────────────────────────────────────────────────
EWI_WINDOW     = 20     # janela de erros recentes para EWI
EWI_INVERT_TH  = 0.60   # threshold para activar inversão
EWI_RESTORE_TH = 0.40   # threshold para desactivar inversão
PEL_MAX_LAG    = 10     # lags máximos para Ljung-Box
PEL_AR_ORDER   = 3      # ordem do modelo AR para PEL
N_TREES        = 100


def _train_base(X: np.ndarray, y: np.ndarray,
                scaler: RobustScaler) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(scaler.transform(X), y.astype(int))
    return clf


def _oof_errors(X: np.ndarray, y: np.ndarray,
                scaler: RobustScaler) -> np.ndarray:
    """Out-of-fold errors para evitar lookahead na estimativa de erro."""
    errors = np.zeros(len(X))
    tscv   = TimeSeriesSplit(n_splits=5)
    for tr_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
        )
        clf.fit(scaler.transform(X[tr_idx]), y[tr_idx].astype(int))
        p = clf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        errors[val_idx] = y[val_idx].astype(float) - p
    return errors


# ── Contrarian Baseline ───────────────────────────────────────────────────

def train_cb(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    base = _train_base(X, y, scaler)
    return {"type": "cb", "base": base, "scaler": scaler}


def predict_cb(model: dict, X: np.ndarray) -> np.ndarray:
    p = model["base"].predict_proba(model["scaler"].transform(X))[:, 1]
    return 1.0 - p


# ── Error-Weighted Inverter ───────────────────────────────────────────────

def train_ewi(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    base   = _train_base(X, y, scaler)
    errors = _oof_errors(X, y, scaler)

    # Calcula taxa de erro deslizante nos últimos EWI_WINDOW passos
    wrong    = (errors != 0).astype(float)
    n        = len(wrong)
    err_rate = np.array([
        wrong[max(0, i - EWI_WINDOW):i].mean() if i > 0 else 0.5
        for i in range(n)
    ])
    # Estado final: inverted se última taxa > threshold
    last_rate = err_rate[-EWI_WINDOW:].mean() if n >= EWI_WINDOW else 0.5
    inverted  = bool(last_rate > EWI_INVERT_TH)

    return {
        "type":      "ewi",
        "base":      base,
        "scaler":    scaler,
        "inverted":  inverted,
        "last_rate": float(last_rate),
    }


def predict_ewi(model: dict, X: np.ndarray) -> np.ndarray:
    p = model["base"].predict_proba(model["scaler"].transform(X))[:, 1]
    if model["inverted"]:
        return 1.0 - p
    return p


# ── Predictive Error Learning ─────────────────────────────────────────────

def _fit_ar(errors: np.ndarray, order: int) -> np.ndarray:
    """Ajusta AR(order) por OLS. Retorna coeficientes."""
    n = len(errors)
    if n <= order:
        return np.zeros(order)
    X_ar = np.column_stack([errors[i:n - order + i] for i in range(order)])
    y_ar = errors[order:]
    try:
        coefs, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        return coefs
    except Exception:
        return np.zeros(order)


def train_pel(X: np.ndarray, y: np.ndarray) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    scaler = RobustScaler()
    scaler.fit(X)
    base   = _train_base(X, y, scaler)
    errors = _oof_errors(X, y, scaler)

    # Testa autocorrelação nos erros
    lb     = acorr_ljungbox(errors, lags=[PEL_MAX_LAG], return_df=True)
    pvalue = float(lb["lb_pvalue"].iloc[-1])
    has_ac = pvalue < 0.05

    ar_coefs  = np.zeros(PEL_AR_ORDER)
    recent_e  = np.zeros(PEL_AR_ORDER)
    correction = 0.0

    if has_ac:
        ar_coefs  = _fit_ar(errors, PEL_AR_ORDER)
        recent_e  = errors[-PEL_AR_ORDER:]
        correction = float(np.dot(ar_coefs, recent_e[::-1]))

    return {
        "type":        "pel",
        "base":        base,
        "scaler":      scaler,
        "has_ac":      has_ac,
        "ar_coefs":    ar_coefs,
        "recent_errors": recent_e,
        "correction":  correction,
        "lb_pvalue":   pvalue,
    }


def predict_pel(model: dict, X: np.ndarray) -> np.ndarray:
    p = model["base"].predict_proba(model["scaler"].transform(X))[:, 1]
    if model["has_ac"]:
        p = p + model["correction"]
    return np.clip(p, 0.0, 1.0)


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "cb":  train_cb(X, y),
        "ewi": train_ewi(X, y),
        "pel": train_pel(X, y),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    p_cb  = predict_cb(model_dict["cb"],   X)
    p_ewi = predict_ewi(model_dict["ewi"], X)
    p_pel = predict_pel(model_dict["pel"], X)
    return (p_cb + p_ewi + p_pel) / 3.0
