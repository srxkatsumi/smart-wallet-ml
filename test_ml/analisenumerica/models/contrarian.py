"""
Família: Contrarian / Error Learning
Modelos: CB, EWI, PEL
Projecto: Mega Sena ML Experiment

Aplicação académica:
  CB  → Se inverter prevê melhor, o ensemble original é pior que acaso.
  EWI → Testa se erros recentes têm estrutura temporal explorável.
  PEL → Testa se a sequência de erros tem autocorrelação nos sorteios.

Resultado esperado: todos os 3 convergem para ~0.10 (= 6/60),
confirmando que não existe estrutura nos erros — os erros são i.i.d.,
não há padrão a aprender nem a inverter.

Referências: Box & Jenkins (1976), Ljung & Box (1978).
"""

import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

EWI_WINDOW     = 10
EWI_INVERT_TH  = 0.60
EWI_RESTORE_TH = 0.40
PEL_MAX_LAG    = 5
PEL_AR_ORDER   = 2
N_TREES        = 50


def _train_base(X, y, scaler):
    clf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(scaler.transform(X), y.astype(int))
    return clf


def _oof_errors(X, y, scaler):
    errors = np.zeros(len(X))
    tscv   = TimeSeriesSplit(n_splits=3)
    for tr_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        clf.fit(scaler.transform(X[tr_idx]), y[tr_idx].astype(int))
        p = clf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        errors[val_idx] = y[val_idx].astype(float) - p
    return errors


def _fit_ar(errors, order):
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


def train_cb(X, y):
    scaler = RobustScaler(); scaler.fit(X)
    return {"type": "cb", "base": _train_base(X, y, scaler), "scaler": scaler}


def predict_cb(model, X):
    return 1.0 - model["base"].predict_proba(model["scaler"].transform(X))[:, 1]


def train_ewi(X, y):
    scaler = RobustScaler(); scaler.fit(X)
    base   = _train_base(X, y, scaler)
    errors = _oof_errors(X, y, scaler)
    wrong  = (errors != 0).astype(float)
    n      = len(wrong)
    last_rate = wrong[-EWI_WINDOW:].mean() if n >= EWI_WINDOW else 0.5
    return {"type": "ewi", "base": base, "scaler": scaler,
            "inverted": bool(last_rate > EWI_INVERT_TH),
            "last_rate": float(last_rate)}


def predict_ewi(model, X):
    p = model["base"].predict_proba(model["scaler"].transform(X))[:, 1]
    return (1.0 - p) if model["inverted"] else p


def train_pel(X, y):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    scaler = RobustScaler(); scaler.fit(X)
    base   = _train_base(X, y, scaler)
    errors = _oof_errors(X, y, scaler)
    lb     = acorr_ljungbox(errors, lags=[PEL_MAX_LAG], return_df=True)
    pvalue = float(lb["lb_pvalue"].iloc[-1])
    has_ac = pvalue < 0.05
    ar_coefs   = _fit_ar(errors, PEL_AR_ORDER) if has_ac else np.zeros(PEL_AR_ORDER)
    recent_e   = errors[-PEL_AR_ORDER:] if len(errors) >= PEL_AR_ORDER else np.zeros(PEL_AR_ORDER)
    correction = float(np.dot(ar_coefs, recent_e[::-1])) if has_ac else 0.0
    return {"type": "pel", "base": base, "scaler": scaler,
            "has_ac": has_ac, "correction": correction, "lb_pvalue": pvalue}


def predict_pel(model, X):
    p = model["base"].predict_proba(model["scaler"].transform(X))[:, 1]
    return np.clip(p + model["correction"], 0.0, 1.0)


def train(X, y):
    return {"cb": train_cb(X, y), "ewi": train_ewi(X, y), "pel": train_pel(X, y)}


def predict(model_dict, X):
    return (predict_cb(model_dict["cb"], X) +
            predict_ewi(model_dict["ewi"], X) +
            predict_pel(model_dict["pel"], X)) / 3.0
