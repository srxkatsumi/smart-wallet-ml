"""
Família: Séries Temporais
Modelos: ARIMA, SARIMA, ETS, Holt-Winters, Prophet

Todos os modelos são ajustados sobre a série temporal de y (sequência de
labels UP=1 / DOWN=0). A saída são probabilidades em [0,1] derivadas dos
valores ajustados (in-sample fitted values), clipped ao intervalo válido.

Contexto académico:
  ARIMA(5,0,1) e SARIMA(1,0,1)(1,0,1,5) testam se existe autocorrelação
  nos retornos diários — o que a hipótese de mercado eficiente (EMH) afirma
  que não deveria existir. ETS e Holt-Winters capturam tendência e
  sazonalidade suaves. Prophet modela sazonalidades semanais e anuais via
  decomposição aditiva (Taylor & Letham, 2018, Meta Research).

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]

Nota: X é usado apenas para determinar o comprimento do output. A série
temporal provém de y no treino. Modelos de séries temporais não usam
features — essa limitação é documentada na análise de ablação.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _to_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _pad_or_trim(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), arr[-1])])


# ── ARIMA ─────────────────────────────────────────────────────────────────

def train_arima(y: np.ndarray) -> dict:
    try:
        result = ARIMA(y.astype(float), order=(5, 0, 1)).fit()
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "arima", "fitted": fitted}


def predict_arima(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


# ── SARIMA ────────────────────────────────────────────────────────────────

def train_sarima(y: np.ndarray) -> dict:
    try:
        result = ARIMA(
            y.astype(float),
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 5),
        ).fit()
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "sarima", "fitted": fitted}


def predict_sarima(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


# ── ETS (Exponential Smoothing simples) ───────────────────────────────────

def train_ets(y: np.ndarray) -> dict:
    try:
        result = ExponentialSmoothing(
            y.astype(float), trend=None, seasonal=None
        ).fit(optimized=True)
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "ets", "fitted": fitted}


def predict_ets(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


# ── Holt-Winters ──────────────────────────────────────────────────────────

def train_holtwinters(y: np.ndarray) -> dict:
    try:
        result = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=5,
        ).fit(optimized=True)
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "holtwinters", "fitted": fitted}


def predict_holtwinters(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


# ── Prophet ───────────────────────────────────────────────────────────────

def train_prophet(y: np.ndarray) -> dict:
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        dates = pd.date_range(start="2015-01-01", periods=len(y), freq="B")
        df = pd.DataFrame({"ds": dates, "y": y.astype(float)})
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        forecast = model.predict(df)
        fitted = _to_prob(forecast["yhat"].values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "prophet", "fitted": fitted}


def predict_prophet(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "arima":       train_arima(y),
        "sarima":      train_sarima(y),
        "ets":         train_ets(y),
        "holtwinters": train_holtwinters(y),
        "prophet":     train_prophet(y),
        "n_train":     len(y),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    n = len(X)
    probs = np.stack([
        predict_arima(model_dict["arima"],       n),
        predict_sarima(model_dict["sarima"],     n),
        predict_ets(model_dict["ets"],           n),
        predict_holtwinters(model_dict["holtwinters"], n),
        predict_prophet(model_dict["prophet"],   n),
    ])
    return probs.mean(axis=0)
