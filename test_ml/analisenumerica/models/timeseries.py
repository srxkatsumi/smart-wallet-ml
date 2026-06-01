"""
Família: Séries Temporais
Modelos: ARIMA, SARIMA, ETS, Holt-Winters, Prophet
Projecto: Mega Sena ML Experiment

Para cada bola (1-60), a frequência histórica de aparição constitui uma
série temporal. Cada modelo é ajustado sobre essa série de frequências e
prevê se a bola aparecerá no próximo sorteio.

Contexto académico:
  Se as bolas fossem sorteadas com memória (não-i.i.d.), os modelos de
  séries temporais deveriam capturar essa autocorrelação. O resultado
  esperado num processo verdadeiramente aleatório é que todos os modelos
  convirjam para P(aparece) ≈ 6/60 = 0.10 sem estrutura temporal detectável.

Interface (compatível com ensemble.py da Mega Sena):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1], shape (N,)
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


def train_arima(y: np.ndarray) -> dict:
    try:
        result = ARIMA(y.astype(float), order=(2, 0, 1)).fit()
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "arima", "fitted": fitted}


def train_sarima(y: np.ndarray) -> dict:
    try:
        result = ARIMA(
            y.astype(float), order=(1, 0, 1), seasonal_order=(1, 0, 0, 3)
        ).fit()
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "sarima", "fitted": fitted}


def train_ets(y: np.ndarray) -> dict:
    try:
        result = ExponentialSmoothing(
            y.astype(float), trend=None, seasonal=None
        ).fit(optimized=True)
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "ets", "fitted": fitted}


def train_holtwinters(y: np.ndarray) -> dict:
    try:
        result = ExponentialSmoothing(
            y.astype(float), trend="add", seasonal=None
        ).fit(optimized=True)
        fitted = _to_prob(result.fittedvalues.values)
    except Exception:
        fitted = np.full(len(y), y.mean())
    return {"type": "holtwinters", "fitted": fitted}


def train_prophet(y: np.ndarray) -> dict:
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # frequência dos sorteios (~3/semana)
        dates = pd.date_range(start="2000-01-01", periods=len(y), freq="2D")
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


def predict_arima(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


def predict_sarima(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


def predict_ets(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


def predict_holtwinters(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


def predict_prophet(model: dict, n: int) -> np.ndarray:
    return _pad_or_trim(model["fitted"], n)


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
        _pad_or_trim(model_dict["arima"]["fitted"],       n),
        _pad_or_trim(model_dict["sarima"]["fitted"],      n),
        _pad_or_trim(model_dict["ets"]["fitted"],         n),
        _pad_or_trim(model_dict["holtwinters"]["fitted"], n),
        _pad_or_trim(model_dict["prophet"]["fitted"],     n),
    ])
    return probs.mean(axis=0)
