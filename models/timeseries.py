"""
Família: Séries Temporais
Modelos: ARIMA, SARIMA, ETS, Holt-Winters, Prophet

Todos os modelos são ajustados sobre a série temporal de y (sequência de
labels UP=1 / DOWN=0). A saída é a previsão do próximo passo, clipped a [0,1].

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

# A previsão depende só de y, então precisa de novo ajuste para incorporar cada dia.
RETRAIN_DAILY = True


def _to_prob(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


# Cada modelo guarda a previsão do próximo passo. Os valores ajustados do treino
# descrevem o passado e não servem como previsão.

def _next_step(forecast_fn, y: np.ndarray) -> float:
    try:
        return _to_prob(np.asarray(forecast_fn())[0])
    except Exception:
        return float(y.mean())


def train_arima(y: np.ndarray) -> dict:
    def forecast():
        return ARIMA(y.astype(float), order=(5, 0, 1)).fit().forecast(1)
    return {"type": "arima", "forecast": _next_step(forecast, y)}


def train_sarima(y: np.ndarray) -> dict:
    def forecast():
        return ARIMA(y.astype(float), order=(1, 0, 1), seasonal_order=(1, 0, 1, 5)).fit().forecast(1)
    return {"type": "sarima", "forecast": _next_step(forecast, y)}


def train_ets(y: np.ndarray) -> dict:
    def forecast():
        return ExponentialSmoothing(y.astype(float), trend=None, seasonal=None).fit(optimized=True).forecast(1)
    return {"type": "ets", "forecast": _next_step(forecast, y)}


def train_holtwinters(y: np.ndarray) -> dict:
    def forecast():
        return ExponentialSmoothing(
            y.astype(float), trend="add", seasonal="add", seasonal_periods=5,
        ).fit(optimized=True).forecast(1)
    return {"type": "holtwinters", "forecast": _next_step(forecast, y)}


def train_prophet(y: np.ndarray) -> dict:
    def forecast():
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
        future = model.make_future_dataframe(periods=1, freq="B").tail(1)
        return model.predict(future)["yhat"].values
    return {"type": "prophet", "forecast": _next_step(forecast, y)}


def _predict_stored(model: dict, n: int) -> np.ndarray:
    return np.full(n, model["forecast"])


predict_arima = predict_sarima = predict_ets = predict_holtwinters = predict_prophet = _predict_stored


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
