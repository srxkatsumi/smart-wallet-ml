"""
Experimento — desafiante de regressão (prevê retorno/preço contínuo).

Roda em paralelo aos classificadores de produção (models/ensemble.py), sem
alterá-los. Alimenta apenas o segundo email "Carteira BOT Experimentos".

Desafiante: RandomForestRegressor calibrado com MAPIE (split conformal
temporal, cv="prefit") para intervalo de confiança de 90%. Alvo contínuo:
retorno percentual futuro (fwd_ret_dN, calculado em features/engineering.py).
"""
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from mapie.regression import MapieRegressor

from features.engineering import FEATURE_COLS
from config.settings import N_ESTIMATORS_RF, MAX_DEPTH_RF, HORIZONS

logger = logging.getLogger(__name__)

ALPHA = 0.10  # 90% prediction interval


def train_regressor_horizon(df: pd.DataFrame, horizon: int, ticker: str) -> dict | None:
    """
    Treina o desafiante para um horizonte e devolve a previsão para o último dia disponível.

    Split conformal temporal (não embaralhado): treina no primeiro ~80% da
    janela, calibra o intervalo de 90% no ~20% mais recente — mesmo espírito
    do split 70/30 manual em models/conformal.py, aqui via MapieRegressor
    com cv="prefit" (evita o jackknife+ com TimeSeriesSplit, que gera scores
    ausentes quando pontos iniciais nunca caem num fold de validação).
    """
    ret_col  = f"fwd_ret_d{horizon}"
    df_train = df.iloc[:-horizon].copy()
    df_train = df_train.dropna(subset=[ret_col])

    if len(df_train) < 30:
        logger.warning("%s D+%d: exemplos insuficientes para regressor (%d)", ticker, horizon, len(df_train))
        return None

    calib_size = max(15, int(len(df_train) * 0.2))
    df_fit   = df_train.iloc[:-calib_size]
    df_calib = df_train.iloc[-calib_size:]
    if len(df_fit) < 15:
        logger.warning("%s D+%d: exemplos insuficientes após split de calibração", ticker, horizon)
        return None

    X_fit = df_fit[FEATURE_COLS].values
    y_fit = df_fit[ret_col].values.astype(float)

    scaler   = RobustScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)

    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_fit_scaled, y_fit)

    X_calib_scaled = scaler.transform(df_calib[FEATURE_COLS].values)
    y_calib        = df_calib[ret_col].values.astype(float)

    mapie = MapieRegressor(estimator=rf, cv="prefit")
    mapie.fit(X_calib_scaled, y_calib)

    X_last = scaler.transform(df[FEATURE_COLS].iloc[[-1]].values)
    pred_ret, pis = mapie.predict(X_last, alpha=ALPHA)

    close_now = float(df["Close"].iloc[-1])
    ret_point = float(pred_ret[0])
    ret_lo    = float(pis[0, 0, 0])
    ret_hi    = float(pis[0, 1, 0])

    return {
        "pred_ret":       ret_point,
        "pred_price":     close_now * (1 + ret_point),
        "interval_lo":    close_now * (1 + min(ret_lo, ret_hi)),
        "interval_hi":    close_now * (1 + max(ret_lo, ret_hi)),
        "close_now":      close_now,
    }


def train_all_regressor(training_data: dict, models_dir=None) -> dict:
    """Espelha models/ensemble.py:train_all, mas para o desafiante de regressão."""
    resultados_exp = {}
    logger.info("[Experimento] Treinando regressor desafiante (D+1, D+2, D+3)...")

    for ticker, df in training_data.items():
        try:
            horizons = {}
            for day in HORIZONS:
                r = train_regressor_horizon(df, day, ticker)
                if r is not None:
                    horizons[day] = r
            if horizons:
                resultados_exp[ticker] = {
                    "close_now": float(df["Close"].iloc[-1]),
                    "horizons":  horizons,
                }
        except Exception as e:
            logger.error("[Experimento] %s: erro no treino do regressor: %s", ticker, e)

    logger.info("[Experimento] Regressor treinado: %d ativos", len(resultados_exp))
    return resultados_exp
