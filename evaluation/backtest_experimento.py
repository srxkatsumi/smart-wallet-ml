"""
Backtest histórico do desafiante de regressão — expanding window sobre os
2 anos de preços já baixados (PRICE_PERIOD), no mesmo espírito de
evaluation/walk_forward.py.

Roda uma vez (ou semanalmente) para dar uma primeira leitura no dia 1,
enquanto o tracking ao vivo (evaluation/experimento_significance.py)
acumula observações reais e não-espiadas.

Baseline usado aqui: retorno zero ("não mudou"), o baseline ingênuo padrão
em previsão de séries financeiras — mais simples e barato que reproduzir a
heurística ATR do campeão em cada passo histórico (exigiria retreinar os
classificadores a cada passo). É um proxy, não o campeão exato do email —
identificado como tal no relatório.
"""
import json
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

from features.engineering import FEATURE_COLS
from config.settings import N_ESTIMATORS_RF, MAX_DEPTH_RF, EXPERIMENTO_BACKTEST_FILE

logger = logging.getLogger(__name__)

BACKTEST_N_STEPS   = 60    # dias úteis de teste (expanding window)
BACKTEST_MIN_TRAIN = 252   # mínimo de linhas antes do primeiro passo


def _backtest_ticker(df: pd.DataFrame, horizon: int, n_steps: int = BACKTEST_N_STEPS) -> dict:
    ret_col    = f"fwd_ret_d{horizon}"
    valid_mask = df[ret_col].notna()
    valid_df   = df[valid_mask]

    if len(valid_df) < BACKTEST_MIN_TRAIN + n_steps:
        return {}

    test_rows = valid_df.iloc[-n_steps:]
    naive_err, model_err = [], []

    for _, test_row in test_rows.iterrows():
        pos      = df.index.get_loc(test_row.name)
        train_df = df.iloc[:pos].dropna(subset=[ret_col])
        if len(train_df) < BACKTEST_MIN_TRAIN:
            continue

        actual = float(test_row[ret_col])

        X_tr = train_df[FEATURE_COLS].values
        y_tr = train_df[ret_col].values.astype(float)
        X_te = test_row[FEATURE_COLS].values.reshape(1, -1)

        scaler  = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        rf = RandomForestRegressor(
            n_estimators=50, max_depth=MAX_DEPTH_RF,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr_sc, y_tr)
        pred = float(rf.predict(X_te_sc)[0])

        naive_err.append(actual ** 2)            # baseline: previsão = 0
        model_err.append((actual - pred) ** 2)

    if not model_err:
        return {}

    return {
        "n":               len(model_err),
        "mae_naive":       round(float(np.mean(np.sqrt(naive_err))), 6),
        "mae_model":       round(float(np.mean(np.sqrt(model_err))), 6),
        "mse_naive":       round(float(np.mean(naive_err)), 6),
        "mse_model":       round(float(np.mean(model_err)), 6),
        "r2":              round(1 - (np.sum(model_err) / np.sum(naive_err)), 4) if np.sum(naive_err) > 0 else None,
    }


def run_backtest_experimento(training_data: dict, my_tickers: list[str]) -> dict:
    results = {}
    for ticker in my_tickers:
        df = training_data.get(ticker)
        if df is None:
            continue
        try:
            per_horizon = {}
            for day in [1, 2, 3]:
                r = _backtest_ticker(df, day)
                if r:
                    per_horizon[f"d{day}"] = r
            if per_horizon:
                results[ticker] = per_horizon
                logger.info(
                    "[Experimento] Backtest %-10s D+1 MSE modelo=%.5f naive=%.5f (n=%d)",
                    ticker,
                    per_horizon.get("d1", {}).get("mse_model", float("nan")),
                    per_horizon.get("d1", {}).get("mse_naive", float("nan")),
                    per_horizon.get("d1", {}).get("n", 0),
                )
        except Exception as e:
            logger.warning("[Experimento] Backtest %s falhou: %s", ticker, e)

    summary = {"tickers": results}
    EXPERIMENTO_BACKTEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTO_BACKTEST_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[Experimento] Backtest completo: %d tickers", len(results))
    return summary


def load_backtest_experimento() -> dict:
    if not EXPERIMENTO_BACKTEST_FILE.exists():
        return {}
    try:
        return json.loads(EXPERIMENTO_BACKTEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
