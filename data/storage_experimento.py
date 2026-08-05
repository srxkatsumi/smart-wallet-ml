"""
Persistência do log do experimento de regressão — totalmente separado do
predictions_log.csv de produção (não compartilha schema nem lógica de
validação/pesos com a carteira principal).
"""
import logging
import numpy as np
import pandas as pd

from config.settings import EXPERIMENTO_PRED_LOG, EXPERIMENTO_PRED_COLS
from data.calendars import target_dates

logger = logging.getLogger(__name__)


def load_experimento_log() -> pd.DataFrame:
    if not EXPERIMENTO_PRED_LOG.exists():
        df = pd.DataFrame(columns=EXPERIMENTO_PRED_COLS)
        df.to_csv(EXPERIMENTO_PRED_LOG, index=False)
        logger.info("[Experimento] predictions_experimentos_log.csv criado em %s", EXPERIMENTO_PRED_LOG)
        return df
    df = pd.read_csv(EXPERIMENTO_PRED_LOG)
    logger.info("[Experimento] log carregado: %d registos", len(df))
    return df


def save_experimento_log(df: pd.DataFrame):
    df.to_csv(EXPERIMENTO_PRED_LOG, index=False)
    logger.info("[Experimento] log guardado: %d registos", len(df))


def validate_experimento_predictions(df_log: pd.DataFrame, training_data: dict) -> pd.DataFrame:
    """Resolve linhas passadas cujo target_date já chegou, usando os preços já baixados."""
    hoje    = pd.Timestamp.now().normalize()
    updated = 0

    for idx, row in df_log.iterrows():
        if pd.notna(row.get("actual_price")):
            continue

        target_date = pd.to_datetime(row["target_date"])
        if target_date > hoje:
            continue

        ticker = row["ticker"]
        if ticker not in training_data:
            continue

        df_tick = training_data[ticker]
        future_prices = df_tick[df_tick.index >= target_date]["Close"]
        if future_prices.empty:
            continue

        actual   = float(future_prices.iloc[0])
        ref      = float(row["ref_price"])
        actual_ret = (actual / ref) - 1

        champion_ret   = float(row["champion_pred_ret"])
        challenger_ret = float(row["challenger_pred_ret"])

        df_log.at[idx, "actual_price"]     = actual
        df_log.at[idx, "actual_ret"]       = actual_ret
        df_log.at[idx, "champion_sq_err"]  = (actual_ret - champion_ret) ** 2
        df_log.at[idx, "challenger_sq_err"]= (actual_ret - challenger_ret) ** 2
        df_log.at[idx, "validated"]        = True
        updated += 1

    if updated > 0:
        save_experimento_log(df_log)
        logger.info("[Experimento] %d previsões validadas", updated)
    else:
        logger.info("[Experimento] nenhuma previsão nova para validar")

    return df_log


def save_new_experimento_predictions(df_log: pd.DataFrame, resultados_ml: dict,
                                      resultados_exp: dict, today: pd.Timestamp) -> pd.DataFrame:
    hoje_str = today.strftime("%Y-%m-%d")
    novas    = []

    for ticker, exp_res in resultados_exp.items():
        ml_res = resultados_ml.get(ticker)
        if ml_res is None:
            continue

        tdates    = target_dates(today, ticker)
        close_now = exp_res["close_now"]

        for day, h_exp in exp_res["horizons"].items():
            if day not in ml_res["preds_dict"]:
                continue
            target_date_str = tdates[day].strftime("%Y-%m-%d")

            ja_existe = (
                (df_log["ticker"]      == ticker) &
                (df_log["pred_date"]   == hoje_str) &
                (df_log["target_date"] == target_date_str)
            )
            if not df_log.empty and ja_existe.any():
                continue

            _, champion_price, _ = ml_res["preds_dict"][day]
            champion_ret = (champion_price / close_now) - 1

            novas.append({
                "ticker":                 ticker,
                "pred_date":              hoje_str,
                "target_date":            target_date_str,
                "horizon":                day,
                "champion_pred_price":    round(champion_price, 4),
                "champion_pred_ret":      round(champion_ret, 6),
                "challenger_pred_price":  round(h_exp["pred_price"], 4),
                "challenger_pred_ret":    round(h_exp["pred_ret"], 6),
                "challenger_interval_lo": round(h_exp["interval_lo"], 4),
                "challenger_interval_hi": round(h_exp["interval_hi"], 4),
                "ref_price":              round(close_now, 4),
                "actual_price":           np.nan,
                "actual_ret":             np.nan,
                "champion_sq_err":        np.nan,
                "challenger_sq_err":      np.nan,
                "validated":              False,
            })

    if novas:
        df_novas = pd.DataFrame(novas)
        df_log   = pd.concat([df_log, df_novas], ignore_index=True)
        save_experimento_log(df_log)
        logger.info("[Experimento] %d novas previsões guardadas", len(novas))
    else:
        logger.info("[Experimento] previsões de hoje já existem no CSV")

    return df_log
