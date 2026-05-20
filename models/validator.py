import logging
import numpy as np
import pandas as pd
import yfinance as yf

from data.storage import save_predictions_log, save_ensemble_weights
from config.settings import HORIZONS, MIN_VALIDATIONS_WEIGHT, WEIGHT_DECAY_FACTOR

logger = logging.getLogger(__name__)


def validate_past_predictions(df_log: pd.DataFrame,
                               featured_data: dict) -> pd.DataFrame:
    hoje  = pd.Timestamp.now().normalize()
    updated     = 0
    skipped_open= 0

    # Fetch data for tickers that left the portfolio but still have open predictions
    tickers_ativos    = set(featured_data.keys())
    tickers_pendentes = set(df_log.loc[df_log["actual_price"].isna(), "ticker"].unique())
    tickers_ausentes  = tickers_pendentes - tickers_ativos

    for ticker in tickers_ausentes:
        try:
            df_extra = yf.download(ticker, period="10d", auto_adjust=True, progress=False)
            if not df_extra.empty:
                if isinstance(df_extra.columns, pd.MultiIndex):
                    df_extra.columns = df_extra.columns.get_level_values(0)
                df_extra.index = pd.to_datetime(df_extra.index).normalize()
                featured_data[ticker] = df_extra
                logger.info("%s: dados obtidos para validação", ticker)
            else:
                logger.warning("%s: sem dados (ticker inválido)", ticker)
        except Exception as e:
            logger.warning("%s: %s", ticker, e)

    for idx, row in df_log.iterrows():
        if pd.notna(row.get("actual_price")):
            continue

        target_date = pd.to_datetime(row["target_date"])
        if target_date > hoje:
            continue

        ticker = row["ticker"]
        if ticker not in featured_data:
            continue

        df_tick = featured_data[ticker]

        if target_date == hoje:
            future_prices = df_tick[df_tick.index == hoje]["Close"]
            if future_prices.empty:
                skipped_open += 1
                continue
        else:
            # Use first available price on or after target_date
            # (handles weekends/holidays automatically)
            future_prices = df_tick[df_tick.index >= target_date]["Close"]

        if future_prices.empty:
            continue

        actual  = float(future_prices.iloc[0])
        correct = (
            (row["direction"] == "up"   and actual > row["pred_price"] * 0.995) or
            (row["direction"] == "down" and actual < row["pred_price"] * 1.005)
        )
        df_log.at[idx, "actual_price"]      = actual
        df_log.at[idx, "actual_change_pct"] = round((actual / row["pred_price"] - 1) * 100, 4)
        df_log.at[idx, "correct"]           = float(correct)
        updated += 1

    if updated > 0:
        save_predictions_log(df_log)
        logger.info("%d previsões validadas", updated)
    if skipped_open > 0:
        logger.info("%d previsões aguardam fecho de mercado", skipped_open)
    if updated == 0 and skipped_open == 0:
        logger.info("Nenhuma previsão nova para validar")

    return df_log


def update_ensemble_weights(df_log: pd.DataFrame,
                             ensemble_weights: dict) -> dict:
    def calc_horizon(row):
        try:
            return (pd.to_datetime(row["target_date"]) - pd.to_datetime(row["pred_date"])).days
        except Exception:
            return None

    validadas = df_log[df_log["correct"].notna()].copy()
    if "horizon" not in validadas.columns or validadas["horizon"].isna().all():
        validadas["horizon"] = validadas.apply(calc_horizon, axis=1)

    weights_updated = False
    for day_n in HORIZONS:
        day_key    = f"d{day_n}"
        validadas_h= validadas[validadas["horizon"] == day_n].tail(30)

        if len(validadas_h) < MIN_VALIDATIONS_WEIGHT:
            continue

        n     = len(validadas_h)
        decay = np.exp(WEIGHT_DECAY_FACTOR * np.arange(n))
        decay = decay / decay.sum()

        new_weights = {}
        for model_col, key in [("model_rf", "rf"), ("model_gb", "gb"), ("model_sgd", "sgd")]:
            if model_col not in validadas_h.columns:
                new_weights[key] = ensemble_weights[day_key].get(key, 1.0)
                continue
            modelo_certo = (
                (validadas_h[model_col].notna()) &
                (
                    ((validadas_h[model_col] == validadas_h["direction"]) &
                     (validadas_h["correct"] == True)) |
                    ((validadas_h[model_col] != validadas_h["direction"]) &
                     (validadas_h["correct"] == False))
                )
            ).astype(float).values
            acc_weighted  = (modelo_certo * decay).sum()
            new_weights[key] = max(0.1, acc_weighted)

        total = sum(new_weights.values())
        ensemble_weights[day_key] = {
            k: round(v * 3.0 / total, 4) for k, v in new_weights.items()
        }
        weights_updated = True

    if weights_updated:
        save_ensemble_weights(ensemble_weights)
        logger.info("Pesos atualizados — d1: RF=%.2f GB=%.2f SGD=%.2f",
                    ensemble_weights["d1"]["rf"],
                    ensemble_weights["d1"]["gb"],
                    ensemble_weights["d1"]["sgd"])
    else:
        logger.info("Histórico insuficiente — pesos mantêm-se")

    return ensemble_weights


def save_new_predictions(df_log: pd.DataFrame, resultados_ml: dict,
                         today: pd.Timestamp) -> pd.DataFrame:
    hoje_str       = today.strftime("%Y-%m-%d")
    novas_previsoes= []

    for ticker, res in resultados_ml.items():
        df_ticker = res.get("df", pd.DataFrame())
        atr_val = (
            float(df_ticker["ATR14"].iloc[-1])
            if "ATR14" in df_ticker.columns and not df_ticker.empty
            else np.nan
        )

        for day, (direction, pred_price, conf) in res["preds_dict"].items():
            # BDay fix: use trading days so Friday D+1 → Monday, not Saturday
            target_date_str = (today + pd.offsets.BDay(day)).strftime("%Y-%m-%d")

            ja_existe = (
                (df_log["ticker"]      == ticker) &
                (df_log["pred_date"]   == hoje_str) &
                (df_log["target_date"] == target_date_str)
            )
            if not df_log.empty and ja_existe.any():
                continue

            h_data = res["horizons"][day]
            novas_previsoes.append({
                "ticker":            ticker,
                "pred_date":         hoje_str,
                "target_date":       target_date_str,
                "horizon":           day,
                "direction":         direction,
                "pred_price":        round(pred_price, 4),
                "confidence":        round(conf, 4),
                "actual_price":      np.nan,
                "actual_change_pct": np.nan,
                "correct":           np.nan,
                "atr_at_prediction": round(atr_val, 4) if not np.isnan(atr_val) else np.nan,
                "predicted_price":   np.nan,
                "model_rf":          h_data["preds_ind"].get("rf", ""),
                "model_gb":          h_data["preds_ind"].get("gb", ""),
                "model_sgd":         h_data["preds_ind"].get("sgd", ""),
            })

    if novas_previsoes:
        df_novas = pd.DataFrame(novas_previsoes)
        df_log   = pd.concat([df_log, df_novas], ignore_index=True)
        save_predictions_log(df_log)
        logger.info("%d novas previsões guardadas", len(novas_previsoes))
    else:
        logger.info("Previsões de hoje já existem no CSV")

    validadas_total = df_log[df_log["correct"].notna()]
    if len(validadas_total) > 0:
        acc_global = validadas_total["correct"].astype(float).mean()
        logger.info("Acurácia global: %.1f%% (%d validadas)",
                    acc_global * 100, len(validadas_total))

    return df_log
