import json
import logging
import numpy as np
import pandas as pd
from config.settings import (
    OUTPUT_DIR, CHARTS_DIR, MODELS_DIR,
    PRED_LOG, PUBLIC_LOG, WEIGHTS_FILE, PRED_COLS, DEFAULT_WEIGHTS,
)

logger = logging.getLogger(__name__)


def ensure_dirs():
    for d in [OUTPUT_DIR, CHARTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_predictions_log() -> pd.DataFrame:
    if not PRED_LOG.exists():
        df = pd.DataFrame(columns=PRED_COLS)
        df.to_csv(PRED_LOG, index=False)
        logger.info("predictions_log.csv criado em %s", PRED_LOG)
        return df

    df = pd.read_csv(PRED_LOG)

    # Migração retrocompatível: model_lr → model_sgd
    if "model_lr" in df.columns and "model_sgd" not in df.columns:
        df.rename(columns={"model_lr": "model_sgd"}, inplace=True)
        logger.info("CSV migrado: model_lr → model_sgd")

    # Migração: adicionar novas colunas se ausentes
    new_cols = ["actual_change_pct", "atr_at_prediction", "predicted_price"]
    added = [c for c in new_cols if c not in df.columns]
    if added:
        for col in added:
            df[col] = np.nan
        # Retrocompatível: calcular actual_change_pct para linhas já validadas
        mask = (
            df["actual_price"].notna() &
            df["pred_price"].notna() &
            df["actual_change_pct"].isna()
        )
        if mask.any():
            df.loc[mask, "actual_change_pct"] = (
                (df.loc[mask, "actual_price"] / df.loc[mask, "pred_price"]) - 1
            ) * 100
        df.to_csv(PRED_LOG, index=False)
        logger.info("CSV migrado: colunas adicionadas — %s", added)

    logger.info("predictions_log carregado: %d registos", len(df))
    return df


def save_predictions_log(df: pd.DataFrame):
    df.to_csv(PRED_LOG, index=False)
    logger.info("predictions_log guardado: %d registos", len(df))


_PUBLIC_COLS = [
    "asset_type", "pred_date", "target_date", "horizon",
    "direction", "confidence", "actual_change_pct", "correct",
    "model_rf", "model_gb", "model_sgd",
]

_PUBLIC_DELAY_BDAYS = 10

def save_public_log(df: pd.DataFrame, portfolio_tickers: list) -> None:
    portfolio_set = set(portfolio_tickers)
    cutoff = (pd.Timestamp.today() - pd.offsets.BDay(_PUBLIC_DELAY_BDAYS)).normalize()

    df_pub = df.copy()
    df_pub["pred_date"] = pd.to_datetime(df_pub["pred_date"])
    df_pub = df_pub[df_pub["pred_date"] <= cutoff].copy()

    df_pub["asset_type"] = df_pub["ticker"].apply(
        lambda t: "portfolio" if t in portfolio_set else "watchlist"
    )
    cols = [c for c in _PUBLIC_COLS if c in df_pub.columns]
    df_pub[cols].to_csv(PUBLIC_LOG, index=False)
    logger.info("predictions_log_public guardado: %d registos (corte: %s)",
                len(df_pub), cutoff.strftime("%Y-%m-%d"))


def load_ensemble_weights() -> dict:
    if not WEIGHTS_FILE.exists():
        weights = DEFAULT_WEIGHTS.copy()
        save_ensemble_weights(weights)
        logger.info("Pesos inicializados (iguais por horizonte)")
        return weights

    with open(WEIGHTS_FILE) as f:
        weights = json.load(f)

    if "d1" not in weights:
        weights = DEFAULT_WEIGHTS.copy()

    # Migração: "lr" → "sgd"
    for dk in weights:
        if "lr" in weights[dk] and "sgd" not in weights[dk]:
            weights[dk]["sgd"] = weights[dk].pop("lr")

    logger.info(
        "Pesos carregados — d1: RF=%.2f GB=%.2f SGD=%.2f",
        weights["d1"]["rf"], weights["d1"]["gb"], weights["d1"]["sgd"],
    )
    return weights


def save_ensemble_weights(weights: dict):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Pesos guardados")


def load_portfolio_config() -> dict:
    with open("config/portfolio.json") as f:
        return json.load(f)


def load_my_tickers() -> list[str]:
    with open("config/my_portfolio.json") as f:
        cfg = json.load(f)
    return list(set(cfg["etoro"] + cfg["etf_acumulacao"]))


def load_watchlist() -> list[str]:
    with open("config/watchlist.json") as f:
        cfg = json.load(f)
    return cfg["watchlist"]


def build_ticker_order(my_tickers: list[str], watchlist: list[str]) -> list[str]:
    extra = [t for t in watchlist if t not in my_tickers]
    return my_tickers + extra
