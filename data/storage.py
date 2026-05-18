import json
import logging
import pandas as pd
from config.settings import (
    OUTPUT_DIR, CHARTS_DIR, MODELS_DIR,
    PRED_LOG, WEIGHTS_FILE, PRED_COLS, DEFAULT_WEIGHTS,
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
        df.to_csv(PRED_LOG, index=False)
        logger.info("CSV migrado: model_lr → model_sgd")

    logger.info("predictions_log carregado: %d registos", len(df))
    return df


def save_predictions_log(df: pd.DataFrame):
    df.to_csv(PRED_LOG, index=False)
    logger.info("predictions_log guardado: %d registos", len(df))


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
