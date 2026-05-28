import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config import PREDICTIONS_FILE, WEIGHTS_FILE, DEFAULT_WEIGHTS, OUTPUT_DIR

logger = logging.getLogger(__name__)

PRED_COLS = [
    "prediction_date", "target_concurso", "target_date", "draw_day",
    "seq_num", "n1", "n2", "n3", "n4", "n5", "n6",
    "matches", "best_prize", "validated",
    "actual_n1", "actual_n2", "actual_n3", "actual_n4", "actual_n5", "actual_n6",
]


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame(columns=PRED_COLS)
    df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    for col in PRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def save_predictions(df: pd.DataFrame):
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info("Previsões guardadas: %d linhas", len(df))


def load_weights() -> dict:
    if not WEIGHTS_FILE.exists():
        return DEFAULT_WEIGHTS.copy()
    with open(WEIGHTS_FILE) as f:
        w = json.load(f)
    return {k: w.get(k, 1.0) for k in DEFAULT_WEIGHTS}


def save_weights(weights: dict):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Pesos guardados: RF=%.3f GB=%.3f SGD=%.3f",
                weights["rf"], weights["gb"], weights["sgd"])
