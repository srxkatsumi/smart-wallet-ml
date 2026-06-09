import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold

from config import (
    N_BALLS, BALLS_PER_DRAW, N_SEQUENCES,
    N_ESTIMATORS_RF, MAX_DEPTH_RF,
    N_ESTIMATORS_GB, MAX_DEPTH_GB, LEARNING_RATE_GB,
    N_SPLITS_CV, SGD_ALPHA,
    DEFAULT_WEIGHTS, MIN_DRAWS_WEIGHT, WEIGHT_DECAY,
)
from features.engineering import FEATURE_COLS, build_prediction_features

logger = logging.getLogger(__name__)

_RNG = np.random.default_rng(seed=42)

def train(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train RF, GB, SGD. Returns (rf, gb, sgd, scaler)."""
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS_GB, max_depth=MAX_DEPTH_GB,
        learning_rate=LEARNING_RATE_GB, random_state=42,
    )
    sgd = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=SGD_ALPHA,
        max_iter=1000, random_state=42, class_weight="balanced",
    )

    rf.fit(X_sc, y)
    gb.fit(X_sc, y)
    sgd.fit(X_sc, y)

    return rf, gb, sgd, scaler


def predict_sequences(results: pd.DataFrame, draw_day: str,
                      models: tuple, weights: dict) -> list[list[int]]:
    """
    Returns N_SEQUENCES lists of BALLS_PER_DRAW numbers each.

    Sequence 1: top-6 by ensemble probability (deterministic)
    Sequences 2-5: weighted random sampling — diversity without full randomness
    """
    rf, gb, sgd, scaler = models
    X = build_prediction_features(results, draw_day)
    X_sc = scaler.transform(X)

    p_rf  = rf.predict_proba(X_sc)[:, 1]
    p_gb  = gb.predict_proba(X_sc)[:, 1]
    p_sgd = sgd.predict_proba(X_sc)[:, 1]

    total_w = weights["rf"] + weights["gb"] + weights["sgd"]
    p_ens   = (p_rf * weights["rf"] + p_gb * weights["gb"] + p_sgd * weights["sgd"]) / total_w

    numbers = np.arange(1, N_BALLS + 1)

    sequences = []

    # Seq 1: top-6 deterministic
    top6 = numbers[np.argsort(p_ens)[::-1][:BALLS_PER_DRAW]].tolist()
    sequences.append(sorted(top6))

    # Seqs 2-5: weighted sampling without replacement (using ensemble probs as weights)
    for _ in range(N_SEQUENCES - 1):
        probs = p_ens / p_ens.sum()
        drawn = _RNG.choice(numbers, size=BALLS_PER_DRAW, replace=False, p=probs)
        sequences.append(sorted(drawn.tolist()))

    return sequences


def predict_per_model(X_sc: np.ndarray, models: tuple) -> dict:
    """Return deterministic top-6 for each model individually.
    Returns {"rf": [...], "gb": [...], "sgd": [...]}
    """
    rf, gb, sgd, _ = models
    numbers = np.arange(1, N_BALLS + 1)
    result = {}
    for name, clf in [("rf", rf), ("gb", gb), ("sgd", sgd)]:
        probs = clf.predict_proba(X_sc)[:, 1]
        top6  = sorted(numbers[np.argsort(probs)[::-1][:BALLS_PER_DRAW]].tolist())
        result[name] = top6
    return result


def count_matches(predicted: list[int], actual: list[int]) -> int:
    return len(set(predicted) & set(actual))


def update_weights(pred_df: pd.DataFrame, weights: dict) -> dict:
    """
    Update weights using per-model deterministic sequences:
      seq_num=6 → RF top-6
      seq_num=7 → GB top-6
      seq_num=8 → SGD top-6

    Falls back to ensemble-level update if per-model data is insufficient.
    """
    SEQ_MAP = {"rf": 6, "gb": 7, "sgd": 8}

    def _decay_weighted_acc(df_model: pd.DataFrame) -> float:
        matches = df_model["matches"].astype(float).values
        n = len(matches)
        decay = np.exp(-WEIGHT_DECAY * np.arange(n)[::-1])
        decay /= decay.sum()
        return float((matches / BALLS_PER_DRAW * decay).sum())

    # Check if per-model sequences exist
    accs = {}
    for model, seq_num in SEQ_MAP.items():
        v = pred_df[(pred_df["seq_num"] == seq_num) & (pred_df["validated"] == True)]
        if len(v) >= MIN_DRAWS_WEIGHT:
            accs[model] = _decay_weighted_acc(v)

    if len(accs) == 3:
        total = sum(accs.values())
        if total == 0:
            return weights
        new_weights = {m: round(accs[m] / total * 3.0, 4) for m in accs}
        logger.info(
            "Pesos (per-model): RF=%.3f GB=%.3f SGD=%.3f  |  acc RF=%.4f GB=%.4f SGD=%.4f",
            new_weights["rf"], new_weights["gb"], new_weights["sgd"],
            accs["rf"], accs["gb"], accs["sgd"],
        )
        return new_weights

    # Fallback: ensemble-level update (while per-model backfill is still running)
    v = pred_df[(pred_df["seq_num"] == 1) & (pred_df["validated"] == True)]
    if len(v) < MIN_DRAWS_WEIGHT:
        logger.info("Pesos não atualizados: dados insuficientes")
        return weights
    acc = _decay_weighted_acc(v)
    signal = max(acc, 0.1) / 0.1
    new_weights = {k: max(0.1, weights[k] * signal) for k in weights}
    total = sum(new_weights.values())
    new_weights = {k: round(v * 3.0 / total, 4) for k, v in new_weights.items()}
    logger.info("Pesos (fallback ensemble): RF=%.3f GB=%.3f SGD=%.3f", *new_weights.values())
    return new_weights
