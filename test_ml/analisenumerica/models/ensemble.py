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


def count_matches(predicted: list[int], actual: list[int]) -> int:
    return len(set(predicted) & set(actual))


def update_weights(pred_df: pd.DataFrame, weights: dict) -> dict:
    """
    Update model weights based on recent validated predictions.

    For each validated draw, compute how many numbers each model's
    top-6 matched vs the ensemble top-6. Weight = decay-weighted hit rate.
    We approximate individual model accuracy by using their individual
    top-6 selections (highest probability numbers from each model alone).
    """
    validated = pred_df[pred_df["validated"] == True].tail(MIN_DRAWS_WEIGHT * 3)
    if len(validated) < MIN_DRAWS_WEIGHT:
        logger.info("Pesos não atualizados: apenas %d sorteios validados (mínimo %d)",
                    len(validated), MIN_DRAWS_WEIGHT)
        return weights

    # Use only seq_num == 1 (deterministic sequence) for weight update
    v = validated[validated["seq_num"] == 1].copy()
    if len(v) < MIN_DRAWS_WEIGHT:
        return weights

    n = len(v)
    decay = np.exp(WEIGHT_DECAY * np.arange(n))
    decay = decay / decay.sum()

    # matches column already tells us how many numbers seq1 got right
    # We use this as proxy for ensemble quality
    # Per-model accuracy can't be tracked without storing individual model sequences
    # → keep uniform weights with small recency adjustment based on overall match rate
    matches = v["matches"].astype(float).values
    norm_matches = matches / BALLS_PER_DRAW      # 0.0 → 1.0
    weighted_acc = (norm_matches * decay).sum()

    # Since we can't distinguish per-model without storing their individual predictions,
    # we use a simplified update: reward all models proportionally to recent accuracy
    # vs random baseline (0.6 / 6 = 0.1)
    baseline = 0.1
    signal   = max(weighted_acc, baseline) / baseline   # > 1 means beating random

    new_weights = {k: max(0.1, weights[k] * signal) for k in weights}
    total = sum(new_weights.values())
    new_weights = {k: round(v * 3.0 / total, 4) for k, v in new_weights.items()}

    logger.info("Pesos: RF=%.3f GB=%.3f SGD=%.3f (acurácia ponderada=%.3f)",
                new_weights["rf"], new_weights["gb"], new_weights["sgd"], weighted_acc)
    return new_weights
