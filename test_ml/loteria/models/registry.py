"""
Central model registry — trains all families, builds mega-ensemble.

Seq-num convention:
  1  → mega-ensemble top-6 (deterministic)
  2–5 → weighted-random diversity (mega-ensemble probabilities)
  6  → RF top-6
  7  → GB top-6
  8  → SGD top-6
  9  → Classical top-6  (XGB+LGBM+CatBoost+SVM)
  10 → Markov top-6     (Markov chain + HMM)
  11 → Neural top-6     (LSTM + GRU)
  12 → Transformer top-6
  13 → Efficient top-6  (TCN)
  14 → TimeSeries top-6 (ARIMA + ETS + Prophet)
  15 → Contrarian top-6
"""

import logging
from importlib import import_module

import numpy as np

from config import (
    N_BALLS, BALLS_PER_DRAW, N_SEQUENCES,
    DEFAULT_WEIGHTS, MIN_DRAWS_WEIGHT, WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)

_RNG = np.random.default_rng(seed=42)

# family → seq_num (order preserved for iteration)
FAMILY_SEQ: dict[str, int] = {
    "rf":          6,
    "gb":          7,
    "sgd":         8,
    "classical":   9,
    "markov":     10,
    "neural":     11,
    "transformer": 12,
    "efficient":   13,
    "timeseries":  14,
    "contrarian":  15,
}

_EXTERNAL_MODULES: dict[str, str] = {
    "classical":   "models.classical",
    "markov":      "models.markov",
    "contrarian":  "models.contrarian",
    "neural":      "models.neural",
    "transformer": "models.transformer",
    "efficient":   "models.efficient",
    "timeseries":  "models.timeseries",
}


# ── Training ──────────────────────────────────────────────────────────────

def train_all(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train all active model families.

    Returns a dict with:
      "_ens"       → (rf, gb, sgd, scaler) tuple from ensemble.py
      "classical"  → model dict from classical.py
      "markov"     → model dict from markov.py
      ... etc.
    """
    models: dict = {}

    # Ensemble (RF / GB / SGD)
    try:
        ens_mod = import_module("models.ensemble")
        models["_ens"] = ens_mod.train(X, y)
        logger.debug("ensemble trained OK")
    except Exception as exc:
        logger.warning("ensemble FAIL: %s", exc)

    # All other families
    for family, module_path in _EXTERNAL_MODULES.items():
        try:
            mod = import_module(module_path)
            models[family] = mod.train(X, y)
            logger.debug("%s trained OK", family)
        except Exception as exc:
            logger.warning("%s FAIL: %s", family, exc)

    n_ok = len([k for k in models if not k.startswith("_")]) + (1 if "_ens" in models else 0)
    logger.info("train_all: %d/%d famílias treinadas", n_ok, len(FAMILY_SEQ))
    return models


# ── Probability extraction ────────────────────────────────────────────────

def _normalize(p: np.ndarray) -> np.ndarray:
    mn, mx = p.min(), p.max()
    rng = mx - mn
    if rng < 1e-9:
        return np.ones_like(p) / len(p)
    return (p - mn) / rng


def _get_proba(family: str, models_dict: dict, X: np.ndarray) -> np.ndarray | None:
    """Return raw probability array shape (len(X),) for one family, or None on error."""
    try:
        if family in ("rf", "gb", "sgd"):
            tup = models_dict.get("_ens")
            if tup is None:
                return None
            rf, gb, sgd, sc = tup
            X_sc = sc.transform(X)
            clf = {"rf": rf, "gb": gb, "sgd": sgd}[family]
            return clf.predict_proba(X_sc)[:, 1]
        else:
            m = models_dict.get(family)
            if m is None:
                return None
            mod = import_module(_EXTERNAL_MODULES[family])
            return mod.predict(m, X)
    except Exception as exc:
        logger.warning("predict %s FAIL: %s", family, exc)
        return None


def predict_proba_all(models_dict: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """Return {family: proba_array (len(X),)} for all available families."""
    return {
        f: p
        for f in FAMILY_SEQ
        if (p := _get_proba(f, models_dict, X)) is not None
    }


# ── Ensemble prediction ───────────────────────────────────────────────────

def mega_ensemble_probs(models_dict: dict, X: np.ndarray, weights: dict) -> np.ndarray:
    """Weighted average of all normalised family probabilities."""
    combined = np.zeros(len(X))
    total_w  = 0.0

    for family, p in predict_proba_all(models_dict, X).items():
        w = weights.get(family, 1.0)
        combined += _normalize(p) * w
        total_w  += w

    if total_w == 0:
        return np.ones(len(X)) / len(X)
    return combined / total_w


def predict_sequences(
    results,
    draw_day: str,
    models_dict: dict,
    weights: dict,
) -> list[list[int]]:
    """
    Returns N_SEQUENCES predictions:
      Seq 1 : mega-ensemble top-6 (deterministic)
      Seq 2–N: weighted random using mega-ensemble probabilities
    """
    from features.engineering import build_prediction_features

    X = build_prediction_features(results, draw_day)
    p = mega_ensemble_probs(models_dict, X, weights)

    numbers   = np.arange(1, N_BALLS + 1)
    sequences = [sorted(numbers[np.argsort(p)[::-1][:BALLS_PER_DRAW]].tolist())]

    norm_p = p / p.sum()
    for _ in range(N_SEQUENCES - 1):
        drawn = _RNG.choice(numbers, size=BALLS_PER_DRAW, replace=False, p=norm_p)
        sequences.append(sorted(drawn.tolist()))

    return sequences


def predict_per_family(
    results,
    draw_day: str,
    models_dict: dict,
) -> dict[str, list[int]]:
    """Return deterministic top-6 for each model family: {family: [n1..n6]}."""
    from features.engineering import build_prediction_features

    X       = build_prediction_features(results, draw_day)
    numbers = np.arange(1, N_BALLS + 1)
    out: dict[str, list[int]] = {}

    for family in FAMILY_SEQ:
        p = _get_proba(family, models_dict, X)
        if p is None:
            continue
        top6 = sorted(numbers[np.argsort(_normalize(p))[::-1][:BALLS_PER_DRAW]].tolist())
        out[family] = top6

    return out


# ── Weight update ─────────────────────────────────────────────────────────

def update_weights(pred_df, weights: dict) -> dict:
    """
    Compute decay-weighted accuracy per family using their dedicated seq_nums.
    Normalises so weights span [0.5, 2.0] relative to each family's performance.
    Falls back to current weights if data is insufficient for a family.
    """
    def _decay_acc(df_model) -> float:
        matches = (
            df_model.sort_values("target_concurso")["matches"]
            .astype(float).values
        )
        n     = len(matches)
        decay = np.exp(-WEIGHT_DECAY * np.arange(n)[::-1])
        decay /= decay.sum()
        return float((matches / BALLS_PER_DRAW * decay).sum())

    new_weights = dict(weights)
    updated_families: list[str] = []

    for family, seq_num in FAMILY_SEQ.items():
        v = pred_df[(pred_df["seq_num"] == seq_num) & (pred_df["validated"] == True)]
        if len(v) >= MIN_DRAWS_WEIGHT:
            new_weights[family] = _decay_acc(v)
            updated_families.append(family)

    if updated_families:
        vals = [new_weights[f] for f in updated_families]
        mn, mx = min(vals), max(vals)
        if mx > mn:
            for f in updated_families:
                new_weights[f] = round(0.5 + 1.5 * (new_weights[f] - mn) / (mx - mn), 4)
        else:
            for f in updated_families:
                new_weights[f] = 1.0

        top = sorted(new_weights.items(), key=lambda x: -x[1])
        logger.info(
            "Pesos actualizados (%d famílias): %s",
            len(updated_families),
            "  ".join(f"{k}={v:.3f}" for k, v in top),
        )

    return new_weights
