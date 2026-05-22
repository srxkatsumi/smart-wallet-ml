import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.engineering import FEATURE_COLS
from models.ensemble import _train_horizon


def _make_training_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["RSI14"]    = rng.uniform(20, 80, n)
    data["BB_pos"]   = rng.uniform(0, 1, n)
    data["sma_cross"] = rng.integers(0, 2, n).astype(float)
    for day in [1, 2, 3]:
        data[f"target_d{day}"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_ensemble_probs_in_bounds():
    df = _make_training_df()
    weights = {"rf": 1.0, "gb": 1.0, "sgd": 1.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _train_horizon(df, "target_d1", weights, "TEST_TICKER", Path(tmpdir))

    prob_ens = result["prob"]
    assert 0.0 <= prob_ens <= 1.0, f"Probabilidade ensemble fora de [0,1]: {prob_ens}"

    for model, p in result["probs_ind"].items():
        assert 0.0 <= p <= 1.0, f"Probabilidade {model} fora de [0,1]: {p}"


def test_ensemble_direction_matches_prob():
    df = _make_training_df(seed=7)
    weights = {"rf": 1.0, "gb": 1.0, "sgd": 1.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _train_horizon(df, "target_d1", weights, "TEST_TICKER2", Path(tmpdir))

    if result["prob"] > 0.5:
        assert result["direction"] == "up"
    else:
        assert result["direction"] == "down"
