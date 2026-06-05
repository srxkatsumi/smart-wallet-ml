import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from features.engineering import FEATURE_COLS
from research.runner import (
    _build_comparison, _build_consensus, _validate_past_predictions,
    _RES_COLS,
)


def _make_featured_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["RSI14"]         = rng.uniform(20, 80, n)
    data["BB_pos"]        = rng.uniform(0, 1, n)
    data["sma_cross"]     = rng.integers(0, 2, n).astype(float)
    data["vix_regime"]    = rng.integers(0, 3, n).astype(float)
    data["asset_class"]   = np.zeros(n)
    for h in [1, 2, 3]:
        data[f"target_d{h}"] = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(data)


# ── _build_comparison ─────────────────────────────────────────────────────

def test_build_comparison_empty_log():
    log = pd.DataFrame(columns=_RES_COLS)
    r   = _build_comparison(log, "2026-06-02")
    assert r == []


def test_build_comparison_returns_sorted():
    rows = [
        {"family": "classico_avancado", "correct_d1": True,  "validated": True},
        {"family": "classico_avancado", "correct_d1": False, "validated": True},
        {"family": "neural_recorrente", "correct_d1": True,  "validated": True},
        {"family": "neural_recorrente", "correct_d1": True,  "validated": True},
    ]
    log = pd.DataFrame(rows)
    r   = _build_comparison(log, "2026-06-02")
    if r:
        accs = [x["accuracy"] for x in r]
        assert accs == sorted(accs, reverse=True)


# ── _build_consensus ──────────────────────────────────────────────────────

def test_build_consensus_all_up():
    rows = [
        {"ticker": "NVDA", "direction_d1": "up",   "family": f"f{i}"}
        for i in range(8)
    ]
    r = _build_consensus(rows, ["NVDA"], {})
    assert len(r) == 1
    assert r[0]["direction"] == "ALTA"
    assert r[0]["up_count"] == 8


def test_build_consensus_all_down():
    rows = [
        {"ticker": "NVDA", "direction_d1": "down", "family": f"f{i}"}
        for i in range(8)
    ]
    r = _build_consensus(rows, ["NVDA"], {})
    assert r[0]["direction"] == "BAIXA"
    assert r[0]["pct_up"] == 0.0


def test_build_consensus_multiple_tickers():
    rows = (
        [{"ticker": "NVDA", "direction_d1": "up",   "family": f"f{i}"} for i in range(5)] +
        [{"ticker": "LLY",  "direction_d1": "down", "family": f"f{i}"} for i in range(5)]
    )
    r = _build_consensus(rows, ["NVDA", "LLY"], {})
    assert len(r) == 2
    tickers = [x["ticker"] for x in r]
    assert "NVDA" in tickers
    assert "LLY"  in tickers


# ── _validate_past_predictions ────────────────────────────────────────────

def test_validate_marks_validated():
    log = pd.DataFrame([{
        "ticker":      "NVDA",
        "family":      "classico_avancado",
        "ref_price":   100.0,
        "direction_d1": "up",
        "direction_d2": "up",
        "direction_d3": "down",
        "validated":   False,
    }])
    close_prices = {"NVDA": 105.0}
    result = _validate_past_predictions(log, close_prices)
    assert result.iloc[0]["validated"] == True
    assert result.iloc[0]["correct_d1"] == True
