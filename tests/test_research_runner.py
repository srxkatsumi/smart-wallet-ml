import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import data.storage as storage
from features.engineering import FEATURE_COLS
from research.runner import (
    _build_comparison, _build_consensus, _validate_past_predictions,
    _update_research_weights, _RES_COLS, _get_xy, _latest_rows, _needs_retrain, _predict_family,
)


@pytest.fixture(autouse=True)
def _no_disk_writes(monkeypatch):
    # _update_research_weights persists via a local `from data.storage import
    # save_research_weights`, straight to output/research_weights.json (real
    # production state). Block that here so these tests can never clobber it.
    monkeypatch.setattr(storage, "save_research_weights", lambda *a, **k: None)


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
        {"family": "classico_avancado", "correct_d1": True,  "validated": True, "week_date": "2026-06-01"},
        {"family": "classico_avancado", "correct_d1": False, "validated": True, "week_date": "2026-06-02"},
        {"family": "neural_recorrente", "correct_d1": True,  "validated": True, "week_date": "2026-06-01"},
        {"family": "neural_recorrente", "correct_d1": True,  "validated": True, "week_date": "2026-06-02"},
    ]
    log = pd.DataFrame(rows)
    r   = _build_comparison(log, "2026-06-02")
    if r:
        accs = [x["accuracy"] for x in r]
        assert accs == sorted(accs, reverse=True)


# ── _update_research_weights ──────────────────────────────────────────────

def test_update_research_weights_uses_distinct_days_not_rows():
    """
    Same tail(30)-on-rows bug as models/validator.py: a single week_date with
    many tickers can dominate a family's own row history. classico_avancado
    is right on 5 older dates (1 row each) and wrong on 40 rows on the single
    most recent date. A row-windowed tail(30) lands entirely inside that
    40-row date (all wrong) and floors the weight to 0.1; a day-windowed
    scheme still sees 5 of 6 days as correct and keeps the weight well above
    the floor.
    """
    def _rows(week_date: str, n: int, correct: bool) -> list[dict]:
        return [
            {"week_date": week_date, "family": "classico_avancado",
             "validated": True, "correct_d1": correct}
            for _ in range(n)
        ]

    older_days = [
        _rows(f"2026-01-{10 + i:02d}", 1, True) for i in range(5)
    ]
    log = pd.DataFrame(
        [row for day in older_days for row in day] +
        _rows("2026-01-15", 40, False)
    )
    current = {f"d{h}": {} for h in [1, 2, 3]}
    weights = _update_research_weights(log, current)

    d1 = weights["d1"]
    assert d1["classico_avancado"] > 0.5, f"collapsed to the floor: {d1}"


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
        "ref_date":    "2026-06-01",
        "direction_d1": "up",
        "direction_d2": "up",
        "direction_d3": "down",
        "validated":   False,
    }])
    closes = pd.DataFrame(
        {"Close": [100.0, 105.0, 95.0, 90.0]},
        index=pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]),
    )
    result = _validate_past_predictions(log, {"NVDA": closes}).iloc[0]
    assert result["validated"] == True
    # cada horizonte usa o próprio fecho: subiu em D+1, caiu em D+2 e D+3
    assert (result["actual_d1"], result["actual_d2"], result["actual_d3"]) == (105.0, 95.0, 90.0)
    assert result["correct_d1"] == True
    assert result["correct_d2"] == False
    assert result["correct_d3"] == True


def test_validate_waits_until_all_horizons_have_a_close():
    log = pd.DataFrame([{
        "ticker": "NVDA", "family": "f", "ref_price": 100.0, "ref_date": "2026-06-01",
        "direction_d1": "up", "direction_d2": "up", "direction_d3": "up", "validated": False,
    }])
    closes = pd.DataFrame({"Close": [100.0, 105.0]}, index=pd.to_datetime(["2026-06-01", "2026-06-02"]))
    result = _validate_past_predictions(log, {"NVDA": closes}).iloc[0]
    assert result["validated"] == False
    assert result["correct_d1"] == True


def test_rows_without_ref_date_close_without_a_result():
    log = pd.DataFrame([{
        "ticker": "NVDA", "family": "f", "ref_price": 100.0, "ref_date": None,
        "direction_d1": "up", "validated": False,
    }])
    closes = pd.DataFrame({"Close": [105.0]}, index=pd.to_datetime(["2026-06-02"]))
    result = _validate_past_predictions(log, {"NVDA": closes}).iloc[0]
    assert result["validated"] == True
    assert "correct_d1" not in result or pd.isna(result["correct_d1"])


def test_failed_family_returns_none_instead_of_a_vote():
    prob = _predict_family("models.nao_existe", np.zeros((60, 3)), np.zeros(60), np.zeros((5, 3)))
    assert prob is None


def test_family_without_prediction_does_not_vote_in_consensus():
    rows = [
        {"ticker": "NVDA", "direction_d1": "down", "family": "a"},
        {"ticker": "NVDA", "direction_d1": None,   "family": "b"},
    ]
    r = _build_consensus(rows, ["NVDA"], {})[0]
    assert r["pct_up"] == 0.0
    assert r["total"] == 1


def test_latest_rows_keep_the_newest_bar_even_without_a_label():
    df = _make_featured_df(n=100)
    df.loc[df.index[-3:], ["target_d1", "target_d2", "target_d3"]] = np.nan
    X, _ = _get_xy(df, 3)
    latest = _latest_rows(df)
    assert len(X) == 97
    np.testing.assert_allclose(latest[-1], df[FEATURE_COLS].iloc[-1].values.astype(np.float32))


def test_stale_model_is_retrained_and_fresh_one_is_not(tmp_path):
    import os, time
    path = tmp_path / "m.pkl"
    assert _needs_retrain(path, always=False)
    path.write_text("x")
    assert not _needs_retrain(path, always=False)
    assert _needs_retrain(path, always=True)
    old = time.time() - 8 * 86400
    os.utime(path, (old, old))
    assert _needs_retrain(path, always=False)
