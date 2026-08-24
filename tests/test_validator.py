import pandas as pd
import pytest
import models.validator as validator
from models.validator import update_ensemble_weights


@pytest.fixture(autouse=True)
def _no_disk_writes(monkeypatch):
    # update_ensemble_weights persists via save_ensemble_weights straight to
    # output/ensemble_weights.json (real production state) whenever it
    # updates weights. Block that here so these tests can never clobber it.
    monkeypatch.setattr(validator, "save_ensemble_weights", lambda *a, **k: None)

_DEFAULT = {
    "d1": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d2": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d3": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
}


def _rows(target_date: str, n: int, correct_model: str) -> list[dict]:
    """n validated D+1 rows for one target_date, where exactly one model
    (correct_model) called the direction right and the other two got it wrong."""
    rows = []
    for i in range(n):
        rows.append({
            "pred_date":   "2026-01-01",
            "target_date": target_date,
            "horizon":     1,
            "direction":   "up",
            "correct":     True,
            "model_rf":  "up" if correct_model == "rf"  else "down",
            "model_gb":  "up" if correct_model == "gb"  else "down",
            "model_sgd": "up" if correct_model == "sgd" else "down",
        })
    return rows


def test_weight_window_uses_distinct_days_not_rows():
    """
    Regression test for the tail(30)-on-rows bug: a single target_date with
    hundreds of tickers used to collapse the intended ~30-day window into one
    day. Here, one recent date has 100 rows where only SGD is right; two
    earlier dates (1 row each) show RF and GB were each right once. A window
    keyed on rows would see only the 100-row date and starve RF/GB down to
    the 0.1 floor; a window keyed on distinct days blends all three days.
    """
    rows = (
        _rows("2026-01-10", 1,   "rf") +
        _rows("2026-01-11", 1,   "gb") +
        _rows("2026-01-12", 100, "sgd")
    )
    df_log = pd.DataFrame(rows)

    weights = update_ensemble_weights(df_log, {k: dict(v) for k, v in _DEFAULT.items()})

    d1 = weights["d1"]
    # All three models were "the only one right" on exactly one day each, so
    # a day-windowed average should keep every model within shouting
    # distance of the neutral weight (1.0), not send two of them to the floor.
    assert d1["rf"] > 0.5, f"rf weight collapsed to the floor: {d1}"
    assert d1["gb"] > 0.5, f"gb weight collapsed to the floor: {d1}"
    # SGD is still the most recent and most-represented day, so it should
    # lead, but not overwhelmingly (that overwhelming swing was the bug).
    assert d1["sgd"] > d1["rf"]
    assert d1["sgd"] < 2.0, f"sgd weight swung as if the window were 1 day: {d1}"


def test_returns_default_when_insufficient_history():
    df_log = pd.DataFrame(_rows("2026-01-10", 1, "rf"))  # 1 row, below MIN_VALIDATIONS_WEIGHT
    weights = update_ensemble_weights(df_log, {k: dict(v) for k, v in _DEFAULT.items()})
    assert weights["d1"] == _DEFAULT["d1"]
