import pandas as pd
import pytest

from maomao.sensors import compute_sensors, compute_market_sensors, real_returns_by_day, CHANNELS

PORTFOLIO = ["AAA.DE", "BBB.DE"]


def _row(ticker, target_date, ref_price, actual_price, rf="up", gb="up", sgd="up"):
    return {
        "ticker": ticker, "horizon": 1, "target_date": target_date,
        "ref_price": ref_price, "actual_price": actual_price,
        "model_rf": rf, "model_gb": gb, "model_sgd": sgd,
    }


def test_empty_log_returns_zeroed_channels_and_no_date():
    df = pd.DataFrame(columns=["ticker", "horizon", "target_date", "ref_price", "actual_price",
                                "model_rf", "model_gb", "model_sgd"])
    out = compute_sensors(df, {}, PORTFOLIO)
    assert out["date"] is None
    assert all(out[c] == 0.0 for c in CHANNELS)


def test_real_returns_by_day_uses_ref_price_not_pred_price():
    """
    Regression guard: predictions_log.csv's actual_change_pct is
    (actual_price/pred_price - 1), i.e. prediction error, not the real
    market return. MaoMao must use ref_price (close at prediction time),
    never pred_price, for anything it treats as "what the market did".
    """
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 102.0)])
    by_day = real_returns_by_day(df, PORTFOLIO)
    assert by_day["2026-01-02"] == pytest.approx(2.0)


def test_ret_up_saturates_at_two_percent():
    df = pd.DataFrame([
        _row("AAA.DE", "2026-01-02", 100.0, 105.0),
        _row("BBB.DE", "2026-01-02", 100.0, 105.0),
    ])
    out = compute_sensors(df, {}, PORTFOLIO)
    assert out["RET_UP"] == 1.0
    assert out["RET_DOWN"] == 0.0


def test_streak_down_counts_only_the_most_recent_consecutive_run():
    rows = []
    # up, down, down, down (mais recente) -> streak_down = 3
    for i, ret in enumerate([1.0, -1.0, -1.0, -1.0]):
        date = f"2026-01-0{i+1}"
        rows.append(_row("AAA.DE", date, 100.0, 100.0 + ret))
        rows.append(_row("BBB.DE", date, 100.0, 100.0 + ret))
    df = pd.DataFrame(rows)
    out = compute_sensors(df, {}, PORTFOLIO)
    assert out["STREAK_DOWN"] == pytest.approx(3 / 5.0)
    assert out["STREAK_UP"] == 0.0


def test_disagreement_is_one_when_all_three_models_differ_on_everything():
    df = pd.DataFrame([
        _row("AAA.DE", "2026-01-02", 100.0, 101.0, rf="up", gb="down", sgd="up"),
        _row("BBB.DE", "2026-01-02", 100.0, 101.0, rf="down", gb="up", sgd="down"),
    ])
    out = compute_sensors(df, {}, PORTFOLIO)
    assert out["DISAGREEMENT"] == 1.0


def test_anomaly_channel_reflects_todays_alerta_flag():
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 100.0)])
    out_true  = compute_sensors(df, {"alerta": True}, PORTFOLIO)
    out_false = compute_sensors(df, {"alerta": False}, PORTFOLIO)
    assert out_true["ANOMALY"] == 1.0
    assert out_false["ANOMALY"] == 0.0


def test_vix_spike_reads_from_market_variacao_1d():
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 100.0)])
    anomaly = {"market": {"variacao_1d": 0.30}}  # +30% num dia -> satura
    out = compute_sensors(df, anomaly, PORTFOLIO)
    assert out["VIX_SPIKE"] == 1.0


def test_vix_falling_does_not_feed_stress_channels():
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 100.0)])
    anomaly = {"alerta": True, "market": {"variacao_1d": -0.20, "spike": True, "transicao": False}, "tickers": {}}
    for fn in (compute_sensors, compute_market_sensors):
        out = fn(df, anomaly, PORTFOLIO)
        assert out["VIX_SPIKE"] == 0.0
        assert out["ANOMALY"] == 0.0


def test_ticker_alert_still_counts_when_vix_falls():
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 100.0)])
    anomaly = {"alerta": True, "market": {"variacao_1d": -0.20, "spike": True, "transicao": False},
               "tickers": {"AAA.DE": {"alerta": True, "multiplo_atr": 3.0}}}
    assert compute_sensors(df, anomaly, PORTFOLIO)["ANOMALY"] == 1.0


def test_day_validated_for_few_tickers_is_ignored():
    tickers = [f"T{i}" for i in range(10)]
    rows = [_row(t, "2026-01-02", 100.0, 101.0) for t in tickers]
    rows += [_row(t, "2026-01-05", 100.0, 99.0) for t in tickers]
    rows.append(_row("T0", "2026-01-06", 100.0, 90.0))
    by_day = real_returns_by_day(pd.DataFrame(rows), tickers)
    assert list(by_day.index) == ["2026-01-02", "2026-01-05"]


def test_single_bad_price_does_not_distort_daily_return():
    rows = [_row(f"T{i}", "2026-01-02", 100.0, 101.0) for i in range(9)]
    rows.append(_row("SPLIT", "2026-01-02", 1.0, 22.0))
    df = pd.DataFrame(rows)
    by_day = real_returns_by_day(df, [r["ticker"] for r in rows])
    assert by_day["2026-01-02"] == pytest.approx(1.0)


def test_calm_streak_grows_with_longer_run_of_no_alert_days():
    df = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 100.0)])
    short = compute_sensors(df, {}, PORTFOLIO, recent_alertas=[False, False])
    long  = compute_sensors(df, {}, PORTFOLIO, recent_alertas=[True] + [False] * 8)
    assert long["CALM_STREAK"] > short["CALM_STREAK"]


def test_direction_and_confidence_columns_are_never_read():
    """
    Design invariant from maomao/README.md: no sensory channel may be
    derived from the ensemble's predicted direction/confidence — only from
    what already happened. A log where direction/confidence are present but
    wrong/garbage must not change any channel's value.
    """
    df1 = pd.DataFrame([_row("AAA.DE", "2026-01-02", 100.0, 101.0)])
    df2 = df1.copy()
    df2["direction"]  = "down"
    df2["confidence"] = 0.99
    assert compute_sensors(df1, {}, PORTFOLIO) == compute_sensors(df2, {}, PORTFOLIO)
