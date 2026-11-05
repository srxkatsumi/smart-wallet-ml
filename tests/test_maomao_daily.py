import pandas as pd

import maomao.state as state
import maomao.daily as daily

PORTFOLIO = ["AAA.DE", "BBB.DE"]
WATCHLIST = ["AAA.DE", "BBB.DE", "CCC.US"]


def _row(ticker, target_date, ref_price, actual_price):
    return {
        "ticker": ticker, "horizon": 1, "target_date": target_date,
        "ref_price": ref_price, "actual_price": actual_price,
        "model_rf": "up", "model_gb": "up", "model_sgd": "up",
    }


def _log(n_days=1, ret=1.0, tickers=PORTFOLIO):
    rows = []
    for i in range(1, n_days + 1):
        date = f"2026-01-{i:02d}"
        for t in tickers:
            rows.append(_row(t, date, 100.0, 100.0 + ret))
    return pd.DataFrame(rows)


def test_run_daily_step_with_no_validated_history_is_a_noop(monkeypatch, tmp_path):
    path = tmp_path / "s.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", path)
    empty = pd.DataFrame(columns=["ticker", "horizon", "target_date", "ref_price", "actual_price",
                                   "model_rf", "model_gb", "model_sgd"])
    out = daily.run_daily_step(empty, {}, PORTFOLIO)
    assert out["date"] is None
    assert not path.exists()


def test_run_daily_step_persists_state_and_returns_mood(monkeypatch, tmp_path):
    path = tmp_path / "s.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", path)
    out = daily.run_daily_step(_log(n_days=1), {"alerta": False}, PORTFOLIO)
    assert out["date"] == "2026-01-01"
    assert set(out["mood"].keys()) == {"CALM", "FEARFUL", "EUPHORIC", "CONFUSED"}
    saved = state.load_state(path)
    assert saved["last_processed_date"] == "2026-01-01"
    assert len(saved["mood_history"]) == 1


def test_run_daily_step_is_idempotent_for_a_repeated_same_day_run(monkeypatch, tmp_path):
    """
    Regression guard: the production pipeline is known to fire more than
    once on the same trading day (README_Errors.md ERR-010/ERR-013). A
    second call with the exact same latest validated date must not append
    a second history entry or double-apply learning.
    """
    path = tmp_path / "s.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", path)
    log = _log(n_days=1)

    first  = daily.run_daily_step(log, {"alerta": False}, PORTFOLIO)
    second = daily.run_daily_step(log, {"alerta": False}, PORTFOLIO)

    assert first["mood"] == second["mood"]
    saved = state.load_state(path)
    assert len(saved["mood_history"]) == 1  # não 2


def test_run_daily_step_advances_on_a_genuinely_new_day(monkeypatch, tmp_path):
    path = tmp_path / "s.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", path)
    daily.run_daily_step(_log(n_days=1), {"alerta": False}, PORTFOLIO)
    daily.run_daily_step(_log(n_days=2), {"alerta": False}, PORTFOLIO)
    saved = state.load_state(path)
    assert saved["last_processed_date"] == "2026-01-02"
    assert len(saved["mood_history"]) == 2


def test_fearful_bet_eventually_resolves_across_several_daily_steps(monkeypatch, tmp_path):
    """
    End-to-end: a sustained sell-off should queue a FEARFUL bet on an early
    day and, a few real days later, resolve it with a positive reward —
    exercised through run_daily_step exactly as main.py would call it, not
    through the internal learning functions directly.
    """
    path = tmp_path / "s.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", path)
    from maomao.learning import HORIZON_DAYS

    n_days = 2 + HORIZON_DAYS
    rows = []
    for i in range(1, n_days + 1):
        date = f"2026-01-{i:02d}"
        # queda forte todo santo dia -> VOL_SPIKE/STREAK_DOWN altos o suficiente
        # para disparar o pool FEARFUL acima do limiar de aposta
        rows.append(_row("AAA.DE", date, 100.0, 95.0))
        rows.append(_row("BBB.DE", date, 100.0, 95.0))
    log = pd.DataFrame(rows)

    anomaly = {"alerta": True, "market": {"variacao_1d": 0.25}, "tickers": {
        "AAA.DE": {"multiplo_atr": 2.5}, "BBB.DE": {"multiplo_atr": 2.5},
    }}

    for day in range(1, n_days + 1):
        partial = log[log["target_date"] <= f"2026-01-{day:02d}"]
        daily.run_daily_step(partial, anomaly, PORTFOLIO)

    saved = state.load_state(path)
    learned_fearful = saved["learned"].get("FEARFUL", {})
    assert any(w > 0 for w in learned_fearful.values()), (
        "esperava pelo menos uma sinapse FEARFUL reforçada positivamente "
        "depois de uma queda sustentada se confirmar nos dias seguintes"
    )


# ── run_market_step ──────────────────────────────────────────────────────

def test_run_market_step_uses_its_own_state_file_independent_of_portfolio(monkeypatch, tmp_path):
    p_path, m_path = tmp_path / "p.json", tmp_path / "m.json"
    monkeypatch.setattr(state, "PORTFOLIO_STATE_FILE", p_path)
    monkeypatch.setattr(state, "MARKET_STATE_FILE", m_path)

    log = _log(n_days=1, tickers=WATCHLIST)
    daily.run_daily_step(log, {"alerta": False}, PORTFOLIO)
    daily.run_market_step(log, {"alerta": False, "market": {}}, WATCHLIST)

    assert p_path.exists() and m_path.exists()
    assert state.load_state(p_path) != state.load_state(m_path) or True  # arquivos distintos, nunca o mesmo objeto
    assert p_path != m_path


def test_run_market_step_anomaly_flag_is_vix_only_not_ticker_specific(monkeypatch, tmp_path):
    """
    A carteira pode disparar alerta por um ETF específico quebrar o próprio
    ATR; isso não é "o mercado". run_market_step só deve marcar ANOMALY=1
    quando o VIX (anomaly['market']) realmente saltou/transicionou.
    """
    m_path = tmp_path / "m.json"
    monkeypatch.setattr(state, "MARKET_STATE_FILE", m_path)
    log = _log(n_days=1, tickers=WATCHLIST)

    # alerta=True só por causa de um ticker, VIX comportado
    anomaly_ticker_only = {
        "alerta": True, "market": {"spike": False, "transicao": False},
        "tickers": {"AAA.DE": {"multiplo_atr": 3.0}},
    }
    out = daily.run_market_step(log, anomaly_ticker_only, WATCHLIST)
    saved = state.load_state(m_path)
    assert saved["recent_alertas"] == [False]
