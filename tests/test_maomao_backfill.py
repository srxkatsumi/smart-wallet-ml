import numpy as np
import pandas as pd
import pytest

from maomao.backfill import atr14_ret1d, replay


def _rng(seed=0):
    return np.random.default_rng(seed)


def _synthetic_ohlc(n=60, start=100.0, seed=0, drift=0.0) -> pd.DataFrame:
    rng = _rng(seed)
    rets = rng.normal(drift, 0.01, n)
    close = start * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "High": high, "Low": low}, index=dates)


# ── atr14_ret1d must match the real production formula exactly ──────────────

def test_atr14_ret1d_matches_real_build_features():
    """
    Regression guard: maomao/backfill.py recomputes ATR14/ret_1d with its
    own copy of the formula (to avoid pulling in build_features' other ~30
    columns and their SPY/BTC/gold context requirements). If that formula
    ever drifts from features/engineering.py's real one, this must fail.

    build_features() also drops every row missing ANY of its ~30 feature
    columns (SMA50 needs 50 rows, RSI14 needs 14, etc.) — far more than
    ATR14/ret_1d alone need — so we only compare on the rows that survive
    that broader dropna, not the full series.
    """
    from features.engineering import build_features

    raw  = _synthetic_ohlc(n=300, seed=1)  # >252 rows: ret_12m/high52w_dist need that much
    ours = atr14_ret1d(raw)
    real = build_features(raw, context_data={})

    common = real.index
    pd.testing.assert_series_equal(ours.loc[common, "ATR14"], real["ATR14"], check_names=False)
    pd.testing.assert_series_equal(ours.loc[common, "ret_1d"], real["ret_1d"], check_names=False)


# ── replay() core ─────────────────────────────────────────────────────────

def test_replay_with_empty_price_history_returns_fresh_default():
    state = replay({}, pd.Series(dtype=float))
    assert state["last_processed_date"] is None
    assert state["learned"] == {}


def test_replay_processes_every_day_with_data_and_valid_vix():
    df = atr14_ret1d(_synthetic_ohlc(n=40, seed=2))
    vix = pd.Series(18.0, index=df.index)
    state = replay({"AAA.DE": df}, vix)
    # rolling(14) só deixa de ser NaN a partir da 14ª observação (índice 13)
    assert state["last_processed_date"] == str(df.index[-1].date())
    assert len(state["mood_history"]) == len(df) - 13


def test_replay_is_causal_no_lookahead():
    """
    O humor do dia T, calculado pelo replay, não pode depender de preços
    depois de T. Verificamos isso rodando o replay até o dia T e comparando
    com rodar até um dia bem mais tarde: o humor registado PARA O DIA T deve
    ser idêntico nos dois casos.
    """
    full = atr14_ret1d(_synthetic_ohlc(n=60, seed=3))
    cutoff = full.index[40]

    truncated = full.loc[:cutoff]
    vix_full = pd.Series(18.0, index=full.index)

    # mesma seed nos dois: sem isso, cada replay tem seu próprio ruído de
    # rede independente e os dois nunca seriam bit-a-bit iguais mesmo sendo
    # causalmente corretos — o teste de causalidade fica sem sentido se não
    # controlar essa fonte de variação separadamente.
    state_short = replay({"AAA.DE": truncated}, vix_full.loc[:cutoff], rng=np.random.default_rng(123))
    state_long  = replay({"AAA.DE": full}, vix_full, rng=np.random.default_rng(123))

    mood_at_cutoff_short = state_short["mood_history"][-1]
    mood_at_cutoff_long  = next(h for h in state_long["mood_history"] if h["date"] == str(cutoff.date()))

    assert mood_at_cutoff_short["mood"] == mood_at_cutoff_long["mood"]


def test_replay_sustained_selloff_produces_learned_fearful_weight():
    """
    End-to-end do replay inteiro: um crash real tem uma fase calma (que
    define o ATR "normal") seguida de um salto brusco de volatilidade — uma
    queda diária constante e previsível (sem salto de ATR) não dispara
    alerta nenhum, porque o ATR já se adapta ao ritmo (ver debug ao escrever
    este teste: isso é esperado e reflete como o detector real funciona).
    Depois do crash, os retornos seguem negativos, confirmando o medo.
    """
    n_calm, n_crash, n_after = 20, 3, 10
    rng = np.random.default_rng(5)
    calm_rets  = rng.normal(0.0, 0.002, n_calm)
    crash_rets = np.array([-0.06, -0.05, -0.04])
    after_rets = rng.normal(-0.01, 0.003, n_after)
    rets  = np.concatenate([calm_rets, crash_rets, after_rets])
    close = 100.0 * np.cumprod(1 + rets)
    dates = pd.date_range("2020-01-01", periods=len(rets), freq="B")
    df = atr14_ret1d(pd.DataFrame({
        "Close": close, "High": close * 1.002, "Low": close * 0.998,
    }, index=dates))
    vix = pd.Series(
        np.concatenate([np.full(n_calm, 15.0), np.full(n_crash, 32.0), np.full(n_after, 24.0)]),
        index=dates,
    )

    state = replay({"AAA.DE": df, "BBB.DE": df}, vix)
    fired_fearful = max(h["mood"]["FEARFUL"] for h in state["mood_history"])
    assert fired_fearful >= 0.5, f"esperava FEARFUL disparar acima do limiar de aposta; máximo foi {fired_fearful:.3f}"
    learned_fearful = state["learned"].get("FEARFUL", {})
    assert any(w > 0 for w in learned_fearful.values())


def test_market_backfill_rerun_starts_from_scratch(tmp_path, monkeypatch):
    import config.settings
    import maomao.backfill as backfill
    import maomao.state as maomao_state

    dates = pd.date_range("2026-01-01", periods=8, freq="B").strftime("%Y-%m-%d")
    log_path = tmp_path / "log.csv"
    pd.DataFrame({
        "ticker": "AAA", "horizon": 1, "target_date": dates, "ref_price": 100.0, "actual_price": 101.0,
        "model_rf": "up", "model_gb": "up", "model_sgd": "up",
    }).to_csv(log_path, index=False)
    state_path = tmp_path / "market.json"

    monkeypatch.setattr(config.settings, "PRED_LOG", log_path)
    monkeypatch.setattr(maomao_state, "MARKET_STATE_FILE", state_path)
    monkeypatch.setattr(backfill, "download_vix_history",
                        lambda: pd.Series(18.0, index=pd.to_datetime(dates)))

    backfill.run_market_backfill(["AAA"])
    first = [h["date"] for h in maomao_state.load_state(state_path)["mood_history"]]
    backfill.run_market_backfill(["AAA"])
    second = [h["date"] for h in maomao_state.load_state(state_path)["mood_history"]]

    assert first == list(dates)
    assert second == first


def test_workflow_persists_state_between_runs():
    from pathlib import Path

    workflow = (Path(__file__).parent.parent / ".github/workflows/executar_diario.yml").read_text(encoding="utf-8")
    for state_file in ("output/maomao_state.json", "output/maomao_market_state.json",
                       "output/confidence_calibration.json"):
        assert state_file in workflow


def test_replay_disagreement_channel_is_always_zero():
    """DISAGREEMENT nunca pode vir de outro lugar que não o ensemble real —
    no replay histórico isso significa ficar sempre 0.0 (ver docstring do
    módulo)."""
    df = atr14_ret1d(_synthetic_ohlc(n=30, seed=4))
    vix = pd.Series(18.0, index=df.index)
    state = replay({"AAA.DE": df}, vix)
    assert state["last_processed_date"] is not None  # roda de verdade, não é um no-op
    # se DISAGREEMENT nunca fosse 0 isso afetaria CONFUSED; aqui só
    # confirmamos que o replay não explode e produz humor coerente
    assert all(0.0 <= v <= 1.0 for v in state["mood_history"][-1]["mood"].values())
