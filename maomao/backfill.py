"""
Replay histórico real do MaoMao: baixa o histórico completo (desde a data
real de início de cada ativo) de cada um dos 7 ETFs da carteira via
yfinance, e revive cada dia em ordem cronológica através do mesmo motor
(sensores + rede LIF + aprendizado) do pipeline em produção — não é dado
inventado, é o mesmo mercado real, só processado em lote em vez de esperar
em tempo real.

Decisões de fidelidade:

- Reusa `evaluation.anomaly.detect_anomaly()` de verdade, não uma
  reimplementação: esse módulo só depende de preço/ATR/VIX, nunca do
  ensemble (ver a docstring dele), então é seguro chamar com qualquer
  recorte histórico causal.
- ATR14/ret_1d são recalculados aqui com a MESMA fórmula de
  `features/engineering.py:build_features` (não a função inteira, que
  também calcula ~30 outras colunas — SMA, RSI, MACD, contexto de
  SPY/BTC/ouro — que `detect_anomaly` nunca lê, e exigiria baixar contexto
  que este replay não precisa). `tests/test_maomao_backfill.py` trava que
  os dois cálculos dão o mesmo número, pra nunca divergir em silêncio.
- SGLN.L é cotado em GBp (pence); a produção converte pra EUR usando a taxa
  de câmbio do DIA DA EXECUÇÃO como fator constante sobre toda a série
  histórica (ver `data/downloader.py`). Multiplicar uma série toda por uma
  constante não muda retorno percentual nem a razão ATR/close — então o
  replay usa o preço bruto em GBp sem converter, com o mesmo resultado
  percentual que a produção teria.
- `DISAGREEMENT` fica sempre 0.0 em todo o replay (nunca existiu ensemble
  correndo nessas datas) — ver maomao/README.md.

Uso: python -m maomao.backfill   (a partir da raiz do repositório)
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from maomao.brain import Brain
from maomao.sensors import compute_sensors_core
from maomao.learning import maybe_queue_bets, resolve_due_bets
from maomao.state import push_history, _fresh_default
from evaluation.anomaly import detect_anomaly

logger = logging.getLogger(__name__)


def atr14_ret1d(df: pd.DataFrame) -> pd.DataFrame:
    """Mesma fórmula de features/engineering.py:build_features (ATR14, ret_1d)."""
    d = df.copy()
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"]  - d["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR14"]  = tr.rolling(14).mean()
    d["ret_1d"] = d["Close"].pct_change(1)
    return d


def _clean_yf_download(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def download_price_history(ticker: str) -> pd.DataFrame:
    df = _clean_yf_download(yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False))
    return atr14_ret1d(df) if not df.empty else df


def download_vix_history() -> pd.Series:
    df = _clean_yf_download(yf.download("^VIX", period="max", interval="1d", auto_adjust=True, progress=False))
    return df["Close"] if not df.empty else pd.Series(dtype=float)


def replay(price_history: dict[str, pd.DataFrame], vix: pd.Series,
          rng: "np.random.Generator | None" = None) -> dict:
    """
    Núcleo puro e testável do replay: recebe já o histórico de preço (com
    ATR14/ret_1d) por ticker e a série de VIX, devolve o estado final
    (mesmo formato de maomao.state.load_state) depois de reviver cada dia
    em ordem cronológica. Separado de download_* pra poder testar sem rede.

    `rng` é opcional (produção não precisa — o ruído da rede é, de
    propósito, não determinístico dia a dia) mas é exposto pra testes que
    precisam comparar dois replays e esperam exatamente o mesmo ruído nos
    dois (ex: teste de causalidade/no-lookahead).
    """
    all_dates = sorted(set().union(*(df.index for df in price_history.values())))
    if not all_dates:
        return _fresh_default()

    # VIX alinhado ao calendário combinado dos tickers, ffill pra feriados/
    # calendários diferentes — mesma convenção já usada na produção (ver
    # README_Errors.md, "Forward fill para NaN em VIX/SPY").
    vix_aligned = vix.reindex(all_dates, method="ffill")

    state = _fresh_default()
    brain = Brain(rng=rng)
    by_day_values: dict[str, float] = {}
    vix_prev = None

    for i, date in enumerate(all_dates):
        available = [
            t for t, df in price_history.items()
            if date in df.index and pd.notna(df.loc[date, "ATR14"]) and pd.notna(df.loc[date, "ret_1d"])
        ]
        vix_today = vix_aligned.iloc[i]
        if not available or pd.isna(vix_today):
            continue

        featured_data = {
            t: pd.DataFrame({
                "Close":  [float(price_history[t].loc[date, "Close"])],
                "ATR14":  [float(price_history[t].loc[date, "ATR14"])],
                "ret_1d": [float(price_history[t].loc[date, "ret_1d"])],
            }, index=[date])
            for t in available
        }
        vix_series = pd.Series([vix_prev, vix_today]) if vix_prev is not None else pd.Series([vix_today])
        vix_prev = float(vix_today)

        anomaly = detect_anomaly({"vix": vix_series}, featured_data, available, as_of=date)

        day_rets = [float(price_history[t].loc[date, "ret_1d"]) * 100 for t in available]
        by_day_values[str(date.date())] = float(np.median(day_rets))
        by_day = pd.Series(by_day_values)

        sensors = compute_sensors_core(by_day, anomaly, state["recent_alertas"], disagreement=0.0)
        if sensors["date"] is None:
            continue

        state["pending"], resolved = resolve_due_bets(state["pending"], by_day, brain)
        mood = brain.step(sensors)
        state["pending"] = maybe_queue_bets(state["pending"], sensors["date"], mood, sensors)
        state["learned"] = brain.snapshot_learned()
        state["last_processed_date"] = sensors["date"]
        push_history(state, sensors["date"], mood, sensors["ANOMALY"] >= 1.0)

    logger.info(
        "Replay: %d dias de mercado processados (%s -> %s), %d apostas ainda pendentes, aprendido=%s",
        len(all_dates),
        all_dates[0].date(), all_dates[-1].date(),
        len(state["pending"]), state["learned"],
    )
    return state


def run(tickers: list[str]) -> dict:
    """Ponto de entrada: baixa o histórico real via yfinance e roda o replay."""
    price_history = {}
    for t in tickers:
        df = download_price_history(t)
        if df.empty:
            logger.warning("%s: sem histórico devolvido pelo yfinance — ignorado no replay", t)
            continue
        price_history[t] = df
        logger.info("%s: %d dias de histórico (%s -> %s)", t, len(df), df.index[0].date(), df.index[-1].date())

    vix = download_vix_history()
    return replay(price_history, vix)


def run_market_backfill(watchlist: list[str]) -> None:
    """
    Pré-treina o humor de MERCADO sobre os ~6 meses que já temos em
    output/predictions_log.csv (não baixa histórico novo de 500+ tickers —
    seria um esforço bem maior, e o usuário pediu pra usar o que já temos
    documentado nesse caso). Só a série de VIX é baixada (pequena, rápida),
    pra reconstruir o anomaly['market'] de cada dia real — o pipeline só
    guarda o anomalia.json de HOJE, não um histórico.

    Reusa maomao.daily.run_market_step tal e qual o pipeline real chamaria,
    só que em loop sobre o histórico causal em vez de uma vez em tempo real
    — por isso também aplica o mesmo idempotência/persistência de sempre.

    Limitação documentada: VOL_SPIKE fica sempre 0 nesse backfill, porque
    anomaly['tickers'] (múltiplo de ATR por ativo) não existe pro watchlist
    inteiro no histórico — só é calculado ao vivo, só pros 7 ETFs da
    carteira. Ver maomao/README.md.
    """
    import pandas as pd
    from config.settings import PRED_LOG
    from maomao.daily import run_market_step
    import maomao.state as maomao_state

    # Parte do zero: sobre um estado existente, o replay aplicaria o aprendizado duas vezes.
    maomao_state.save_state(_fresh_default(), maomao_state.MARKET_STATE_FILE)

    df_log = pd.read_csv(PRED_LOG)
    vix = download_vix_history()

    dates = sorted(df_log["target_date"].dropna().unique())
    logger.info("Backfill de mercado: %d dias reais em predictions_log.csv (%s -> %s)",
                len(dates), dates[0], dates[-1])

    vix_aligned = vix.reindex(pd.to_datetime(dates), method="ffill")

    for i, d in enumerate(dates):
        partial = df_log[df_log["target_date"] <= d]
        vix_today = vix_aligned.iloc[i]
        vix_series = vix_aligned.iloc[max(0, i - 1): i + 1].dropna()
        anomaly = detect_anomaly({"vix": vix_series}, {}, []) if pd.notna(vix_today) else {"alerta": False, "market": {}}
        run_market_step(partial, anomaly, watchlist)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from data.storage import load_portfolio_config, load_my_tickers, load_watchlist, build_ticker_order, portfolio_tickers
    from maomao.state import save_state, load_state, PORTFOLIO_STATE_FILE, MARKET_STATE_FILE

    portfolio = portfolio_tickers(load_portfolio_config())
    final_state = run(portfolio)
    save_state(final_state, PORTFOLIO_STATE_FILE)
    print(f"\n[Carteira] Estado final gravado em {PORTFOLIO_STATE_FILE}")
    print(f"[Carteira] Último dia processado: {final_state['last_processed_date']}")
    print(f"[Carteira] Pesos aprendidos: {final_state['learned']}")

    all_tickers = build_ticker_order(load_my_tickers(), load_watchlist())
    run_market_backfill(all_tickers)
    market_state = load_state(MARKET_STATE_FILE)
    print(f"\n[Mercado] Estado final gravado em {MARKET_STATE_FILE}")
    print(f"[Mercado] Último dia processado: {market_state['last_processed_date']}")
    print(f"[Mercado] Pesos aprendidos: {market_state['learned']}")
