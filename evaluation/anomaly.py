"""
Deteção de anomalia de volatilidade — alimenta o email "Carteira BOT Alerta",
que só é enviado quando algo fora do normal acontece (não todo dia).

Duas pernas, qualquer uma dispara:
  1. Mercado (VIX): transição pra regime alto, ou salto diário grande —
     calculado direto da série já em memória (context_data["vix"]), sem
     precisar guardar estado entre execuções.
  2. Por ativo (só a carteira, não a watchlist): retorno diário maior que
     ALERTA_ATR_MULTIPLICADOR vezes o ATR normal do próprio ativo.
"""
import logging
import json
import numpy as np
import pandas as pd

from config.settings import ALERTA_FILE, ALERTA_ATR_MULTIPLICADOR, ALERTA_VIX_SPIKE_PCT

logger = logging.getLogger(__name__)


def bin_vix_regime(level: float) -> int:
    """Mesmos bins de features/engineering.py:104-106 (0=baixo <15, 1=médio 15-25, 2=alto >25)."""
    if level > 25:
        return 2
    if level > 15:
        return 1
    return 0


def _detect_market(context_data: dict) -> dict:
    vix = context_data.get("vix")
    if vix is None or len(vix.dropna()) < 2:
        return {"transicao": False, "spike": False}

    vix_clean = vix.dropna()
    vix_hoje  = float(vix_clean.iloc[-1])
    vix_ontem = float(vix_clean.iloc[-2])
    regime_hoje  = bin_vix_regime(vix_hoje)
    regime_ontem = bin_vix_regime(vix_ontem)
    variacao_1d  = (vix_hoje / vix_ontem - 1) if vix_ontem else 0.0

    transicao = regime_hoje == 2 and regime_ontem < 2
    spike     = abs(variacao_1d) >= ALERTA_VIX_SPIKE_PCT

    return {
        "vix_hoje": round(vix_hoje, 2),
        "vix_ontem": round(vix_ontem, 2),
        "regime_hoje": regime_hoje,
        "regime_ontem": regime_ontem,
        "variacao_1d": round(variacao_1d, 4),
        "transicao": bool(transicao),
        "spike": bool(spike),
    }


def _detect_tickers(featured_data: dict, portfolio_tickers: list[str]) -> dict:
    resultado = {}
    for ticker in portfolio_tickers:
        df = featured_data.get(ticker)
        if df is None or df.empty or "ATR14" not in df.columns or "ret_1d" not in df.columns:
            continue
        close = float(df["Close"].iloc[-1])
        atr   = float(df["ATR14"].iloc[-1])
        if close <= 0 or np.isnan(atr):
            continue
        atr_pct      = atr / close
        retorno_1d   = float(df["ret_1d"].iloc[-1])
        multiplo_atr = abs(retorno_1d) / atr_pct if atr_pct > 0 else 0.0

        resultado[ticker] = {
            "retorno_1d":  round(retorno_1d, 4),
            "atr_pct":     round(atr_pct, 4),
            "multiplo_atr": round(multiplo_atr, 2),
            "alerta":      bool(multiplo_atr >= ALERTA_ATR_MULTIPLICADOR),
        }
    return resultado


def detect_anomaly(context_data: dict, featured_data: dict, portfolio_tickers: list[str]) -> dict:
    market  = _detect_market(context_data)
    tickers = _detect_tickers(featured_data, portfolio_tickers)

    market_alerta = market.get("transicao", False) or market.get("spike", False)
    tickers_em_alerta = [t for t, r in tickers.items() if r["alerta"]]

    motivos = []
    if market.get("transicao"):
        motivos.append(f"VIX transitou pra regime alto ({market['vix_ontem']}→{market['vix_hoje']})")
    elif market.get("spike"):
        motivos.append(f"VIX saltou {market['variacao_1d']*100:.0f}% num dia ({market['vix_ontem']}→{market['vix_hoje']})")
    for t in tickers_em_alerta:
        motivos.append(f"{t}: retorno {tickers[t]['multiplo_atr']:.1f}x o ATR normal")

    alerta = bool(market_alerta or tickers_em_alerta)

    return {
        "date":    str(pd.Timestamp.now().normalize().date()),
        "alerta":  alerta,
        "market":  market,
        "tickers": tickers,
        "motivos": motivos,
    }


def save_anomaly(result: dict) -> None:
    ALERTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERTA_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if result.get("alerta"):
        logger.info("[Alerta] Anomalia detetada: %s", "; ".join(result.get("motivos", [])))
    else:
        logger.info("[Alerta] Nenhuma anomalia hoje")


def load_anomaly() -> dict:
    if not ALERTA_FILE.exists():
        return {}
    try:
        return json.loads(ALERTA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
