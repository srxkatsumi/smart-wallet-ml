import logging

logger = logging.getLogger(__name__)


def generate_signals(resumo_etoro: list[dict], resultados_ml: dict) -> list[dict]:
    signals = []
    for r in resumo_etoro:
        ticker    = r["ticker"]
        if ticker not in resultados_ml:
            continue

        close     = r["close_eur"]
        breakeven = r["breakeven_eur"]
        alvo15    = r["alvo_15_eur"]
        gp_pct    = r["gp_pct"]
        res       = resultados_ml[ticker]
        df_t      = res["df"]

        rsi     = float(df_t["RSI14"].iloc[-1])
        sma20   = float(df_t["SMA20"].iloc[-1])
        sma50   = float(df_t["SMA50"].iloc[-1])
        vix_now = float(df_t["vix_level"].iloc[-1]) if "vix_level" in df_t.columns else 20.0
        h1      = res["horizons"][1]["direction"]
        h2      = res["horizons"][2]["direction"]
        h3      = res["horizons"][3]["direction"]
        conf    = res["horizons"][1]["confidence"]

        if h1 == h2 == h3 == "down":
            consenso = "BEARISH nos 3 horizontes"
        elif h1 == h2 == h3 == "up":
            consenso = "BULLISH nos 3 horizontes"
        else:
            consenso = f"Misto (D+1:{h1} D+2:{h2} D+3:{h3})"

        if close >= alvo15 and rsi > 65:
            recomendacao = "VENDER"
            motivo       = "alvo +15% atingido + RSI elevado"
        elif close >= breakeven and h1 == h2 == "down" and conf > 0.60:
            recomendacao = "ATENÇÃO"
            motivo       = "acima do breakeven mas ML bearish em D+1 e D+2"
        elif gp_pct < -20 and h1 == h2 == h3 == "down":
            recomendacao = "ATENÇÃO"
            motivo       = "queda prolongada + consenso bearish nos 3 horizontes"
        else:
            recomendacao = "MANTER"
            motivo       = "critérios de saída não atingidos"

        signals.append({
            "ticker":        ticker,
            "nome":          r["nome"],
            "close":         close,
            "breakeven":     breakeven,
            "alvo15":        alvo15,
            "rsi":           rsi,
            "sma_trend":     "up" if sma20 > sma50 else "down",
            "vix":           vix_now,
            "consenso":      consenso,
            "recomendacao":  recomendacao,
            "motivo":        motivo,
        })
        logger.info("%s: %s — %s", ticker, recomendacao, motivo)

    return signals
