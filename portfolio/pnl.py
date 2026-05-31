import logging
from data.downloader import to_eur

logger = logging.getLogger(__name__)


def calculate_etoro_pnl(etoro: list[dict], resultados_ml: dict,
                         EUR_USD: float) -> tuple[list[dict], dict]:
    resumo = []
    for ativo in etoro:
        ticker = ativo["ticker"]
        if ticker not in resultados_ml:
            logger.warning("%s: sem dados de mercado", ticker)
            continue

        close         = resultados_ml[ticker]["close_now"]
        moeda         = ativo["moeda"]
        fee           = ativo["fee_euro"]
        uni           = ativo["unidades"]
        pa            = ativo["preco_abertura"]
        pa_eur        = pa / EUR_USD if moeda == "USD" else pa
        close_eur     = close / EUR_USD if moeda == "USD" else close
        investido_eur = pa_eur * uni
        atual_eur     = close_eur * uni
        gp_eur        = atual_eur - investido_eur
        gp_pct        = gp_eur / investido_eur * 100 if investido_eur else 0
        breakeven_eur = pa_eur + (fee / uni if uni > 0 else 0)
        alvo_15_eur   = pa_eur * 1.15 + (fee / uni if uni > 0 else 0)

        resumo.append({
            "ticker":        ticker,
            "nome":          ativo["nome"],
            "close_eur":     close_eur,
            "investido_eur": investido_eur,
            "atual_eur":     atual_eur,
            "gp_eur":        gp_eur,
            "gp_pct":        gp_pct,
            "breakeven_eur": breakeven_eur,
            "alvo_15_eur":   alvo_15_eur,
            "fee":           fee,
            "unidades":      uni,
        })

    totals = {
        "investido": sum(r["investido_eur"] for r in resumo),
        "atual":     sum(r["atual_eur"]     for r in resumo),
        "gp":        sum(r["gp_eur"]        for r in resumo),
    }
    logger.info("eToro P&L: investido=%.2f€ atual=%.2f€ G/P=%+.2f€",
                totals["investido"], totals["atual"], totals["gp"])
    return resumo, totals


def calculate_etf_pnl(etf_acumulacao: list[dict], resultados_ml: dict) -> tuple[list[dict], dict]:
    resumo = []
    for etf in etf_acumulacao:
        ticker = etf["ticker"]
        if ticker not in resultados_ml:
            logger.warning("%s: sem dados de mercado", ticker)
            continue

        close_raw     = resultados_ml[ticker]["close_now"]
        moeda         = etf["moeda"]
        uni           = etf["unidades"]
        close_eur     = to_eur(close_raw, "GBP", etf.get("gbp_pence", False)) \
                        if moeda == "GBP" else close_raw
        atual_eur     = close_eur * uni
        investido_eur = etf["euros_investidos"]
        gp_eur        = atual_eur - investido_eur
        gp_pct        = gp_eur / investido_eur * 100 if investido_eur else 0

        resumo.append({
            "ticker":        ticker,
            "nome":          etf["nome"],
            "close_eur":     close_eur,
            "investido_eur": investido_eur,
            "atual_eur":     atual_eur,
            "gp_eur":        gp_eur,
            "gp_pct":        gp_pct,
            "unidades":      uni,
            "aporte_mensal": etf["aporte_mensal"],
        })

    totals = {
        "investido": sum(r["investido_eur"] for r in resumo),
        "atual":     sum(r["atual_eur"]     for r in resumo),
        "gp":        sum(r["gp_eur"]        for r in resumo),
    }
    logger.info("ETFs P&L: investido=%.2f€ atual=%.2f€ G/P=%+.2f€",
                totals["investido"], totals["atual"], totals["gp"])
    return resumo, totals


def expand_etoro_lots(etoro: list[dict], resultados_ml: dict, EUR_USD: float) -> list[dict]:
    """Returns one dict per purchase lot for the eToro price forecast table."""
    rows = []
    for ativo in etoro:
        ticker = ativo["ticker"]
        if ticker not in resultados_ml:
            continue
        close_eur = resultados_ml[ticker]["close_now"]
        moeda = ativo["moeda"]
        if moeda == "USD":
            close_eur = close_eur / EUR_USD

        lotes = ativo.get("lotes") or []
        if not lotes:
            pa = ativo["preco_abertura"]
            pa_eur = pa / EUR_USD if moeda == "USD" else pa
            fee = ativo.get("fee_euro", 0.0)
            uni = ativo["unidades"]
            alvo = pa_eur * 1.15 + (fee / uni if uni > 0 else 0)
            rows.append({
                "ticker":         ticker,
                "nome":           ativo["nome"],
                "data_compra":    None,
                "preco_compra_eur": pa_eur,
                "close_eur":      close_eur,
                "alvo_15_eur":    alvo,
            })
            continue

        fee = ativo.get("fee_euro", 0.0)
        total_uni = ativo["unidades"]
        for lote in lotes:
            pa = lote["preco_abertura"]
            uni = lote["unidades"]
            pa_eur = pa / EUR_USD if moeda == "USD" else pa
            alvo = pa_eur * 1.15 + (fee / total_uni if total_uni > 0 else 0)
            rows.append({
                "ticker":           ticker,
                "nome":             ativo["nome"],
                "data_compra":      lote.get("data_compra"),
                "preco_compra_eur": pa_eur,
                "close_eur":        close_eur,
                "alvo_15_eur":      alvo,
            })
    return rows


def expand_etf_lots(etf_acumulacao: list[dict], resultados_ml: dict) -> list[dict]:
    """Returns one dict per purchase lot for the ETF section."""
    from data.downloader import to_eur
    rows = []
    for etf in etf_acumulacao:
        ticker = etf["ticker"]
        if ticker not in resultados_ml:
            continue
        close_raw = resultados_ml[ticker]["close_now"]
        moeda = etf["moeda"]
        close_eur = to_eur(close_raw, "GBP", etf.get("gbp_pence", False)) \
                    if moeda == "GBP" else close_raw

        lotes = etf.get("lotes") or []
        if not lotes:
            euros_inv = etf["euros_investidos"]
            uni = etf["unidades"]
            preco_eur = euros_inv / uni if uni > 0 else 0.0
            rows.append({
                "ticker":           ticker,
                "nome":             etf["nome"],
                "data_compra":      None,
                "preco_unidade_eur": preco_eur,
                "close_eur":        close_eur,
            })
            continue

        for lote in lotes:
            rows.append({
                "ticker":           ticker,
                "nome":             etf["nome"],
                "data_compra":      lote.get("data_compra"),
                "preco_unidade_eur": lote["preco_unidade_eur"],
                "close_eur":        close_eur,
            })
    return rows
