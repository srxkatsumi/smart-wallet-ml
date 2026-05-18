import logging
from config.settings import TAXA_CRESCIMENTO_BASE

logger = logging.getLogger(__name__)


def _valor_futuro_dca(valor_atual: float, aporte_mensal: float,
                      taxa_anual: float, anos: int) -> float:
    taxa_mensal      = (1 + taxa_anual) ** (1 / 12) - 1
    meses            = anos * 12
    capital_crescido = valor_atual * (1 + taxa_anual) ** anos
    if taxa_mensal > 0:
        aportes_futuro = aporte_mensal * (((1 + taxa_mensal) ** meses - 1) / taxa_mensal)
    else:
        aportes_futuro = aporte_mensal * meses
    return capital_crescido + aportes_futuro


def simulate_dca(resumo_etfs: list[dict], resultados_ml: dict) -> list[dict]:
    results = []
    for r in resumo_etfs:
        if r["aporte_mensal"] == 0:
            continue
        ticker = r["ticker"]
        atual  = r["atual_eur"]
        aporte = r["aporte_mensal"]

        if ticker in resultados_ml:
            df_t   = resultados_ml[ticker]["df"]
            n_dias = (df_t.index[-1] - df_t.index[0]).days
            taxa   = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0]) ** (365 / n_dias) - 1 \
                     if n_dias > 0 else TAXA_CRESCIMENTO_BASE["base"]
            taxa   = max(-0.30, min(taxa, 0.50))
        else:
            taxa = TAXA_CRESCIMENTO_BASE["base"]

        horizons = []
        for anos in [1, 2, 3, 5, 10]:
            vf      = _valor_futuro_dca(atual, aporte, taxa, anos)
            aportes = aporte * anos * 12
            rend    = vf - atual - aportes
            horizons.append({
                "anos":    anos,
                "vf":      vf,
                "aportes": aportes,
                "rend":    rend,
            })

        results.append({
            "ticker":   ticker,
            "nome":     r["nome"],
            "atual":    atual,
            "aporte":   aporte,
            "taxa":     taxa,
            "horizons": horizons,
        })
        logger.info("%s DCA: taxa=%.1f%% | 10a=%.0f€",
                    ticker, taxa * 100, horizons[-1]["vf"])
    return results
