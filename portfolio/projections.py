import logging
from config.settings import HORIZONTE_ANOS, TAXA_CRESCIMENTO_BASE

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


def _taxa_historica(ticker: str, resultados_ml: dict) -> float:
    if ticker not in resultados_ml:
        return TAXA_CRESCIMENTO_BASE["base"]
    df_t   = resultados_ml[ticker]["df"]
    n_dias = (df_t.index[-1] - df_t.index[0]).days
    if n_dias <= 0:
        return TAXA_CRESCIMENTO_BASE["base"]
    taxa = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0]) ** (365 / n_dias) - 1
    return max(-0.30, min(taxa, 0.50))


def calculate_etoro_projections(resumo_etoro: list[dict],
                                 resultados_ml: dict) -> list[dict]:
    rows = []
    for r in resumo_etoro:
        ticker = r["ticker"]
        atual  = r["atual_eur"]
        taxa   = _taxa_historica(ticker, resultados_ml)
        projs  = {a: _valor_futuro_dca(atual, 0, taxa, a) for a in HORIZONTE_ANOS}
        rows.append({
            "ticker":  ticker,
            "nome":    r["nome"],
            "atual":   atual,
            "taxa":    taxa,
            "projs":   projs,
            "metodo":  f"hist={taxa:+.0%}",
        })
        logger.info("%s: taxa_hist=%.1f%% → 10a=%.0f€", ticker, taxa * 100, projs[10])
    return rows


def calculate_etf_projections(resumo_etfs: list[dict],
                               resultados_ml: dict) -> list[dict]:
    rows = []
    for r in resumo_etfs:
        ticker = r["ticker"]
        atual  = r["atual_eur"]
        aporte = r["aporte_mensal"]
        taxa   = _taxa_historica(ticker, resultados_ml)
        projs  = {a: _valor_futuro_dca(atual, aporte, taxa, a) for a in HORIZONTE_ANOS}
        rows.append({
            "ticker":  ticker,
            "nome":    r["nome"],
            "atual":   atual,
            "aporte":  aporte,
            "taxa":    taxa,
            "projs":   projs,
            "label":   f"hist={taxa:+.0%} +{aporte}€/m",
        })
    return rows


def calculate_etf_scenarios(resumo_etfs: list[dict],
                             resultados_ml: dict) -> dict:
    total_etf   = sum(r["atual_eur"] for r in resumo_etfs)
    total_aport = sum(r["aporte_mensal"] for r in resumo_etfs)
    scenarios   = {}
    for cenario, taxa in TAXA_CRESCIMENTO_BASE.items():
        scenarios[cenario] = {
            "taxa": taxa,
            "projs": {a: _valor_futuro_dca(total_etf, total_aport, taxa, a)
                      for a in HORIZONTE_ANOS},
        }
    return scenarios
