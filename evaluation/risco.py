"""
Contexto de risco do mercado e risco da carteira, para o email diário.

Nada aqui depende de quanto está investido em cada ativo: o risco é lido ativo
por ativo, contra o próprio histórico, e o estado geral da carteira resume
quantos ativos estão sob tensão e se eles estão se movendo juntos.

Contexto: VIX (nível e variação), juro de 10 anos dos EUA (^TNX), EUR/USD e ouro.
Por ativo, em euro (ações em dólar convertidas pelo EUR/USD de cada dia):
  - volatilidade de 20 dias, anualizada, e a razão contra a do último ano
    (acima de RAZAO_ELEVADA = mais agitado que o normal DESSE ativo);
  - queda desde o pico de 1 ano;
  - VaR 95% de 1 dia histórico (em 1 de cada 20 dias cai mais que isso).
Geral:
  - nível: "baixo", "normal" ou "elevado", pela fração de ativos agitados,
    pelo regime do VIX e pela correlação média;
  - correlação média de 60 dias contra 1 ano (diversificação).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIAS_ANO      = 252
JANELA_VOL    = 20
JANELA_CORR   = 60
NIVEL_VAR     = 0.95
RAZAO_ELEVADA = 1.3    # vol 20d 30% acima da do último ano
RAZAO_BAIXA   = 0.8
CORR_SUBIU    = 0.10   # correlação média 60d acima da de 1 ano por mais que isto


def _hist(ticker: str, period: str = "3mo") -> pd.Series | None:
    import yfinance as yf
    try:
        d = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s if len(s) >= 2 else None
    except Exception as e:
        logger.info("Histórico de %s indisponível: %s", ticker, e)
        return None


def _var_1d(s: pd.Series | None) -> tuple[float, float] | None:
    if s is None or len(s.dropna()) < 2:
        return None
    s = s.dropna()
    return float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)


def contexto_mercado(context_data: dict, baixar=_hist) -> dict:
    """{"vix": (nível, var), "juro10": (nível %, var em p.p.), "eurusd": (nível, var), "ouro": (nível, var)}"""
    out = {"vix": _var_1d((context_data or {}).get("vix")),
           "ouro": _var_1d((context_data or {}).get("gold"))}
    tnx = baixar("^TNX")
    out["juro10"] = (float(tnx.iloc[-1]), float(tnx.iloc[-1] - tnx.iloc[-2])) if tnx is not None and len(tnx) >= 2 else None
    out["eurusd"] = _var_1d(baixar("EURUSD=X"))
    return out


def _closes_em_euro(resultados_ml: dict, carteira: list, usd_eur: pd.Series | None) -> pd.DataFrame:
    cols = {}
    for t in carteira:
        df = (resultados_ml.get(t) or {}).get("df")
        if df is None or "Close" not in df:
            continue
        c = df["Close"].dropna().copy()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        if "." not in t and usd_eur is not None and not usd_eur.empty:
            c = c * usd_eur.reindex(c.index, method="ffill").bfill()
        cols[t] = c
    return pd.DataFrame(cols).sort_index()


def _nivel(frac_agitados: float, vix: float | None, corr_subiu: bool) -> str:
    pontos = 0
    pontos += 2 if frac_agitados >= 0.5 else (1 if frac_agitados >= 0.25 else 0)
    if vix is not None:
        pontos += 2 if vix > 25 else (0 if vix < 15 else 1)
    pontos += 1 if corr_subiu else 0
    # baixo só sem nenhum sinal de tensão; VIX médio sozinho já é "normal"
    return "elevado" if pontos >= 3 else ("baixo" if pontos == 0 else "normal")


def risco_carteira(resultados_ml: dict, carteira: list, vix_nivel: float | None = None,
                   baixar=_hist) -> dict | None:
    eurusd = baixar("EURUSD=X", "2y")
    usd_eur = (1 / eurusd) if eurusd is not None else None
    precos = _closes_em_euro(resultados_ml, carteira, usd_eur)
    if precos.empty or len(precos) < JANELA_VOL + 1:
        return None
    ret = precos.pct_change(fill_method=None)

    por_ativo = []
    for t in precos.columns:
        r = ret[t].dropna()
        c = precos[t].dropna().tail(DIAS_ANO)
        if len(r) < JANELA_VOL:
            continue
        vol20 = float(r.tail(JANELA_VOL).std(ddof=1) * np.sqrt(DIAS_ANO))
        vol_ano = float(r.tail(DIAS_ANO).std(ddof=1) * np.sqrt(DIAS_ANO))
        razao = vol20 / vol_ano if vol_ano > 0 else 1.0
        from research.volatilidade import ewma_var_ultima
        v_ewma = ewma_var_ultima(r)
        por_ativo.append({
            "ticker": t,
            "sigma_amanha": float(np.sqrt(v_ewma)) if v_ewma == v_ewma else None,
            "vol20": vol20,
            "vol_ano": vol_ano,
            "razao": razao,
            "estado": "agitado" if razao >= RAZAO_ELEVADA else ("calmo" if razao <= RAZAO_BAIXA else "normal"),
            "queda_pico": float(c.iloc[-1] / c.max() - 1),
            "var95": float(-np.quantile(r.tail(DIAS_ANO), 1 - NIVEL_VAR)),
        })
    if not por_ativo:
        return None
    por_ativo.sort(key=lambda a: -a["razao"])

    corr60 = corr_ano = None
    if precos.shape[1] >= 2:
        tri = np.triu_indices(precos.shape[1], 1)
        corr60 = float(np.nanmean(ret.tail(JANELA_CORR).corr().values[tri]))
        corr_ano = float(np.nanmean(ret.tail(DIAS_ANO).corr().values[tri]))
    corr_subiu = corr60 is not None and corr_ano is not None and corr60 - corr_ano > CORR_SUBIU

    agitados = [a["ticker"] for a in por_ativo if a["estado"] == "agitado"]
    frac = len(agitados) / len(por_ativo)
    return {
        "nivel": _nivel(frac, vix_nivel, corr_subiu),
        "agitados": agitados,
        "n_ativos": len(por_ativo),
        "corr60": corr60,
        "corr_ano": corr_ano,
        "corr_subiu": corr_subiu,
        "por_ativo": por_ativo,
    }
