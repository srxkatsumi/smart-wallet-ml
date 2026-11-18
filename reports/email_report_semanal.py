"""
Email semanal "Carteira BOT Semanal": sai com a execução do último pregão da
semana (normalmente a de sexta, que chega no sábado de manhã).

Concentra o que muda devagar e não precisa ser lido todo dia:
  1. A semana da carteira: retorno de cada ativo e dos fatores de mercado.
  2. Acurácia da produção na semana e no acumulado, com intervalo de confiança.
  3. As 13 famílias: N, intervalo de confiança, p-valor corrigido (BH) e tendência.
  4. Calibração: confiança dita × acerto real, Brier score por horizonte.
  5. Drift das features e correlações de 60 contra 120 dias.
  6. Placar do ensemble ampliado contra a produção.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import BARCELONA_TZ, OUTPUT_DIR
from data.storage import aligned_rows
from evaluation.confianca import ic_acuracia, faixas_confianca
from evaluation.significance import apply_bh_correction, daily_cluster_stats, MIN_DIAS

logger = logging.getLogger(__name__)

SEMANAL_HTML_REPORT = OUTPUT_DIR / "resumo_semanal.html"
JANELA_TENDENCIA = 20   # dias de referência para a "tendência recente" das famílias

_TH = ("padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;"
       "text-transform:uppercase;border-bottom:1px solid #e6e3dc")
_MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def e_fim_de_semana_de_pregao(ref_date: pd.Timestamp, ticker: str = "PPFB.DE") -> bool:
    """
    True quando o próximo pregão (no calendário real da bolsa do ticker) cai noutra
    semana: a sexta, ou o último dia antes de um feriado no fim da semana (Natal).
    """
    from data.calendars import target_dates
    ref = pd.Timestamp(ref_date).normalize()
    try:
        prox = pd.Timestamp(target_dates(ref, ticker)[1]).normalize()
    except Exception:
        prox = ref + pd.offsets.BDay(1)
    return prox.isocalendar()[1] != ref.isocalendar()[1]


def _pct(v, d=1, sinal=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:+.{d}f}%" if sinal else f"{v*100:.{d}f}%"


def _cor(v):
    return "#1e7a4c" if v is not None and v >= 0 else "#b8453a"


def _secao(titulo: str, sub: str, corpo: str, fundo: str = "#fbfaf7") -> str:
    return f"""
  <div style="padding:24px 36px;border-top:1px solid #efece4;background:{fundo}">
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:17px;font-weight:500;color:#1a1740">{titulo}</div>
    <div style="font-size:11.5px;color:#8a8a8a;margin:3px 0 14px;line-height:1.5">{sub}</div>
    {corpo}
  </div>"""


# ── 1. A semana ──────────────────────────────────────────────────────────

def semana_da_carteira(resultados_ml: dict, carteira: list, context_data: dict, baixar=None) -> dict:
    """Retorno de 5 sessões de cada ativo EM EURO (ações em dólar pelo EUR/USD de cada dia)."""
    from evaluation.risco import _closes_em_euro, _hist
    eurusd = (baixar or _hist)("EURUSD=X", "3mo")
    precos = _closes_em_euro(resultados_ml, carteira, (1 / eurusd) if eurusd is not None else None)
    ativos = {}
    for t in precos.columns:
        c = precos[t].dropna()
        if len(c) < 6:
            continue
        ativos[t] = float(c.iloc[-1] / c.iloc[-6] - 1)
    fatores = {}
    nomes = {"spy": "Ações EUA (SPY)", "gold": "Ouro (GLD)", "btc": "Bitcoin"}
    for k, nome in nomes.items():
        s = (context_data or {}).get(k)
        if s is not None and len(s.dropna()) >= 6:
            s = s.dropna()
            fatores[nome] = float(s.iloc[-1] / s.iloc[-6] - 1)
    vix = (context_data or {}).get("vix")
    vix_txt = None
    if vix is not None and len(vix.dropna()) >= 6:
        v = vix.dropna()
        vix_txt = (float(v.iloc[-6]), float(v.iloc[-1]))
    return {"ativos": ativos, "fatores": fatores, "vix": vix_txt}


def _html_semana(sem: dict) -> str:
    if not sem["ativos"]:
        return ""
    linhas = "".join(
        f'<tr style="border-bottom:1px solid #f0ede5"><td style="padding:6px 8px 6px 0;{_MONO};font-size:12px;font-weight:600;color:#1a1740">{t}</td>'
        f'<td style="padding:6px 0;text-align:right;{_MONO};font-size:12px;font-weight:600;color:{_cor(r)}">{_pct(r)}</td></tr>'
        for t, r in sorted(sem["ativos"].items(), key=lambda kv: -kv[1])
    )
    fat = " &nbsp;·&nbsp; ".join(f'{n} <strong style="color:{_cor(r)}">{_pct(r)}</strong>' for n, r in sem["fatores"].items())
    vix = (f' &nbsp;·&nbsp; VIX {sem["vix"][0]:.1f} → <strong>{sem["vix"][1]:.1f}</strong>' if sem["vix"] else "")
    corpo = f"""
    <table style="border-collapse:collapse;width:100%;max-width:320px">{linhas}</table>
    <div style="margin-top:12px;font-size:11px;color:#5a5a5a;line-height:1.7">Mercado na semana: {fat}{vix}</div>
    <div style="margin-top:6px;font-size:10.5px;color:#a8a39a">Retorno das últimas 5 sessões em euro (a AFRM convertida pelo EUR/USD de cada dia). Os fatores de mercado ficam na moeda deles.</div>"""
    return _secao("A semana da carteira", "Retorno de cada ativo e dos fatores que mais pesam nela", corpo)


# ── 2. Acurácia da produção ──────────────────────────────────────────────

def acuracia_producao(df_log: pd.DataFrame, carteira: list, ref_date: pd.Timestamp) -> dict:
    v = aligned_rows(df_log[df_log["correct"].notna() & df_log["ticker"].isin(carteira)]).copy()
    v["target_date"] = pd.to_datetime(v["target_date"])
    ini_semana = pd.Timestamp(ref_date).normalize() - pd.offsets.BDay(4)
    out = {}
    for h in [1, 2, 3]:
        g = v[v["horizon"] == h]
        sem = g[g["target_date"] >= ini_semana]
        out[h] = {"semana": ic_acuracia(sem["correct"], sem["target_date"]),
                  "total": ic_acuracia(g["correct"], g["target_date"])}
    return out


def _fmt_ic(ic: dict) -> str:
    if ic["acc"] is None:
        return '<span style="color:#ccc">-</span>'
    return (f'<strong>{ic["acc"]*100:.0f}%</strong> <span style="color:#a0a0a0;font-size:10px">'
            f'n={ic["n"]} · IC {ic["lo"]*100:.0f}-{ic["hi"]*100:.0f}% · {ic["n_dias"]}d</span>')


def _html_producao(acc: dict) -> str:
    linhas = "".join(
        f'<tr style="border-bottom:1px solid #f0ede5"><td style="padding:7px 8px 7px 0;font-size:12px;font-weight:600;color:#1a1740">D+{h}</td>'
        f'<td style="padding:7px 8px;font-size:11.5px">{_fmt_ic(acc[h]["semana"])}</td>'
        f'<td style="padding:7px 0 7px 8px;font-size:11.5px">{_fmt_ic(acc[h]["total"])}</td></tr>'
        for h in [1, 2, 3]
    )
    corpo = f"""
    <table style="border-collapse:collapse;width:100%">
      <thead><tr><th style="{_TH};text-align:left;padding-left:0">Horizonte</th><th style="{_TH};text-align:left">Esta semana</th><th style="{_TH};text-align:left">Desde o alinhamento</th></tr></thead>
      <tbody>{linhas}</tbody>
    </table>
    <div style="margin-top:10px;font-size:10.5px;color:#a8a39a;line-height:1.6">Ensemble de produção (RF+GB+SGD), ativos da carteira, só previsões com horizonte alinhado.
    IC 95% com o dia como unidade a partir de 5 dias; abaixo disso, Wilson (otimista).</div>"""
    return _secao("Acurácia da produção", "O que vai para o email diário, na semana e no acumulado", corpo, "#f6f3eb")


# ── 3. As 13 famílias ────────────────────────────────────────────────────

def ranking_familias(research_log: pd.DataFrame, carteira: list) -> list[dict]:
    if research_log is None or research_log.empty:
        return []
    r = research_log[research_log["ticker"].isin(carteira) & research_log["ref_date"].notna()].copy()
    if r.empty:
        return []
    r["ref_date"] = r["ref_date"].astype(str)
    datas = sorted(r["ref_date"].unique())
    recentes = set(datas[-JANELA_TENDENCIA:])
    out = []
    for fam, g in r.groupby("family"):
        linha = {"family": fam}
        for h in [1, 2, 3]:
            col = f"correct_d{h}"
            gh = g[g[col].notna()] if col in g else g.iloc[0:0]
            ic = ic_acuracia(gh[col], gh["ref_date"]) if len(gh) else {"acc": None, "n": 0, "n_dias": 0, "lo": None, "hi": None, "metodo": None}
            linha[f"d{h}"] = ic
        g1 = g[g["correct_d1"].notna()]
        st = daily_cluster_stats(g1["correct_d1"], g1["ref_date"]) if len(g1) else {"p": 1.0}
        linha["p"] = st["p"]
        rec = g1[g1["ref_date"].isin(recentes)]
        linha["acc_recente"] = float(rec["correct_d1"].astype(float).mean()) if len(rec) else None
        out.append(linha)
    apply_bh_correction(out)
    out.sort(key=lambda x: -(x["d1"]["acc"] or 0))
    return out


def _html_familias(rank: list[dict]) -> str:
    if not rank:
        return ""
    from reports.email_report import _NOMES_FAMILIA
    linhas = ""
    for r in rank:
        cells = ""
        for h in [1, 2, 3]:
            ic = r[f"d{h}"]
            cells += (f'<td style="padding:6px 6px;text-align:center;{_MONO};font-size:11px">'
                      + (f'<strong>{ic["acc"]*100:.0f}%</strong><div style="font-size:9px;color:#a0a0a0">{ic["lo"]*100:.0f}-{ic["hi"]*100:.0f}</div>'
                         if ic["acc"] is not None else '<span style="color:#ccc">-</span>') + '</td>')
        p_adj = r.get("p_adj", r.get("p"))
        sig = "★" if r.get("sig") else ""
        linhas += (f'<tr style="border-bottom:1px solid #f0ede5"><td style="padding:6px 6px 6px 0;font-size:11.5px;color:#1a1a1a">'
                   f'{_NOMES_FAMILIA.get(r["family"], r["family"])} {sig}</td>{cells}'
                   f'<td style="padding:6px 0 6px 6px;text-align:right;{_MONO};font-size:11px;color:#5a5a5a">{p_adj:.2f}</td></tr>')
    n_dias = max(r["d1"]["n_dias"] for r in rank)
    corpo = f"""
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%;min-width:420px">
      <thead><tr><th style="{_TH};text-align:left;padding-left:0">Família</th><th style="{_TH};text-align:center">D+1</th><th style="{_TH};text-align:center">D+2</th><th style="{_TH};text-align:center">D+3</th><th style="{_TH};text-align:right">p corr.</th></tr></thead>
      <tbody>{linhas}</tbody>
    </table>
    </div>
    <div style="margin-top:10px;font-size:10.5px;color:#a8a39a;line-height:1.6">
      Acurácia com IC 95% (números menores) nos ativos da carteira, {n_dias} dias com horizonte alinhado.
      "p corr." = p-valor do D+1 contra 50%, corrigido por Benjamini-Hochberg pelas 13 famílias; ★ só abaixo de 0,05.
      {"Com menos de " + str(MIN_DIAS) + " dias, diferenças entre famílias são sobretudo ruído: os intervalos se sobrepõem quase todos." if n_dias < MIN_DIAS else ""}
    </div>"""
    return _secao("As 13 famílias", "Quem acerta mais, e se isso já é distinguível do acaso", corpo)


# ── 4. Calibração ────────────────────────────────────────────────────────

def brier_por_horizonte(df_log: pd.DataFrame) -> dict:
    """Brier da confiança na direção prevista; referência "sempre 50%" = 0,25. BSS > 0 = melhor que ela."""
    v = aligned_rows(df_log[df_log["correct"].notna() & df_log["confidence"].notna()])
    out = {}
    for h in [1, 2, 3]:
        g = v[v["horizon"] == h]
        if g.empty:
            continue
        b = float(((g["confidence"].astype(float) - g["correct"].astype(float)) ** 2).mean())
        out[h] = {"brier": b, "bss": 1 - b / 0.25, "n": len(g)}
    return out


def _html_calibracao(df_log: pd.DataFrame) -> str:
    from reports.email_report import _build_calibracao_html
    tabela = _build_calibracao_html(faixas_confianca(df_log))
    br = brier_por_horizonte(df_log)
    if not tabela and not br:
        return ""
    linhas = " &nbsp;·&nbsp; ".join(
        f'D+{h}: Brier <strong>{v["brier"]:.3f}</strong> (BSS <strong style="color:{_cor(v["bss"])}">{v["bss"]:+.2f}</strong>)'
        for h, v in br.items())
    extra = (f'<div style="padding:0 36px 20px;background:#f6f3eb;font-size:11px;color:#5a5a5a;line-height:1.7">{linhas}<br>'
             f'<span style="font-size:10.5px;color:#a8a39a">Brier = erro quadrático entre a confiança e o acerto (0 = perfeito). '
             f'BSS compara com dizer sempre 50%: acima de 0 a confiança ajuda, abaixo de 0 atrapalha.</span></div>')
    return tabela + extra


# ── 5. Correlações 60 × 120 dias ─────────────────────────────────────────

def correlacoes_60_120(resultados_ml: dict, carteira: list) -> dict | None:
    closes = {t: resultados_ml[t]["df"]["Close"] for t in carteira
              if t in resultados_ml and "df" in resultados_ml[t]}
    if len(closes) < 2:
        return None
    ret = pd.DataFrame(closes).pct_change(fill_method=None)
    c60, c120 = ret.tail(60).corr(), ret.tail(120).corr()
    pares = []
    ts = list(c60.columns)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ts[i], ts[j]
            pares.append((a, b, float(c60.loc[a, b]), float(c120.loc[a, b])))
    tri = np.triu_indices(len(ts), 1)
    return {"pares": sorted(pares, key=lambda p: -abs(p[2] - p[3])),
            "media60": float(np.nanmean(c60.values[tri])), "media120": float(np.nanmean(c120.values[tri]))}


def _html_correlacoes(cor: dict | None, resultados_ml: dict, carteira: list) -> str:
    if not cor:
        return ""
    from reports.email_report import _build_corr_section
    linhas = "".join(
        f'<div>{a} × {b}: <strong>{c60:+.2f}</strong> em 60d contra {c120:+.2f} em 120d '
        f'<span style="color:{"#b8453a" if c60 > c120 else "#1e7a4c"}">({c60 - c120:+.2f})</span></div>'
        for a, b, c60, c120 in cor["pares"][:4])
    sobe = cor["media60"] > cor["media120"] + 0.05
    resumo = (f'Correlação média entre os ativos: <strong>{cor["media60"]:.2f}</strong> em 60 dias contra '
              f'{cor["media120"]:.2f} em 120 dias.'
              + (' <span style="color:#b8453a;font-weight:600">A diversificação piorou: os ativos estão se movendo mais juntos.</span>' if sobe else ''))
    corpo = (f'<div style="font-size:11.5px;color:#5a5a5a;line-height:1.8">{resumo}<div style="margin-top:6px">'
             f'Pares que mais mudaram:</div>{linhas}</div>')
    return _secao("Diversificação", "Os ativos estão se movendo mais juntos do que o normal?", corpo) + \
        _build_corr_section(resultados_ml, carteira)


# ── 6. Ensemble ampliado ─────────────────────────────────────────────────

def _html_ampliado(ampliado: dict | None) -> str:
    if not ampliado or not ampliado.get("placar"):
        return ""
    linhas = ""
    for h in [1, 2, 3]:
        pl = ampliado["placar"].get(f"d{h}") or {}
        if not pl.get("n"):
            linhas += f'<div>D+{h}: ainda sem previsões conferidas</div>'
            continue
        linhas += (f'<div>D+{h}: ampliado <strong>{pl["acc_ampliado"]*100:.1f}%</strong> vs produção '
                   f'<strong>{pl["acc_producao"]*100:.1f}%</strong> '
                   f'<span style="color:#a0a0a0">({pl["n"]} prev., {pl["n_dias"]} de {MIN_DIAS} dias)</span></div>')
    corpo = (f'<div style="font-size:11.5px;color:#5a5a5a;line-height:1.9">{linhas}</div>'
             f'<div style="margin-top:6px;font-size:10.5px;color:#a8a39a">Com {MIN_DIAS} dias, o ampliado passa pelo Jinshi contra a produção. '
             f'Só vira a previsão principal se for comprovadamente melhor.</div>')
    return _secao("Ensemble ampliado", "Produção + 13 famílias, nas mesmas previsões conferidas", corpo, "#f6f3eb")


def _html_regime(regime: dict | None) -> str:
    if not regime or not regime.get("placar"):
        return ""
    linhas = ""
    for h in [1, 2, 3]:
        pl = regime["placar"].get(f"d{h}") or {}
        if not pl.get("n"):
            linhas += f'<div>D+{h}: ainda sem previsões conferidas</div>'
            continue
        linhas += (f'<div>D+{h}: com regime <strong>{pl["acc_ampliado"]*100:.1f}%</strong> vs produção '
                   f'<strong>{pl["acc_producao"]*100:.1f}%</strong> '
                   f'<span style="color:#a0a0a0">({pl["n"]} prev., {pl["n_dias"]} de {MIN_DIAS} dias)</span></div>')
    corpo = (f'<div style="font-size:11.5px;color:#5a5a5a;line-height:1.9">{linhas}</div>'
             f'<div style="margin-top:6px;font-size:10.5px;color:#a8a39a">Mesmo trio da produção com VIX/VIX de 3 meses, juro de 10 anos e dólar. '
             f'No backtest de 2 anos só o D+3 passou; aqui é a confirmação com dados novos.</div>')
    return _secao("Desafiante de regime", "As features de regime confirmam fora do backtest?", corpo)


def _html_volatilidade(volat: dict | None) -> str:
    pl = (volat or {}).get("placar") or {}
    if not pl:
        return _secao("Volatilidade ao vivo", "Desafiantes do estudo de 18/11",
                      '<div style="font-size:11.5px;color:#a0a0a0">Ainda sem previsões conferidas.</div>')
    partes = []
    g = pl.get("grande")
    if g:
        bss = g["bss"]
        conf = "".join(
            f'<tr><td style="padding:3px 8px 3px 0;font-size:11px">{c["faixa"]}</td>'
            f'<td style="padding:3px 8px;text-align:right;{_MONO};font-size:11px">{c["prob_media"]*100:.0f}%</td>'
            f'<td style="padding:3px 8px;text-align:right;{_MONO};font-size:11px;font-weight:600">{c["real"]*100:.0f}%</td>'
            f'<td style="padding:3px 0;text-align:right;font-size:10px;color:#a0a0a0">n={c["n"]}</td></tr>'
            for c in g["confiabilidade"])
        partes.append(
            f'<div style="font-size:11.5px;color:#5a5a5a;line-height:1.7"><strong>Movimento grande amanhã</strong> '
            f'(|retorno| acima de 1 ATR): aconteceu em {g["taxa_real"]*100:.0f}% dos casos. '
            f'Logística contra a taxa de base: BSS <strong style="color:{_cor(bss)}">{bss*100:+.1f}%</strong> '
            f'<span style="color:#a0a0a0">({g["n"]} prev., {g["n_dias"]} de {MIN_DIAS} dias)</span></div>'
            f'<table style="border-collapse:collapse;margin:6px 0 4px"><thead><tr><th style="{_TH};text-align:left;padding-left:0">Faixa</th>'
            f'<th style="{_TH};text-align:right">Dita</th><th style="{_TH};text-align:right">Real</th><th style="{_TH}"></th></tr></thead>'
            f'<tbody>{conf}</tbody></table>')
    linhas = []
    for n in [1, 3, 5]:
        v = pl.get(f"vol{n}")
        if v:
            melhora = 1 - v["qlike_ewma"] / v["qlike_const"] if v["qlike_const"] else 0
            linhas.append(f'{n} dia{"s" if n > 1 else ""}: QLIKE EWMA {v["qlike_ewma"]:.3f} vs média histórica '
                          f'{v["qlike_const"]:.3f} (<strong style="color:{_cor(melhora)}">{melhora*100:+.0f}%</strong>, '
                          f'{v["n"]} prev.)')
    if linhas:
        partes.append('<div style="margin-top:8px;font-size:11.5px;color:#5a5a5a;line-height:1.8"><strong>Volatilidade esperada</strong><br>'
                      + "<br>".join(linhas) + '</div>')
    corpo = "".join(partes) + (
        f'<div style="margin-top:8px;font-size:10.5px;color:#a8a39a;line-height:1.6">No backtest de 2 anos a logística teve BSS +1,4% e AUC 0,61, '
        f'mas p corrigido 0,146; aqui é a confirmação com dados novos. Com {MIN_DIAS} dias vai para o Jinshi.</div>')
    return _secao("Volatilidade ao vivo", "O tamanho do movimento é previsível onde a direção não é?", corpo, "#f6f3eb")


# ── montagem ─────────────────────────────────────────────────────────────

def build_html_semanal(df_log: pd.DataFrame, research_log: pd.DataFrame, carteira: list,
                       resultados_ml: dict, context_data: dict, ref_date: pd.Timestamp,
                       ampliado: dict | None = None, regime: dict | None = None,
                       rastreio: dict | None = None, volat: dict | None = None) -> str:
    from reports.email_report import _build_drift_section, _rodape_versao
    agora = datetime.now(BARCELONA_TZ)
    ini = (pd.Timestamp(ref_date) - pd.offsets.BDay(4)).strftime("%d/%m")
    fim = pd.Timestamp(ref_date).strftime("%d/%m/%Y")

    blocos = [
        _html_semana(semana_da_carteira(resultados_ml, carteira, context_data)),
        _html_producao(acuracia_producao(df_log, carteira, ref_date)),
        _html_familias(ranking_familias(research_log, carteira)),
        _html_calibracao(df_log),
        _build_drift_section(),
        _html_correlacoes(correlacoes_60_120(resultados_ml, carteira), resultados_ml, carteira),
        _html_ampliado(ampliado),
        _html_regime(regime),
        _html_volatilidade(volat),
    ]
    corpo = "".join(b for b in blocos if b)
    return f"""<!DOCTYPE html>
<html lang="pt">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carteira BOT Semanal · {fim}</title><style>body {{ margin: 0; }}</style></head>
<body style="margin:0;padding:24px 16px;background:#ece9e2;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1a1a1a;-webkit-font-smoothing:antialiased">
<div style="max-width:680px;margin:0 auto;background:#fbfaf7;border-radius:6px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,0.04),0 12px 40px -12px rgba(38,33,92,0.18)">
  <div style="background:#1a1740;padding:30px 36px 26px">
    <div style="color:#a8a0ff;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase">Carteira Inteligente · Semanal</div>
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:30px;line-height:1.1;font-weight:500;margin-top:14px">Semana de {ini} a {fim}</div>
    <div style="color:#7d75c4;font-size:12px;margin-top:6px">Gerado em {agora.strftime('%d/%m/%Y %H:%M')}, Barcelona</div>
  </div>
  {corpo}
  <div style="padding:18px 36px;border-top:1px solid #efece4;font-size:10.5px;color:#a8a39a;line-height:1.55">
    <span style="color:#1a1740;font-weight:600">Aviso.</span> Previsões são probabilísticas e nenhum modelo passou ainda pelo Jinshi.
    Não tomar decisões financeiras baseadas exclusivamente neste relatório.
  </div>
</div>
<div style="max-width:680px;margin:14px auto 0;text-align:center;font-size:10px;color:#a8a39a;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">Carteira BOT Semanal · gerado automaticamente{_rodape_versao(rastreio)}</div>
</body></html>"""


def save_html_semanal(html: str) -> None:
    SEMANAL_HTML_REPORT.write_text(html, encoding="utf-8")
    logger.info("[Semanal] HTML guardado em %s", SEMANAL_HTML_REPORT)
