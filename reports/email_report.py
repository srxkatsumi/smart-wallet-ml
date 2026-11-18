import logging
import pandas as pd
from datetime import datetime
from data.storage import aligned_rows
from config.settings import (
    HTML_REPORT, BARCELONA_TZ, METADATA_FILE, TAXA_CRESCIMENTO_BASE,
)

logger = logging.getLogger(__name__)

_DIAS_PT  = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
             "Sexta-feira", "Sábado", "Domingo"]

_TAXA_PESS = TAXA_CRESCIMENTO_BASE["pessimista"]   # 0.03
_TAXA_BASE = TAXA_CRESCIMENTO_BASE["base"]          # 0.08
_TAXA_OTIM = TAXA_CRESCIMENTO_BASE["optimista"]     # 0.15


def _build_correlation_html(resultados_ml: dict, my_tickers: list) -> str:
    try:
        closes = {}
        for ticker in my_tickers:
            if ticker in resultados_ml:
                closes[ticker] = resultados_ml[ticker]["df"]["Close"].tail(120)
        if len(closes) < 2:
            return ""

        df_c  = pd.DataFrame(closes).pct_change(fill_method=None).dropna()
        corr  = df_c.corr()
        ticks = list(corr.columns)

        def _bg(v: float) -> str:
            if v >= 0:
                r = int(0xf6 + (0x1e - 0xf6) * v)
                g = int(0xf3 + (0x7a - 0xf3) * v)
                b = int(0xeb + (0x4c - 0xeb) * v)
            else:
                t = abs(v)
                r = int(0xf6 + (0xb8 - 0xf6) * t)
                g = int(0xf3 + (0x45 - 0xf3) * t)
                b = int(0xeb + (0x3a - 0xeb) * t)
            return f"#{r:02x}{g:02x}{b:02x}"

        def _fg(v: float) -> str:
            return "white" if abs(v) > 0.65 else "#2a2a2a"

        th = '<td style="padding:4px;font-size:9px;color:#a89e85"></td>'
        for t in ticks:
            th += (f'<td style="padding:4px 6px;font-size:9px;font-weight:600;'
                   f'color:#5a5a5a;text-align:center;white-space:nowrap">{t}</td>')
        rows = f"<tr>{th}</tr>"

        for i, t_row in enumerate(ticks):
            row = (f'<td style="padding:4px 10px 4px 0;font-size:9px;font-weight:600;'
                   f'color:#5a5a5a;white-space:nowrap">{t_row}</td>')
            for j in range(len(ticks)):
                v      = corr.values[i, j]
                diag   = "border:2px solid #1a1740;" if i == j else ""
                row   += (f'<td style="padding:5px 4px;text-align:center;background:{_bg(v)};'
                           f'{diag}border-radius:3px;font-family:ui-monospace,SFMono-Regular,'
                           f'Menlo,monospace;font-size:10px;font-weight:600;color:{_fg(v)};'
                           f'min-width:42px">{v:.2f}</td>')
            rows += f"<tr>{row}</tr>"

        return f'<table style="border-collapse:separate;border-spacing:2px">{rows}</table>'
    except Exception as e:
        logger.warning("Correlation HTML failed: %s", e)
        return ""
_MESES_PT = ['janeiro','fevereiro','março','abril','maio','junho',
             'julho','agosto','setembro','outubro','novembro','dezembro']


def _dir_cell(direction: str, conf: float) -> str:
    if direction.lower() == "up":
        return (
            '<span style="display:inline-block;color:#1e7a4c;font-size:11px;'
            'font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
            f'▲&nbsp;{conf*100:.0f}</span>'
        )
    return (
        '<span style="color:#b8453a;font-size:11px;font-weight:600;'
        'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
        f'▼&nbsp;{conf*100:.0f}</span>'
    )


def _var_cell(var_pct: float) -> str:
    if var_pct >= 0:
        return (
            '<span style="display:inline-block;color:#1e7a4c;font-size:11px;'
            'font-weight:500;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
            f'▲&nbsp;{abs(var_pct):.1f}</span>'
        )
    return (
        '<span style="color:#b8453a;font-size:11px;font-weight:500;'
        'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
        f'▼&nbsp;{abs(var_pct):.1f}</span>'
    )


def _consenso_badge(consenso: str) -> str:
    styles = {
        "BULLISH": ("#e7f3eb", "#1e7a4c"),
        "BEARISH": ("#f7e8e5", "#b8453a"),
        "MISTO":   ("#f1ede4", "#8a7a3a"),
    }
    bg, col = styles.get(consenso, styles["MISTO"])
    return (
        f'<span style="display:inline-block;background:{bg};color:{col};border-radius:3px;'
        f'padding:3px 8px;font-size:10px;font-weight:700;letter-spacing:0.08em">'
        f'{consenso}</span>'
    )








def _acertou_ontem(ticker: str, df_log: pd.DataFrame, ontem_str: str):
    """Returns 'up_correct' | 'down_correct' | 'wrong' | None."""
    mask = (
        (df_log["ticker"]      == ticker) &
        (df_log["target_date"] == ontem_str) &
        (df_log["horizon"]     == 1) &
        (df_log["correct"].notna())
    )
    rows = df_log[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if not bool(row["correct"]):
        return "wrong"
    return "up_correct" if row.get("direction") == "up" else "down_correct"


def _recentes_d1(df_log: pd.DataFrame, portfolio_tickers: list) -> pd.DataFrame:
    """D+1 da carteira, alinhadas, com alvo nos últimos 30 dias úteis (a janela do bloco de acurácia)."""
    v = aligned_rows(df_log[
        (df_log["correct"].notna()) &
        (df_log["ticker"].isin(portfolio_tickers)) &
        (df_log["horizon"] == 1)
    ]).copy()
    v["target_date"] = pd.to_datetime(v["target_date"])
    return v[v["target_date"] >= pd.Timestamp.today() - pd.offsets.BDay(30)]


def _calcular_tendencia(df_log: pd.DataFrame, portfolio_tickers: list) -> tuple:
    validadas = aligned_rows(df_log[
        (df_log["correct"].notna()) &
        (df_log["ticker"].isin(portfolio_tickers)) &
        (df_log["horizon"] == 1)
    ]).copy()
    validadas["target_date"] = pd.to_datetime(validadas["target_date"])
    validadas = validadas.sort_values("target_date")
    hoje_ts  = pd.Timestamp.today()
    corte30  = hoje_ts - pd.offsets.BDay(30)
    corte60  = hoje_ts - pd.offsets.BDay(60)
    recentes = validadas[validadas["target_date"] >= corte30]
    ant      = validadas[(validadas["target_date"] >= corte60) &
                         (validadas["target_date"] <  corte30)]
    MIN_VAL  = 15
    if len(recentes) < MIN_VAL:
        return "treino", len(recentes), MIN_VAL - len(recentes), None, None, None, 0, None, 0
    acc_rec = recentes["correct"].astype(float).mean()
    acc_ant = ant["correct"].astype(float).mean() if len(ant) >= MIN_VAL else None
    delta   = acc_rec - acc_ant if acc_ant is not None else None

    up_preds = recentes[recentes["direction"] == "up"]
    dn_preds = recentes[recentes["direction"] == "down"]
    prec_up  = up_preds["correct"].astype(float).mean() if len(up_preds) >= 5 else None
    prec_dn  = dn_preds["correct"].astype(float).mean() if len(dn_preds) >= 5 else None

    return "ok", len(recentes), None, acc_rec, delta, prec_up, len(up_preds), prec_dn, len(dn_preds)


def _compute_drift() -> dict:
    try:
        df = pd.read_csv(METADATA_FILE)
    except Exception:
        return {"enough_data": False}

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols or "date" not in df.columns:
        return {"enough_data": False}

    df = df[df["model"] == "rf"].copy()
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return {"enough_data": False}

    today_imp = df[df["date"] == dates[-1]][feat_cols].mean()
    ref_imp   = df[df["date"].isin(dates[:-1])][feat_cols].mean()

    corr = float(today_imp.corr(ref_imp, method="spearman"))

    def label(name):
        return name.replace("feat_", "")

    today_ranked = [label(f) for f in today_imp.sort_values(ascending=False).index]
    ref_ranked   = [label(f) for f in ref_imp.sort_values(ascending=False).index]
    ref_rank_map = {f: i + 1 for i, f in enumerate(ref_ranked)}

    top5 = []
    for rank_today, feat in enumerate(today_ranked[:5], start=1):
        rank_ref = ref_rank_map.get(feat, len(feat_cols))
        delta    = rank_ref - rank_today
        top5.append((feat, rank_today, rank_ref, delta))

    rho_baixo   = corr < 0.70
    top_mudou   = today_ranked[0] != ref_ranked[0]
    is_drift    = rho_baixo or top_mudou
    motivo = ("ρ abaixo de 0,70" if rho_baixo else
              f"a feature nº1 mudou ({ref_ranked[0]} → {today_ranked[0]})" if top_mudou else "")

    return {
        "enough_data": True,
        "is_drift":    is_drift,
        "motivo":      motivo,
        "corr":        corr,
        "top5":        top5,
        "n_ref_days":  len(dates) - 1,
    }


def _build_research_section(research_data: dict) -> str:
    """Secção diária com comparação das 13 famílias e consenso ponderado."""
    if not research_data:
        return ""

    comparison = research_data.get("comparison", [])
    consensus  = research_data.get("consensus", [])

    # ── tabela de famílias ────────────────────────────────────────────────
    family_labels = {
        "classico_avancado": "Clássico avançado",
        "estado_oculto":     "Estado oculto (HMM)",
        "series_temporais":  "Séries temporais",
        "neural_recorrente": "Neural recorrente",
        "neural_atencao":    "Neural com atenção",
        "bayesiano":         "Bayesiano (GP+BNN)",
        "generativo":        "Generativo (VAE+GAN)",
        "reinforcement":     "Reinforcement (DQN+PPO)",
        "contrarian":        "Contrarian (CB+EWI+PEL) ⚠",
        "eficiente":         "Eficiente (TCN/DLinear/PatchTST)",
        "foundation":        "Foundation (Chronos/TimesFM/Moirai)",
        "conformal":         "Conformal (calibração)",
        "drift":             "Drift detection (ADWIN/PH)",
    }
    contrarian_warning = any(
        r["family"] == "contrarian" and r == comparison[0]
        for r in comparison
    ) if comparison else False
    comp_rows = ""
    for r in comparison:
        acc_pct  = r["accuracy"] * 100
        vs_pct   = r["vs_acaso"] * 100
        sign     = "+" if vs_pct >= 0 else ""
        vs_color = "#1e7a4c" if vs_pct >= 0 else "#b8453a"
        star     = " ★" if r == comparison[0] and r.get("sig") else ""
        comp_rows += (
            f'<tr><td style="padding:7px 8px 7px 0;font-size:12px;color:#1a1a1a">'
            f'{family_labels.get(r["family"], r["family"])}{star}</td>'
            f'<td style="padding:7px 8px;text-align:right;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;'
            f'color:#1a1740">{acc_pct:.0f}%'
            f'<span style="font-size:9.5px;color:#a0a0a0;font-weight:400"> n={r.get("n", "?")}</span></td>'
            f'<td style="padding:7px 0 7px 8px;text-align:right;font-size:12px;'
            f'color:{vs_color};font-weight:600">{sign}{vs_pct:.0f}%</td></tr>'
        )

    # ── tabela de consenso ────────────────────────────────────────────────
    cons_rows = ""
    n_dias_fam = max((r.get("n_dias", 0) for r in comparison), default=0)
    desde_fam  = min((r.get("desde", "") for r in comparison if r.get("desde")), default="")
    desde_txt  = (f"desde {pd.Timestamp(desde_fam).strftime('%d/%m')} · {n_dias_fam} dias"
                  if desde_fam else "")
    for r in consensus:
        arrows   = "▲▲▲" if r["pct_up"] >= 0.75 else ("▲▲" if r["pct_up"] >= 0.6
                   else ("▲" if r["pct_up"] >= 0.5 else "▼"))
        color    = "#1e7a4c" if r["direction"] == "ALTA" else "#b8453a"
        strength = f'({r["strength"]})' if r["strength"] == "fraco" else ""
        cons_rows += (
            f'<tr><td style="padding:7px 8px 7px 0;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;'
            f'color:#1a1740">{r["ticker"]}</td>'
            f'<td style="padding:7px 8px;text-align:center;font-size:12px;'
            f'color:{color};font-weight:700">{r["direction"]} {strength}</td>'
            f'<td style="padding:7px 8px;text-align:center;font-size:11px;'
            f'color:#5a5a5a">{(r["up_count"] if r["direction"] == "ALTA" else r["total"] - r["up_count"])}'
            f' de {r["total"]} a favor · {(r["pct_up"] if r["direction"] == "ALTA" else 1 - r["pct_up"])*100:.0f}% ponderado</td>'
            f'<td style="padding:7px 0 7px 8px;text-align:right;font-size:13px;'
            f'color:{color}">{arrows}</td></tr>'
        )

    if not comp_rows and not cons_rows:
        return ""

    comp_html = f"""
    <table style="width:100%;border-collapse:collapse;margin-bottom:8px">
      <thead><tr>
        <th style="text-align:left;padding:0 8px 6px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Família</th>
        <th style="text-align:right;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Acurácia D+1</th>
        <th style="text-align:right;padding:0 0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">vs Acaso</th>
      </tr></thead>
      <tbody>{comp_rows}</tbody>
    </table>""" if comp_rows else "<p style='font-size:12px;color:#a0a0a0'>Dados de comparação disponíveis após a primeira semana completa.</p>"

    cons_html = f"""
    <table style="width:100%;border-collapse:collapse">
      <thead><tr>
        <th style="text-align:left;padding:0 8px 6px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ativo</th>
        <th style="text-align:center;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Direção</th>
        <th style="text-align:center;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Consenso</th>
        <th style="text-align:right;padding:0 0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Força</th>
      </tr></thead>
      <tbody>{cons_rows}</tbody>
    </table>""" if cons_rows else ""

    return f"""
  <div style="padding:24px 36px;border-top:2px solid #e0d8f0;background:#faf8ff">
    <div style="font-size:10px;font-weight:600;color:#7d75c4;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:16px">13 famílias de pesquisa · atualizado todo dia · só informativo</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
      <div>
        <div style="font-size:13px;font-weight:600;color:#1a1740;margin-bottom:10px">Acurácia D+1 por família{(' · ' + desde_txt) if desde_txt else ''}</div>
        {comp_html}
        <div style="margin-top:8px;font-size:10px;color:#a8a39a">Só previsões com horizonte alinhado, todos os ativos da carteira &nbsp;·&nbsp; ★ melhor família, apenas se significativa após correção de Benjamini-Hochberg pelas 13 famílias &nbsp;·&nbsp; vs Acaso = diferença face a 50% &nbsp;·&nbsp; ⚠ = teste de sanidade do ensemble &nbsp;·&nbsp; com menos de 20 dias, as diferenças entre famílias são sobretudo ruído</div>
        {"<div style='margin-top:10px;padding:8px 10px;background:#fff3cd;border-left:3px solid #d4a017;border-radius:3px;font-size:11px;color:#7a5c00'><strong>Alerta Contrarian:</strong> a família Contrarian é a mais precisa no período. Se CB supera o ensemble principal, o ensemble pode estar a errar de forma sistemática e previsível. Verificar acurácia recente do ensemble.</div>" if contrarian_warning else ""}
      </div>
      <div>
        <div style="font-size:13px;font-weight:600;color:#1a1740;margin-bottom:10px">Consenso das 13 famílias · D+1</div>
        {cons_html}
        <div style="margin-top:8px;font-size:10px;color:#a8a39a">Direção = maioria ponderada pelos pesos das famílias &nbsp;·&nbsp; "a favor" = famílias que votaram nessa direção &nbsp;·&nbsp; abaixo de 60% ponderado = sinal fraco &nbsp;·&nbsp; ▲▲▲ = ≥75% a favor da alta</div>
      </div>
    </div>
  </div>"""


_NOMES_FAMILIA = {
    "producao": "Produção", "classico_avancado": "Clássico", "estado_oculto": "HMM",
    "series_temporais": "Séries temporais", "neural_recorrente": "Neural recorrente",
    "neural_atencao": "Transformer", "bayesiano": "Bayesiano", "generativo": "VAE/GAN",
    "reinforcement": "Reforço", "contrarian": "Contrarian", "eficiente": "TCN/PatchTST",
    "foundation": "Foundation", "conformal": "Conformal", "drift": "Drift",
}


def _ampliado_row(ampliado: dict | None, ticker: str, h: int) -> dict | None:
    if not ampliado:
        return None
    return (ampliado.get("hoje") or {}).get(ticker, {}).get(h)


def _familias_line(ampliado: dict | None, ticker: str, h: int, prod_dir: str) -> str:
    """Segunda linha da célula D+N: consenso ponderado das 13 famílias e aviso de divergência."""
    r = _ampliado_row(ampliado, ticker, h)
    if not r or r.get("prob_up_familias") is None or pd.isna(r.get("prob_up_familias")):
        return ""
    p = float(r["prob_up_familias"])
    fam_dir = "up" if p >= 0.5 else "down"
    conf = max(p, 1 - p) * 100
    seta = "▲" if fam_dir == "up" else "▼"
    if fam_dir != prod_dir.lower():
        return (f'<div style="font-size:9.5px;color:#c07a1a;font-weight:600;margin-top:3px;'
                f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">≠ 13F {seta}{conf:.0f}</div>')
    return (f'<div style="font-size:9.5px;color:#a8a39a;margin-top:3px;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">13F {seta}{conf:.0f}</div>')


def _build_ampliado_section(ampliado: dict | None, tickers: list) -> str:
    if not ampliado or not ampliado.get("hoje"):
        return ""
    from evaluation.significance import MIN_DIAS

    rows = ""
    linhas = [t for t in tickers if t in ampliado["hoje"]]
    for i, t in enumerate(linhas):
        cells = ""
        for h in [1, 2, 3]:
            r = _ampliado_row(ampliado, t, h)
            if not r:
                cells += '<td style="padding:8px 6px;text-align:center;color:#ccc">-</td>'
                continue
            p = float(r["prob_up"])
            cells += f'<td style="padding:8px 6px;text-align:center">{_dir_cell(r["direction"], max(p, 1 - p))}</td>'
        border = "border-bottom:1px solid #f0ede5" if i < len(linhas) - 1 else ""
        rows += (f'<tr style="{border}"><td style="padding:8px 6px 8px 0;font-family:ui-monospace,SFMono-Regular,'
                 f'Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740">{t}</td>{cells}</tr>')

    placar_html = ""
    for h in [1, 2, 3]:
        pl = (ampliado.get("placar") or {}).get(f"d{h}") or {}
        if not pl.get("n"):
            placar_html += f'<div>D+{h}: ainda sem previsões conferidas</div>'
            continue
        estado = ("em coleta" if pl["n_dias"] < MIN_DIAS else "pronto para o Jinshi")
        placar_html += (f'<div>D+{h}: ampliado <strong>{pl["acc_ampliado"]*100:.1f}%</strong> vs produção '
                        f'<strong>{pl["acc_producao"]*100:.1f}%</strong> '
                        f'<span style="color:#a0a0a0">({pl["n"]} prev. · {pl["n_dias"]} de {MIN_DIAS} dias · {estado})</span></div>')

    pesos_d1 = ((ampliado.get("pesos") or {}).get("d1") or {}).get("pesos", {})
    top = sorted(pesos_d1.items(), key=lambda kv: -kv[1])
    pesos_html = " &nbsp;·&nbsp; ".join(
        f'{_NOMES_FAMILIA.get(k, k)} <strong>{v:.2f}</strong>' for k, v in top[:4]
    )
    piso = " &nbsp;·&nbsp; ".join(f'{_NOMES_FAMILIA.get(k, k)} {v:.2f}' for k, v in top[-2:]) if len(top) > 4 else ""

    th = ("padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;"
          "text-transform:uppercase;border-bottom:1px solid #e6e3dc")
    return f"""
  <div style="padding:24px 36px;border-top:1px solid #efece4">
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:17px;font-weight:500;color:#1a1740">Ensemble ampliado <span style="font-size:11px;color:#a89e85;font-family:-apple-system,Helvetica,Arial,sans-serif;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">desafiante</span></div>
    <div style="font-size:12px;color:#8a8a8a;margin:3px 0 14px;line-height:1.5">Produção + 13 famílias votando juntas, com pesos que só crescem com desempenho sustentado. Não substitui as previsões acima até passar pelo Jinshi.</div>
    <table style="border-collapse:collapse;width:100%;max-width:420px">
      <thead><tr>
        <th style="text-align:left;{th.replace('padding:0 6px 8px', 'padding:0 6px 8px 0')}">Ativo</th>
        <th style="text-align:center;{th}">D+1</th><th style="text-align:center;{th}">D+2</th><th style="text-align:center;{th}">D+3</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div style="margin-top:14px;font-size:11px;color:#5a5a5a;line-height:1.9">{placar_html}</div>
    <div style="margin-top:10px;font-size:10.5px;color:#a8a39a;line-height:1.7">
      Maiores pesos D+1: {pesos_html}{('<br>Menores: ' + piso) if piso else ''}<br>
      Peso 1,00 = voto igual. Vai de 0,25 a 2,00 conforme a acurácia, encolhida para 50% enquanto há poucas previsões.
    </div>
  </div>"""


def _conf_cells(direction: str, conf: float, h: int, curves: dict | None, faixas: dict) -> str:
    """
    Célula D+N: com calibração disponível, mostra a confiança calibrada e a bruta
    em letra pequena; sem calibração, a bruta e o acerto real da faixa.
    """
    from models.calibration import apply_calibration
    curve = (curves or {}).get(f"d{h}")
    if not curve:
        return _dir_cell(direction, conf) + _faixa_line(faixas, h, conf)
    cal = apply_calibration(conf, h, curves)
    return (_dir_cell(direction, cal) +
            f'<div style="font-size:9.5px;color:#a0a0a0;margin-top:3px;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">bruta {conf*100:.0f}</div>')


def _faixa_line(faixas: dict, h: int, conf: float) -> str:
    from evaluation.confianca import acerto_da_faixa
    f = acerto_da_faixa(faixas, h, conf) if faixas else None
    if not f:
        return ""
    return (f'<div style="font-size:9.5px;color:#7d75c4;margin-top:3px;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">real {f["acc"]*100:.0f}%</div>')


def _build_calibracao_html(faixas: dict) -> str:
    if not faixas or not any(f["n"] for h in faixas for f in faixas[h]):
        return ""
    th = ("padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;"
          "text-transform:uppercase;border-bottom:1px solid #e6e3dc;text-align:center")
    rows = ""
    for i, base in enumerate(faixas[1]):
        cells = ""
        for h in [1, 2, 3]:
            f = faixas[h][i]
            if not f["n"]:
                cells += '<td style="padding:6px 8px;text-align:center;color:#ccc;font-size:11px">-</td>'
                continue
            acc = f["acc"] * 100
            meio = (f["lo"] + min(f["hi"], 1.0)) / 2 * 100
            cor = "#1e7a4c" if acc >= meio - 5 else ("#8a7a3a" if acc >= 50 else "#b8453a")
            cells += (f'<td style="padding:6px 8px;text-align:center;font-size:11px;font-family:ui-monospace,'
                      f'SFMono-Regular,Menlo,Consolas,monospace"><strong style="color:{cor}">{acc:.0f}%</strong>'
                      f' <span style="color:#a0a0a0;font-size:9.5px">n={f["n"]} · {f["ic"][0]*100:.0f}-{f["ic"][1]*100:.0f}</span></td>')
        rotulo = f'{base["lo"]*100:.0f}-{min(base["hi"], 1.0)*100:.0f}%'
        rows += (f'<tr style="border-bottom:1px solid #ede9e0"><td style="padding:6px 8px 6px 0;font-size:11px;'
                 f'color:#1a1740;font-weight:600">{rotulo}</td>{cells}</tr>')
    return f"""
  <div style="padding:20px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:6px">Confiança dita × acerto real</div>
    <div style="font-size:10.5px;color:#a89e85;margin-bottom:12px;line-height:1.5">Todas as previsões conferidas com horizonte alinhado, carteira e watchlist. Num modelo bem calibrado, a faixa 80-90% acerta perto de 85%.</div>
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%">
      <thead><tr><th style="{th};text-align:left;padding-left:0">Confiança</th><th style="{th}">D+1</th><th style="{th}">D+2</th><th style="{th}">D+3</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    <div style="margin-top:10px;font-size:10.5px;color:#a8a39a;line-height:1.6">
      <span style="color:#1e7a4c;font-weight:600">verde</span> = acerta perto do que promete &nbsp;·&nbsp;
      <span style="color:#8a7a3a;font-weight:600">amarelo</span> = acima de 50%, mas abaixo do prometido &nbsp;·&nbsp;
      <span style="color:#b8453a;font-weight:600">vermelho</span> = abaixo de 50%<br>
      Intervalo de Wilson sobre as linhas: otimista, porque previsões do mesmo dia não são independentes.
    </div>
  </div>"""


def _build_saude_html(saude: dict) -> str:
    estilos = {
        "ok":       ("#e7f3eb", "#1e7a4c", "✅ Sistema OK"),
        "atencao":  ("#fffbea", "#7a6010", "⚠️ Atenção"),
        "problema": ("#f7e8e5", "#b8453a", "⛔ Dados inconsistentes"),
    }
    bg, cor, titulo = estilos[saude["estado"]]
    resumo = f'{saude["n_ok"]} de {saude["n_ativos"]} ativos com dados do último pregão'
    itens = "".join(f'<div style="margin-top:3px">• {x}</div>' for x in saude["problemas"] + saude["atencoes"])
    nota = ('<div style="margin-top:6px;font-weight:600">As previsões dos ativos acima não são confiáveis hoje.</div>'
            if saude["estado"] == "problema" else "")
    return f"""
  <div style="padding:12px 36px;background:{bg};border-top:2px solid {cor}">
    <div style="font-size:11.5px;color:{cor};line-height:1.6">
      <strong>{titulo}</strong> &nbsp;·&nbsp; {resumo}{itens}{nota}
    </div>
  </div>"""


def _build_eventos_html(eventos: dict | None, tickers: list) -> str:
    if not eventos:
        return ""
    lista = eventos.get("eventos") or []
    aviso = eventos.get("aviso_calendario")
    linhas = "".join(
        f'<div style="margin-top:3px"><strong>D+{e["sessao"]}</strong> · {pd.Timestamp(e["data"]).strftime("%d/%m")} · '
        f'{e["nome"]} <span style="color:#a0a0a0">(afeta {e["afeta"]})</span></div>'
        for e in lista
    )
    corpo = (linhas + '<div style="margin-top:6px;color:#8a7a3a">O modelo não vê esses eventos: '
             'em dias assim, movimentos grandes e erros com confiança alta são mais prováveis.</div>') if lista else \
            '<div style="margin-top:3px">Nenhum evento relevante nas próximas 3 sessões.</div>'
    aviso_html = f'<div style="margin-top:6px;color:#b8453a">{aviso}</div>' if aviso else ""
    return f"""
  <div style="padding:14px 36px;background:#f6f3eb;border-top:1px solid #efece4">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase">Eventos nas próximas 3 sessões</div>
    <div style="font-size:11.5px;color:#5a5a5a;line-height:1.6;margin-top:4px">{corpo}{aviso_html}</div>
  </div>"""


def _build_corr_section(resultados_ml: dict, my_tickers: list) -> str:
    # ── Correlation matrix ────────────────────────────────────────────────
    corr_table = _build_correlation_html(resultados_ml, my_tickers)
    if corr_table:
        corr_html = f"""
  <div style="padding:20px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px">Correlação de retornos</div>
    <div style="overflow-x:auto">{corr_table}</div>
    <div style="margin-top:12px;font-size:10.5px;color:#a8a39a;line-height:1.8">
      Retornos diários dos últimos 120 dias úteis.<br>
      <span style="display:inline-block;width:11px;height:11px;background:#1e7a4c;border-radius:2px;vertical-align:middle;margin-right:5px"></span><strong style="color:#5a5a5a">+1.0</strong> = movem-se sempre na mesma direção &nbsp;·&nbsp;
      <span style="display:inline-block;width:11px;height:11px;background:#f6f3eb;border:1px solid #ccc;border-radius:2px;vertical-align:middle;margin-right:5px"></span><strong style="color:#5a5a5a">0.0</strong> = sem relação &nbsp;·&nbsp;
      <span style="display:inline-block;width:11px;height:11px;background:#b8453a;border-radius:2px;vertical-align:middle;margin-right:5px"></span><strong style="color:#5a5a5a">-1.0</strong> = movem-se em direções opostas
    </div>
  </div>"""
    else:
        corr_html = ""
    return corr_html


def _build_drift_section() -> str:
    # ── Feature importance drift ──────────────────────────────────────────
    drift = _compute_drift()
    if not drift["enough_data"]:
        drift_html = ""
    else:
        corr      = drift["corr"]
        is_drift  = drift["is_drift"]
        n_ref     = drift["n_ref_days"]
        badge_bg  = "#f7e8e5" if is_drift else "#e7f3eb"
        badge_col = "#b8453a" if is_drift else "#1e7a4c"
        badge_txt = (f"⚠️ Drift: {drift['motivo']} · ρ={corr:.2f}" if is_drift
                     else f"✅ Estável · ρ={corr:.2f}")

        rows_drift = ""
        for feat, rank_today, rank_ref, delta in drift["top5"]:
            if delta > 0:
                delta_html = f'<span style="color:#1e7a4c;font-weight:600">↑ +{delta}</span>'
            elif delta < 0:
                delta_html = f'<span style="color:#b8453a;font-weight:600">↓ {delta}</span>'
            else:
                delta_html = '<span style="color:#a0a0a0">→</span>'
            rows_drift += f"""
            <tr>
              <td style="padding:5px 8px 5px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#1a1740">{feat}</td>
              <td style="padding:5px 8px;text-align:center;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#1a1740;font-weight:600">#{rank_today}</td>
              <td style="padding:5px 8px;text-align:center;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#8a8a8a">#{rank_ref}</td>
              <td style="padding:5px 0 5px 8px;text-align:center;font-size:11px">{delta_html}</td>
            </tr>"""

        drift_html = f"""
  <div style="padding:20px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:6px">
      <tr>
        <td style="vertical-align:middle">
          <span style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase">Feature importance drift</span>
        </td>
        <td style="vertical-align:middle;text-align:right">
          <span style="display:inline-block;background:{badge_bg};color:{badge_col};border-radius:3px;padding:3px 9px;font-size:10px;font-weight:700;letter-spacing:0.06em">{badge_txt}</span>
        </td>
      </tr>
    </table>
    <div style="font-size:10.5px;color:#a89e85;margin-bottom:12px;line-height:1.5">
      Compara o ranking de importância das features de hoje com os últimos {n_ref} dias (RF, média por horizonte).<br>
      <strong style="color:#7a6a5a">ρ</strong> = correlação de Spearman entre os dois rankings · ρ próximo de 1 = padrão estável · drift = ρ &lt; 0,70 <strong>ou</strong> a feature mais importante mudou.<br>
      <strong style="color:#1e7a4c">↑</strong> subiu no ranking &nbsp;·&nbsp; <strong style="color:#b8453a">↓</strong> desceu no ranking &nbsp;·&nbsp; → sem alteração
    </div>
    <table style="border-collapse:collapse">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 8px 6px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Feature</th>
          <th style="text-align:center;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Hoje</th>
          <th style="text-align:center;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ref. ({n_ref}d)</th>
          <th style="text-align:center;padding:0 0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Δ rank</th>
        </tr>
      </thead>
      <tbody>{rows_drift}
      </tbody>
    </table>
  </div>"""
    return drift_html


def _fmt_num_pt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _build_risco_html(risco: dict | None) -> str:
    if not risco:
        return ""
    ctx, cart = risco.get("contexto") or {}, risco.get("carteira")
    def var(v, pct=True, pp=False):
        cor = "#1e7a4c" if v >= 0 else "#b8453a"
        txt = (f"{v:+.2f}".replace(".", ",") + " p.p.") if pp else (f"{v*100:+.1f}".replace(".", ",") + "%")
        return f'<span style="color:{cor}">{txt}</span>'
    partes = []
    if ctx.get("vix"):
        n, v = ctx["vix"]
        regime = "alto" if n > 25 else ("médio" if n > 15 else "baixo")
        partes.append(f'VIX <strong>{_fmt_num_pt(n, 1)}</strong> {var(v)} <span style="color:#a0a0a0">(regime {regime})</span>')
    if ctx.get("juro10"):
        n, v = ctx["juro10"]
        partes.append(f'Juro 10 anos EUA <strong>{_fmt_num_pt(n)}%</strong> {var(v, pp=True)}')
    if ctx.get("eurusd"):
        n, v = ctx["eurusd"]
        partes.append(f'EUR/USD <strong>{_fmt_num_pt(n, 4)}</strong> {var(v)}')
    if ctx.get("ouro"):
        partes.append(f'Ouro (GLD) {var(ctx["ouro"][1])}')
    ctx_html = " &nbsp;·&nbsp; ".join(partes)

    cart_html = ""
    if cart:
        cores = {"agitado": "#b8453a", "normal": "#5a5a5a", "calmo": "#1e7a4c"}
        linhas = "".join(
            f'<tr style="border-bottom:1px solid #ede9e0"><td style="padding:5px 8px 5px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;font-weight:600;color:#1a1740">{a["ticker"]}</td>'
            f'<td style="padding:5px 8px;text-align:right;font-size:11px;font-weight:600">{("±" + format(a["sigma_amanha"]*100, ".1f").replace(".", ",") + "%") if a.get("sigma_amanha") else "-"}</td>'
            f'<td style="padding:5px 8px;text-align:right;font-size:11px">{a["vol20"]*100:.0f}% <span style="color:#a0a0a0">/ {a["vol_ano"]*100:.0f}%</span></td>'
            f'<td style="padding:5px 8px;text-align:center;font-size:11px;font-weight:600;color:{cores[a["estado"]]}">{a["estado"]}</td>'
            f'<td style="padding:5px 8px;text-align:right;font-size:11px;color:{"#b8453a" if a["queda_pico"] < -0.1 else "#5a5a5a"}">{a["queda_pico"]*100:.0f}%</td>'
            f'<td style="padding:5px 0 5px 8px;text-align:right;font-size:11px">{a["var95"]*100:.1f}%</td></tr>'
            for a in cart["por_ativo"])
        th = "padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:9.5px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc;text-align:right"
        nivel_cor = {"baixo": "#1e7a4c", "normal": "#8a7a3a", "elevado": "#b8453a"}[cart["nivel"]]
        agit = (f'{len(cart["agitados"])} de {cart["n_ativos"]} ativos mais agitados que o normal ({", ".join(cart["agitados"])})'
                if cart["agitados"] else f'nenhum dos {cart["n_ativos"]} ativos mais agitado que o normal')
        corr_txt = ""
        if cart["corr60"] is not None:
            corr_txt = (f' &nbsp;·&nbsp; correlação média {cart["corr60"]:.2f} em 60 dias contra {cart["corr_ano"]:.2f} em 1 ano'
                        + (' <span style="color:#b8453a">(os ativos estão se movendo mais juntos: diversificação menor)</span>' if cart["corr_subiu"] else ''))
        cart_html = f"""
    <div style="margin-top:12px;font-size:11.5px;color:#5a5a5a;line-height:1.7">
      Risco da carteira: <strong style="color:{nivel_cor};text-transform:uppercase;letter-spacing:0.04em">{cart["nivel"]}</strong>
      &nbsp;·&nbsp; {agit}{corr_txt}
    </div>
    <table style="border-collapse:collapse;margin-top:10px;width:100%;max-width:480px">
      <thead><tr><th style="{th};text-align:left;padding-left:0">Ativo</th><th style="{th}">Esperado amanhã</th><th style="{th}">Vol. 20d / 1 ano</th><th style="{th};text-align:center">Estado</th><th style="{th}">Do pico</th><th style="{th};padding-right:0">VaR 1 dia</th></tr></thead>
      <tbody>{linhas}</tbody>
    </table>
    <div style="margin-top:6px;font-size:10px;color:#a8a39a;line-height:1.5">
      "Esperado amanhã" = oscilação típica prevista para o próximo pregão (1 desvio, EWMA λ=0,94): em ~2 de cada 3 dias o movimento fica dentro disso.
      Num teste de 2 anos, essa previsão bateu a média histórica de cada ativo (QLIKE 5% a 10% menor).<br>
      Cada ativo comparado com ele mesmo, em euro: "agitado" = volatilidade dos últimos 20 dias 30% acima da do último ano.
      VaR 1 dia = em 1 de cada 20 dias o ativo cai mais que isso. Nada depende de quanto está investido em cada um.
    </div>"""
    return f"""
  <div style="padding:16px 36px;border-top:1px solid #efece4">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase">Risco hoje</div>
    <div style="font-size:11.5px;color:#5a5a5a;line-height:1.7;margin-top:4px">{ctx_html}</div>{cart_html}
  </div>"""


def _rodape_versao(rastreio: dict | None, resultados_ml: dict | None = None, tickers: list | None = None) -> str:
    """Linha do rodapé com o que gerou este email: código, calibração e data dos dados."""
    if not rastreio:
        return ""
    ultimas = [pd.Timestamp(resultados_ml[t]["last_date"]) for t in (tickers or [])
               if resultados_ml and t in resultados_ml and "last_date" in resultados_ml[t]]
    dados = f" · dados até {max(ultimas).strftime('%d/%m/%Y')}" if ultimas else ""
    cal = rastreio.get("calibracao", {})
    cal_txt = " · ".join(f"D+{h} {cal[h]}" for h in sorted(cal))
    return (f'<div style="margin-top:6px;font-size:9.5px;color:#b8b2a6;letter-spacing:0.02em;text-transform:none;font-weight:400;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
            f'código {rastreio.get("versao_codigo", "?")}{dados} · gerado {rastreio.get("executado_em", "")}<br>'
            f'calibração: {cal_txt}</div>')


def build_html(resultados_ml: dict, etf_tickers: list[str],
               df_log: pd.DataFrame, my_tickers: list[str],
               ensemble_weights: dict,
               research_data: dict | None = None,
               context_warnings: list[str] | None = None,
               ampliado: dict | None = None,
               saude: dict | None = None,
               eventos: dict | None = None,
               risco: dict | None = None,
               calib_curves: dict | None = None,
               rastreio: dict | None = None) -> str:

    barcelona_tz = BARCELONA_TZ
    agora        = datetime.now(barcelona_tz)
    dia_semana   = _DIAS_PT[agora.weekday()]
    mes          = _MESES_PT[agora.month - 1]
    data_str     = (f"{dia_semana} · {agora.day} de {mes} de {agora.year} "
                    f"· {agora.strftime('%H:%M')}, Barcelona")

    validadas_portfolio = aligned_rows(df_log[
        (df_log["correct"].notna()) &
        (df_log["ticker"].isin(my_tickers))
    ])
    n_ativos      = len([t for t in my_tickers if t in resultados_ml])
    n_etfs        = len(etf_tickers)
    n_val         = len(validadas_portfolio)
    acc_global    = validadas_portfolio["correct"].astype(float).mean() if n_val >= 15 else None
    acc_kpi       = f"{acc_global*100:.0f}" if acc_global is not None else "—"
    hoje          = pd.Timestamp.now().normalize()
    ontem_bday    = (hoje - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    # ── Histórico das faixas de confiança (universo alinhado) ─────────────
    from evaluation.confianca import (faixas_confianca, ic_acuracia, acc_sempre_alta,
                                      p_binomial, MIN_N_FAIXA)
    try:
        faixas = faixas_confianca(df_log)
    except Exception as e:
        logger.warning("Faixas de confiança falharam (não bloqueia): %s", e)
        faixas = {}

    # ── ML prediction rows ────────────────────────────────────────────────
    tickers_sorted = sorted([t for t in my_tickers if t in resultados_ml])
    ml_rows = ""
    for i, ticker in enumerate(tickers_sorted):
        res      = resultados_ml[ticker]
        close    = res.get("close_eur", res["close_now"])
        var_1d   = res.get("var_1d", 0.0)
        preds    = res["preds_dict"]
        dirs     = [preds[day][0] for day in [1, 2, 3]]
        consenso = ("BULLISH" if all(x == "up"   for x in dirs) else
                    "BEARISH" if all(x == "down" for x in dirs) else "MISTO")

        ontem_ok = _acertou_ontem(ticker, df_log, ontem_bday)
        icon     = {"up_correct": "✅", "down_correct": "📉", "wrong": "❌"}.get(ontem_ok, "")
        icon_td  = f'<span style="font-size:12px;margin-right:3px">{icon}</span>' if icon else ""

        d1_dir, _, d1_conf = preds[1]
        d2_dir, _, d2_conf = preds[2]
        d3_dir, _, d3_conf = preds[3]

        border = "border-bottom:1px solid #f0ede5" if i < len(tickers_sorted) - 1 else ""
        ml_rows += f"""
        <tr style="{border}">
          <td style="padding:11px 6px 11px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740;letter-spacing:0.02em;white-space:nowrap">{icon_td}{ticker}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">€{close:,.2f}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_var_cell(var_1d)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_conf_cells(d1_dir, d1_conf, 1, calib_curves, faixas)}{_familias_line(ampliado, ticker, 1, d1_dir)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_conf_cells(d2_dir, d2_conf, 2, calib_curves, faixas)}{_familias_line(ampliado, ticker, 2, d2_dir)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_conf_cells(d3_dir, d3_conf, 3, calib_curves, faixas)}{_familias_line(ampliado, ticker, 3, d3_dir)}</td>
          <td style="padding:11px 0 11px 6px;text-align:right;white-space:nowrap">{_consenso_badge(consenso)}</td>
        </tr>"""

    etf_section_html = ""

    # ── Accuracy section ──────────────────────────────────────────────────
    estado, n_rec, faltam, acc_rec, delta, prec_up, n_up, prec_dn, n_dn = _calcular_tendencia(df_log, my_tickers)

    if estado == "treino":
        acc_number_html = f"""
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:48px;font-weight:500;color:#1a1740;line-height:1;letter-spacing:-0.02em">
            —
          </div>
          <div style="font-size:11.5px;color:#8a8a8a;margin-top:6px;line-height:1.4">
            {n_rec} previsões do portfólio validadas · ainda em treino<br>
            Acurácia disponível após {faltam} validações.
          </div>"""
        bar_width = 0
        tendencia_html = ""
    else:
        acc_pct        = acc_rec * 100
        bar_width      = min(100, max(0, acc_pct))
        acc_int        = f"{acc_pct:.1f}".rstrip("0").rstrip(".")
        if delta is None:
            tendencia_html = '<div style="margin-top:14px;font-size:11px;color:#8a8a8a;line-height:1.5;font-style:italic">Comparação com período anterior disponível em breve.</div>'
        elif delta > 0.01:
            tendencia_html = f'<div style="margin-top:14px;font-size:11px;color:#1e7a4c;font-weight:600">↑ +{delta:.1%} vs período anterior</div>'
        elif delta < -0.01:
            tendencia_html = f'<div style="margin-top:14px;font-size:11px;color:#b8453a;font-weight:600">↓ {delta:.1%} vs período anterior</div>'
        else:
            tendencia_html = '<div style="margin-top:14px;font-size:11px;color:#8a8a8a">→ Estável vs período anterior</div>'

        prec_up_str = f'{prec_up*100:.0f}%' if prec_up is not None else '-'
        prec_dn_str = f'{prec_dn*100:.0f}%' if prec_dn is not None else '-'
        p_up = p_binomial(round(prec_up * n_up), n_up) if prec_up is not None else None
        p_dn = p_binomial(round(prec_dn * n_dn), n_dn) if prec_dn is not None else None
        p_up_str = f' · p={p_up:.2f}' if p_up is not None else ''
        p_dn_str = f' · p={p_dn:.2f}' if p_dn is not None else ''
        rec_df = _recentes_d1(df_log, my_tickers)
        ic = ic_acuracia(rec_df["correct"], rec_df["target_date"])
        sempre = acc_sempre_alta(rec_df)
        ic_html = (f'<div style="font-size:11px;color:#5a5a5a;margin-top:6px">IC 95%: '
                   f'<strong>{ic["lo"]*100:.0f}% a {ic["hi"]*100:.0f}%</strong> '
                   f'<span style="color:#a0a0a0">({ic["metodo"]}, {ic["n_dias"]} dias)</span></div>'
                   if ic["acc"] is not None else "")
        sempre_html = (f'<div style="font-size:11px;color:#5a5a5a;margin-top:2px">Referência: "sempre ▲" acertaria '
                       f'<strong>{sempre*100:.0f}%</strong> no mesmo período</div>' if sempre is not None else "")
        tendencia_html += f"""
        <div style="margin-top:14px;font-size:11px;color:#5a5a5a;line-height:2.1;border-top:1px solid #ede9e0;padding-top:12px">
          <div>Quando prevê <span style="color:#1e7a4c;font-weight:700">▲</span> &nbsp;→&nbsp; <strong>{prec_up_str}</strong> de precisão <span style="color:#a0a0a0">({n_up} prev.{p_up_str})</span></div>
          <div>Quando prevê <span style="color:#b8453a;font-weight:700">▼</span> &nbsp;→&nbsp; <strong>{prec_dn_str}</strong> de precisão <span style="color:#a0a0a0">({n_dn} prev.{p_dn_str})</span></div>
          <div style="font-size:10px;color:#a8a39a;line-height:1.5;margin-top:4px">p alto (acima de 0,05) = diferença de 50% compatível com acaso. Precisão baixa numa direção não quer dizer que inverter o sinal funcionaria.</div>
        </div>"""
        acc_number_html = f"""
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:48px;font-weight:500;color:#1a1740;line-height:1;letter-spacing:-0.02em">
            {acc_int}<span style="font-size:24px;color:#7d75c4">%</span>
          </div>
          <div style="font-size:11.5px;color:#8a8a8a;margin-top:6px;line-height:1.4">
            {n_rec} previsões D+1 da carteira validadas<br>nos últimos 30 dias úteis
          </div>{ic_html}{sempre_html}"""

    # ── Context data warnings (VIX/SPY NaN forward-filled) ───────────────
    if context_warnings:
        warning_items = "".join(
            f'<div style="margin-bottom:3px">⚠️ {w}</div>' for w in context_warnings
        )
        ctx_warning_html = f"""
  <div style="padding:12px 36px;background:#fffbea;border-top:2px solid #f0c040">
    <div style="font-size:11px;color:#7a6010;line-height:1.6;font-weight:500">
      {warning_items}
    </div>
  </div>"""
    else:
        ctx_warning_html = ""

    html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carteira Inteligente · {agora.strftime('%d/%m/%Y')}</title>
<style>body {{ margin: 0; }}</style>
</head>
<body style="margin:0;padding:24px 16px;background:#ece9e2;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1a1a1a;-webkit-font-smoothing:antialiased">

<div style="max-width:680px;margin:0 auto;background:#fbfaf7;border-radius:6px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,0.04),0 12px 40px -12px rgba(38,33,92,0.18)">

  <!-- CABEÇALHO -->
  <div style="background:rgb(64,2,127);padding:32px 36px 28px">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:28px">
      <tr>
        <td style="vertical-align:middle">
          <div style="display:inline-block;width:8px;height:8px;background:#a8a0ff;border-radius:50%;margin-right:8px;vertical-align:middle"></div>
          <span style="color:#a8a0ff;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;vertical-align:middle">Carteira Inteligente</span>
        </td>
        <td style="vertical-align:middle;text-align:right">
          <span style="display:inline-block;border:1px solid rgba(168,160,255,0.35);border-radius:3px;padding:3px 9px;color:#a8a0ff;font-size:10px;font-weight:600;letter-spacing:0.12em;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">BOT</span>
        </td>
      </tr>
    </table>
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:34px;line-height:1.1;font-weight:500;letter-spacing:-0.01em;margin-bottom:6px">Relatório diário</div>
    <div style="color:#7d75c4;font-size:13px;letter-spacing:0.02em">{data_str}</div>
    <div style="color:#a8a0ff;font-size:11px;letter-spacing:0.06em;margin-top:4px">Actualizado: {agora.strftime('%d/%m/%Y - %H:%M')}</div>
    <table role="presentation" style="border-collapse:collapse;width:100%;margin-top:28px">
      <tr>
        <td style="padding:0;vertical-align:top;width:25%">
          <div style="color:#7d75c4;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Ativos</div>
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:28px;font-weight:500;line-height:1">{n_ativos}</div>
        </td>
        <td style="padding:0;vertical-align:top;width:25%;border-left:1px solid rgba(125,117,196,0.25);padding-left:18px">
          <div style="color:#7d75c4;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">ETFs</div>
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:28px;font-weight:500;line-height:1">{n_etfs}</div>
        </td>
        <td style="padding:0;vertical-align:top;width:25%;border-left:1px solid rgba(125,117,196,0.25);padding-left:18px">
          <div style="color:#7d75c4;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Validadas</div>
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:28px;font-weight:500;line-height:1">{n_val}<span style="font-size:12px;color:#7d75c4;margin-left:4px">portf.</span></div>
          <div style="color:#7d75c4;font-size:9.5px;margin-top:4px">alinhadas · D+1 a D+3</div>
        </td>
        <td style="padding:0;vertical-align:top;width:25%;border-left:1px solid rgba(125,117,196,0.25);padding-left:18px">
          <div style="color:#7d75c4;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Acurácia</div>
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:28px;font-weight:500;line-height:1">{acc_kpi}<span style="color:#7d75c4;font-size:18px">{'%' if acc_global is not None else ''}</span></div>
          <div style="color:#7d75c4;font-size:9.5px;margin-top:4px">carteira · todo o período</div>
        </td>
      </tr>
    </table>
    <a href="https://srxkatsumi.github.io/smart-wallet-ml/pt/" style="display:inline-block;margin-top:22px;background:rgba(168,160,255,0.15);border:1px solid rgba(168,160,255,0.35);border-radius:4px;padding:9px 18px;color:#f4f2ff;font-size:12px;font-weight:600;text-decoration:none;letter-spacing:0.02em">Ver no site →</a>
  </div>

  {_build_saude_html(saude) if saude else ctx_warning_html}
  {_build_eventos_html(eventos, tickers_sorted)}
  {_build_risco_html(risco)}

  <!-- PREVISÕES ML -->
  <div style="padding:32px 36px 24px">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:18px">
      <tr>
        <td style="vertical-align:baseline">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:20px;font-weight:500;color:#1a1740;letter-spacing:-0.005em">Previsões de curto prazo</div>
          <div style="font-size:12px;color:#8a8a8a;margin-top:2px">Probabilidade direcional · próximos 3 dias úteis</div>
        </td>
        <td style="vertical-align:baseline;text-align:right;font-size:10px;color:#aaa;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">{n_ativos} ativos</td>
      </tr>
    </table>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -36px;padding:0 36px">
    <table style="width:100%;border-collapse:collapse;min-width:520px">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 6px 8px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ativo</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Preço</th>
          <th style="text-align:center;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Var%</th>
          <th style="text-align:center;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+1</th>
          <th style="text-align:center;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+2</th>
          <th style="text-align:center;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+3</th>
          <th style="text-align:right;padding:0 0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Consenso</th>
        </tr>
      </thead>
      <tbody>{ml_rows}
      </tbody>
    </table>
    </div>
    <div style="margin-top:14px;font-size:10.5px;color:#a8a39a;line-height:1.7">
      <span style="color:#1e7a4c;font-weight:600">▲</span> probabilidade de alta &nbsp;·&nbsp;
      <span style="color:#b8453a;font-weight:600">▼</span> probabilidade de queda &nbsp;·&nbsp; valor em % de confiança<br>
      ✅ previu ▲ e subiu &nbsp;·&nbsp; 📉 previu ▼ e caiu &nbsp;·&nbsp; ❌ errou a previsão D+1 do dia anterior<br>
      Número principal = confiança <strong>calibrada</strong> contra os acertos reais (50 = o modelo não mostrou vantagem) &nbsp;·&nbsp; <span style="color:#a0a0a0">bruta</span> = o que os modelos disseram antes da calibração<br>
      13F: consenso das 13 famílias (ponderado) &nbsp;·&nbsp; <span style="color:#c07a1a;font-weight:600">≠</span> famílias discordam da previsão principal
    </div>
  </div>

  <!-- ENSEMBLE AMPLIADO (DESAFIANTE) -->
  {_build_ampliado_section(ampliado, tickers_sorted)}

  <!-- ETFs VISÃO DE LONGO PRAZO -->
  {etf_section_html}

  <!-- ACURÁCIA DO MODELO -->
  <div style="padding:28px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px">Acurácia do modelo · D+1 · últimos 30 dias úteis</div>
    <table role="presentation" style="width:100%;border-collapse:collapse">
      <tr>
        <td style="vertical-align:middle;width:50%;padding-right:24px">
          {acc_number_html}
        </td>
        <td style="vertical-align:middle;width:50%">
          <div style="background:#e6e1d2;height:6px;border-radius:3px;overflow:hidden;margin-bottom:8px">
            <div style="background:#1a1740;width:{bar_width:.1f}%;height:100%;border-radius:3px"></div>
          </div>
          <table role="presentation" style="width:100%;border-collapse:collapse">
            <tr>
              <td style="font-size:10px;color:#a89e85;letter-spacing:0.06em">0%</td>
              <td style="font-size:10px;color:#a89e85;letter-spacing:0.06em;text-align:right">100%</td>
            </tr>
          </table>
          {tendencia_html}
        </td>
      </tr>
    </table>
  </div>

  <!-- FEATURE IMPORTANCE DRIFT -->
  {_build_drift_section()}

  <!-- CORRELAÇÃO DE RETORNOS -->
  {_build_corr_section(resultados_ml, my_tickers)}

  <!-- 13 FAMÍLIAS (diário) -->
  {_build_research_section(research_data) if research_data else ""}

  <div style="padding:12px 36px;border-top:1px solid #efece4;font-size:10.5px;color:#a8a39a;line-height:1.6">
    Leitura mais longa (famílias com intervalo de confiança, calibração, diversificação e desafiantes): no <strong style="color:#5a5a5a">email semanal</strong>, que chega no sábado.
  </div>

  <!-- AVISO LEGAL -->
  <div style="padding:18px 36px;border-top:1px solid #efece4">
    <div style="font-size:10.5px;color:#a8a39a;line-height:1.55">
      <span style="color:#1a1740;font-weight:600">Aviso.</span> Previsões são probabilísticas. Não tomar decisões financeiras baseadas exclusivamente neste relatório.
    </div>
  </div>

  <!-- ASSINATURA -->
  <div style="padding:24px 36px;border-top:1px solid #efece4;background:#fbfaf7">
    <table role="presentation" style="width:100%;border-collapse:collapse">
      <tr>
        <td style="vertical-align:middle">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:17px;font-weight:500;color:#1a1740;letter-spacing:-0.005em;margin-bottom:3px">Vicky Costa Sanches</div>
          <div style="font-size:12px;color:#8a8a8a;font-style:italic">Tenha um excelente dia.</div>
        </td>
        <td style="vertical-align:middle;text-align:right;width:48px">
          <div style="width:42px;height:42px;border-radius:50%;background:#1a1740;color:#f4f2ff;text-align:center;line-height:42px;font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:15px;font-weight:500;letter-spacing:0.02em;display:inline-block">VC</div>
        </td>
      </tr>
    </table>
  </div>

</div>

<div style="max-width:680px;margin:14px auto 0;text-align:center;font-size:10px;color:#a8a39a;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">
  Carteira Inteligente · gerado automaticamente
  {_rodape_versao(rastreio, resultados_ml, my_tickers)}
</div>

</body>
</html>"""

    return html


def save_html(html: str):
    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML do email guardado em %s", HTML_REPORT)
