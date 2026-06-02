import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from config.settings import (
    HTML_REPORT, BARCELONA_UTC_OFFSET, METADATA_FILE, TAXA_CRESCIMENTO_BASE,
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

        df_c  = pd.DataFrame(closes).pct_change().dropna()
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


def _alvo_cell(close_eur: float, alvo_15_eur: float) -> str:
    if close_eur >= alvo_15_eur:
        pct = (close_eur / (alvo_15_eur / 1.15) - 1) * 100
        return (
            '<span style="display:inline-block;background:#e7f3eb;color:#1e7a4c;border-radius:3px;'
            'padding:3px 7px;font-size:10px;font-weight:700;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;white-space:nowrap">✅ +{pct:.0f}%</span>'
        )
    falta_pct = (alvo_15_eur - close_eur) / alvo_15_eur * 100
    if falta_pct <= 5:
        return (
            '<span style="display:inline-block;background:#fffbea;color:#7a6010;border-radius:3px;'
            'padding:3px 7px;font-size:10px;font-weight:700;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;white-space:nowrap">⚠️ −{falta_pct:.1f}%</span>'
        )
    return (
        f'<span style="font-size:11px;color:#a0a0a0;font-family:ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,monospace;white-space:nowrap">−{falta_pct:.0f}%</span>'
    )


def _pct_feito_cell(preco_compra_eur: float, close_eur: float) -> str:
    pct = (close_eur - preco_compra_eur) / preco_compra_eur * 100
    if pct >= 15:
        return (
            '<span style="display:inline-block;background:#e7f3eb;color:#1e7a4c;border-radius:3px;'
            'padding:2px 6px;font-size:10px;font-weight:700;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;white-space:nowrap">✅ +{pct:.1f}%</span>'
        )
    if pct >= 0:
        return (
            f'<span style="font-size:11px;color:#1e7a4c;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;white-space:nowrap">+{pct:.1f}%</span>'
        )
    return (
        f'<span style="font-size:11px;color:#b8453a;font-family:ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,monospace;white-space:nowrap">{pct:.1f}%</span>'
    )


def _pct_pendente_cell(close_eur: float, alvo_15_eur: float, preco_compra_eur: float) -> str:
    if close_eur >= alvo_15_eur:
        return '<span style="font-size:10px;color:#1e7a4c;font-weight:700">✅ Atingido</span>'
    falta_pct = (alvo_15_eur - close_eur) / preco_compra_eur * 100
    if falta_pct <= 2:
        return (
            f'<span style="display:inline-block;background:#fffbea;color:#7a6010;border-radius:3px;'
            f'padding:2px 6px;font-size:10px;font-weight:700;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;white-space:nowrap">⚠️ {falta_pct:.1f}%</span>'
        )
    return (
        f'<span style="font-size:11px;color:#a0a0a0;font-family:ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,monospace;white-space:nowrap">{falta_pct:.1f}%</span>'
    )


def _delta_vs_compra_cell(preco_compra_eur: float, close_eur: float) -> str:
    delta = (close_eur - preco_compra_eur) / preco_compra_eur * 100
    if delta >= 0:
        return (
            f'<span style="font-size:11px;color:#1e7a4c;font-weight:600;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap">+{delta:.1f}%</span>'
        )
    return (
        f'<span style="font-size:11px;color:#b8453a;font-weight:600;font-family:ui-monospace,'
        f'SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap">{delta:.1f}%</span>'
    )


def _is_first_business_week() -> bool:
    today = pd.Timestamp.today()
    return today.day <= 7 and today.weekday() < 5


def _build_etf_monthly_recommendation(etf_lotes: list[dict], resultados_ml: dict) -> str:
    if not _is_first_business_week():
        return ""
    seen = set()
    rec_rows = ""
    today = pd.Timestamp.today().normalize()
    for row in etf_lotes:
        ticker = row["ticker"]
        if ticker in seen or ticker not in resultados_ml:
            continue
        seen.add(ticker)
        preds = resultados_ml[ticker].get("preds_dict", {})
        best_day, best_price = None, float("inf")
        day_labels = {
            h: (today + pd.offsets.BDay(h)).strftime("%d/%m/%Y")
            for h in [1, 2, 3]
        }
        for h in [1, 2, 3]:
            if h not in preds:
                continue
            direction, pred_price, _ = preds[h]
            if direction == "down" and pred_price < best_price:
                best_price = pred_price
                best_day = h
        if best_day is None:
            for h in [1, 2, 3]:
                if h in preds:
                    _, pred_price, _ = preds[h]
                    if pred_price < best_price:
                        best_price = pred_price
                        best_day = h
        if best_day is None:
            continue
        label = day_labels.get(best_day, f"D+{best_day}")
        rec_rows += (
            f'<tr><td style="padding:6px 8px 6px 0;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740">{ticker}</td>'
            f'<td style="padding:6px 8px;font-size:12px;color:#1a1a1a">{label}</td>'
            f'<td style="padding:6px 0 6px 8px;text-align:right;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1e7a4c;font-weight:600">'
            f'{best_price:,.2f} €</td></tr>'
        )
    if not rec_rows:
        return ""
    return f"""
  <div style="padding:20px 36px;border-top:2px solid #e7f3eb;background:#f4faf6">
    <div style="font-size:10px;font-weight:600;color:#1e7a4c;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px">Recomendação de Compra — Esta Semana</div>
    <div style="font-size:11.5px;color:#5a5a5a;margin-bottom:12px">Melhor dia previsto para comprar na baixa (D+1 a D+3):</div>
    <table style="border-collapse:collapse">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 8px 6px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #c8e6d0">ETF</th>
          <th style="text-align:left;padding:0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #c8e6d0">Melhor dia</th>
          <th style="text-align:right;padding:0 0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #c8e6d0">Preço estimado</th>
        </tr>
      </thead>
      <tbody>{rec_rows}</tbody>
    </table>
  </div>"""


def _pred_price_cell(pred_price: float, close_eur: float, direction: str) -> str:
    color = "#1e7a4c" if direction == "up" else "#b8453a"
    arrow = "▲" if direction == "up" else "▼"
    return (
        f'<span style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
        f'font-size:11px;color:{color};font-variant-numeric:tabular-nums;white-space:nowrap">'
        f'{arrow}&nbsp;{pred_price:,.2f}</span>'
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


def _calcular_tendencia(df_log: pd.DataFrame, portfolio_tickers: list) -> tuple:
    validadas = df_log[
        (df_log["correct"].notna()) &
        (df_log["ticker"].isin(portfolio_tickers))
    ].copy()
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

    is_drift = corr < 0.70 or today_ranked[0] != ref_ranked[0]

    return {
        "enough_data": True,
        "is_drift":    is_drift,
        "corr":        corr,
        "top5":        top5,
        "n_ref_days":  len(dates) - 1,
    }


def _build_research_section(research_data: dict) -> str:
    """Secção de segunda-feira com comparação de famílias e consenso."""
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
    }
    comp_rows = ""
    for r in comparison:
        acc_pct  = r["accuracy"] * 100
        vs_pct   = r["vs_acaso"] * 100
        sign     = "+" if vs_pct >= 0 else ""
        vs_color = "#1e7a4c" if vs_pct >= 0 else "#b8453a"
        star     = " ★" if r == comparison[0] else ""
        comp_rows += (
            f'<tr><td style="padding:7px 8px 7px 0;font-size:12px;color:#1a1a1a">'
            f'{family_labels.get(r["family"], r["family"])}{star}</td>'
            f'<td style="padding:7px 8px;text-align:right;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;'
            f'color:#1a1740">{acc_pct:.0f}%</td>'
            f'<td style="padding:7px 0 7px 8px;text-align:right;font-size:12px;'
            f'color:{vs_color};font-weight:600">{sign}{vs_pct:.0f}%</td></tr>'
        )

    # ── tabela de consenso ────────────────────────────────────────────────
    cons_rows = ""
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
            f'color:#5a5a5a">{r["up_count"]}/{r["total"]} modelos</td>'
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
    <div style="font-size:10px;font-weight:600;color:#7d75c4;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:16px">Segunda-feira · Análise de investigação semanal</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
      <div>
        <div style="font-size:13px;font-weight:600;color:#1a1740;margin-bottom:10px">Comparação de famílias — semana anterior</div>
        {comp_html}
        <div style="margin-top:8px;font-size:10px;color:#a8a39a">★ melhor família da semana &nbsp;·&nbsp; vs Acaso = diferença face a 50%</div>
      </div>
      <div>
        <div style="font-size:13px;font-weight:600;color:#1a1740;margin-bottom:10px">Consenso dos 25 modelos — amanhã</div>
        {cons_html}
        <div style="margin-top:8px;font-size:10px;color:#a8a39a">Consenso abaixo de 60% = sinal fraco &nbsp;·&nbsp; ▲▲▲ = ≥75% modelos de acordo</div>
      </div>
    </div>
  </div>"""


def build_html(resultados_ml: dict, resumo_etfs: list[dict],
               df_log: pd.DataFrame, my_tickers: list[str],
               ensemble_weights: dict,
               resumo_etoro: list[dict] | None = None,
               resumo_etoro_lotes: list[dict] | None = None,
               resumo_etf_lotes: list[dict] | None = None,
               research_data: dict | None = None,
               context_warnings: list[str] | None = None) -> str:

    barcelona_tz = timezone(timedelta(hours=BARCELONA_UTC_OFFSET))
    agora        = datetime.now(barcelona_tz)
    dia_semana   = _DIAS_PT[agora.weekday()]
    mes          = _MESES_PT[agora.month - 1]
    data_str     = (f"{dia_semana} · {agora.day} de {mes} de {agora.year} "
                    f"· {agora.strftime('%H:%M')}, Barcelona")

    validadas_portfolio = df_log[
        (df_log["correct"].notna()) &
        (df_log["ticker"].isin(my_tickers))
    ]
    n_ativos      = len([t for t in my_tickers if t in resultados_ml])
    n_etfs        = len(resumo_etfs)
    n_val         = len(validadas_portfolio)
    acc_global    = validadas_portfolio["correct"].astype(float).mean() if n_val >= 15 else None
    acc_kpi       = f"{acc_global*100:.0f}" if acc_global is not None else "—"
    hoje          = pd.Timestamp.now().normalize()
    ontem_bday    = (hoje - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

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
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{close:,.2f}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_var_cell(var_1d)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_dir_cell(d1_dir, d1_conf)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_dir_cell(d2_dir, d2_conf)}</td>
          <td style="padding:11px 6px;text-align:center;white-space:nowrap">{_dir_cell(d3_dir, d3_conf)}</td>
          <td style="padding:11px 0 11px 6px;text-align:right;white-space:nowrap">{_consenso_badge(consenso)}</td>
        </tr>"""

    # ── ETF per-lot rows ──────────────────────────────────────────────────
    etf_lot_rows = ""
    if resumo_etf_lotes:
        for i, r in enumerate(resumo_etf_lotes):
            ticker    = r["ticker"]
            preco_c   = r["preco_unidade_eur"]
            close     = r["close_eur"]
            data_c    = r["data_compra"] or "—"
            preds     = resultados_ml.get(ticker, {}).get("preds_dict", {})
            d1 = _pred_price_cell(preds[1][1], close, preds[1][0]) if 1 in preds else "—"
            d2 = _pred_price_cell(preds[2][1], close, preds[2][0]) if 2 in preds else "—"
            d3 = _pred_price_cell(preds[3][1], close, preds[3][0]) if 3 in preds else "—"
            border = "border-bottom:1px solid #f0ede5" if i < len(resumo_etf_lotes) - 1 else ""
            etf_lot_rows += f"""
        <tr style="{border}">
          <td style="padding:11px 6px 11px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740;white-space:nowrap">{ticker}</td>
          <td style="padding:11px 6px;font-size:11px;color:#8a8a8a;white-space:nowrap">{data_c}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#7c7c7c;font-variant-numeric:tabular-nums;white-space:nowrap">{preco_c:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{close:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{d1}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{d2}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{d3}</td>
          <td style="padding:11px 0 11px 6px;text-align:right;white-space:nowrap">{_delta_vs_compra_cell(preco_c, close)}</td>
        </tr>"""

    # ── Price forecast table (eToro stocks) — per lot ─────────────────────
    prev_rows = ""
    if resumo_etoro_lotes:
        for i, r in enumerate(resumo_etoro_lotes):
            ticker  = r["ticker"]
            if ticker not in resultados_ml:
                continue
            close   = r["close_eur"]
            preco_c = r["preco_compra_eur"]
            alvo    = r["alvo_15_eur"]
            data_c  = r["data_compra"] or "—"
            preds   = resultados_ml[ticker]["preds_dict"]
            border  = "border-bottom:1px solid #f0ede5" if i < len(resumo_etoro_lotes) - 1 else ""
            prev_rows += f"""
        <tr style="{border}">
          <td style="padding:11px 6px 11px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740;white-space:nowrap">{ticker}</td>
          <td style="padding:11px 6px;font-size:11px;color:#8a8a8a;white-space:nowrap">{data_c}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#7c7c7c;font-variant-numeric:tabular-nums;white-space:nowrap">{preco_c:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{close:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{_pred_price_cell(preds[1][1], close, preds[1][0])}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{_pred_price_cell(preds[2][1], close, preds[2][0])}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{_pred_price_cell(preds[3][1], close, preds[3][0])}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#5a5a5a">{alvo:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;white-space:nowrap">{_pct_feito_cell(preco_c, close)}</td>
          <td style="padding:11px 0 11px 6px;text-align:right;white-space:nowrap">{_pct_pendente_cell(close, alvo, preco_c)}</td>
        </tr>"""

    if prev_rows:
        prev_section_html = f"""
  <div style="padding:24px 36px;border-top:1px solid #efece4">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:18px">
      <tr>
        <td style="vertical-align:baseline">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:20px;font-weight:500;color:#1a1740;letter-spacing:-0.005em">Previsões de preço</div>
          <div style="font-size:12px;color:#8a8a8a;margin-top:2px">Ações eToro · estimativa ATR por horizonte · alvo +15% por lote</div>
        </td>
      </tr>
    </table>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -36px;padding:0 36px">
    <table style="width:100%;border-collapse:collapse;min-width:700px">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 6px 8px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ativo</th>
          <th style="text-align:left;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Compra</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Preço €</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ontem</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+1</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+2</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+3</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Alvo €</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">% feito</th>
          <th style="text-align:right;padding:0 0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">% falta</th>
        </tr>
      </thead>
      <tbody>{prev_rows}
      </tbody>
    </table>
    </div>
    <div style="margin-top:14px;font-size:10.5px;color:#a8a39a;line-height:1.55;border-left:2px solid #efece4;padding-left:10px">
      Preço € = valor de compra em euros · Alvo = preço de compra × 1,15 · % feito = ganho atual · % falta = quanto resta para atingir o alvo
    </div>
  </div>"""
    else:
        prev_section_html = ""

    # ── ETF lot section HTML ──────────────────────────────────────────────
    monthly_rec_html = _build_etf_monthly_recommendation(resumo_etf_lotes or [], resultados_ml)
    if etf_lot_rows:
        etf_section_html = f"""
  {monthly_rec_html}
  <div style="padding:24px 36px;border-top:1px solid #efece4">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:18px">
      <tr>
        <td style="vertical-align:baseline">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:20px;font-weight:500;color:#1a1740;letter-spacing:-0.005em">ETFs — Visão de Longo Prazo</div>
          <div style="font-size:12px;color:#8a8a8a;margin-top:2px">Por lote de compra · previsão curto prazo · variação vs entrada</div>
        </td>
        <td style="vertical-align:baseline;text-align:right;font-size:10px;color:#aaa;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">{n_etfs} ETFs</td>
      </tr>
    </table>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -36px;padding:0 36px">
    <table style="width:100%;border-collapse:collapse;min-width:600px">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 6px 8px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">ETF</th>
          <th style="text-align:left;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Compra</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Preço €</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ontem</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+1</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+2</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">D+3</th>
          <th style="text-align:right;padding:0 0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Δ vs compra</th>
        </tr>
      </thead>
      <tbody>{etf_lot_rows}
      </tbody>
    </table>
    </div>
    <div style="margin-top:14px;font-size:10.5px;color:#a8a39a;line-height:1.55;border-left:2px solid #efece4;padding-left:10px">
      Preço € = valor de entrada por unidade · Δ = variação desde a compra · ETFs são posições de longo prazo sem alvo de saída fixo.
    </div>
  </div>"""
    else:
        etf_section_html = monthly_rec_html

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

        prec_up_str = f'{prec_up*100:.0f}%' if prec_up is not None else '—'
        prec_dn_str = f'{prec_dn*100:.0f}%' if prec_dn is not None else '—'
        tendencia_html += f"""
        <div style="margin-top:14px;font-size:11px;color:#5a5a5a;line-height:2.1;border-top:1px solid #ede9e0;padding-top:12px">
          <div>Quando prevê <span style="color:#1e7a4c;font-weight:700">▲</span> &nbsp;→&nbsp; <strong>{prec_up_str}</strong> de precisão <span style="color:#a0a0a0">({n_up} prev.)</span></div>
          <div>Quando prevê <span style="color:#b8453a;font-weight:700">▼</span> &nbsp;→&nbsp; <strong>{prec_dn_str}</strong> de precisão <span style="color:#a0a0a0">({n_dn} prev.)</span></div>
        </div>"""
        acc_number_html = f"""
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:48px;font-weight:500;color:#1a1740;line-height:1;letter-spacing:-0.02em">
            {acc_int}<span style="font-size:24px;color:#7d75c4">%</span>
          </div>
          <div style="font-size:11.5px;color:#8a8a8a;margin-top:6px;line-height:1.4">
            {n_rec} previsões do portfólio validadas<br>nos últimos 30 dias úteis
          </div>"""

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
        badge_txt = f"⚠️ Drift detectado · ρ={corr:.2f}" if is_drift else f"✅ Estável · ρ={corr:.2f}"

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
      <strong style="color:#7a6a5a">ρ</strong> = correlação de Spearman entre os dois rankings · ρ próximo de 1 = padrão estável · ρ &lt; 0,70 = sinal de drift.<br>
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
        </td>
        <td style="padding:0;vertical-align:top;width:25%;border-left:1px solid rgba(125,117,196,0.25);padding-left:18px">
          <div style="color:#7d75c4;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Acurácia</div>
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#f4f2ff;font-size:28px;font-weight:500;line-height:1">{acc_kpi}<span style="color:#7d75c4;font-size:18px">{'%' if acc_global is not None else ''}</span></div>
        </td>
      </tr>
    </table>
  </div>

  {ctx_warning_html}

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
      ✅ previu ▲ e subiu &nbsp;·&nbsp; 📉 previu ▼ e caiu &nbsp;·&nbsp; ❌ errou a previsão D+1 do dia anterior
    </div>
  </div>

  <!-- ETFs VISÃO DE LONGO PRAZO -->
  {etf_section_html}

  <!-- PREVISÕES DE PREÇO -->
  {prev_section_html}

  <!-- ACURÁCIA DO MODELO -->
  <div style="padding:28px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px">Acurácia do modelo</div>
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
  {drift_html}

  <!-- CORRELAÇÃO DE RETORNOS -->
  {corr_html}

  <!-- INVESTIGAÇÃO SEMANAL (apenas segunda-feira) -->
  {_build_research_section(research_data) if research_data else ""}

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
</div>

</body>
</html>"""

    return html


def save_html(html: str):
    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML do email guardado em %s", HTML_REPORT)
