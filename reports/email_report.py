import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from config.settings import HTML_REPORT, BARCELONA_UTC_OFFSET, METADATA_FILE

logger = logging.getLogger(__name__)

_DIAS_PT  = ['Segunda-feira','Terça-feira','Quarta-feira','Quinta-feira',
             'Sexta-feira','Sábado','Domingo']
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
    mask = (
        (df_log["ticker"]    == ticker) &
        (df_log["pred_date"] == ontem_str) &
        (df_log["horizon"]   == 1) &
        (df_log["correct"].notna())
    )
    rows = df_log[mask]
    if rows.empty:
        return None
    return bool(rows.iloc[0]["correct"])


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
        return "treino", len(recentes), MIN_VAL - len(recentes), None, None
    acc_rec = recentes["correct"].astype(float).mean()
    acc_ant = ant["correct"].astype(float).mean() if len(ant) >= MIN_VAL else None
    delta   = acc_rec - acc_ant if acc_ant is not None else None
    return "ok", len(recentes), None, acc_rec, delta


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


def build_html(resultados_ml: dict, resumo_etfs: list[dict],
               df_log: pd.DataFrame, my_tickers: list[str],
               ensemble_weights: dict) -> str:

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
        icon     = ("✅" if ontem_ok is True else "❌" if ontem_ok is False else "")
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

    # ── ETF long-term projection rows ─────────────────────────────────────
    etfs_sorted = sorted(resumo_etfs, key=lambda x: x["ticker"])
    etf_rows = ""
    for i, r in enumerate(etfs_sorted):
        ticker = r["ticker"]
        preco  = r["close_eur"]
        if ticker in resultados_ml:
            df_t   = resultados_ml[ticker]["df"]
            n_dias = (df_t.index[-1] - df_t.index[0]).days
            taxa   = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0]) ** (365 / n_dias) - 1 \
                     if n_dias > 0 else 0.08
            taxa   = max(-0.30, min(taxa, 0.50))
        else:
            taxa = 0.08

        p1  = preco * (1 + taxa) ** 1
        p3  = preco * (1 + taxa) ** 3
        p5  = preco * (1 + taxa) ** 5
        p10 = preco * (1 + taxa) ** 10

        taxa_pct = taxa * 100
        taxa_col = ("#b8453a" if taxa_pct > 30 else
                    "#c2891c" if taxa_pct > 15 else "#1e7a4c")

        border = "border-bottom:1px solid #f0ede5" if i < len(etfs_sorted) - 1 else ""
        etf_rows += f"""
        <tr style="{border}">
          <td style="padding:11px 6px 11px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;font-weight:600;color:#1a1740;letter-spacing:0.02em;white-space:nowrap">{ticker}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#7c7c7c;font-variant-numeric:tabular-nums;white-space:nowrap">{preco:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{p1:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{p3:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{p5:,.2f}</td>
          <td style="padding:11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1a1a1a;font-variant-numeric:tabular-nums;white-space:nowrap">{p10:,.2f}</td>
          <td style="padding:11px 0 11px 6px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;font-weight:700;color:{taxa_col};font-variant-numeric:tabular-nums;white-space:nowrap">+{taxa_pct:.1f}%</td>
        </tr>"""

    # ── Accuracy section ──────────────────────────────────────────────────
    estado, n_rec, faltam, acc_rec, delta = _calcular_tendencia(df_log, my_tickers)

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
        acc_number_html = f"""
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:48px;font-weight:500;color:#1a1740;line-height:1;letter-spacing:-0.02em">
            {acc_int}<span style="font-size:24px;color:#7d75c4">%</span>
          </div>
          <div style="font-size:11.5px;color:#8a8a8a;margin-top:6px;line-height:1.4">
            {n_rec} previsões do portfólio validadas<br>nos últimos 30 dias úteis
          </div>"""

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
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:12px">
      <tr>
        <td style="vertical-align:middle">
          <span style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase">Feature importance drift</span>
        </td>
        <td style="vertical-align:middle;text-align:right">
          <span style="display:inline-block;background:{badge_bg};color:{badge_col};border-radius:3px;padding:3px 9px;font-size:10px;font-weight:700;letter-spacing:0.06em">{badge_txt}</span>
        </td>
      </tr>
    </table>
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
      ✅ acertou a previsão D+1 do dia anterior &nbsp;·&nbsp; ❌ errou a previsão D+1 do dia anterior
    </div>
  </div>

  <!-- ETFs LONGO PRAZO -->
  <div style="padding:24px 36px;border-top:1px solid #efece4">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:18px">
      <tr>
        <td style="vertical-align:baseline">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:20px;font-weight:500;color:#1a1740;letter-spacing:-0.005em">Projeção de longo prazo</div>
          <div style="font-size:12px;color:#8a8a8a;margin-top:2px">ETFs · preço por unidade, taxa histórica anualizada</div>
        </td>
        <td style="vertical-align:baseline;text-align:right;font-size:10px;color:#aaa;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">{n_etfs} ETFs</td>
      </tr>
    </table>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -36px;padding:0 36px">
    <table style="width:100%;border-collapse:collapse;min-width:480px">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 6px 8px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">ETF</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Atual</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">1a</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">3a</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">5a</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">10a</th>
          <th style="text-align:right;padding:0 0 8px 6px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Taxa</th>
        </tr>
      </thead>
      <tbody>{etf_rows}
      </tbody>
    </table>
    </div>
    <div style="margin-top:14px;font-size:10.5px;color:#a8a39a;line-height:1.55;border-left:2px solid #efece4;padding-left:10px">
      Valores em € por unidade · taxa anualizada com base no histórico. Não inclui aportes mensais nem garante retorno futuro.
    </div>
  </div>

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
