import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from config.settings import EXPERIMENTO_HTML_REPORT, BARCELONA_UTC_OFFSET, MIN_N_EXPERIMENTO

logger = logging.getLogger(__name__)

_DIAS_PT  = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
             "Sexta-feira", "Sábado", "Domingo"]
_MESES_PT = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
             'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']


def _ret_cell(ret: float) -> str:
    pct = ret * 100
    color = "#1e7a4c" if pct >= 0 else "#b8453a"
    arrow = "▲" if pct >= 0 else "▼"
    return (f'<span style="color:{color};font-weight:600;font-size:11px;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
            f'{arrow}&nbsp;{abs(pct):.2f}%</span>')


def _build_previsoes_html(resultados_exp: dict, resultados_ml: dict, my_tickers: list) -> str:
    rows = ""
    for ticker in sorted(t for t in my_tickers if t in resultados_exp):
        exp_res = resultados_exp[ticker]
        ml_res  = resultados_ml.get(ticker, {})
        close_now = exp_res["close_now"]
        row_cells = (
            f'<td style="padding:8px 6px;font-size:12px;font-weight:600;color:#1a1740">{ticker}</td>'
            f'<td style="padding:8px 6px;text-align:right;font-size:11px;color:#5a5a5a;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace">€{close_now:,.2f}</td>'
        )
        for day in [1, 2, 3]:
            h = exp_res["horizons"].get(day)
            if h is None:
                row_cells += '<td style="padding:8px 6px;text-align:center;color:#ccc">—</td>'
                continue
            champion_price = None
            if ml_res and day in ml_res.get("preds_dict", {}):
                champion_price = ml_res["preds_dict"][day][1]
            cell = (
                f'<div>{_ret_cell(h["pred_ret"])}</div>'
                f'<div style="font-size:9.5px;color:#a8a39a;margin-top:2px">'
                f'€{h["pred_price"]:,.2f} [€{h["interval_lo"]:,.2f}–€{h["interval_hi"]:,.2f}]</div>'
            )
            if champion_price is not None:
                cell += (f'<div style="font-size:9px;color:#c9c2b3;margin-top:1px">'
                         f'campeão: €{champion_price:,.2f}</div>')
            row_cells += f'<td style="padding:8px 6px;text-align:center">{cell}</td>'
        rows += f'<tr style="border-bottom:1px solid #f0ede4">{row_cells}</tr>'

    header_cols = "".join(
        f'<th style="text-align:center;padding:0 6px 8px;color:#a0a0a0;font-weight:500;'
        f'font-size:10px;letter-spacing:0.1em;text-transform:uppercase;'
        f'border-bottom:1px solid #e6e3dc">D+{d}</th>'
        for d in [1, 2, 3]
    )
    return f"""
    <div style="overflow-x:auto;margin:0 -36px;padding:0 36px">
    <table style="width:100%;border-collapse:collapse;min-width:520px">
      <thead>
        <tr>
          <th style="text-align:left;padding:0 6px 8px 0;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Ativo</th>
          <th style="text-align:right;padding:0 6px 8px;color:#a0a0a0;font-weight:500;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #e6e3dc">Preço</th>
          {header_cols}
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    <div style="margin-top:12px;font-size:10.5px;color:#a8a39a;line-height:1.7">
      Retorno % previsto pelo desafiante (regressão) · preço-alvo e intervalo de 90% entre colchetes ·
      "campeão" = heurística ATR já usada no email principal, para comparação.
    </div>"""


def _build_estado_experimento_html(sig: dict) -> str:
    rows = ""
    for day in [1, 2, 3]:
        r = sig.get(f"d{day}", {})
        if not r.get("ready"):
            n = r.get("n", 0)
            rows += (
                f'<tr style="border-bottom:1px solid #f0ede4">'
                f'<td style="padding:8px 6px;font-size:12px;font-weight:600;color:#1a1740">D+{day}</td>'
                f'<td colspan="4" style="padding:8px 6px;font-size:11px;color:#a8a39a">'
                f'⏳ coletando dados — N={n}/{MIN_N_EXPERIMENTO}</td></tr>'
            )
            continue
        veredito = ("🏆 desafiante venceu" if r["challenger_wins"] else
                     ("— campeão manteve" if r["sig"] else "— sem diferença significativa"))
        guardrail = "✅" if r["guardrail_mae_ok"] else "⚠️"
        rows += (
            '<tr style="border-bottom:1px solid #f0ede4">'
            f'<td style="padding:8px 6px;font-size:12px;font-weight:600;color:#1a1740">D+{day}</td>'
            f'<td style="padding:8px 6px;font-size:11px;text-align:center;color:#5a5a5a">n={r["n"]}</td>'
            f'<td style="padding:8px 6px;font-size:11px;text-align:center;color:#5a5a5a">'
            f'MAE campeão {r["mae_champion"]*100:.2f}% · desafiante {r["mae_challenger"]*100:.2f}% {guardrail}</td>'
            f'<td style="padding:8px 6px;font-size:11px;text-align:center;color:#5a5a5a">DM p={r["dm_p"]:.3f}</td>'
            f'<td style="padding:8px 6px;font-size:11px;text-align:right;font-weight:600;color:#1a1740">{veredito}</td>'
            '</tr>'
        )
    return f"""
    <table style="width:100%;border-collapse:collapse">
      <tbody>{rows}</tbody>
    </table>
    <div style="margin-top:12px;font-size:10.5px;color:#a8a39a;line-height:1.7">
      OEC: teste de Diebold-Mariano no erro quadrático pareado (desafiante vs. campeão), N mínimo = {MIN_N_EXPERIMENTO}
      por horizonte antes de qualquer veredito — sem espiar resultados prematuros.
      Guardrail: MAE do desafiante não pode ser &gt;20% pior que o campeão (⚠️ se violado).
    </div>"""


def _build_backtest_html(backtest: dict) -> str:
    tickers = backtest.get("tickers", {})
    if not tickers:
        return '<div style="font-size:11px;color:#a8a39a">Backtest ainda não disponível.</div>'
    rows = ""
    for ticker in sorted(tickers):
        d1 = tickers[ticker].get("d1")
        if not d1:
            continue
        r2 = d1.get("r2")
        r2_str = f"{r2:.3f}" if r2 is not None else "—"
        rows += (
            '<tr style="border-bottom:1px solid #f0ede4">'
            f'<td style="padding:6px;font-size:11px;font-weight:600;color:#1a1740">{ticker}</td>'
            f'<td style="padding:6px;font-size:11px;text-align:center;color:#5a5a5a">n={d1["n"]}</td>'
            f'<td style="padding:6px;font-size:11px;text-align:center;color:#5a5a5a">MSE modelo {d1["mse_model"]:.5f}</td>'
            f'<td style="padding:6px;font-size:11px;text-align:center;color:#5a5a5a">MSE naive {d1["mse_naive"]:.5f}</td>'
            f'<td style="padding:6px;font-size:11px;text-align:right;color:#5a5a5a">R² {r2_str}</td>'
            '</tr>'
        )
    return f"""
    <table style="width:100%;border-collapse:collapse">
      <tbody>{rows}</tbody>
    </table>
    <div style="margin-top:10px;font-size:10.5px;color:#a8a39a;line-height:1.7">
      D+1, expanding window sobre 2 anos de histórico. Baseline "naive" = prever retorno zero
      (proxy simplificado do campeão para esta leitura retrospectiva — não é a heurística ATR exata).
      R² &gt; 0 significa que o modelo bate o baseline ingênuo.
    </div>"""


def build_html_experimentos(resultados_exp: dict, resultados_ml: dict, df_exp_log: pd.DataFrame,
                             sig: dict, backtest: dict, my_tickers: list) -> str:
    tz    = timezone(timedelta(hours=BARCELONA_UTC_OFFSET))
    agora = datetime.now(tz)
    data_str = f"{_DIAS_PT[agora.weekday()]} · {agora.day} de {_MESES_PT[agora.month-1]} de {agora.year}"

    n_ativos    = len(resultados_exp)
    n_validadas = int((df_exp_log["validated"] == True).sum()) if not df_exp_log.empty else 0

    previsoes_html = _build_previsoes_html(resultados_exp, resultados_ml, my_tickers)
    estado_html    = _build_estado_experimento_html(sig)
    backtest_html  = _build_backtest_html(backtest)

    html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carteira BOT Experimentos · {agora.strftime('%d/%m/%Y')}</title>
<style>body {{ margin: 0; }}</style>
</head>
<body style="margin:0;padding:24px 16px;background:#ece9e2;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1a1a1a;-webkit-font-smoothing:antialiased">

<div style="max-width:680px;margin:0 auto;background:#fbfaf7;border-radius:6px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,0.04),0 12px 40px -12px rgba(127,60,2,0.18)">

  <!-- CABEÇALHO -->
  <div style="background:rgb(127,60,2);padding:32px 36px 28px">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:14px">
      <tr>
        <td style="vertical-align:middle">
          <div style="display:inline-block;width:8px;height:8px;background:#ffc98a;border-radius:50%;margin-right:8px;vertical-align:middle"></div>
          <span style="color:#ffc98a;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;vertical-align:middle">Carteira BOT Experimentos</span>
        </td>
        <td style="vertical-align:middle;text-align:right">
          <span style="display:inline-block;border:1px solid rgba(255,201,138,0.35);border-radius:3px;padding:3px 9px;color:#ffc98a;font-size:10px;font-weight:600;letter-spacing:0.12em;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">🧪 EXPERIMENTAL</span>
        </td>
      </tr>
    </table>
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#fff6ea;font-size:30px;line-height:1.1;font-weight:500;letter-spacing:-0.01em;margin-bottom:6px">Desafiante de regressão</div>
    <div style="color:#e0b98a;font-size:13px;letter-spacing:0.02em">{data_str}</div>
    <div style="background:rgba(255,255,255,0.1);border-radius:4px;padding:10px 14px;margin-top:18px;font-size:11.5px;color:#fff6ea;line-height:1.5">
      ⚠️ Não usar para decisões financeiras. Isto é um teste A/B: RandomForestRegressor (desafiante) vs.
      heurística ATR atual (campeão), acompanhado com rigor estatístico (Diebold-Mariano, sem espiar antes de N={MIN_N_EXPERIMENTO}).
    </div>
  </div>

  <!-- PREVISÕES DE HOJE -->
  <div style="padding:32px 36px 24px">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:18px">
      <tr>
        <td style="vertical-align:baseline">
          <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;font-size:20px;font-weight:500;color:#1a1740;letter-spacing:-0.005em">Previsões de hoje</div>
          <div style="font-size:12px;color:#8a8a8a;margin-top:2px">Retorno previsto · preço-alvo · intervalo de 90%</div>
        </td>
        <td style="vertical-align:baseline;text-align:right;font-size:10px;color:#aaa;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">{n_ativos} ativos</td>
      </tr>
    </table>
    {previsoes_html}
  </div>

  <!-- ESTADO DO EXPERIMENTO -->
  <div style="padding:28px 36px;border-top:1px solid #efece4;background:#f6f3eb">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px">Estado do experimento (ao vivo) — {n_validadas} previsões validadas</div>
    {estado_html}
  </div>

  <!-- BACKTEST HISTÓRICO -->
  <div style="padding:28px 36px">
    <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px">Backtest histórico (leitura inicial, 2 anos)</div>
    {backtest_html}
  </div>

  <!-- AVISO LEGAL -->
  <div style="padding:18px 36px;border-top:1px solid #efece4">
    <div style="font-size:10.5px;color:#a8a39a;line-height:1.55">
      <span style="color:#7f3c02;font-weight:600">Aviso.</span> Email experimental separado da Carteira Inteligente principal.
      Previsões de valor não têm validação estatística suficiente até N={MIN_N_EXPERIMENTO} por horizonte.
    </div>
  </div>

</div>

<div style="max-width:680px;margin:14px auto 0;text-align:center;font-size:10px;color:#a8a39a;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">
  Carteira BOT Experimentos · gerado automaticamente
</div>

</body>
</html>"""

    return html


def save_html_experimentos(html: str):
    with open(EXPERIMENTO_HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("[Experimento] HTML do email guardado em %s", EXPERIMENTO_HTML_REPORT)
