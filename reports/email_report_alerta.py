import logging
from datetime import datetime, timezone, timedelta
from config.settings import ALERTA_HTML_REPORT, BARCELONA_UTC_OFFSET

logger = logging.getLogger(__name__)

_DIAS_PT = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
            "Sexta-feira", "Sábado", "Domingo"]
_MESES_PT = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
             'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']


def _ticker_row(ticker: str, r: dict) -> str:
    pct = r["retorno_1d"] * 100
    color = "#1e7a4c" if pct >= 0 else "#b8453a"
    arrow = "▲" if pct >= 0 else "▼"
    return (
        '<tr style="border-bottom:1px solid #f0ede4">'
        f'<td style="padding:8px 6px;font-size:12px;font-weight:600;color:#1a1740">{ticker}</td>'
        f'<td style="padding:8px 6px;text-align:center;font-size:12px;font-weight:600;color:{color}">'
        f'{arrow}&nbsp;{abs(pct):.2f}%</td>'
        f'<td style="padding:8px 6px;text-align:center;font-size:11px;color:#5a5a5a">'
        f'{r["multiplo_atr"]:.1f}x o ATR normal</td>'
        '</tr>'
    )


def build_html_alerta(anomaly: dict) -> str:
    tz    = timezone(timedelta(hours=BARCELONA_UTC_OFFSET))
    agora = datetime.now(tz)
    data_str = f"{_DIAS_PT[agora.weekday()]} · {agora.day} de {_MESES_PT[agora.month-1]} de {agora.year}"

    motivos_html = "".join(f'<li style="margin-bottom:4px">{m}</li>' for m in anomaly.get("motivos", []))

    market = anomaly.get("market", {})
    market_html = ""
    if market.get("vix_hoje") is not None:
        market_html = f"""
    <div style="padding:20px 36px;border-top:1px solid #efece4">
      <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px">VIX</div>
      <div style="font-size:13px;color:#3a3a3a">
        {market['vix_ontem']} → <strong>{market['vix_hoje']}</strong>
        ({market['variacao_1d']*100:+.1f}% no dia · regime {market['regime_ontem']}→{market['regime_hoje']})
      </div>
    </div>"""

    tickers_em_alerta = {t: r for t, r in anomaly.get("tickers", {}).items() if r.get("alerta")}
    tickers_html = ""
    if tickers_em_alerta:
        rows = "".join(_ticker_row(t, r) for t, r in tickers_em_alerta.items())
        tickers_html = f"""
    <div style="padding:20px 36px;border-top:1px solid #efece4">
      <div style="font-size:10px;font-weight:600;color:#a89e85;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px">Ativos fora do normal</div>
      <table style="width:100%;border-collapse:collapse">
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carteira BOT Alerta · {agora.strftime('%d/%m/%Y')}</title>
<style>body {{ margin: 0; }}</style>
</head>
<body style="margin:0;padding:24px 16px;background:#ece9e2;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1a1a1a;-webkit-font-smoothing:antialiased">

<div style="max-width:680px;margin:0 auto;background:#fbfaf7;border-radius:6px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,0.04),0 12px 40px -12px rgba(184,69,58,0.25)">

  <!-- CABEÇALHO -->
  <div style="background:rgb(140,30,30);padding:28px 36px 24px">
    <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:14px">
      <tr>
        <td style="vertical-align:middle">
          <div style="display:inline-block;width:8px;height:8px;background:#ffb3a8;border-radius:50%;margin-right:8px;vertical-align:middle"></div>
          <span style="color:#ffb3a8;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;vertical-align:middle">Carteira BOT Alerta</span>
        </td>
      </tr>
    </table>
    <div style="font-family:'Iowan Old Style','Palatino Linotype',Georgia,serif;color:#fff0ee;font-size:28px;line-height:1.1;font-weight:500;letter-spacing:-0.01em;margin-bottom:6px">🚨 Anomalia detetada</div>
    <div style="color:#ffcfc7;font-size:13px;letter-spacing:0.02em">{data_str}</div>
  </div>

  <!-- MOTIVOS -->
  <div style="padding:20px 36px">
    <ul style="font-size:13px;color:#3a3a3a;line-height:1.6;margin:0;padding-left:18px">
      {motivos_html}
    </ul>
  </div>

  {market_html}
  {tickers_html}

  <!-- AVISO -->
  <div style="padding:16px 36px;border-top:1px solid #efece4">
    <div style="font-size:10.5px;color:#a8a39a;line-height:1.55">
      <span style="color:#8c1e1e;font-weight:600">Informativo, não é recomendação.</span> Este alerta indica volatilidade fora do padrão recente (VIX ou movimento de um ativo vs. seu próprio ATR) — não diz se é bom ou ruim, nem o que fazer a respeito.
    </div>
  </div>

</div>

<div style="max-width:680px;margin:14px auto 0;text-align:center;font-size:10px;color:#a8a39a;letter-spacing:0.08em;text-transform:uppercase;font-weight:600">
  Carteira BOT Alerta · gerado automaticamente
</div>

</body>
</html>"""

    return html


def save_html_alerta(html: str):
    with open(ALERTA_HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("[Alerta] HTML do email guardado em %s", ALERTA_HTML_REPORT)
