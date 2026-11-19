"""
Eventos das próximas sessões que podem mexer com a carteira e que o modelo NÃO vê
(nenhuma feature sabe que amanhã há decisão de juros ou resultado trimestral).

Duas fontes:
  - config/eventos_macro.json: Fed, BCE, Banco do Japão e CPI dos EUA, datas
    conferidas nas fontes oficiais e mantidas à mão.
  - yfinance: próxima data de resultado das ações da carteira e das mega caps
    listadas em "resultados_empresas" (elas mexem com o mercado todo).

O caso que motivou: uma ação da carteira com resultado trimestral em 05/11/2026,
no mesmo dia de uma venda forte de tecnologia; o modelo previu alta com 72-76% e errou as 6.
"""
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EVENTOS_FILE   = Path("config/eventos_macro.json")
JANELA_SESSOES = 3     # D+1, D+2, D+3
AVISO_FIM_DIAS = 60    # avisa quando o calendário macro acaba em menos que isto


def _janela(hoje: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Da próxima sessão até a terceira (dias úteis), que é o que D+1..D+3 cobrem."""
    ini = (hoje + pd.offsets.BDay(1)).normalize()
    fim = (hoje + pd.offsets.BDay(JANELA_SESSOES)).normalize()
    return ini, fim


def _proxima_data_resultado(ticker: str) -> pd.Timestamp | None:
    import yfinance as yf
    try:
        cal = yf.Ticker(ticker).calendar
        datas = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if datas:
            return pd.Timestamp(datas[0]).normalize()
    except Exception as e:
        logger.info("Data de resultado de %s indisponível: %s", ticker, e)
    return None


def proximos_eventos(hoje: pd.Timestamp, acoes_carteira: list[str],
                     buscar_resultado=_proxima_data_resultado) -> dict:
    """
    {"eventos": [{"data", "nome", "afeta", "sessao"}], "aviso_calendario": str|None}
    `sessao` = 1, 2 ou 3 (em qual D+N o evento cai). `buscar_resultado` é injetável nos testes.
    """
    hoje = pd.Timestamp(hoje).normalize()
    ini, fim = _janela(hoje)
    sessoes = list(pd.bdate_range(ini, fim))
    out, aviso = [], None

    try:
        cfg = json.loads(EVENTOS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Calendário macro indisponível: %s", e)
        cfg = {"eventos": [], "tipos": {}, "resultados_empresas": []}

    tipos = cfg.get("tipos", {})
    datas = []
    for ev in cfg.get("eventos", []):
        d = pd.Timestamp(ev["data"]).normalize()
        datas.append(d)
        if ini <= d <= fim:
            t = tipos.get(ev["tipo"], {})
            afeta = t.get("afeta", "todos")
            out.append({"data": d, "nome": t.get("nome", ev["tipo"]),
                        "afeta": "toda a carteira" if afeta == "todos" else ", ".join(afeta)})
    if datas and max(datas) - hoje < pd.Timedelta(days=AVISO_FIM_DIAS):
        aviso = (f"O calendário de eventos macro termina em {max(datas).strftime('%d/%m/%Y')}: "
                 f"atualizar config/eventos_macro.json.")

    for t in acoes_carteira:
        d = buscar_resultado(t)
        if d is not None and ini <= d <= fim:
            out.append({"data": d, "nome": f"Resultado trimestral de {t}", "afeta": t})
    for t in cfg.get("resultados_empresas", []):
        if t in acoes_carteira:
            continue
        d = buscar_resultado(t)
        if d is not None and ini <= d <= fim:
            out.append({"data": d, "nome": f"Resultado trimestral de {t}",
                        "afeta": "mercado todo (mega cap)"})

    for e in out:
        e["sessao"] = next((i + 1 for i, s in enumerate(sessoes) if s >= e["data"]), JANELA_SESSOES)
    out.sort(key=lambda e: (e["data"], e["nome"]))
    return {"eventos": out, "aviso_calendario": aviso}
