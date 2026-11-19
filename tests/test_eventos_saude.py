"""Calendário de eventos das próximas sessões e saúde do sistema."""
import json

import pandas as pd

import evaluation.eventos as ev
from evaluation.saude import avaliar_saude


def test_calendario_macro_e_json_valido_e_ordenavel():
    cfg = json.loads(ev.EVENTOS_FILE.read_text(encoding="utf-8"))
    assert all(e["tipo"] in cfg["tipos"] for e in cfg["eventos"])
    pd.to_datetime([e["data"] for e in cfg["eventos"]])


def test_evento_macro_na_janela_e_sessao_certa():
    # 08/12/2026 (terça): a decisão do Fed em 09/12 cai em D+1, o CPI em 10/12 em D+2
    r = ev.proximos_eventos(pd.Timestamp("2026-12-08"), [], buscar_resultado=lambda t: None)
    nomes = {(e["sessao"], e["nome"]) for e in r["eventos"]}
    assert (1, "Decisão de juros do Fed (EUA)") in nomes
    assert (2, "Inflação dos EUA (CPI)") in nomes


def test_resultado_da_acao_da_carteira_aparece_e_fora_da_janela_nao():
    datas = {"ACAO": pd.Timestamp("2026-11-20"), "NVDA": pd.Timestamp("2026-12-30")}
    r = ev.proximos_eventos(pd.Timestamp("2026-11-18"), ["ACAO"], buscar_resultado=lambda t: datas.get(t))
    assert [e["afeta"] for e in r["eventos"]] == ["ACAO"]
    assert r["eventos"][0]["sessao"] == 2


def test_aviso_quando_calendario_acaba():
    r = ev.proximos_eventos(pd.Timestamp("2027-11-01"), [], buscar_resultado=lambda t: None)
    assert r["aviso_calendario"] and "eventos_macro.json" in r["aviso_calendario"]


def test_saude_ok_problema_e_atencao():
    now = pd.Timestamp("2026-11-18 03:00", tz="UTC")       # último pregão encerrado: 17/11
    fresco = {"last_date": pd.Timestamp("2026-11-17")}
    velho  = {"last_date": pd.Timestamp("2026-11-13")}
    assert avaliar_saude(["ACAO"], {"ACAO": fresco}, [], now=now)["estado"] == "ok"
    s = avaliar_saude(["ACAO", "PPFB.DE"], {"ACAO": fresco, "PPFB.DE": velho}, [], now=now)
    assert s["estado"] == "problema" and s["n_ok"] == 1
    s = avaliar_saude(["ACAO"], {"ACAO": fresco}, [("SHAP", "boom")], now=now)
    assert s["estado"] == "atencao"
    assert avaliar_saude(["ACAO"], {}, [], now=now)["estado"] == "problema"
