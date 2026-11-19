"""Ativos privados: só nos emails, nunca no que é publicado."""
import json

from data.storage import load_privados, portfolio_tickers, load_portfolio_config
from evaluation.anomaly import versao_publica


def test_privados_vem_do_secret(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRIVADO", json.dumps({"acoes": ["XXX"]}))
    assert load_privados() == ["XXX"]
    monkeypatch.setenv("PORTFOLIO_PRIVADO", json.dumps(["YYY"]))
    assert load_privados() == ["YYY"]


def test_privados_invalido_ou_vazio_nao_quebra(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PORTFOLIO_PRIVADO", "isto não é json")
    assert load_privados() == []
    monkeypatch.delenv("PORTFOLIO_PRIVADO")
    assert load_privados() == []


def test_carteira_publica_nao_tem_privados(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRIVADO", json.dumps({"acoes": ["XXX"]}))
    assert "XXX" not in portfolio_tickers(load_portfolio_config())


def test_anomalia_publica_tira_o_privado_e_recalcula_o_alerta():
    r = {"alerta": True, "market": {"transicao": False, "spike": False},
         "tickers": {"XXX": {"alerta": True}, "PPFB.DE": {"alerta": False}},
         "motivos": ["XXX: retorno 3.1x o ATR normal"]}
    pub = versao_publica(r, ["XXX"])
    assert "XXX" not in pub["tickers"] and pub["motivos"] == [] and pub["alerta"] is False
    assert r["alerta"] is True          # o do email continua completo
