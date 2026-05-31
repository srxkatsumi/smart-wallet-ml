import pytest
from unittest.mock import patch
import pandas as pd

from reports.email_report import (
    _pct_feito_cell,
    _pct_pendente_cell,
    _delta_vs_compra_cell,
    _is_first_business_week,
    _build_etf_monthly_recommendation,
)
from portfolio.pnl import expand_etoro_lots, expand_etf_lots


# ── _pct_feito_cell ───────────────────────────────────────────────────────

def test_pct_feito_ganho_positivo():
    html = _pct_feito_cell(100.0, 107.7)
    assert "+7.7%" in html
    assert "1e7a4c" in html  # verde


def test_pct_feito_alvo_atingido():
    html = _pct_feito_cell(100.0, 116.0)
    assert "✅" in html
    assert "+16.0%" in html


def test_pct_feito_perda():
    html = _pct_feito_cell(100.0, 90.0)
    assert "-10.0%" in html
    assert "b8453a" in html  # vermelho


# ── _pct_pendente_cell ────────────────────────────────────────────────────

def test_pct_pendente_alvo_atingido():
    html = _pct_pendente_cell(116.0, 115.0, 100.0)
    assert "Atingido" in html


def test_pct_pendente_muito_perto():
    # alvo = 115, close = 113.8 → falta 1.2% do preco_compra 100
    html = _pct_pendente_cell(113.8, 115.0, 100.0)
    assert "⚠️" in html


def test_pct_pendente_longe():
    html = _pct_pendente_cell(100.0, 115.0, 100.0)
    assert "15.0%" in html
    assert "a0a0a0" in html  # cinzento


# ── _delta_vs_compra_cell ─────────────────────────────────────────────────

def test_delta_positivo():
    html = _delta_vs_compra_cell(80.0, 85.76)
    assert "+" in html
    assert "1e7a4c" in html


def test_delta_negativo():
    html = _delta_vs_compra_cell(85.76, 78.13)
    assert "b8453a" in html


# ── expand_etoro_lots ─────────────────────────────────────────────────────

def _make_resultados_ml(ticker, close=200.0):
    return {
        ticker: {
            "close_now": close,
            "preds_dict": {
                1: ("up", 205.0, 0.6),
                2: ("up", 208.0, 0.55),
                3: ("down", 203.0, 0.52),
            },
        }
    }


def test_expand_etoro_lots_single():
    etoro = [{
        "ticker": "ALV.DE", "nome": "Allianz", "moeda": "EUR",
        "fee_euro": 2.0, "unidades": 0.14, "preco_abertura": 354.30,
        "lotes": [{"data_compra": "2025-11-03", "preco_abertura": 354.30, "unidades": 0.14}],
    }]
    result = expand_etoro_lots(etoro, _make_resultados_ml("ALV.DE", 381.60), 1.0)
    assert len(result) == 1
    assert result[0]["ticker"] == "ALV.DE"
    assert result[0]["data_compra"] == "2025-11-03"
    assert abs(result[0]["preco_compra_eur"] - 354.30) < 0.01
    fee_por_unidade = 2.0 / 0.14
    assert abs(result[0]["alvo_15_eur"] - (354.30 * 1.15 + fee_por_unidade)) < 0.01


def test_expand_etoro_lots_multiple():
    etoro = [{
        "ticker": "NVDA", "nome": "NVIDIA", "moeda": "USD",
        "fee_euro": 2.0, "unidades": 0.48115, "preco_abertura": 207.83,
        "lotes": [
            {"data_compra": "2025-10-29", "preco_abertura": 210.75, "unidades": 0.23725},
            {"data_compra": "2025-11-03", "preco_abertura": 205.00, "unidades": 0.24390},
        ],
    }]
    result = expand_etoro_lots(etoro, _make_resultados_ml("NVDA", 182.91), EUR_USD=1.08)
    assert len(result) == 2
    assert result[0]["data_compra"] == "2025-10-29"
    assert result[1]["data_compra"] == "2025-11-03"


def test_expand_etoro_lots_fallback_no_lotes():
    etoro = [{
        "ticker": "LLY", "nome": "Eli Lilly", "moeda": "USD",
        "fee_euro": 2.0, "unidades": 0.13, "preco_abertura": 1088.32,
        "lotes": [],
    }]
    result = expand_etoro_lots(etoro, _make_resultados_ml("LLY", 946.0), EUR_USD=1.08)
    assert len(result) == 1
    assert result[0]["data_compra"] is None


# ── expand_etf_lots ───────────────────────────────────────────────────────

def test_expand_etf_lots_known_lotes():
    etfs = [{
        "ticker": "SGLN.L", "nome": "Gold ETC", "moeda": "GBP",
        "gbp_pence": True, "unidades": 7.13, "euros_investidos": 600.0,
        "aporte_mensal": 50, "fee_euro": 0.0,
        "lotes": [
            {"data_compra": "2026-02-23", "unidades": 5.83, "euros_investidos": 500.0, "preco_unidade_eur": 85.76},
            {"data_compra": "2026-04-07", "unidades": 0.64, "euros_investidos": 50.0,  "preco_unidade_eur": 78.13},
        ],
    }]
    resultados = {"SGLN.L": {"close_now": 9000.0, "preds_dict": {}}}
    result = expand_etf_lots(etfs, resultados)
    assert len(result) == 2
    assert result[0]["preco_unidade_eur"] == 85.76
    assert result[1]["preco_unidade_eur"] == 78.13


def test_expand_etf_lots_empty_lotes():
    etfs = [{
        "ticker": "EMIM.AS", "nome": "EM IMI", "moeda": "EUR",
        "gbp_pence": False, "unidades": 1.06, "euros_investidos": 50.0,
        "aporte_mensal": 50, "fee_euro": 0.0, "lotes": [],
    }]
    resultados = {"EMIM.AS": {"close_now": 52.0, "preds_dict": {}}}
    result = expand_etf_lots(etfs, resultados)
    assert len(result) == 1
    assert result[0]["data_compra"] is None
    assert abs(result[0]["preco_unidade_eur"] - 50.0 / 1.06) < 0.01
