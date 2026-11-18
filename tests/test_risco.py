"""Risco por ativo e estado geral da carteira, sem depender de valores investidos."""
import numpy as np
import pandas as pd
import pytest

from evaluation.risco import risco_carteira, contexto_mercado, _nivel


def _serie(seed, n=300, vol=0.01, vol_final=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    if vol_final:
        r[-20:] = rng.normal(0, vol_final, 20)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2025-09-01", periods=n))


def test_ativo_agitado_contra_o_proprio_historico():
    res = {"A.DE": {"df": pd.DataFrame({"Close": _serie(1, vol=0.01, vol_final=0.03)})},
           "B.DE": {"df": pd.DataFrame({"Close": _serie(2, vol=0.01)})}}
    r = risco_carteira(res, ["A.DE", "B.DE"], vix_nivel=20, baixar=lambda *a, **k: None)
    assert r["agitados"] == ["A.DE"] and r["por_ativo"][0]["ticker"] == "A.DE"
    assert all(a["var95"] > 0 and a["queda_pico"] <= 0 for a in r["por_ativo"])
    assert "peso" not in r["por_ativo"][0]


def test_conversao_do_dolar_nao_muda_o_risco_com_cambio_constante():
    eurusd = pd.Series(2.0, index=pd.bdate_range("2025-01-01", periods=600))
    s = _serie(3)
    sem = risco_carteira({"AAA": {"df": pd.DataFrame({"Close": s})}}, ["AAA"], baixar=lambda *a, **k: None)
    com = risco_carteira({"AAA": {"df": pd.DataFrame({"Close": s})}}, ["AAA"], baixar=lambda *a, **k: eurusd)
    assert com["por_ativo"][0]["vol20"] == pytest.approx(sem["por_ativo"][0]["vol20"])


@pytest.mark.parametrize("frac,vix,corr,esperado", [
    (0.0, 12, False, "baixo"), (0.3, 20, False, "normal"), (0.6, 28, True, "elevado"),
])
def test_nivel_geral(frac, vix, corr, esperado):
    assert _nivel(frac, vix, corr) == esperado


def test_contexto_sem_rede_nao_quebra():
    vix = pd.Series([20.0, 22.0], index=pd.bdate_range("2026-11-16", periods=2))
    c = contexto_mercado({"vix": vix}, baixar=lambda *a, **k: None)
    assert c["vix"] == (22.0, pytest.approx(0.1)) and c["juro10"] is None
