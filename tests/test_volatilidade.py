"""Desafiantes de volatilidade: previsão, conferência e placar."""
import numpy as np
import pandas as pd
import pytest

import research.volatilidade as V


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    vol = np.where(np.arange(n) % 100 < 50, 0.005, 0.02)          # regimes calmo/agitado
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol))), index=idx)
    d = pd.DataFrame({"Close": close})
    r = close.pct_change()
    d["ret_1d"], d["ret_5d"] = r, close.pct_change(5)
    d["ATR14"] = (close * r.abs()).rolling(14).mean()
    d["vol_10d"] = r.rolling(10).std()
    d["vol_ratio"], d["vix_level"], d["vix_change"] = 1.0, 20.0, 0.0
    d["BB_width"] = close.rolling(20).std() * 4 / close.rolling(20).mean()
    return d


def test_ewma_responde_ao_regime():
    r_calmo = pd.Series(np.random.default_rng(1).normal(0, 0.005, 200))
    r_agit = pd.Series(np.random.default_rng(1).normal(0, 0.02, 200))
    assert V.ewma_var_ultima(r_agit) > 4 * V.ewma_var_ultima(r_calmo)


def test_prever_ativo_devolve_probabilidade_e_referencias():
    p = V.prever_ativo(_df())
    assert 0 <= p["p_grande"] <= 1 and 0 < p["p_base"] < 1
    assert p["var_ewma"] > 0 and p["var_const"] > 0 and p["atr_pct"] > 0


def test_validar_preenche_o_realizado_com_os_precos_seguintes():
    idx = pd.bdate_range("2026-11-02", periods=8)
    precos = pd.Series([100, 100, 103, 103, 103, 103, 103, 103.0], index=idx)
    log = pd.DataFrame([{"ticker": "AAA", "ref_date": "2026-11-03", "atr_pct": 0.01, "p_grande": 0.3,
                         "p_base": 0.2, "var_ewma": 1e-4, "var_const": 1e-4,
                         "grande_real": np.nan, "rv1": np.nan, "rv3": np.nan, "rv5": np.nan}])
    out = V.validar(log, {"AAA": precos})
    assert out.loc[0, "grande_real"] == 1.0 and out.loc[0, "rv1"] == pytest.approx(0.03)
    assert out.loc[0, "rv5"] == pytest.approx(np.sqrt(0.03 ** 2 / 5))
    p = V.placar(out)
    assert p["grande"]["n"] == 1 and "vol5" in p
