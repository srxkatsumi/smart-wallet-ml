"""Desafiante de regime: features sem olhar o futuro e previsão do dia."""
import numpy as np
import pandas as pd

from features.engineering import FEATURE_COLS
from research.regime import features_regime, prever_hoje, NOVAS


def _ctx(idx):
    rng = np.random.default_rng(0)
    s = lambda base: pd.Series(base + rng.normal(0, 1, len(idx)).cumsum() * 0.01, index=idx)  # noqa: E731
    tnx = s(4.0).iloc[:-1]                      # sem o último dia, como o ^TNX real às vezes
    return {"^VIX": s(20), "^VIX3M": s(21), "^TNX": tnx, "DX-Y.NYB": s(100)}


def test_features_usam_so_o_passado_e_preenchem_o_dia_sem_dado():
    idx = pd.bdate_range("2026-01-01", periods=40)
    ctx = _ctx(idx)
    f = features_regime(ctx, idx)
    assert not f.iloc[-1].isna().any()
    esperado = ctx["^VIX"].iloc[-2] / ctx["^VIX3M"].iloc[-2]      # shift(1): ontem
    assert f["vix_term"].iloc[-1] == esperado


def test_prever_hoje_devolve_probabilidades_nos_tres_horizontes():
    idx = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 1, (300, len(FEATURE_COLS))), index=idx, columns=FEATURE_COLS)
    for h in [1, 2, 3]:
        df[f"target_d{h}"] = rng.integers(0, 2, 300).astype(float)
        df.iloc[-h:, df.columns.get_loc(f"target_d{h}")] = np.nan
    p = prever_hoje(df, _ctx(idx))
    assert set(p) == {1, 2, 3} and all(0 <= v <= 1 for v in p.values())
    assert set(NOVAS) == {"vix_term", "juro10_chg5", "usd_ret5"}
