"""Leituras de acurácia do email: Wilson, IC por dia, sempre ▲ e faixas de confiança."""
import numpy as np
import pandas as pd
import pytest

from evaluation.confianca import (wilson, ic_acuracia, acc_sempre_alta, faixas_confianca,
                                  acerto_da_faixa, MIN_N_FAIXA)
from evaluation.statistical_tests import dm_from_loss_diff


def test_wilson_24_de_48():
    lo, hi = wilson(24, 48)
    assert lo == pytest.approx(0.364, abs=0.005) and hi == pytest.approx(0.636, abs=0.005)


def test_ic_usa_o_dia_como_unidade_com_dias_suficientes():
    dates = pd.Series(np.repeat(pd.date_range("2026-11-02", periods=6, freq="B"), 10))
    correct = pd.Series(np.tile([1, 0], 30))
    ic = ic_acuracia(correct, dates)
    assert ic["metodo"] == "por dia" and ic["n_dias"] == 6 and ic["acc"] == 0.5


def test_ic_poucos_dias_cai_para_wilson_rotulado():
    ic = ic_acuracia(pd.Series([1, 0, 1]), pd.Series(["2026-11-02"] * 3))
    assert ic["metodo"].startswith("Wilson")


def test_sempre_alta_conta_zero_como_alta():
    assert acc_sempre_alta(pd.DataFrame({"actual_change_pct": [1.0, 0.0, -1.0, -2.0]})) == 0.5


def test_faixas_e_minimo_de_casos():
    n = MIN_N_FAIXA
    df = pd.DataFrame({
        "ticker": "AAA", "horizon": 1, "ref_date": "2026-11-10", "target_date": "2026-11-11",
        "confidence": [0.75] * n + [0.85] * 3, "correct": [1.0] * (n // 2) + [0.0] * (n - n // 2) + [1.0] * 3,
    })
    fx = faixas_confianca(df)
    assert acerto_da_faixa(fx, 1, 0.72)["acc"] == pytest.approx(0.5)
    assert acerto_da_faixa(fx, 1, 0.85) is None   # só 3 casos


def test_hln_e_mais_conservador_que_a_normal():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, 12)
    from scipy import stats
    t = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    p_normal = 2 * stats.norm.sf(abs(t))
    assert dm_from_loss_diff(d, h=1)["p_value"] > p_normal
