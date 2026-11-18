"""Ensemble ampliado: pesos encolhidos, voto combinado, conferência e placar."""
import numpy as np
import pandas as pd
import pytest

import research.ampliado as A


def test_peso_encolhido_comeca_neutro_e_so_cresce_com_amostra():
    assert A.peso_encolhido(0, 0) == pytest.approx(1.0)
    pequeno = A.peso_encolhido(27, 36)        # 75% em 36 previsões
    grande  = A.peso_encolhido(750, 1000)     # 75% em 1000 previsões
    assert 1.0 < pequeno < 1.5 < grande
    assert A.peso_encolhido(0, 10_000) == A.PESO_MIN
    assert A.peso_encolhido(10_000, 10_000) == A.PESO_MAX


def _prod(ticker="AAA", ref="2026-11-13", direction="up", conf=0.8, correct=np.nan, chg=np.nan):
    return pd.DataFrame([{"ticker": ticker, "ref_date": ref, "horizon": h, "direction": direction,
                          "confidence": conf, "correct": correct, "actual_change_pct": chg}
                         for h in [1, 2, 3]])


def _research(ticker="AAA", ref="2026-11-13", probs=(0.3, 0.4)):
    rows = []
    for i, p in enumerate(probs):
        rows.append({"ticker": ticker, "ref_date": ref, "family": f"f{i}",
                     "prob_d1": p, "prob_d2": p, "prob_d3": p})
    return pd.DataFrame(rows)


def test_build_ampliado_combina_producao_e_familias_pelos_pesos():
    w = {f"d{h}": {"pesos": {"producao": 2.0, "f0": 1.0, "f1": 1.0}} for h in [1, 2, 3]}
    rows = A.build_ampliado(_prod(conf=0.8), _research(probs=(0.3, 0.4)),
                            {"AAA": "2026-11-13"}, w, "2026-11-14")
    r = [x for x in rows if x["horizon"] == 1][0]
    assert r["prob_up"] == pytest.approx((2 * 0.8 + 0.3 + 0.4) / 4)
    assert r["prob_up_familias"] == pytest.approx(0.35)
    assert r["direction"] == "up" and r["n_membros"] == 3


def test_build_ampliado_sem_familias_nao_gera_linha():
    rows = A.build_ampliado(_prod(), _research(probs=()), {"AAA": "2026-11-13"}, {}, "2026-11-14")
    assert rows == []


def test_validate_e_placar_usam_a_conferencia_da_producao():
    log = pd.DataFrame([{"pred_date": "2026-11-14", "ticker": "AAA", "ref_date": "2026-11-13", "horizon": h,
                         "direction": "down", "prob_up": 0.4, "n_membros": 3, "prob_up_producao": 0.8,
                         "prob_up_familias": 0.3, "actual_change_pct": np.nan, "correct": np.nan}
                        for h in [1, 2, 3]])
    prod = _prod(direction="up", correct=0.0, chg=-1.2)   # caiu: produção errou, ampliado acertou
    log = A.validate_ampliado(log, prod)
    assert (log["correct"] == 1.0).all()
    placar = A.scoreboard(log, prod)
    assert placar["d1"] == {"n": 1, "n_dias": 1, "acc_ampliado": 1.0, "acc_producao": 0.0}


def test_validate_deixa_pendente_quando_producao_ainda_nao_conferiu():
    log = pd.DataFrame([{"ticker": "AAA", "ref_date": "2026-11-13", "horizon": 1, "direction": "up",
                         "actual_change_pct": np.nan, "correct": np.nan}])
    out = A.validate_ampliado(log, _prod())
    assert out["correct"].isna().all()
