"""Email semanal: quando sai, métricas e montagem do HTML."""
import numpy as np
import pandas as pd
import pytest

from reports.email_report_semanal import (
    e_fim_de_semana_de_pregao, brier_por_horizonte, correlacoes_60_120,
    ranking_familias, build_html_semanal,
)


@pytest.mark.parametrize("data,esperado", [
    ("2026-11-13", True),    # sexta
    ("2026-11-17", False),   # terça
    ("2026-12-23", True),    # quarta antes do Natal (Xetra fecha 24 e 25)
])
def test_sai_no_ultimo_pregao_da_semana(data, esperado):
    assert e_fim_de_semana_de_pregao(pd.Timestamp(data)) is esperado


def _log():
    rows = []
    for d, (conf, ok) in zip(pd.bdate_range("2026-11-02", periods=6), [(0.9, 1), (0.9, 0), (0.6, 1), (0.6, 0), (0.7, 1), (0.8, 0)]):
        for h in [1, 2, 3]:
            rows.append({"ticker": "AAA", "horizon": h, "ref_date": str(d.date()),
                         "target_date": str((d + pd.offsets.BDay(h)).date()),
                         "confidence": conf, "correct": float(ok), "direction": "up",
                         "actual_change_pct": 1.0 if ok else -1.0})
    return pd.DataFrame(rows)


def test_brier_e_bss():
    br = brier_por_horizonte(_log())
    esperado = np.mean([(0.9 - 1) ** 2, 0.9 ** 2, 0.4 ** 2, 0.6 ** 2, 0.3 ** 2, 0.8 ** 2])
    assert br[1]["brier"] == pytest.approx(esperado)
    assert br[1]["bss"] == pytest.approx(1 - esperado / 0.25)


def test_correlacoes_60_120():
    idx = pd.bdate_range("2026-01-01", periods=150)
    rng = np.random.default_rng(1)
    a = pd.Series(100 + rng.normal(0, 1, 150).cumsum(), index=idx)
    res = {"A": {"df": pd.DataFrame({"Close": a})}, "B": {"df": pd.DataFrame({"Close": a * 1.01})}}
    c = correlacoes_60_120(res, ["A", "B"])
    assert c["media60"] == pytest.approx(1.0, abs=1e-6) and c["pares"][0][:2] == ("A", "B")


def test_ranking_familias_e_html_monta_sem_dados_de_mercado():
    rl = pd.DataFrame([{"ticker": "AAA", "family": f, "ref_date": str(d.date()),
                        "correct_d1": float(i % 2), "correct_d2": 1.0, "correct_d3": 0.0}
                       for f in ["drift", "conformal"] for i, d in enumerate(pd.bdate_range("2026-11-02", periods=6))])
    rk = ranking_familias(rl, ["AAA"])
    assert {r["family"] for r in rk} == {"drift", "conformal"} and "p_adj" in rk[0]
    html = build_html_semanal(_log(), rl, ["AAA"], {}, {}, pd.Timestamp("2026-11-13"))
    assert "Semana de" in html and "As 13 famílias" in html
