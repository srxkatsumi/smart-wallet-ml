"""Rastreabilidade: versões gravadas em cada previsão e no rodapé do email."""
import pandas as pd

from evaluation.rastreio import versao_calibracao, hash_dados, contexto_execucao, versao_codigo


def test_versao_calibracao():
    curves = {"updated": "2026-11-18", "d1": {"tipo": "encolhimento", "lambda": 0.0},
              "d2": {"tipo": "isotonica", "n": 5400, "x": [0.5], "y": [0.5]}}
    assert versao_calibracao(curves, 1) == "enc λ=0.00 @2026-11-18"
    assert versao_calibracao(curves, 2) == "iso n=5400 @2026-11-18"
    assert versao_calibracao(curves, 3) == "sem calibração (bruta)"


def test_hash_muda_quando_o_dado_muda_e_e_estavel():
    idx = pd.bdate_range("2026-11-16", periods=2)
    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1.0, 2.0], "Volume": [10, 20]}, index=idx)
    h1, h2 = hash_dados(df), hash_dados(df.copy())
    df2 = df.copy(); df2.iloc[-1, df2.columns.get_loc("Close")] = 2.01
    assert h1 == h2 and len(h1) == 10 and hash_dados(df2) != h1


def test_contexto_tem_codigo_e_calibracao_por_horizonte(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567")
    ctx = contexto_execucao({})
    assert ctx["versao_codigo"] == "abcdef1" and set(ctx["calibracao"]) == {1, 2, 3}
    assert versao_codigo() == "abcdef1"
