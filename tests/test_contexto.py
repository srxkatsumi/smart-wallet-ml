"""Contexto (VIX, SPY, BTC, GLD) com a linha do dia vazia: monta pelo 1h em vez de repetir a véspera."""
import numpy as np
import pandas as pd

import data.downloader as dl


def _diario(com_nan_no_fim=True):
    idx = pd.bdate_range("2026-11-02", "2026-11-18")
    close = np.linspace(500, 520, len(idx))
    if com_nan_no_fim:
        close[-1] = np.nan
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)


def _horario():
    idx = pd.date_range("2026-11-18 14:30", periods=7, freq="h")
    return pd.DataFrame({"Open": 521.0, "High": 523.0, "Low": 519.0, "Close": [521, 522, 522, 523, 522, 522, 525.0],
                         "Volume": 10.0}, index=idx)


def test_contexto_usa_candles_de_1h_em_vez_da_vespera(monkeypatch):
    def fake_download(t, period=None, interval="1d", **k):
        return _horario() if interval == "1h" else _diario()
    monkeypatch.setattr(dl.yf, "download", fake_download)
    monkeypatch.setattr(dl, "last_completed_session", lambda t, now=None: pd.Timestamp("2026-11-18"))
    ctx, avisos = dl.download_context()
    assert ctx["spy"].index[-1] == pd.Timestamp("2026-11-18")
    assert ctx["spy"].iloc[-1] == 525.0           # o fechamento do 1h, não a véspera
    assert avisos == []


def test_contexto_avisa_quando_nem_o_1h_tem_o_dia(monkeypatch):
    def fake_download(t, period=None, interval="1d", **k):
        return pd.DataFrame() if interval == "1h" else _diario()
    monkeypatch.setattr(dl.yf, "download", fake_download)
    monkeypatch.setattr(dl, "last_completed_session", lambda t, now=None: pd.Timestamp("2026-11-18"))
    ctx, avisos = dl.download_context()
    assert ctx["spy"].index[-1] == pd.Timestamp("2026-11-17")
    assert any("SPY" in a and "2026-11-18" in a for a in avisos)
