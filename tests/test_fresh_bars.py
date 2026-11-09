"""O pregão europeu de ontem falta na série diária da madrugada (ERR-015)."""
import numpy as np
import pandas as pd
import pytest

import data.downloader as downloader
import models.validator as validator
from config.settings import PRED_COLS

AGORA = pd.Timestamp("2026-11-07 00:18", tz="UTC")


def _diario(fim="2026-11-05"):
    idx = pd.bdate_range(end=fim, periods=5)
    return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": [10.0, 11, 12, 13, 14], "Volume": 100.0}, index=idx)


def _intradiario():
    idx = pd.date_range("2026-11-06 08:00", periods=9, freq="h", tz="Europe/Berlin")
    return pd.DataFrame({"Open": np.arange(9) + 20.0, "High": 40.0, "Low": 5.0, "Close": np.arange(9) + 21.0, "Volume": 10.0}, index=idx)


def test_missing_session_is_built_from_hourly_bars(monkeypatch):
    monkeypatch.setattr(downloader.yf, "download", lambda *a, **k: _intradiario())
    out = downloader._fill_missing_sessions(_diario(), "IS3N.DE", now=AGORA)
    novo = out.loc["2026-11-06"]
    assert out.index[-1] == pd.Timestamp("2026-11-06")
    assert (novo["Open"], novo["High"], novo["Low"], novo["Close"], novo["Volume"]) == (20.0, 40.0, 5.0, 29.0, 90.0)


def test_up_to_date_series_is_left_alone(monkeypatch):
    def proibido(*a, **k):
        raise AssertionError("não devia baixar candles de 1h")
    monkeypatch.setattr(downloader.yf, "download", proibido)
    df = _diario(fim="2026-11-06")
    assert downloader._fill_missing_sessions(df, "IS3N.DE", now=AGORA).equals(df)


def test_hourly_failure_keeps_the_daily_series(monkeypatch):
    def falha(*a, **k):
        raise RuntimeError("sem rede")
    monkeypatch.setattr(downloader.yf, "download", falha)
    df = _diario()
    assert downloader._fill_missing_sessions(df, "IS3N.DE", now=AGORA).equals(df)


def test_validation_uses_the_official_close_of_the_reference_bar(monkeypatch):
    monkeypatch.setattr(validator, "save_predictions_log", lambda *a, **k: None)
    row = {c: np.nan for c in PRED_COLS}
    # previsto com o fecho intradiário 100.4; o oficial foi 100.0 e o alvo fechou 100.2
    row.update(ticker="IS3N.DE", pred_date="2026-11-07", ref_date="2026-11-06", target_date="2026-11-09",
               horizon=1, direction="up", ref_price=100.4, pred_price=101.0)
    precos = pd.DataFrame({"Close": [100.0, 100.2]}, index=pd.to_datetime(["2026-11-06", "2026-11-09"]))
    monkeypatch.setattr(validator.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-11-10")))
    log = validator.validate_past_predictions(pd.DataFrame([row]), {"IS3N.DE": precos})
    assert log.iloc[0]["correct"] == 1.0
    assert log.iloc[0]["actual_change_pct"] == pytest.approx(0.2)


def test_official_close_falls_back_when_the_bar_is_missing():
    precos = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-11-05"]))
    assert downloader.official_close(precos, "2026-11-06") is None
    assert downloader.official_close(precos, None) is None
    assert downloader.official_close(precos, "2026-11-05") == 100.0
