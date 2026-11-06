"""Regressão da auditoria de 2026-11-06 (README_Errors.md, ERR-014)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.downloader as downloader
import models.validator as validator
from config.settings import PRED_COLS
from data.storage import aligned_rows
from evaluation.significance import compute_significance, daily_cluster_stats


def _result(last_date):
    return {
        "df": pd.DataFrame(),
        "last_date": pd.Timestamp(last_date),
        "close_now": 45.0,
        "preds_dict": {d: ("up", 46.0, 0.6) for d in (1, 2, 3)},
        "horizons": {d: {"preds_ind": {"rf": "up", "gb": "up", "sgd": "up"}} for d in (1, 2, 3)},
    }


@pytest.fixture
def no_disk(monkeypatch):
    monkeypatch.setattr(validator, "save_predictions_log", lambda *a, **k: None)


def test_target_counts_from_last_close_not_from_the_clock(no_disk):
    log = validator.save_new_predictions(
        pd.DataFrame(columns=PRED_COLS), {"IS3N.DE": _result("2026-11-04")}, pd.Timestamp("2026-11-06"),
    )
    d1 = log[log["horizon"] == 1].iloc[0]
    assert d1["ref_date"] == "2026-11-04"
    assert d1["target_date"] == "2026-11-05"
    assert d1["pred_date"] == "2026-11-06"


def test_same_bar_is_not_predicted_twice(no_disk):
    results = {"IS3N.DE": _result("2026-11-04")}
    log = validator.save_new_predictions(pd.DataFrame(columns=PRED_COLS), results, pd.Timestamp("2026-11-05"))
    log = validator.save_new_predictions(log, results, pd.Timestamp("2026-11-06"))
    assert len(log) == 3


def test_ticker_with_stale_data_gets_no_prediction(no_disk):
    log = validator.save_new_predictions(
        pd.DataFrame(columns=PRED_COLS), {"IS3N.DE": _result("2026-10-20")}, pd.Timestamp("2026-11-06"),
    )
    assert log.empty


def test_aligned_rows_drops_only_rows_without_ref_date():
    df = pd.DataFrame({"ref_date": [None, "2026-11-04"], "x": [1, 2]})
    assert aligned_rows(df)["x"].tolist() == [2]
    assert len(aligned_rows(pd.DataFrame({"x": [1]}))) == 1


def test_weights_ignore_misaligned_rows(monkeypatch):
    monkeypatch.setattr(validator, "save_ensemble_weights", lambda *a, **k: None)
    default = {f"d{d}": {"rf": 1.0, "gb": 1.0, "sgd": 1.0} for d in (1, 2, 3)}
    df = pd.DataFrame([{
        "pred_date": "2026-01-01", "target_date": f"2026-01-{i + 10}", "horizon": 1, "direction": "up",
        "correct": True, "model_rf": "up", "model_gb": "down", "model_sgd": "down", "ref_date": None,
    } for i in range(10)])
    assert validator.update_ensemble_weights(df, default)["d1"] == {"rf": 1.0, "gb": 1.0, "sgd": 1.0}


def test_pence_series_uses_each_days_fx_rate(monkeypatch):
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    fx = pd.DataFrame({"Close": [0.80, 0.90]}, index=idx)
    monkeypatch.setattr(downloader.yf, "download", lambda *a, **k: fx)
    df = pd.DataFrame({"Close": [6000.0, 6000.0]}, index=idx)
    out = downloader._gbp_pence_to_eur(df, "2y")
    assert out["Close"].tolist() == pytest.approx([75.0, 60000.0 / 900.0])


def test_significance_treats_the_day_as_the_unit():
    # 100 tickers por dia, todos iguais dentro do dia: 17 dias certos em 30.
    rows = [{
        "horizon": 1, "target_date": f"d{i:02d}", "correct": float(i < 17), "ref_date": "x",
    } for i in range(30) for _ in range(100)]
    d1 = compute_significance(pd.DataFrame(rows))["d1"]
    assert d1["ready"] is True
    assert d1["n_dias"] == 30
    assert d1["acc"] == pytest.approx(17 / 30, abs=1e-3)
    assert d1["sig"] is False


def test_significance_stays_silent_with_few_days():
    rows = [{"horizon": 1, "target_date": "d0", "correct": 1.0, "ref_date": "x"} for _ in range(500)]
    assert compute_significance(pd.DataFrame(rows))["d1"]["ready"] is False


def test_cluster_stats_detect_a_real_daily_edge():
    rng = np.random.default_rng(0)
    dates = pd.Series(np.repeat(np.arange(60), 20))
    correct = pd.Series((rng.uniform(size=1200) < 0.7).astype(float))
    assert daily_cluster_stats(correct, dates)["p"] < 0.05


def _export_data():
    path = Path(__file__).parent.parent / "docs" / "export_data.py"
    spec = importlib.util.spec_from_file_location("export_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_does_not_convert_eur_prices_twice():
    export = _export_data()
    assert export.log_price_to_eur(69.93, "SGLN.L", 1.15) == 69.93
    assert export.log_price_to_eur(6800.0, "SGLN.L", 1.15) == pytest.approx(78.2)
    assert export.log_price_to_eur(6800.0, "AAPL", 1.15) == 6800.0


def test_english_alert_reason_says_vix_fell():
    export = _export_data()
    reasons = export.anomaly_reasons_en({"market": {"spike": True, "variacao_1d": -0.2, "vix_ontem": 20, "vix_hoje": 16}})
    assert reasons == ["VIX fell 20% in a day (20 → 16)"]


def test_ticker_without_a_new_bar_stays_on_the_site():
    export = _export_data()
    log = pd.DataFrame({
        "ticker": ["A", "A", "B"], "pred_date": ["2026-11-05", "2026-11-06", "2026-11-05"], "x": [1, 2, 3],
    })
    assert export.latest_rows_per_ticker(log)["x"].tolist() == [2, 3]


def test_workflow_checks_recent_bars_not_the_run_date():
    workflow = (Path(__file__).parent.parent / ".github/workflows/executar_diario.yml").read_text(encoding="utf-8")
    assert "log['ref_date'].dropna().astype(str) >= " in workflow
    assert "log['pred_date'].isin" not in workflow
