import json

import numpy as np
import pandas as pd
import pytest

import models.calibration as calibration
from models.calibration import (
    fit_calibration, apply_calibration, apply_to_log,
    needs_refit, monthly_calibration_refresh,
)


def _rng(seed=42):
    return np.random.default_rng(seed)


def _validated_rows(n_per_horizon: int, well_calibrated: bool) -> pd.DataFrame:
    """
    Synthetic validated predictions log.

    well_calibrated=True: confidence genuinely tracks the chance of being
    right (higher confidence -> higher P(correct)), so isotonic fit should
    recover a clearly increasing curve.

    well_calibrated=False: confidence is pure noise (independent of outcome),
    mirroring what the real predictions_log.csv currently shows — the fit
    should stay close to flat (~overall accuracy) everywhere.
    """
    rng  = _rng()
    rows = []
    for day in [1, 2, 3]:
        confidence = rng.uniform(0.5, 1.0, n_per_horizon)
        if well_calibrated:
            p_correct = confidence  # perfectly calibrated by construction
        else:
            p_correct = np.full(n_per_horizon, 0.5)  # confidence carries no info
        correct = rng.uniform(size=n_per_horizon) < p_correct
        for c, ok in zip(confidence, correct):
            rows.append({"horizon": day, "confidence": c, "correct": float(ok)})
    return pd.DataFrame(rows)


# ── fit_calibration ─────────────────────────────────────────────────────────

def test_fit_calibration_skips_horizon_below_min_n(monkeypatch):
    monkeypatch.setattr(calibration, "MIN_N_CALIBRATION", 200)
    df = _validated_rows(n_per_horizon=50, well_calibrated=True)  # below threshold
    curves = fit_calibration(df)
    assert curves == {}


def test_fit_calibration_recovers_increasing_curve_when_genuinely_calibrated(monkeypatch):
    monkeypatch.setattr(calibration, "MIN_N_CALIBRATION", 200)
    df = _validated_rows(n_per_horizon=2000, well_calibrated=True)
    curves = fit_calibration(df)
    assert set(curves.keys()) == {"d1", "d2", "d3"}
    for key in curves:
        y = curves[key]["y"]
        # isotonic fit must be non-decreasing by construction
        assert all(b >= a - 1e-9 for a, b in zip(y, y[1:]))
        # and should span a meaningfully wide range (genuine signal recovered)
        assert max(y) - min(y) > 0.2


def test_fit_calibration_stays_flat_when_confidence_is_noise(monkeypatch):
    """
    Regression guard for the actual finding behind ERR-013: when confidence
    carries no information (current production reality), the fitted curve
    should stay close to overall accuracy across the whole range, not
    fabricate a spread that isn't in the data.
    """
    monkeypatch.setattr(calibration, "MIN_N_CALIBRATION", 200)
    df = _validated_rows(n_per_horizon=3000, well_calibrated=False)
    curves = fit_calibration(df)
    for key in curves:
        y = curves[key]["y"]
        assert max(y) - min(y) < 0.1
        assert all(abs(v - 0.5) < 0.1 for v in y)


def test_fit_calibration_reports_brier_improvement_when_there_is_signal(monkeypatch):
    monkeypatch.setattr(calibration, "MIN_N_CALIBRATION", 200)
    # Confiança superestimada: há o que corrigir, e o Brier é medido fora da amostra.
    rng = _rng()
    confidence = rng.uniform(0.5, 1.0, 2000)
    correct = rng.uniform(size=2000) < 0.5 + (confidence - 0.5) * 0.4
    df = pd.DataFrame({"horizon": 1, "confidence": confidence, "correct": correct.astype(float)})
    curves = fit_calibration(df)
    assert curves["d1"]["brier_calibrated"] < curves["d1"]["brier_raw"]


def test_fit_calibration_requires_enough_distinct_days():
    df = _validated_rows(n_per_horizon=2000, well_calibrated=True)
    df["target_date"] = "2026-01-02"
    assert fit_calibration(df) == {}


def test_fit_calibration_ignores_rows_without_ref_date():
    df = _validated_rows(n_per_horizon=2000, well_calibrated=True)
    df["ref_date"] = None
    assert fit_calibration(df) == {}


# ── apply_calibration ────────────────────────────────────────────────────────

def test_apply_calibration_returns_raw_value_without_curve():
    assert apply_calibration(0.73, 1, {}) == 0.73
    assert apply_calibration(0.73, 1, {"d2": {"x": [0.5, 1.0], "y": [0.5, 0.6]}}) == 0.73


def test_apply_calibration_interpolates_known_curve():
    curve = {"d1": {"x": [0.5, 0.75, 1.0], "y": [0.5, 0.6, 0.9]}}
    assert apply_calibration(0.75, 1, curve) == pytest.approx(0.6)
    assert apply_calibration(0.625, 1, curve) == pytest.approx(0.55)


# ── apply_to_log ──────────────────────────────────────────────────────────

def test_apply_to_log_fills_calibrated_column_per_horizon():
    df = pd.DataFrame({
        "horizon":    [1, 1, 2],
        "confidence": [0.5, 1.0, 0.5],
    })
    curves = {
        "d1": {"x": [0.5, 1.0], "y": [0.5, 0.9]},
        "d2": {"x": [0.5, 1.0], "y": [0.5, 0.5]},
    }
    out = apply_to_log(df, curves)
    assert out.loc[0, "confidence_calibrated"] == pytest.approx(0.5)
    assert out.loc[1, "confidence_calibrated"] == pytest.approx(0.9)
    assert out.loc[2, "confidence_calibrated"] == pytest.approx(0.5)


def test_apply_to_log_leaves_horizon_untouched_without_curve():
    df = pd.DataFrame({"horizon": [3], "confidence": [0.8]})
    out = apply_to_log(df, {"d1": {"x": [0.5, 1.0], "y": [0.5, 0.9]}})
    assert pd.isna(out.loc[0, "confidence_calibrated"])


# ── needs_refit ───────────────────────────────────────────────────────────

def test_needs_refit_true_without_prior_curve():
    assert needs_refit({}) is True


def test_needs_refit_false_within_window():
    curves = {"updated": pd.Timestamp.today().strftime("%Y-%m-%d")}
    assert needs_refit(curves, days=30) is False


def test_needs_refit_true_after_window():
    old = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    assert needs_refit({"updated": old}, days=30) is True


# ── monthly_calibration_refresh ──────────────────────────────────────────────

def test_monthly_refresh_skips_recent_curve(monkeypatch, tmp_path):
    calib_path = tmp_path / "confidence_calibration.json"
    monkeypatch.setattr(calibration, "CONFIDENCE_CALIB_FILE", calib_path)
    existing = {"updated": pd.Timestamp.today().strftime("%Y-%m-%d"), "d1": {"x": [0.5], "y": [0.5], "n": 999}}
    calib_path.write_text(json.dumps(existing))

    called = {"fit": False}
    def _fake_fit(df):
        called["fit"] = True
        return {}
    monkeypatch.setattr(calibration, "fit_calibration", _fake_fit)

    df = pd.DataFrame({"horizon": [1], "confidence": [0.7], "correct": [1.0]})
    out_df, curves = monthly_calibration_refresh(df)
    assert called["fit"] is False
    assert curves["d1"]["n"] == 999


def test_monthly_refresh_refits_and_persists_when_due(monkeypatch, tmp_path):
    calib_path = tmp_path / "confidence_calibration.json"
    monkeypatch.setattr(calibration, "CONFIDENCE_CALIB_FILE", calib_path)
    monkeypatch.setattr(calibration, "MIN_N_CALIBRATION", 10)

    saved = {}
    monkeypatch.setattr("data.storage.save_predictions_log", lambda df: saved.setdefault("df", df))

    df = _validated_rows(n_per_horizon=50, well_calibrated=True)
    df["ticker"] = "FAKE"
    out_df, curves = monthly_calibration_refresh(df)

    assert calib_path.exists()
    assert "d1" in curves and "d2" in curves and "d3" in curves
    assert "confidence_calibrated" in out_df.columns
    assert out_df["confidence_calibrated"].notna().all()
    assert "df" in saved  # save_predictions_log was called to persist the reapplied column
