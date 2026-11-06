"""
Regressão: as famílias rodavam, mas as sequenciais recebiam 1 linha e devolviam
sempre a taxa-base, e as de séries temporais devolviam o primeiro valor ajustado
do treino. Os testes antigos só conferiam formato e limites, com muitas linhas.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from research.runner import LATEST_ROWS


def _data(n=120, n_features=33, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(float)
    return X, y


@pytest.mark.parametrize("module", ["models.neural", "models.transformer", "models.efficient", "models.foundation"])
def test_sequence_families_react_to_the_latest_rows(module, monkeypatch):
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "EPOCHS", 3)
    if module == "models.foundation":
        monkeypatch.setattr(mod, "_CHRONOS_AVAILABLE", False)
    X, y = _data()
    model = mod.train(X, y)

    a = mod.predict(model, X[-LATEST_ROWS:])[-1]
    b = mod.predict(model, X[-LATEST_ROWS - 7:-7])[-1]

    assert a != pytest.approx(b, abs=1e-6)
    assert a != pytest.approx(float(y.mean()), abs=1e-6)


def test_timeseries_forecasts_next_step_not_first_fitted_value():
    from models.timeseries import train_ets, predict_ets, RETRAIN_DAILY

    y = np.r_[np.zeros(60), np.ones(60)]
    assert predict_ets(train_ets(y), 1)[0] > 0.5
    assert RETRAIN_DAILY is True


def test_ewi_does_not_invert_a_model_that_is_right():
    from models.contrarian import train_ewi

    X, y = _data(n=200)
    model = train_ewi(X, y)
    assert model["last_rate"] < 0.6
    assert model["inverted"] is False


def test_pel_correction_applies_lags_in_the_right_order(monkeypatch):
    import models.contrarian as contrarian

    rng = np.random.default_rng(1)
    errors = np.zeros(400)
    for t in range(1, 400):
        errors[t] = 0.9 * errors[t - 1] + rng.normal(0, 0.05)
    monkeypatch.setattr(contrarian, "_oof_errors", lambda *a: errors)

    X, y = _data()
    model = contrarian.train_pel(X, y)
    assert model["has_ac"]
    assert model["correction"] == pytest.approx(0.9 * errors[-1], abs=0.03)


def test_conformal_probability_is_centered_when_there_is_no_signal():
    from models.conformal import train, predict

    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 10))
    y = (rng.uniform(size=400) < 0.6).astype(float)
    probs = predict(train(X, y), rng.standard_normal((200, 10)))
    assert 0.35 < probs.mean() < 0.65


def test_gan_discriminator_separates_up_from_down(monkeypatch):
    import models.generative as gen

    monkeypatch.setattr(gen, "EPOCHS_GAN", 300)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 5))
    X[:, 0] = np.where(rng.uniform(size=300) < 0.5, 3.0, -3.0)
    y = (X[:, 0] > 0).astype(float)
    D = gen._train_gan(X, y)
    p = gen._predict_gan(D, X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2


def test_generative_survives_an_extreme_outlier():
    from models.generative import train, predict

    X, y = _data(n=120)
    X = X.astype(np.float64)
    X[50, 3] = 1e5
    probs = predict(train(X, y), X[-LATEST_ROWS:])
    assert np.isfinite(probs).all()


def test_obv_trend_stays_bounded_when_its_average_crosses_zero():
    from features.engineering import build_features, OBV_TREND_LIMIT

    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.sin(np.arange(n) / 3.0) * 5
    raw = pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1000.0}, index=idx)
    assert build_features(raw, {})["obv_trend"].abs().max() <= OBV_TREND_LIMIT


def _featured(n=40):
    from features.engineering import FEATURE_COLS

    idx = pd.bdate_range("2026-01-01", periods=n)
    df = pd.DataFrame(np.ones((n, len(FEATURE_COLS))), columns=FEATURE_COLS, index=idx)
    for h in (1, 2, 3):
        df[f"target_d{h}"] = 1.0
        df[f"fwd_ret_d{h}"] = 0.01
    return df


class _SpyModel:
    sizes: list = []

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        _SpyModel.sizes.append(len(y))
        return self

    def predict(self, X):
        return np.array([1])


def test_walk_forward_only_trains_on_labels_known_that_day(monkeypatch):
    import evaluation.walk_forward as wfv

    _SpyModel.sizes = []
    monkeypatch.setattr(wfv, "RandomForestClassifier", _SpyModel)
    monkeypatch.setattr(wfv, "WFV_MIN_TRAIN", 10)
    out = wfv.run_walk_forward(_featured(40), n_steps=1)
    assert _SpyModel.sizes == [39, 38, 37]
    assert out["acc_d1"] == 1.0


def test_experiment_backtest_only_trains_on_labels_known_that_day(monkeypatch):
    import evaluation.backtest_experimento as bt

    _SpyModel.sizes = []
    monkeypatch.setattr(bt, "RandomForestRegressor", _SpyModel)
    monkeypatch.setattr(bt, "BACKTEST_MIN_TRAIN", 10)
    bt._backtest_ticker(_featured(40), 3, n_steps=1)
    assert _SpyModel.sizes == [37]
