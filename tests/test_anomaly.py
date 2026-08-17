import pandas as pd
from evaluation.anomaly import detect_anomaly, bin_vix_regime


def _vix_series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2026-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def _featured_ticker(close: float, ret_1d: float, atr14: float) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=5)
    df = pd.DataFrame(index=idx)
    df["Close"] = close
    df["ret_1d"] = 0.001
    df["ATR14"] = atr14
    df.loc[idx[-1], "ret_1d"] = ret_1d
    return df


def test_bin_vix_regime():
    assert bin_vix_regime(10) == 0
    assert bin_vix_regime(20) == 1
    assert bin_vix_regime(30) == 2


def test_market_transicao_dispara_alerta():
    context_data = {"vix": _vix_series([18, 19, 20, 21, 27])}
    result = detect_anomaly(context_data, {}, [])
    assert result["market"]["transicao"] is True
    assert result["alerta"] is True
    assert any("VIX" in m for m in result["motivos"])


def test_market_calmo_nao_dispara():
    context_data = {"vix": _vix_series([18, 18.5, 19, 18.7, 19.2])}
    result = detect_anomaly(context_data, {}, [])
    assert result["market"]["transicao"] is False
    assert result["market"]["spike"] is False
    assert result["alerta"] is False


def test_ticker_movimento_grande_dispara_alerta():
    # ATR normal = 1% do preço; retorno de hoje = 4% -> 4x o ATR normal
    featured_data = {"SGLN.L": _featured_ticker(close=100.0, ret_1d=0.04, atr14=1.0)}
    result = detect_anomaly({}, featured_data, ["SGLN.L"])
    assert result["tickers"]["SGLN.L"]["alerta"] is True
    assert result["alerta"] is True
    assert any("SGLN.L" in m for m in result["motivos"])


def test_ticker_movimento_normal_nao_dispara():
    # retorno de hoje = 0.5% vs ATR normal de 1% -> 0.5x, bem abaixo do limiar
    featured_data = {"SGLN.L": _featured_ticker(close=100.0, ret_1d=0.005, atr14=1.0)}
    result = detect_anomaly({}, featured_data, ["SGLN.L"])
    assert result["tickers"]["SGLN.L"]["alerta"] is False
    assert result["alerta"] is False
