import numpy as np
import pandas as pd
import pytest
from features.engineering import build_features


def _make_price_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def test_rsi_within_bounds():
    df = _make_price_df(150)
    result = build_features(df, {})
    rsi = result["RSI14"].dropna()
    assert not rsi.empty, "RSI não foi calculado"
    assert (rsi >= 0).all(), f"RSI abaixo de 0: min={rsi.min():.4f}"
    assert (rsi <= 100).all(), f"RSI acima de 100: max={rsi.max():.4f}"


def test_rsi_bounds_monotone_up():
    """Preço sempre a subir → RSI deve estar próximo de 100."""
    n = 150
    close = np.linspace(100, 200, n)
    df = pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    result = build_features(df, {})
    rsi = result["RSI14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
