import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "SMA20_dist", "SMA50_dist", "sma_cross",
    "RSI14", "MACD", "MACD_sig", "MACD_hist",
    "BB_width", "BB_pos", "ATR14",
    "ret_1d", "ret_5d", "vol_10d",
    "spy_ret_1d", "vix_level", "vix_change", "vix_regime",
]


def build_features(df: pd.DataFrame, context_data: dict) -> pd.DataFrame:
    d = df.copy()

    d["SMA20"]      = d["Close"].rolling(20).mean()
    d["SMA50"]      = d["Close"].rolling(50).mean()
    d["SMA20_dist"] = (d["Close"] - d["SMA20"]) / d["SMA20"]
    d["SMA50_dist"] = (d["Close"] - d["SMA50"]) / d["SMA50"]
    d["sma_cross"]  = (d["SMA20"] > d["SMA50"]).astype(int)

    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    rs         = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))

    ema12           = d["Close"].ewm(span=12, adjust=False).mean()
    ema26           = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]       = ema12 - ema26
    d["MACD_sig"]   = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]  = d["MACD"] - d["MACD_sig"]

    sma20          = d["Close"].rolling(20).mean()
    std20          = d["Close"].rolling(20).std()
    d["BB_upper"]  = sma20 + 2 * std20
    d["BB_lower"]  = sma20 - 2 * std20
    d["BB_width"]  = (d["BB_upper"] - d["BB_lower"]) / sma20
    band_range     = (d["BB_upper"] - d["BB_lower"]).replace(0, np.nan)
    d["BB_pos"]    = (d["Close"] - d["BB_lower"]) / band_range

    hl         = d["High"] - d["Low"]
    hc         = (d["High"] - d["Close"].shift()).abs()
    lc         = (d["Low"]  - d["Close"].shift()).abs()
    tr         = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()

    d["ret_1d"]  = d["Close"].pct_change(1)
    d["ret_5d"]  = d["Close"].pct_change(5)
    d["vol_10d"] = d["ret_1d"].rolling(10).std()

    if "spy" in context_data:
        spy_ret         = context_data["spy"].pct_change(1).shift(1)
        d["spy_ret_1d"] = spy_ret.reindex(d.index, method="ffill").values
    else:
        d["spy_ret_1d"] = 0.0

    if "vix" in context_data:
        vix             = context_data["vix"].shift(1)
        vix_chg         = context_data["vix"].pct_change(1).shift(1)
        d["vix_level"]  = vix.reindex(d.index, method="ffill").values
        d["vix_change"] = vix_chg.reindex(d.index, method="ffill").values
        vix_vals        = vix.reindex(d.index, method="ffill")
        d["vix_regime"] = pd.cut(
            vix_vals, bins=[-np.inf, 15, 25, np.inf], labels=[0, 1, 2]
        ).astype(float).values
    else:
        d["vix_level"]  = 20.0
        d["vix_change"] = 0.0
        d["vix_regime"] = 1.0

    # Targets por horizonte (dias de calendário — convertidos a dias úteis no validator)
    d["target_d1"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d["target_d2"] = (d["Close"].shift(-2) > d["Close"]).astype(int)
    d["target_d3"] = (d["Close"].shift(-3) > d["Close"]).astype(int)

    return d.dropna()


def build_all_features(raw_data: dict, context_data: dict,
                       min_samples: int = 60) -> dict:
    featured_data = {}
    for ticker, df in raw_data.items():
        try:
            fd = build_features(df, context_data)
            if len(fd) >= min_samples:
                featured_data[ticker] = fd
                logger.info("%s: %d dias com features completas", ticker, len(fd))
            else:
                logger.warning("%s: dados insuficientes (%d dias)", ticker, len(fd))
        except Exception as e:
            logger.error("%s: erro nas features: %s", ticker, e)
    logger.info("Features prontas para %d ativos", len(featured_data))
    return featured_data
