import logging
import yfinance as yf
import pandas as pd
from config.settings import PRICE_PERIOD, EUR_USD_FALLBACK, EUR_GBP_FALLBACK

logger = logging.getLogger(__name__)

# Module-level FX cache (populated once per run)
EUR_USD: float = EUR_USD_FALLBACK
EUR_GBP: float = EUR_GBP_FALLBACK
GBP_EUR: float = 1 / EUR_GBP_FALLBACK


def _fetch_fx(pair_ticker: str, fallback: float) -> float:
    try:
        rate = yf.Ticker(pair_ticker).fast_info["last_price"]
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    logger.warning("%s indisponível — usando fallback %.4f", pair_ticker, fallback)
    return fallback


def load_fx_rates() -> tuple[float, float, float]:
    global EUR_USD, EUR_GBP, GBP_EUR
    EUR_USD = _fetch_fx("EURUSD=X", EUR_USD_FALLBACK)
    EUR_GBP = _fetch_fx("EURGBP=X", EUR_GBP_FALLBACK)
    GBP_EUR = 1 / EUR_GBP
    logger.info("EUR/USD=%.4f  EUR/GBP=%.4f", EUR_USD, EUR_GBP)
    return EUR_USD, EUR_GBP, GBP_EUR


def to_eur(preco: float, moeda: str, gbp_pence: bool = False) -> float:
    if moeda == "USD":
        return preco / EUR_USD
    if moeda == "GBP":
        return (preco / 100 * GBP_EUR) if gbp_pence else (preco * GBP_EUR)
    return preco


def download_context() -> dict:
    context_data = {}
    for ctx_ticker, ctx_name in [("^VIX", "vix"), ("SPY", "spy")]:
        try:
            df = yf.download(ctx_ticker, period=PRICE_PERIOD, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("sem dados")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()
            context_data[ctx_name] = df["Close"].rename(ctx_name)
            logger.info("%s: %d dias | último=%.2f", ctx_ticker, len(df), df["Close"].iloc[-1])
        except Exception as e:
            logger.warning("%s: %s — features de contexto usarão valores neutros", ctx_ticker, e)
    return context_data


def download_prices(tickers: list[str], etf_acumulacao: list[dict]) -> dict:
    raw_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=PRICE_PERIOD, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                logger.warning("%s: sem dados", ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()
            raw_data[ticker] = df

            etf_info = next((e for e in etf_acumulacao if e["ticker"] == ticker), None)
            if etf_info and etf_info["moeda"] == "GBP":
                eur_p = to_eur(df["Close"].iloc[-1], "GBP", etf_info.get("gbp_pence", False))
                logger.info("%s: %dd | raw=%.2f | EUR=%.4f€", ticker, len(df),
                            df["Close"].iloc[-1], eur_p)
            else:
                logger.info("%s: %dd | último=%.2f", ticker, len(df), df["Close"].iloc[-1])
        except Exception as e:
            logger.error("%s: %s", ticker, e)
    logger.info("%d ativos descarregados", len(raw_data))
    return raw_data
