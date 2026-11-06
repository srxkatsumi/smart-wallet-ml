import logging
import time
import yfinance as yf
import pandas as pd
from config.settings import (
    PRICE_PERIOD, EUR_USD_FALLBACK, EUR_GBP_FALLBACK,
    DOWNLOAD_BATCH_SIZE, DOWNLOAD_BATCH_SLEEP, GBP_PENCE_TICKERS,
)

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


def _gbp_pence_to_eur(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converte com o câmbio de cada dia, não com o de hoje aplicado à série toda:
    com taxa única, o ref_price gravado ontem e o preço real de hoje ficariam em
    câmbios diferentes, e essa diferença entraria no retorno e no `correct`.
    """
    try:
        fx = yf.download("EURGBP=X", period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        fx.index = pd.to_datetime(fx.index).normalize()
        eur_gbp = fx["Close"].dropna().sort_index()
        if eur_gbp.empty:
            raise ValueError("sem dados")
        rate = 1 / eur_gbp.reindex(df.index, method="ffill").bfill()
    except Exception as e:
        logger.warning("EURGBP=X histórico indisponível (%s), usando taxa única", e)
        rate = pd.Series(GBP_EUR, index=df.index)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] / 100 * rate
    return df


def download_context(period: str = PRICE_PERIOD) -> tuple[dict, list[str]]:
    context_data = {}
    warnings     = []
    for ctx_ticker, ctx_name in [("^VIX", "vix"), ("SPY", "spy"), ("BTC-USD", "btc"), ("GLD", "gold")]:
        try:
            df = yf.download(ctx_ticker, period=period, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("sem dados")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()
            series = df["Close"].rename(ctx_name)

            # Forward fill any NaN in the last 3 days; warn if applied
            nan_recent = series.tail(3)
            nan_dates  = nan_recent[nan_recent.isna()].index
            if len(nan_dates) > 0:
                series = series.ffill()
                dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in nan_dates)
                msg = f"{ctx_ticker}: dados em falta ({dates_str}) — valor anterior usado"
                warnings.append(msg)
                logger.warning(msg)

            context_data[ctx_name] = series
            logger.info("%s: %d dias | último=%.2f", ctx_ticker, len(df), df["Close"].iloc[-1])
        except Exception as e:
            logger.warning("%s: %s — features de contexto usarão valores neutros", ctx_ticker, e)
    return context_data, warnings


def download_prices(tickers: list[str], period: str = PRICE_PERIOD) -> dict:
    raw_data  = {}
    n_batches = (len(tickers) + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE

    for batch_idx in range(n_batches):
        batch = tickers[batch_idx * DOWNLOAD_BATCH_SIZE : (batch_idx + 1) * DOWNLOAD_BATCH_SIZE]
        logger.info("Batch %d/%d (%d tickers)...", batch_idx + 1, n_batches, len(batch))

        for ticker in batch:
            try:
                df = yf.download(ticker, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
                if df.empty:
                    logger.warning("%s: sem dados", ticker)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).normalize()
                df = df.sort_index()
                # Remove linhas sem Close (yfinance adiciona linha vazia do dia em curso
                # para mercados que ainda não abriram, causando NaN que propaga pelo pipeline)
                df = df[df["Close"].notna()]
                if df.empty:
                    logger.warning("%s: sem dados válidos após limpeza", ticker)
                    continue
                if ticker in GBP_PENCE_TICKERS:
                    df = _gbp_pence_to_eur(df, period)
                    logger.info("%s: preços convertidos de GBp para EUR, câmbio diário", ticker)
                raw_data[ticker] = df
                logger.info("%s: %dd | último=%.2f", ticker, len(df), df["Close"].iloc[-1])
            except Exception as e:
                logger.error("%s: %s", ticker, e)

        if batch_idx < n_batches - 1:
            time.sleep(DOWNLOAD_BATCH_SLEEP)

    logger.info("%d ativos descarregados", len(raw_data))
    return raw_data
