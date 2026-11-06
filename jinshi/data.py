"""
Dados de longo histórico para o backtest do Jinshi. Reusa os downloaders e o feature
engineering REAIS da produção (data/downloader.py, features/engineering.py),
só pedindo `period="max"` em vez do `PRICE_PERIOD="2y"` que a produção usa
— não reimplementa nada, só passa um período diferente pros mesmos
caminhos de código já testados.
"""
import logging

from config.settings import ASSET_CLASSES
from data.downloader import download_context, download_prices
from features.engineering import build_features

logger = logging.getLogger(__name__)


def build_long_history(tickers: list[str]) -> dict:
    """
    Baixa o histórico completo disponível (period="max") pros `tickers` e
    pelo contexto (VIX/SPY/BTC/ouro), e devolve {ticker: DataFrame com
    features}, exatamente o mesmo formato que training_data em main.py —
    pra poder reusar models/regressor_experimento.py sem adaptação.
    """
    context_data, warnings = download_context(period="max")
    if warnings:
        logger.warning("Avisos no contexto: %s", warnings)

    raw_data = download_prices(tickers, period="max")

    featured = {}
    for ticker, df in raw_data.items():
        try:
            ac = ASSET_CLASSES.get(ticker, 0)
            feat = build_features(df, context_data, asset_class=ac)
            if not feat.empty:
                featured[ticker] = feat
                logger.info("%s: %d dias de histórico com features completas (%s -> %s)",
                           ticker, len(feat), feat.index[0].date(), feat.index[-1].date())
        except Exception as e:
            logger.error("%s: erro ao construir features: %s", ticker, e)

    return featured
