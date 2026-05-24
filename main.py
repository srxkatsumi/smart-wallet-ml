import logging
import sys
from datetime import date

import numpy as np
import pandas as pd

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


def main():
    logger.info("=== Carteira Inteligente — %s ===", date.today())

    # ── Storage setup ─────────────────────────────────────────────────────
    from data.storage import (
        ensure_dirs, load_predictions_log, load_ensemble_weights,
        load_portfolio_config, load_my_tickers, load_watchlist,
        build_ticker_order,
    )
    ensure_dirs()
    df_log          = load_predictions_log()
    ensemble_weights= load_ensemble_weights()
    portfolio_cfg   = load_portfolio_config()
    my_tickers      = load_my_tickers()
    watchlist       = load_watchlist()
    all_tickers     = build_ticker_order(my_tickers, watchlist)

    etoro        = portfolio_cfg["etoro"]
    etf_acumul   = portfolio_cfg["etf_acumulacao"]

    # ── FX rates ──────────────────────────────────────────────────────────
    from data.downloader import load_fx_rates, download_context, download_prices
    EUR_USD, EUR_GBP, GBP_EUR = load_fx_rates()

    # ── Market data download ──────────────────────────────────────────────
    logger.info("Downloading context (VIX, SPY)...")
    context_data, ctx_warnings = download_context()

    logger.info("Downloading %d tickers...", len(all_tickers))
    raw_data = download_prices(all_tickers, etf_acumul)

    # ── Feature engineering ───────────────────────────────────────────────
    from features.engineering import build_all_features
    from config.settings import MIN_SAMPLES_FEATURES
    featured_data = build_all_features(raw_data, context_data, MIN_SAMPLES_FEATURES)

    # ── Validate past predictions + update weights (before training) ─────
    from models.validator import (
        validate_past_predictions, update_ensemble_weights, save_new_predictions,
    )
    hoje = pd.Timestamp.now().normalize()
    df_log           = validate_past_predictions(df_log, featured_data)
    ensemble_weights = update_ensemble_weights(df_log, ensemble_weights)

    # ── ML training (uses today's updated weights) ────────────────────────
    from config.settings import MODELS_DIR
    from models.ensemble import train_all, save_model_metadata, monthly_recalibration

    monthly_recalibration(featured_data, my_tickers, MODELS_DIR)
    resultados_ml = train_all(featured_data, ensemble_weights, MODELS_DIR)
    save_model_metadata(resultados_ml, my_tickers)

    # ── Save today's predictions ──────────────────────────────────────────
    df_log = save_new_predictions(df_log, resultados_ml, hoje)

    # ── Portfolio P&L ─────────────────────────────────────────────────────
    from portfolio.pnl import calculate_etoro_pnl, calculate_etf_pnl
    resumo_etoro, totals_etoro = calculate_etoro_pnl(etoro, resultados_ml, EUR_USD)
    resumo_etfs,  totals_etfs  = calculate_etf_pnl(etf_acumul, resultados_ml)

    # Enrich resultados_ml with EUR-converted price for email display
    close_eur_map = {r["ticker"]: r["close_eur"] for r in resumo_etoro + resumo_etfs}
    for ticker, res in resultados_ml.items():
        res["close_eur"] = close_eur_map.get(ticker, res["close_now"])

    # ── Charts ────────────────────────────────────────────────────────────
    from config.settings import CHARTS_DIR, CHARTS_RETENTION_DAYS
    from reports.charts import cleanup_old_charts, generate_charts
    cleanup_old_charts(CHARTS_DIR, CHARTS_RETENTION_DAYS)
    generate_charts(my_tickers, resultados_ml, df_log, portfolio_cfg, CHARTS_DIR)

    # ── Email report ──────────────────────────────────────────────────────
    from reports.email_report import build_html, save_html
    html = build_html(resultados_ml, resumo_etfs, df_log, my_tickers, ensemble_weights,
                      context_warnings=ctx_warnings)
    save_html(html)

    logger.info("=== Completed successfully ===")


if __name__ == "__main__":
    main()
