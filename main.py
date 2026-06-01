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
        build_ticker_order, save_public_log,
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
    save_public_log(df_log, my_tickers)

    # ── MLflow: registar runs diários ────────────────────────────────────
    try:
        from evaluation.tracking import log_run
        from config.settings import (
            N_ESTIMATORS_RF, MAX_DEPTH_RF,
            N_ESTIMATORS_GB, MAX_DEPTH_GB, LEARNING_RATE_GB,
            N_SPLITS_CV, SGD_ALPHA,
        )
        _mlflow_params = {
            "n_estimators_rf":  N_ESTIMATORS_RF,
            "max_depth_rf":     MAX_DEPTH_RF,
            "n_estimators_gb":  N_ESTIMATORS_GB,
            "max_depth_gb":     MAX_DEPTH_GB,
            "learning_rate_gb": LEARNING_RATE_GB,
            "n_splits_cv":      N_SPLITS_CV,
            "sgd_alpha":        SGD_ALPHA,
        }
        for ticker, res in resultados_ml.items():
            if ticker not in my_tickers:
                continue
            for day in [1, 2, 3]:
                h = res["horizons"][day]
                log_run(
                    experiment_name=f"carteira_ensemble_d{day}",
                    model_name="rf_gb_sgd_ensemble",
                    params={"ticker": ticker, "horizon": str(day), **_mlflow_params},
                    metrics={
                        "acc_rf":        h["acc_media"]["rf"],
                        "acc_gb":        h["acc_media"]["gb"],
                        "acc_sgd":       h["acc_media"]["sgd"],
                        "prob_ensemble": h["prob"],
                        "confidence":    h["confidence"],
                    },
                    tags={"ticker": ticker, "date": str(date.today())},
                )
        logger.info("MLflow: runs registados para %d ativos", len(my_tickers))
    except Exception as e:
        logger.warning("MLflow logging falhou (nao bloqueia): %s", e)

    # ── SHAP: portfolio tickers, D+1, modelo RF ───────────────────────────
    try:
        import json as _json
        from pathlib import Path as _Path
        from evaluation.explainability import shap_tree
        from features.engineering import FEATURE_COLS
        _xai_dir = _Path("output/xai")
        _xai_dir.mkdir(parents=True, exist_ok=True)
        portfolio_set = {a["ticker"] for a in etoro} | {a["ticker"] for a in etf_acumul}
        _shap_count = 0
        for ticker in portfolio_set:
            res = resultados_ml.get(ticker)
            if not res:
                continue
            rf_cal = res["horizons"][1].get("rf_model")
            df_t   = res.get("df")
            if rf_cal is None or df_t is None:
                continue
            X = df_t[FEATURE_COLS].dropna().values
            if len(X) < 10:
                continue
            try:
                shap_result = shap_tree(rf_cal, X, feature_names=FEATURE_COLS)
            except Exception:
                inner = rf_cal.calibrated_classifiers_[0].estimator
                shap_result = shap_tree(inner, X, feature_names=FEATURE_COLS)
            out = {
                "ticker": ticker,
                "date":   str(date.today()),
                "top5":   shap_result["top5_features"],
                "mean_abs_shap": dict(zip(FEATURE_COLS,
                                          shap_result["mean_abs_shap"].tolist())),
            }
            safe = ticker.replace(".", "_").replace("-", "_")
            with open(_xai_dir / f"{safe}_d1_shap.json", "w") as f:
                _json.dump(out, f, indent=2)
            _shap_count += 1
        logger.info("SHAP: gerado para %d ativos", _shap_count)
    except Exception as e:
        logger.warning("SHAP falhou (nao bloqueia): %s", e)

    # ── Portfolio P&L ─────────────────────────────────────────────────────
    from portfolio.pnl import (calculate_etoro_pnl, calculate_etf_pnl,
                               expand_etoro_lots, expand_etf_lots)
    resumo_etoro, totals_etoro = calculate_etoro_pnl(etoro, resultados_ml, EUR_USD)
    resumo_etfs,  totals_etfs  = calculate_etf_pnl(etf_acumul, resultados_ml)
    resumo_etoro_lotes         = expand_etoro_lots(etoro, resultados_ml, EUR_USD)
    resumo_etf_lotes           = expand_etf_lots(etf_acumul, resultados_ml)

    # Enrich resultados_ml with EUR-converted price for email display
    close_eur_map = {r["ticker"]: r["close_eur"] for r in resumo_etoro + resumo_etfs}
    for ticker, res in resultados_ml.items():
        res["close_eur"] = close_eur_map.get(ticker, res["close_now"])

    # ── Charts ────────────────────────────────────────────────────────────
    from config.settings import CHARTS_DIR, CHARTS_RETENTION_DAYS
    from reports.charts import cleanup_old_charts, generate_charts
    cleanup_old_charts(CHARTS_DIR, CHARTS_RETENTION_DAYS)
    generate_charts(my_tickers, resultados_ml, df_log, portfolio_cfg, CHARTS_DIR)

    # ── Research pipeline (apenas às segundas-feiras) ─────────────────────
    research_data = None
    is_monday     = pd.Timestamp.today().weekday() == 0
    if is_monday:
        logger.info("Segunda-feira — a correr research pipeline (25 modelos)...")
        try:
            from research.runner import run_monday_research
            from features.engineering import FEATURE_COLS
            featured_data = {t: res["df"] for t, res in resultados_ml.items()
                             if "df" in res}
            close_prices  = {t: res.get("close_eur", res["close_now"])
                             for t, res in resultados_ml.items()}
            etoro_tickers = [a["ticker"] for a in etoro]
            research_data = run_monday_research(
                featured_data, etoro_tickers, close_prices
            )
        except Exception as e:
            logger.warning("Research pipeline falhou (não bloqueia): %s", e)

    # ── Email report ──────────────────────────────────────────────────────
    from reports.email_report import build_html, save_html
    html = build_html(resultados_ml, resumo_etfs, df_log, my_tickers, ensemble_weights,
                      resumo_etoro=resumo_etoro,
                      resumo_etoro_lotes=resumo_etoro_lotes,
                      resumo_etf_lotes=resumo_etf_lotes,
                      research_data=research_data,
                      context_warnings=ctx_warnings)
    save_html(html)

    logger.info("=== Completed successfully ===")


if __name__ == "__main__":
    main()
