import logging
import sys
from datetime import date

import numpy as np
import pandas as pd


def _blend_research(resultados_ml: dict, consensus: list[dict]) -> None:
    """
    Blends 13-family research consensus into production ensemble (50/50).
    Updates resultados_ml in-place for tickers that appear in consensus.
    Only affects tickers in the research portfolio (etoro_tickers).
    """
    if not consensus:
        return
    cons_map = {c["ticker"]: c for c in consensus}
    blended_count = 0
    for ticker, res in resultados_ml.items():
        c = cons_map.get(ticker)
        if c is None:
            continue
        for day in [1, 2, 3]:
            h = res["horizons"].get(day)
            if h is None:
                continue
            research_prob = c.get(f"pct_up_d{day}", c.get("pct_up"))
            if research_prob is None:
                continue
            blended        = 0.5 * h["prob"] + 0.5 * float(research_prob)
            h["prob"]      = round(blended, 4)
            h["direction"] = "up" if blended > 0.5 else "down"
            h["confidence"]= round(max(blended, 1 - blended), 4)
        h1 = res["horizons"].get(1)
        if h1:
            res["direction"]  = h1["direction"]
            res["confidence"] = h1["confidence"]
            res["prob"]       = h1["prob"]
        blended_count += 1
    if blended_count:
        logger.info("Blend research→produção: %d tickers actualizados", blended_count)

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

    # Fixado logo no arranque: runs longos (~1h30-2h) podem atravessar a
    # meia-noite UTC. Se "hoje" fosse calculado mais tarde (como antes),
    # ficaria com a data errada — rotulando dados do fecho de ontem como se
    # fossem de hoje, e fazendo o "verificar" do workflow pular o próximo
    # dia real por achar que ele já tinha corrido.
    hoje = pd.Timestamp.now().normalize()

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

    etoro      = portfolio_cfg.get("etoro", [])
    etf_acumul = portfolio_cfg["etf_acumulacao"]

    # ── FX rates ──────────────────────────────────────────────────────────
    from data.downloader import load_fx_rates, download_context, download_prices
    EUR_USD, EUR_GBP, GBP_EUR = load_fx_rates()

    # ── Market data download ──────────────────────────────────────────────
    logger.info("Downloading context (VIX, SPY)...")
    context_data, ctx_warnings = download_context()

    logger.info("Downloading %d tickers...", len(all_tickers))
    raw_data = download_prices(all_tickers)
    logger.info("DEBUG raw_data portfolio (%d/%d): %s",
                len([t for t in my_tickers if t in raw_data]),
                len(my_tickers),
                sorted([t for t in my_tickers if t in raw_data]))

    # ── Feature engineering ───────────────────────────────────────────────
    from features.engineering import build_all_features, FEATURE_COLS
    from config.settings import MIN_SAMPLES_FEATURES
    featured_data = build_all_features(raw_data, context_data, MIN_SAMPLES_FEATURES)
    logger.info("DEBUG featured_data portfolio (%d/%d): %s",
                len([t for t in my_tickers if t in featured_data]),
                len(my_tickers),
                sorted([t for t in my_tickers if t in featured_data]))

    # Retry portfolio tickers ausentes: se algum falhou no build_all_features
    # (ex: amostras insuficientes após limpeza de NaN), tenta sem mínimo de amostras
    missing_portfolio = [t for t in my_tickers if t not in featured_data and t in raw_data]
    if missing_portfolio:
        logger.warning("Portfolio tickers ausentes de featured_data — retry com min_samples=1: %s",
                       missing_portfolio)
        retry_features = build_all_features(
            {t: raw_data[t] for t in missing_portfolio},
            context_data,
            min_samples=1,
        )
        for t, fd in retry_features.items():
            if set(FEATURE_COLS).issubset(fd.columns):
                featured_data[t] = fd
                logger.info("Retry OK: %s adicionado (%d linhas)", t, len(fd))
            else:
                logger.error("Retry falhou: %s sem FEATURE_COLS após build", t)

    # ── Validate past predictions + update weights (before training) ─────
    from models.validator import (
        validate_past_predictions, update_ensemble_weights, save_new_predictions,
    )
    df_log           = validate_past_predictions(df_log, featured_data)
    logger.info("DEBUG featured_data após validação portfolio (%d/%d): %s",
                len([t for t in my_tickers if t in featured_data]),
                len(my_tickers),
                sorted([t for t in my_tickers if t in featured_data]))
    ensemble_weights = update_ensemble_weights(df_log, ensemble_weights)

    # ── Significância estatística (binomial test por horizonte) ───────────
    try:
        from evaluation.significance import compute_significance, save_significance
        sig = compute_significance(df_log)
        save_significance(sig)
    except Exception as e:
        logger.warning("Cálculo de significância falhou (não bloqueia): %s", e)

    # ── Deteção de anomalia (email separado, só dispara em evento fora do normal) ─
    try:
        from evaluation.anomaly import detect_anomaly, save_anomaly
        anomaly = detect_anomaly(context_data, featured_data, etf_acumul)
        save_anomaly(anomaly)
        if anomaly.get("alerta"):
            from reports.email_report_alerta import build_html_alerta, save_html_alerta
            save_html_alerta(build_html_alerta(anomaly))
    except Exception as e:
        logger.warning("Deteção de anomalia falhou (não bloqueia): %s", e)

    # ── ML training (uses today's updated weights) ────────────────────────
    from config.settings import MODELS_DIR
    from models.ensemble import train_all, save_model_metadata, monthly_recalibration

    # Filtrar entradas sem FEATURE_COLS (dados brutos adicionados por validate_past_predictions)
    training_data = {
        t: fd for t, fd in featured_data.items()
        if set(FEATURE_COLS).issubset(fd.columns)
    }
    if len(training_data) < len(featured_data):
        logger.warning(
            "Filtro treino: %d entradas sem features excluídas (%d total → %d válidas)",
            len(featured_data) - len(training_data), len(featured_data), len(training_data),
        )

    monthly_recalibration(training_data, my_tickers, MODELS_DIR)
    resultados_ml = train_all(training_data, ensemble_weights, MODELS_DIR)
    save_model_metadata(resultados_ml, my_tickers)

    # ── Research pipeline → blend com produção antes de gravar previsões ──
    research_data = None
    try:
        from research.runner import run_research
        _rd_features   = {t: res["df"] for t, res in resultados_ml.items() if "df" in res}
        _rd_prices     = {t: res["close_now"] for t, res in resultados_ml.items()}
        _research_tickers = list(etoro) if etoro else list(etf_acumul)
        research_data  = run_research(_rd_features, _research_tickers, _rd_prices)
        _blend_research(resultados_ml, (research_data or {}).get("consensus", []))
    except Exception as e:
        logger.warning("Research pipeline falhou (não bloqueia): %s", e)

    # ── Save today's predictions (já com blend research+produção) ────────
    df_log = save_new_predictions(df_log, resultados_ml, hoje)
    save_public_log(df_log, my_tickers)

    # ── Experimento de regressão (desafiante vs. campeão, email separado) ──
    # Isolado do pipeline principal: qualquer falha aqui não afeta a carteira.
    try:
        from data.storage_experimento import (
            load_experimento_log, validate_experimento_predictions,
            save_new_experimento_predictions,
        )
        from models.regressor_experimento import train_all_regressor
        from evaluation.experimento_significance import (
            compute_experimento_significance, save_significance_experimentos,
        )
        from evaluation.backtest_experimento import run_backtest_experimento
        from reports.email_report_experimentos import build_html_experimentos, save_html_experimentos
        from config.settings import EXPERIMENTO_MODELS_DIR

        EXPERIMENTO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        df_exp_log = load_experimento_log()
        df_exp_log = validate_experimento_predictions(df_exp_log, training_data)

        resultados_exp = train_all_regressor(training_data, EXPERIMENTO_MODELS_DIR)
        df_exp_log = save_new_experimento_predictions(df_exp_log, resultados_ml, resultados_exp, hoje)

        sig_exp  = compute_experimento_significance(df_exp_log)
        save_significance_experimentos(sig_exp)

        backtest_exp = run_backtest_experimento(training_data, my_tickers)

        html_exp = build_html_experimentos(
            resultados_exp, resultados_ml, df_exp_log, sig_exp, backtest_exp, my_tickers,
        )
        save_html_experimentos(html_exp)
    except Exception as e:
        logger.warning("Experimento de regressão falhou (não bloqueia): %s", e)

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
        portfolio_set = set(etoro) | set(etf_acumul)
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

    # ── Charts (publicados como artefacto do run, não persistem entre execuções —
    # checkout é sempre novo, então não há gráficos antigos a limpar aqui) ──────
    from config.settings import CHARTS_DIR
    from reports.charts import generate_charts
    generate_charts(my_tickers, resultados_ml, df_log, portfolio_cfg, CHARTS_DIR)

    # ── Walk-Forward Validation (todos os dias úteis, portfolio tickers) ────
    try:
        from evaluation.walk_forward import run_portfolio_wfv
        wfv_featured = {t: resultados_ml[t]["df"]
                        for t in my_tickers if t in resultados_ml}
        run_portfolio_wfv(wfv_featured, my_tickers, run_date=hoje.date())
    except Exception as e:
        logger.warning("Walk-Forward Validation falhou (não bloqueia): %s", e)

    # ── Email report ──────────────────────────────────────────────────────
    logger.info("DEBUG my_tickers (%d): %s", len(my_tickers), sorted(my_tickers))
    logger.info("DEBUG resultados_ml (%d): %s", len(resultados_ml), sorted(resultados_ml.keys())[:15])
    logger.info("DEBUG intersecção: %d", len([t for t in my_tickers if t in resultados_ml]))
    from reports.email_report import build_html, save_html
    html = build_html(resultados_ml, etf_acumul, df_log, my_tickers, ensemble_weights,
                      research_data=research_data,
                      context_warnings=ctx_warnings)
    save_html(html)

    logger.info("=== Completed successfully ===")


if __name__ == "__main__":
    main()
