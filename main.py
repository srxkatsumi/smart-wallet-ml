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


def load_privados_cedo() -> list[str]:
    from data.storage import load_privados
    return load_privados()


def main():
    logger.info("=== Carteira Inteligente — %s ===", date.today())

    # Fixado logo no arranque: runs longos (~1h30-2h) podem atravessar a
    # meia-noite UTC. Se "hoje" fosse calculado mais tarde (como antes),
    # ficaria com a data errada — rotulando dados do fecho de ontem como se
    # fossem de hoje, e fazendo o "verificar" do workflow pular o próximo
    # dia real por achar que ele já tinha corrido.
    hoje = pd.Timestamp.now().normalize()
    falhas: list[tuple[str, str]] = []   # etapas que falharam sem bloquear, para o bloco de saúde do email

    # ── Storage setup ─────────────────────────────────────────────────────
    from data.storage import (
        ensure_dirs, load_predictions_log, load_ensemble_weights,
        load_portfolio_config, load_my_tickers, load_watchlist,
        build_ticker_order, save_public_log, portfolio_tickers,
    )
    ensure_dirs()
    df_log          = load_predictions_log()
    ensemble_weights= load_ensemble_weights()
    portfolio_cfg   = load_portfolio_config()
    my_tickers      = load_my_tickers()
    watchlist       = load_watchlist()
    all_tickers     = build_ticker_order(list(dict.fromkeys(my_tickers + load_privados_cedo())), watchlist)

    etoro      = portfolio_cfg.get("etoro", [])
    etf_acumul = portfolio_cfg["etf_acumulacao"]
    carteira   = portfolio_tickers(portfolio_cfg)  # carteira PÚBLICA: tudo o que é publicado usa esta
    # Ativos privados (secret PORTFOLIO_PRIVADO): só nos emails, nunca em nada commitado ou no site.
    from data.storage import load_privados
    privados         = [t for t in load_privados() if t not in carteira]
    carteira_email   = carteira + privados
    my_tickers_email = list(dict.fromkeys(my_tickers + privados))

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
    missing_portfolio = [t for t in my_tickers_email if t not in featured_data and t in raw_data]
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
        falhas.append(("Significância", str(e)))

    # ── Calibração de confiança (ERR-013): reajusta a curva isotônica contra
    # o histórico validado e reaplica a todo o log quando é hora de reajustar
    # (mesma cadência do recalibration do SGD) ────────────────────────────
    calib_curves = {}
    try:
        from models.calibration import monthly_calibration_refresh
        df_log, calib_curves = monthly_calibration_refresh(df_log)
    except Exception as e:
        logger.warning("Calibração de confiança falhou (não bloqueia): %s", e)
        falhas.append(("Calibração", str(e)))

    # ── Deteção de anomalia (email separado, só dispara em evento fora do normal) ─
    try:
        from evaluation.anomaly import detect_anomaly, save_anomaly
        from evaluation.anomaly import versao_publica
        anomaly_email = detect_anomaly(context_data, featured_data, carteira_email)
        anomaly = versao_publica(anomaly_email, privados)   # a que é commitada e alimenta a MaoMao
        save_anomaly(anomaly)
        if anomaly_email.get("alerta"):
            from reports.email_report_alerta import build_html_alerta, save_html_alerta
            save_html_alerta(build_html_alerta(anomaly_email))
    except Exception as e:
        logger.warning("Deteção de anomalia falhou (não bloqueia): %s", e)
        falhas.append(("Anomalia", str(e)))
        anomaly = {}

    # ── MaoMao: rede neural pulsada do pet do site, evolui com o mercado real
    # (ver maomao/README.md — inspirado no Pestinha, implementação própria).
    # Dois humores independentes: carteira (só os 7 ETFs) e mercado (watchlist
    # inteiro) ──────────────────────────────────────────────────────────────
    try:
        from maomao.daily import run_daily_step
        run_daily_step(df_log, anomaly, carteira)
    except Exception as e:
        logger.warning("MaoMao (carteira) falhou (não bloqueia): %s", e)
        falhas.append(("MaoMao carteira", str(e)))

    try:
        from maomao.daily import run_market_step
        run_market_step(df_log, anomaly, all_tickers)
    except Exception as e:
        logger.warning("MaoMao (mercado) falhou (não bloqueia): %s", e)
        falhas.append(("MaoMao mercado", str(e)))

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

    # Ações da carteira em dólar aparecem em euro no email (moeda em que você compra);
    # o modelo continua prevendo o preço em USD da bolsa americana.
    for t_ in carteira_email:
        if t_ in resultados_ml and "." not in t_ and EUR_USD:
            resultados_ml[t_]["close_eur"] = resultados_ml[t_]["close_now"] / EUR_USD

    # ── Research pipeline: só relatório, não altera as previsões de produção ──
    research_data = None
    try:
        from research.runner import run_research
        _rd_features   = {t: res["df"] for t, res in resultados_ml.items() if "df" in res}
        _rd_prices     = {t: res["close_now"] for t, res in resultados_ml.items()}
        _research_tickers = list(carteira)
        research_data  = run_research(_rd_features, _research_tickers, _rd_prices)
    except Exception as e:
        logger.warning("Research pipeline falhou (não bloqueia): %s", e)
        falhas.append(("13 famílias", str(e)))

    # ── Save today's predictions ─────────────────────────────────────────
    from evaluation.rastreio import contexto_execucao
    rastreio = contexto_execucao(calib_curves)
    df_log = save_new_predictions(df_log, resultados_ml, hoje, calib_curves, rastreio)
    save_public_log(df_log, my_tickers)

    # ── Ensemble ampliado (desafiante): produção + 13 famílias com pesos
    # encolhidos. Só registra e confere; não muda as previsões do email ──────
    ampliado = None
    try:
        from research.ampliado import run_ampliado
        _ref_dates = {t: str(pd.Timestamp(resultados_ml[t]["last_date"]).date())
                      for t in carteira if t in resultados_ml and "last_date" in resultados_ml[t]}
        ampliado = run_ampliado(df_log, carteira, _ref_dates, str(hoje.date()))
    except Exception as e:
        logger.warning("Ensemble ampliado falhou (não bloqueia): %s", e)
        falhas.append(("Ensemble ampliado", str(e)))

    # ── Desafiante de regime: produção + VIX/VIX3M, juro 10a e dólar. Só
    # registra e confere; não muda as previsões do email ─────────────────
    regime = None
    try:
        from research.regime import run_regime
        regime = run_regime(resultados_ml, carteira, df_log, str(hoje.date()))
    except Exception as e:
        logger.warning("Desafiante de regime falhou (não bloqueia): %s", e)
        falhas.append(("Desafiante de regime", str(e)))

    # ── Desafiantes de volatilidade: movimento grande (logística) e EWMA.
    # Só registram e conferem; não mudam nada do que o email afirma ──────
    volat = None
    try:
        from research.volatilidade import run_volatilidade
        volat = run_volatilidade(resultados_ml, carteira, str(hoje.date()))
    except Exception as e:
        logger.warning("Desafiantes de volatilidade falharam (não bloqueia): %s", e)
        falhas.append(("Volatilidade", str(e)))

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
            resultados_exp, resultados_ml, df_exp_log, sig_exp, backtest_exp, my_tickers_email,
        )
        save_html_experimentos(html_exp)
    except Exception as e:
        logger.warning("Experimento de regressão falhou (não bloqueia): %s", e)
        falhas.append(("Experimento", str(e)))

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
        falhas.append(("MLflow", str(e)))

    # ── SHAP: portfolio tickers, D+1, modelo RF ───────────────────────────
    try:
        import json as _json
        from pathlib import Path as _Path
        from evaluation.explainability import shap_tree
        from features.engineering import FEATURE_COLS
        _xai_dir = _Path("output/xai")
        _xai_dir.mkdir(parents=True, exist_ok=True)
        portfolio_set = set(carteira)
        _shap_count = 0
        for ticker in portfolio_set:
            res = resultados_ml.get(ticker)
            if not res:
                continue
            rf_cal = res["horizons"][1].get("rf_model")
            scaler = res["horizons"][1].get("scaler")
            df_t   = res.get("df")
            if rf_cal is None or df_t is None or scaler is None:
                continue
            # O RF foi treinado em dados escalados: sem isto, os SHAP saem de splits errados.
            X = scaler.transform(df_t[FEATURE_COLS].dropna().values)
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
        falhas.append(("SHAP", str(e)))

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
        falhas.append(("Walk-forward", str(e)))

    # ── Email report ──────────────────────────────────────────────────────
    logger.info("DEBUG my_tickers (%d): %s", len(my_tickers), sorted(my_tickers))
    logger.info("DEBUG resultados_ml (%d): %s", len(resultados_ml), sorted(resultados_ml.keys())[:15])
    logger.info("DEBUG intersecção: %d", len([t for t in my_tickers if t in resultados_ml]))
    from reports.email_report import build_html, save_html
    # ── Email semanal: sai com o último pregão da semana (chega no sábado) ──
    try:
        from reports.email_report_semanal import (
            e_fim_de_semana_de_pregao, build_html_semanal, save_html_semanal,
        )
        _ultimas = [pd.Timestamp(resultados_ml[t]["last_date"]) for t in carteira_email
                    if t in resultados_ml and "last_date" in resultados_ml[t]]
        if _ultimas and e_fim_de_semana_de_pregao(max(_ultimas), etf_acumul[0]):
            from research.runner import RESEARCH_LOG as _rlog_path
            _rlog = pd.read_csv(_rlog_path, low_memory=False) if _rlog_path.exists() else pd.DataFrame()
            save_html_semanal(build_html_semanal(df_log, _rlog, carteira_email, resultados_ml,
                                                 context_data, max(_ultimas), ampliado, regime, rastreio,
                                                 volat))
    except Exception as e:
        logger.warning("Email semanal falhou (não bloqueia): %s", e)
        falhas.append(("Email semanal", str(e)))

    # ── Saúde do sistema e eventos das próximas sessões (topo do email) ──
    saude, eventos = None, None
    try:
        from evaluation.saude import avaliar_saude
        saude = avaliar_saude(carteira_email, resultados_ml, falhas, ctx_warnings)
    except Exception as e:
        logger.warning("Saúde do sistema falhou (não bloqueia): %s", e)
    risco = None
    try:
        from evaluation.risco import contexto_mercado, risco_carteira
        _ctx = contexto_mercado(context_data)
        risco = {"contexto": _ctx,
                 "carteira": risco_carteira(resultados_ml, carteira_email, (_ctx.get("vix") or (None,))[0])}
    except Exception as e:
        logger.warning("Risco da carteira falhou (não bloqueia): %s", e)
        falhas.append(("Risco", str(e)))
    try:
        from evaluation.eventos import proximos_eventos
        eventos = proximos_eventos(hoje, list(dict.fromkeys(portfolio_cfg.get("acoes", []) + privados)))
    except Exception as e:
        logger.warning("Calendário de eventos falhou (não bloqueia): %s", e)

    html = build_html(resultados_ml, etf_acumul, df_log, my_tickers_email, ensemble_weights,
                      research_data=research_data,
                      context_warnings=ctx_warnings,
                      ampliado=ampliado, saude=saude, eventos=eventos, risco=risco,
                      calib_curves=calib_curves, rastreio=rastreio)
    save_html(html)

    logger.info("=== Completed successfully ===")


if __name__ == "__main__":
    main()
