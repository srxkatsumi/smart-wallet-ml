"""
Research Runner: executa todos os dias úteis, só para relatório: nada aqui
altera as previsões de produção.

Cada modelo é retreinado quando fica mais velho que RETRAIN_AFTER_DAYS (por
idade, não por dia da semana: as execuções atrasam e mudam de data), dentro de
RETRAIN_BUDGET_MIN por noite. Nos outros dias carrega o modelo guardado e corre
apenas predict() com as features do dia. Famílias com RETRAIN_DAILY ajustam sempre.

Gera para o email a tabela de comparação das 13 famílias e o consenso
ponderado por ticker. Gera também output/telegram_resumo.txt com a
tendência diária de acurácia por família (melhora / piora).
"""

import joblib
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

from config.settings import WEIGHT_DECAY_FACTOR, MIN_VALIDATIONS_WEIGHT
from data.downloader import official_close
from data.storage import aligned_rows
from evaluation.significance import apply_bh_correction, daily_cluster_stats

logger = logging.getLogger(__name__)

RESEARCH_LOG        = Path("output/predictions_research_log.csv")
RESEARCH_MODELS_DIR = Path("output/models/research")
TELEGRAM_RESUMO     = Path("output/telegram_resumo.txt")

RETRAIN_AFTER_DAYS  = 7
RETRAIN_BUDGET_MIN  = 60
LATEST_ROWS         = 60    # histórico recente para predict(); as famílias sequenciais usam 20
# Mudou quando as sequências passaram a incluir a barra do dia: invalida modelos antigos.
MODEL_VERSION       = "v2"

# Famílias e módulos de modelos
_FAMILIES = {
    "classico_avancado": "models.classical",
    "estado_oculto":     "models.markov",
    "series_temporais":  "models.timeseries",
    "neural_recorrente": "models.neural",
    "neural_atencao":    "models.transformer",
    "bayesiano":         "models.bayesian",
    "generativo":        "models.generative",
    "reinforcement":     "models.reinforcement",
    "contrarian":        "models.contrarian",
    "eficiente":         "models.efficient",
    "foundation":        "models.foundation",
    "conformal":         "models.conformal",
    "drift":             "models.drift",
}

_FAMILY_LABELS_SHORT: dict[str, str] = {
    "classico_avancado": "Clássico",
    "estado_oculto":     "HMM",
    "series_temporais":  "Séries temporais",
    "neural_recorrente": "Neural recorrente",
    "neural_atencao":    "Transformer",
    "bayesiano":         "Bayesiano",
    "generativo":        "VAE/GAN",
    "reinforcement":     "Reforço",
    "contrarian":        "Contrarian",
    "eficiente":         "TCN/PatchTST",
    "foundation":        "Foundation",
    "conformal":         "Conformal",
    "drift":             "Drift",
}


def _model_path(ticker: str, family: str, horizon: int) -> Path:
    safe = ticker.replace(".", "_").replace("-", "_")
    return RESEARCH_MODELS_DIR / f"{safe}_{family}_{horizon}_{MODEL_VERSION}.pkl"


def _remove_old_versions() -> None:
    for old in RESEARCH_MODELS_DIR.glob("*.pkl"):
        if not old.name.endswith(f"_{MODEL_VERSION}.pkl"):
            old.unlink()


_RES_COLS = [
    "week_date", "ticker", "family",
    "prob_d1", "direction_d1",
    "prob_d2", "direction_d2",
    "prob_d3", "direction_d3",
    "ref_price",
    "actual_d1", "actual_d2", "actual_d3",
    "correct_d1", "correct_d2", "correct_d3",
    "validated", "ref_date",
]


def _load_log() -> pd.DataFrame:
    if RESEARCH_LOG.exists():
        return pd.read_csv(RESEARCH_LOG)
    return pd.DataFrame(columns=_RES_COLS)


def _save_log(df: pd.DataFrame):
    RESEARCH_LOG.parent.mkdir(exist_ok=True)
    df.to_csv(RESEARCH_LOG, index=False)


def _get_xy(df_ticker: pd.DataFrame, horizon: int = 1):
    """Extrai X e y para o horizonte dado a partir de um DataFrame com features."""
    from features.engineering import FEATURE_COLS
    target_col = f"target_d{horizon}"
    if target_col not in df_ticker.columns:
        return None, None
    valid = df_ticker.dropna(subset=FEATURE_COLS + [target_col])
    if len(valid) < 50:
        return None, None
    X = valid[FEATURE_COLS].values.astype(np.float32)
    y = valid[target_col].values.astype(float)
    return X, y


def _latest_rows(df_ticker: pd.DataFrame) -> np.ndarray | None:
    """Features mais recentes, inclusive as linhas cujo rótulo ainda não existe."""
    from features.engineering import FEATURE_COLS
    rows = df_ticker[FEATURE_COLS].dropna().tail(LATEST_ROWS)
    return rows.values.astype(np.float32) if len(rows) else None


def _needs_retrain(path: Path, always: bool) -> bool:
    if always or not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) / 86400 >= RETRAIN_AFTER_DAYS


def _predict_family(family: str, X: np.ndarray, y: np.ndarray,
                    X_latest: np.ndarray,
                    ticker: str = "",
                    horizon: int = 1,
                    deadline: float = float("inf")) -> float | None:
    """
    Devolve a probabilidade da última linha de X_latest, ou None se não houver
    previsão: 0.5 viraria um voto em alta e entraria na acurácia como se fosse real.
    """
    import importlib
    try:
        mod  = importlib.import_module(family)
        path = _model_path(ticker, family, horizon)
        retrain = _needs_retrain(path, getattr(mod, "RETRAIN_DAILY", False))

        if retrain and time.time() > deadline and not getattr(mod, "RETRAIN_DAILY", False):
            if not path.exists():
                logger.info("Orçamento de retreino esgotado: %s fica sem previsão hoje", path.name)
                return None
            retrain = False

        if not retrain:
            try:
                model = joblib.load(path)
                probs = mod.predict(model, X_latest)
                return float(np.clip(probs[-1], 0.0, 1.0))
            except Exception as load_err:
                logger.warning("Carregar %s falhou — a retreinar: %s", path.name, load_err)

        model = mod.train(X, y)

        try:
            RESEARCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path, compress=1)
        except Exception as save_err:
            logger.warning("Guardar %s falhou: %s", path.name, save_err)

        probs = mod.predict(model, X_latest)
        return float(np.clip(probs[-1], 0.0, 1.0))
    except Exception as e:
        logger.warning("Família %s falhou: %s", family, e)
        return None


def _update_research_weights(log: pd.DataFrame, current_weights: dict) -> dict:
    """Recalcula pesos de cada família com base no histórico de acertos (últimos 30 dias validados)."""
    validated = aligned_rows(log[log["validated"] == True]).copy()  # noqa: E712
    updated = False

    for day_n in [1, 2, 3]:
        day_key   = f"d{day_n}"
        correct_col = f"correct_d{day_n}"
        if correct_col not in validated.columns:
            continue

        new_weights = {}
        for family in _FAMILIES:
            fam_rows = validated[(validated["family"] == family) & validated[correct_col].notna()]

            # Janela pelos últimos 30 week_date distintos, não pelas últimas
            # 30 linhas: cada dia valida vários tickers da carteira de uma vez
            # (ver o mesmo fix em models/validator.py:update_ensemble_weights).
            dates = sorted(fam_rows["week_date"].unique())[-30:]
            rows  = fam_rows[fam_rows["week_date"].isin(dates)]
            if len(rows) < MIN_VALIDATIONS_WEIGHT:
                # Peso neutro: os pesos antigos vêm de validações por horizonte inválidas.
                new_weights[family] = 1.0
                continue

            n_dias    = len(dates)
            decay_dia = np.exp(WEIGHT_DECAY_FACTOR * np.arange(n_dias))
            decay_dia = decay_dia / decay_dia.sum()
            peso_por_data = dict(zip(dates, decay_dia))

            acc_por_dia  = rows[correct_col].astype(float).groupby(rows["week_date"]).mean().reindex(dates)
            acc_weighted = sum(acc_por_dia[d] * peso_por_data[d] for d in dates)
            new_weights[family] = max(0.1, acc_weighted)

        total = sum(new_weights.values())
        n_fam = len(_FAMILIES)
        current_weights[day_key] = {
            k: round(v * n_fam / total, 4) for k, v in new_weights.items()
        }
        updated = True

    if updated:
        from data.storage import save_research_weights
        save_research_weights(current_weights)
        best = max(current_weights["d1"], key=current_weights["d1"].get)
        logger.info("Research weights atualizados — d1 melhor: %s (%.3f)",
                    best, current_weights["d1"][best])
    else:
        logger.info("Research weights: histórico insuficiente — mantêm-se")

    return current_weights


def run_research(featured_data: dict,
                 portfolio_tickers: list,
                 close_prices: dict) -> dict:
    """
    Ponto de entrada principal — chamado todos os dias úteis.

    Parameters
    ----------
    featured_data     : dict {ticker: pd.DataFrame} com features calculadas
    portfolio_tickers : lista de tickers da carteira (eToro)
    close_prices      : dict {ticker: float} preço de fecho de hoje

    Returns
    -------
    dict com chaves 'comparison' e 'consensus' para o email
    """
    from data.storage import load_research_weights
    today_str        = date.today().isoformat()
    deadline         = time.time() + RETRAIN_BUDGET_MIN * 60
    log              = _load_log()
    if "ref_date" not in log.columns:
        log["ref_date"] = pd.Series(dtype="object")
    research_weights = load_research_weights()
    if RESEARCH_MODELS_DIR.exists():
        _remove_old_versions()

    # ── 1. Validar previsões anteriores ───────────────────────────────────
    log = _validate_past_predictions(log, featured_data)

    # ── 1b. Atualizar pesos com base nos acertos validados ────────────────
    research_weights = _update_research_weights(log, research_weights)

    # ── 2. Treinar (se preciso) e prever ──────────────────────────────────
    new_rows = []
    for ticker in portfolio_tickers:
        if ticker not in featured_data or ticker not in close_prices:
            continue
        df = featured_data[ticker]
        ref_price = close_prices[ticker]
        ref_date  = str(df.index[-1].date())
        X_latest  = _latest_rows(df)

        for family, module in _FAMILIES.items():
            ja_existe = (
                (log["ticker"] == ticker) & (log["family"] == family) & (log["ref_date"] == ref_date)
            ).any() if not log.empty else False
            if ja_existe:
                continue

            row = {
                "week_date":   today_str,
                "ticker":      ticker,
                "family":      family,
                "ref_price":   ref_price,
                "ref_date":    ref_date,
                "validated":   False,
            }
            for h in [1, 2, 3]:
                X, y = _get_xy(df, h)
                prob = None
                if X is not None and X_latest is not None:
                    prob = _predict_family(module, X, y, X_latest,
                                           ticker=ticker, horizon=h,
                                           deadline=deadline)
                row[f"prob_d{h}"]      = round(prob, 4) if prob is not None else None
                row[f"direction_d{h}"] = None if prob is None else ("up" if prob >= 0.5 else "down")

            for col in ["actual_d1", "actual_d2", "actual_d3",
                        "correct_d1", "correct_d2", "correct_d3"]:
                row[col] = None

            new_rows.append(row)
            logger.info("  %s %s: D+1=%s D+2=%s D+3=%s",
                        ticker, family, row["prob_d1"], row["prob_d2"], row["prob_d3"])

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    _save_log(log)

    # ── 3. MLflow: registar previsões e acurácia validada ─────────────────
    try:
        from evaluation.tracking import log_run

        # Previsões desta semana — uma run por família × ticker
        for row in new_rows:
            log_run(
                experiment_name=f"research_{row['family']}",
                model_name=row["family"],
                params={
                    "ticker":    row["ticker"],
                    "week_date": row["week_date"],
                },
                metrics={
                    f"prob_d{h}": float(row[f"prob_d{h}"])
                    for h in [1, 2, 3] if row[f"prob_d{h}"] is not None
                },
                tags={
                    "ticker": row["ticker"],
                    "family": row["family"],
                    "date":   row["week_date"],
                    "domain": "carteira_research",
                },
            )

        # Acurácia validada da semana anterior — uma run por família
        validated = aligned_rows(log[log["validated"] == True]).copy()  # noqa: E712
        if not validated.empty:
            for family in _FAMILIES:
                fam_rows = validated[validated["family"] == family]
                if fam_rows.empty:
                    continue
                acc_d1 = float(fam_rows["correct_d1"].astype(float).mean())
                acc_d2 = float(fam_rows["correct_d2"].astype(float).mean())
                acc_d3 = float(fam_rows["correct_d3"].astype(float).mean())
                log_run(
                    experiment_name=f"research_{family}_validation",
                    model_name=family,
                    params={"family": family, "week_date": today_str},
                    metrics={
                        "acc_d1":      acc_d1,
                        "acc_d2":      acc_d2,
                        "acc_d3":      acc_d3,
                        "vs_acaso_d1": round(acc_d1 - 0.5, 3),
                    },
                    tags={"domain": "carteira_research_validation", "date": today_str},
                )

        logger.info("MLflow: %d runs de investigação registados (%d famílias × %d ativos)",
                    len(new_rows), len(_FAMILIES), len(portfolio_tickers))
    except Exception as e:
        logger.warning("MLflow research logging falhou (nao bloqueia): %s", e)

    # ── 5. Gerar dados para o email ───────────────────────────────────────
    comparison = _build_comparison(log, today_str)
    consensus  = _build_consensus(new_rows, portfolio_tickers, research_weights)

    # ── 6. Gerar resumo Telegram ──────────────────────────────────────────
    _generate_telegram_resumo(log)

    logger.info("Research run concluído: %d famílias × %d ativos",
                len(_FAMILIES), len(portfolio_tickers))
    return {"comparison": comparison, "consensus": consensus}


def _build_telegram_message(log: pd.DataFrame) -> str:
    """
    Mensagem diária de tendência dos modelos para Telegram.
    Mostra acurácia D+1 dos últimos 10 dias e seta de tendência
    comparando com os 10 dias anteriores (se disponíveis).
    """
    validated = aligned_rows(log[log["correct_d1"].notna()]).copy() if "correct_d1" in log.columns else log.iloc[0:0]
    today_str = date.today().strftime("%d/%m/%Y")

    if validated.empty:
        return (f"🤖 Modelos — {today_str}\n\n"
                "Sem dados validados ainda — acurácia disponível a partir de amanhã.")

    results = []
    for family in _FAMILIES:
        rows = validated[validated["family"] == family].copy()
        if rows.empty:
            continue
        # Por dias distintos, não por linhas: cada dia tem uma linha por ticker.
        dates   = sorted(rows["week_date"].unique())
        recent  = rows[rows["week_date"].isin(dates[-10:])]["correct_d1"].astype(float)
        prev    = rows[rows["week_date"].isin(dates[-20:-10])]["correct_d1"].astype(float)
        acc     = recent.mean()
        trend   = acc - prev.mean() if len(prev) >= 5 else 0.0
        results.append((family, acc, trend, len(rows)))

    if not results:
        return (f"🤖 Modelos — {today_str}\n\n"
                "Sem acurácia validada ainda.")

    results.sort(key=lambda x: -x[1])

    lines = [f"🤖 Modelos — {today_str}", "", "D+1 Acurácia (últ. 10 dias):"]
    for fam, acc, trend, n in results:
        icon  = "🟢" if acc >= 0.60 else ("⚪" if acc >= 0.50 else "🔴")
        if abs(trend) < 0.02:
            arrow = "→"
        elif trend > 0:
            arrow = f"↑ +{trend*100:.0f}pp"
        else:
            arrow = f"↓ {trend*100:.0f}pp"
        label = _FAMILY_LABELS_SHORT.get(fam, fam)
        lines.append(f"{icon} {label:<18} {acc*100:.0f}%  {arrow}")

    best  = results[0]
    worst = results[-1]
    lines.append("")
    lines.append(f"🏆 Melhor: {_FAMILY_LABELS_SHORT.get(best[0], best[0])} ({best[1]*100:.0f}%)")
    if worst[1] < 0.50:
        lines.append(f"⚠️  Pior:   {_FAMILY_LABELS_SHORT.get(worst[0], worst[0])} ({worst[1]*100:.0f}%)")

    # Walk-Forward Validation summary
    try:
        import json as _json
        from pathlib import Path as _Path
        wfv_path = _Path("output/wfv_results.json")
        if wfv_path.exists():
            wfv = _json.loads(wfv_path.read_text())
            pm  = wfv.get("portfolio_mean", {})
            acc = pm.get("acc_d1")
            if acc is not None:
                wfv_icon = "🟢" if acc >= 0.55 else ("⚪" if acc >= 0.50 else "🔴")
                lines.append("")
                lines.append(f"📊 WFV {wfv['date']} — D+1 real: {wfv_icon} {acc*100:.1f}%")
    except Exception:
        pass

    # Significância estatística (silenciosa até n≥100 por horizonte)
    try:
        from evaluation.significance import load_significance, format_telegram_line
        sig_line = format_telegram_line(load_significance())
        if sig_line:
            lines.append(sig_line)
    except Exception:
        pass

    return "\n".join(lines)


def _generate_telegram_resumo(log: pd.DataFrame) -> None:
    """Escreve output/telegram_resumo.txt para o workflow enviar via Telegram."""
    try:
        msg = _build_telegram_message(log)
        TELEGRAM_RESUMO.parent.mkdir(parents=True, exist_ok=True)
        TELEGRAM_RESUMO.write_text(msg, encoding="utf-8")
        logger.debug("telegram_resumo.txt gerado (%d chars)", len(msg))
    except Exception as e:
        logger.warning("Falha ao gerar telegram_resumo.txt: %s", e)


def _validate_past_predictions(log: pd.DataFrame, featured_data: dict) -> pd.DataFrame:
    """
    Cada horizonte é comparado com o fecho N pregões depois de ref_date. A linha
    só fica validated quando os três horizontes têm resultado. Linhas sem
    ref_date são de antes desta regra: fecham sem resultado, ficam como histórico.
    """
    if log.empty:
        return log
    pending = log[log["validated"] == False]  # noqa: E712
    for idx, row in pending.iterrows():
        ref_date = row.get("ref_date")
        if pd.isna(ref_date):
            log.at[idx, "validated"] = True
            continue
        df = featured_data.get(row["ticker"])
        ref = row.get("ref_price")
        if df is None or not ref or pd.isna(ref):
            continue
        ref = official_close(df, ref_date) or ref

        after = df["Close"][df.index > pd.Timestamp(ref_date)]
        for h in [1, 2, 3]:
            if len(after) < h:
                break
            actual = float(after.iloc[h - 1])
            log.at[idx, f"actual_d{h}"] = actual
            direction = row.get(f"direction_d{h}")
            if isinstance(direction, str):
                log.at[idx, f"correct_d{h}"] = bool((direction == "up") == (actual >= ref))
        if len(after) >= 3:
            log.at[idx, "validated"] = True

    return log


def _build_comparison(log: pd.DataFrame, today_str: str) -> list[dict]:
    """
    Constrói tabela de acurácia D+1 por família.

    Cada família tem um p-valor (unilateral, H1: acurácia > 50%, com o dia como
    unidade), e o conjunto passa por correção de Benjamini-Hochberg antes de
    ordenar: com 13 famílias testadas, uma parecer "significativa" por puro acaso
    é comum se cada p-valor for lido isoladamente. 'sig' reflete o p-valor corrigido.
    """
    if log.empty or "correct_d1" not in log.columns:
        return []
    validated = aligned_rows(log[log["correct_d1"].notna()])
    if validated.empty:
        return []

    results = []
    for family in _FAMILIES:
        fam_rows = validated[validated["family"] == family]
        if fam_rows.empty:
            continue
        correct = fam_rows["correct_d1"].astype(float)
        n     = len(fam_rows)
        acc   = correct.sum() / n
        stats = daily_cluster_stats(correct, fam_rows["week_date"])
        results.append({
            "family":   family,
            "accuracy": round(float(acc), 3),
            "n":        n,
            "vs_acaso": round(float(acc) - 0.5, 3),
            "p":        round(stats["p"], 4),
        })

    apply_bh_correction(results)
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results


def _build_consensus(new_rows: list, portfolio_tickers: list,
                     research_weights: dict) -> list[dict]:
    """
    Consenso ponderado por família para D+1, D+2 e D+3. Famílias sem previsão não votam.
    """
    consensus = []
    for ticker in portfolio_tickers:
        ticker_rows = [r for r in new_rows if r["ticker"] == ticker]
        if not ticker_rows:
            continue

        row: dict = {"ticker": ticker, "total": sum(1 for r in ticker_rows if r.get("direction_d1"))}

        for h in [1, 2, 3]:
            w = research_weights.get(f"d{h}", {})
            weight_up    = sum(w.get(r["family"], 1.0)
                               for r in ticker_rows if r.get(f"direction_d{h}") == "up")
            weight_total = sum(w.get(r["family"], 1.0) for r in ticker_rows if r.get(f"direction_d{h}"))
            pct          = weight_up / max(weight_total, 1e-9)
            row[f"pct_up_d{h}"] = round(pct, 4)

        # Campos D+1 em retrocompatibilidade com o email report
        pct_d1 = row["pct_up_d1"]
        row.update({
            "pct_up":    pct_d1,
            "up_count":  sum(1 for r in ticker_rows if r.get("direction_d1") == "up"),
            "direction": "ALTA" if pct_d1 >= 0.5 else "BAIXA",
            "strength":  "forte" if abs(pct_d1 - 0.5) >= 0.25 else "fraco",
        })
        consensus.append(row)

    consensus.sort(key=lambda x: x["pct_up"], reverse=True)
    return consensus