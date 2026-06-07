"""
Research Runner — executa apenas às segundas-feiras.

Treina todos os 25 modelos de investigação em cada ativo da carteira,
compara a acurácia da semana anterior por família e calcula o consenso
dos 25 modelos para a semana seguinte.

Nunca substitui o ensemble principal (RF/GB/SGD). Corre DEPOIS do
pipeline diário e gera secções adicionais para o email de segunda.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

from config.settings import WEIGHT_DECAY_FACTOR, MIN_VALIDATIONS_WEIGHT

logger = logging.getLogger(__name__)

RESEARCH_LOG = Path("output/predictions_research_log.csv")

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
}

_RES_COLS = [
    "week_date", "ticker", "family",
    "prob_d1", "direction_d1",
    "prob_d2", "direction_d2",
    "prob_d3", "direction_d3",
    "ref_price",
    "actual_d1", "actual_d2", "actual_d3",
    "correct_d1", "correct_d2", "correct_d3",
    "validated",
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


def _predict_family(family: str, X: np.ndarray, y: np.ndarray,
                    X_latest: np.ndarray) -> float:
    """Treina uma família e retorna P(UP) para X_latest."""
    import importlib
    try:
        mod    = importlib.import_module(family)
        model  = mod.train(X, y)
        probs  = mod.predict(model, X_latest)
        return float(np.clip(probs[-1], 0.0, 1.0))
    except Exception as e:
        logger.warning("Família %s falhou: %s", family, e)
        return 0.5


def _update_research_weights(log: pd.DataFrame, current_weights: dict) -> dict:
    """Recalcula pesos de cada família com base no histórico de acertos (últimas 30 semanas)."""
    validated = log[log["validated"] == True].copy()  # noqa: E712
    updated = False

    for day_n in [1, 2, 3]:
        day_key   = f"d{day_n}"
        correct_col = f"correct_d{day_n}"
        if correct_col not in validated.columns:
            continue

        new_weights = {}
        for family in _FAMILIES:
            rows = validated[validated["family"] == family].tail(30)
            if len(rows) < MIN_VALIDATIONS_WEIGHT or correct_col not in rows.columns:
                new_weights[family] = current_weights[day_key].get(family, 1.0)
                continue
            vals  = rows[correct_col].astype(float).values
            n     = len(vals)
            decay = np.exp(WEIGHT_DECAY_FACTOR * np.arange(n))
            decay = decay / decay.sum()
            acc_weighted      = (vals * decay).sum()
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


def run_monday_research(featured_data: dict,
                        portfolio_tickers: list,
                        close_prices: dict) -> dict:
    """
    Ponto de entrada principal — chamado em main.py apenas às segundas.

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
    log              = _load_log()
    research_weights = load_research_weights()

    # ── 1. Validar previsões da semana anterior ───────────────────────────
    log = _validate_past_predictions(log, close_prices)

    # ── 1b. Atualizar pesos com base nos acertos validados ────────────────
    research_weights = _update_research_weights(log, research_weights)

    # ── 2. Treinar e prever para esta semana ──────────────────────────────
    new_rows = []
    for ticker in portfolio_tickers:
        if ticker not in featured_data or ticker not in close_prices:
            continue
        df = featured_data[ticker]
        ref_price = close_prices[ticker]

        for family, module in _FAMILIES.items():
            row = {
                "week_date":   today_str,
                "ticker":      ticker,
                "family":      family,
                "ref_price":   ref_price,
                "validated":   False,
            }
            for h in [1, 2, 3]:
                X, y = _get_xy(df, h)
                if X is None:
                    row[f"prob_d{h}"]      = 0.5
                    row[f"direction_d{h}"] = "up"
                    continue
                X_latest = X[-1:].copy()
                prob = _predict_family(module, X[:-1], y[:-1], X_latest)
                row[f"prob_d{h}"]      = round(prob, 4)
                row[f"direction_d{h}"] = "up" if prob >= 0.5 else "down"

            for col in ["actual_d1", "actual_d2", "actual_d3",
                        "correct_d1", "correct_d2", "correct_d3"]:
                row[col] = None

            new_rows.append(row)
            logger.info("  %s — %s: D+1=%.0f%% D+2=%.0f%% D+3=%.0f%%",
                        ticker, family,
                        row["prob_d1"] * 100,
                        row["prob_d2"] * 100,
                        row["prob_d3"] * 100)

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
                    "prob_d1": float(row["prob_d1"]),
                    "prob_d2": float(row["prob_d2"]),
                    "prob_d3": float(row["prob_d3"]),
                },
                tags={
                    "ticker": row["ticker"],
                    "family": row["family"],
                    "date":   row["week_date"],
                    "domain": "carteira_research",
                },
            )

        # Acurácia validada da semana anterior — uma run por família
        validated = log[log["validated"] == True].copy()  # noqa: E712
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

    logger.info("Research run concluído: %d famílias × %d ativos",
                len(_FAMILIES), len(portfolio_tickers))
    return {"comparison": comparison, "consensus": consensus}


def _validate_past_predictions(log: pd.DataFrame,
                                close_prices: dict) -> pd.DataFrame:
    """Valida previsões de semanas anteriores que ainda não foram validadas."""
    if log.empty:
        return log
    pending = log[log["validated"] == False].copy()  # noqa: E712
    if pending.empty:
        return log

    today = pd.Timestamp.today().normalize()
    for idx, row in pending.iterrows():
        ticker = row["ticker"]
        if ticker not in close_prices:
            continue
        current_price = close_prices[ticker]
        ref           = row.get("ref_price")
        if not ref or pd.isna(ref):
            continue

        for h in [1, 2, 3]:
            direction = row.get(f"direction_d{h}", "up")
            predicted_up = (direction == "up")
            actual_up    = current_price >= ref
            log.at[idx, f"actual_d{h}"] = current_price
            log.at[idx, f"correct_d{h}"] = bool(predicted_up == actual_up)

        log.at[idx, "validated"] = True

    return log


def _build_comparison(log: pd.DataFrame, today_str: str) -> list[dict]:
    """Constrói tabela de acurácia por família (semana anterior)."""
    validated = log[log["validated"] == True].copy()  # noqa: E712
    if validated.empty:
        return []

    results = []
    for family in _FAMILIES:
        fam_rows = validated[validated["family"] == family]
        if fam_rows.empty:
            continue
        acc = fam_rows["correct_d1"].astype(float).mean()
        results.append({
            "family":   family,
            "accuracy": round(float(acc), 3),
            "n":        len(fam_rows),
            "vs_acaso": round(float(acc) - 0.5, 3),
        })

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results


def _build_consensus(new_rows: list, portfolio_tickers: list,
                     research_weights: dict) -> list[dict]:
    """Constrói tabela de consenso ponderado pelo histórico de acertos de cada família."""
    w_d1   = research_weights.get("d1", {})
    consensus = []
    for ticker in portfolio_tickers:
        ticker_rows = [r for r in new_rows if r["ticker"] == ticker]
        if not ticker_rows:
            continue

        weight_up   = sum(w_d1.get(r["family"], 1.0)
                          for r in ticker_rows if r.get("direction_d1") == "up")
        weight_total= sum(w_d1.get(r["family"], 1.0) for r in ticker_rows)
        pct         = weight_up / max(weight_total, 1e-9)

        up_count = sum(1 for r in ticker_rows if r.get("direction_d1") == "up")
        consensus.append({
            "ticker":    ticker,
            "up_count":  up_count,
            "total":     len(ticker_rows),
            "pct_up":    round(pct, 2),
            "direction": "ALTA" if pct >= 0.5 else "BAIXA",
            "strength":  "forte" if abs(pct - 0.5) >= 0.25 else "fraco",
        })
    consensus.sort(key=lambda x: x["pct_up"], reverse=True)
    return consensus