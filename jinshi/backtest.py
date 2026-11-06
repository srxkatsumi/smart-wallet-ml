"""
Fase 1 do Jinshi: valida se treinar o regressor que já existe
(models/regressor_experimento.py:train_regressor_horizon) com histórico bem
mais longo (até ~20 anos, via jinshi/data.py) realmente melhora algo sobre
o que já temos (2 anos, PRICE_PERIOD atual) — ANTES de construir qualquer
infraestrutura nova em produção. Backtest-only: nenhuma previsão viva,
nenhum email, nada publicado ainda.

Walk-forward causal: a cada RETRAIN_EVERY dias úteis, retreina do zero com
tudo disponível até aquele ponto (sem olhar o futuro) e usa esse modelo pra
prever só esse dia de teste. Comparado contra:
  - a mesma receita, mas só com a janela de ~2 anos mais recente disponível
    nesse ponto (replica o que a produção faz hoje)
  - baseline ingênuo: "o preço de daqui a N dias é o preço de hoje"
    (passeio aleatório sem deriva)

Métricas: MAE (ponto), cobertura do intervalo de 90% (deveria conter o
valor real ~90% das vezes), e Diebold-Mariano
(evaluation/statistical_tests.py, já usado em outras partes do projeto)
comparando histórico longo vs. curto, e longo vs. ingênuo.
"""
import logging

import numpy as np
import pandas as pd

from models.regressor_experimento import train_regressor_horizon
from evaluation.statistical_tests import diebold_mariano

logger = logging.getLogger(__name__)

SHORT_WINDOW = pd.DateOffset(years=2)  # calendário, não linhas: o BTC negocia 365 dias/ano
RETRAIN_EVERY     = 21    # dias úteis entre retreinos (~1 mês) — custo computacional
MIN_LONG_HISTORY  = 756   # só testa depois de ~3 anos acumulados, senão "longo" e
                          # "curto" seriam quase a mesma janela e a comparação não diria nada


def backtest_ticker(df: pd.DataFrame, ticker: str, horizon: int = 1) -> pd.DataFrame | None:
    """
    Walk-forward causal para um ticker: devolve um DataFrame, uma linha por
    dia de teste, com previsão dos três candidatos (longo, curto, ingênuo)
    e se o valor real caiu dentro do intervalo de 90% do modelo de histórico
    longo.
    """
    ret_col = f"fwd_ret_d{horizon}"
    valid = df.dropna(subset=[ret_col, "Close"])
    if len(valid) < MIN_LONG_HISTORY + RETRAIN_EVERY:
        logger.warning("%s: histórico insuficiente (%d linhas) — pulando", ticker, len(valid))
        return None

    rows = []
    test_idx = range(MIN_LONG_HISTORY, len(valid) - horizon, RETRAIN_EVERY)

    for i in test_idx:
        df_so_far  = valid.iloc[: i + 1]     # tudo até (e incluindo) o dia de teste — causal
        target_row = valid.iloc[i]
        close_now   = float(target_row["Close"])
        actual_ret  = float(target_row[ret_col])
        actual_price = close_now * (1 + actual_ret)

        long_res  = train_regressor_horizon(df_so_far, horizon, f"{ticker}-longo")
        short_df  = df_so_far[df_so_far.index > df_so_far.index[-1] - SHORT_WINDOW]
        short_res = train_regressor_horizon(short_df, horizon, f"{ticker}-curto")
        if long_res is None or short_res is None:
            continue

        rows.append({
            "date":         df_so_far.index[-1],
            "actual_price": actual_price,
            "long_pred":    long_res["pred_price"],
            "short_pred":   short_res["pred_price"],
            "naive_pred":   close_now,  # passeio aleatório: previsão = preço de hoje
            "long_lo":      long_res["interval_lo"],
            "long_hi":      long_res["interval_hi"],
            "long_covered": long_res["interval_lo"] <= actual_price <= long_res["interval_hi"],
        })

    return pd.DataFrame(rows) if rows else None


def summarize(results: pd.DataFrame, ticker: str) -> dict:
    err_long  = (results["actual_price"] - results["long_pred"]).values
    err_short = (results["actual_price"] - results["short_pred"]).values
    err_naive = (results["actual_price"] - results["naive_pred"]).values

    mae_long  = float(np.mean(np.abs(err_long)))
    mae_short = float(np.mean(np.abs(err_short)))
    mae_naive = float(np.mean(np.abs(err_naive)))

    dm_vs_short = diebold_mariano(err_long, err_short, loss="absolute")
    dm_vs_naive = diebold_mariano(err_long, err_naive, loss="absolute")
    coverage = float(results["long_covered"].mean())

    return {
        "ticker":              ticker,
        "n":                   len(results),
        "mae_long_20y":        round(mae_long, 4),
        "mae_short_2y":        round(mae_short, 4),
        "mae_naive":           round(mae_naive, 4),
        "long_beats_short":    bool(mae_long < mae_short),
        "dm_long_vs_short_p":  round(dm_vs_short["p_value"], 4),
        "long_beats_naive":    bool(mae_long < mae_naive),
        "dm_long_vs_naive_p":  round(dm_vs_naive["p_value"], 4),
        "interval_90_coverage": round(coverage, 4),
    }


def run(featured_data: dict, horizon: int = 1) -> list[dict]:
    summaries = []
    for ticker, df in featured_data.items():
        logger.info("Backtest %s D+%d (%d dias disponíveis)...", ticker, horizon, len(df))
        results = backtest_ticker(df, ticker, horizon)
        if results is None:
            continue
        summary = summarize(results, ticker)
        summaries.append(summary)
        logger.info(
            "%s: n=%d  MAE 20y=%.4f  MAE 2y=%.4f  MAE ingênuo=%.4f  cobertura90=%.1f%%",
            ticker, summary["n"], summary["mae_long_20y"], summary["mae_short_2y"],
            summary["mae_naive"], summary["interval_90_coverage"] * 100,
        )
    return summaries
