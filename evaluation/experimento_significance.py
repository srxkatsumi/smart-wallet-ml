"""
OEC do experimento de regressão: teste de Diebold-Mariano (erro quadrático
pareado, desafiante vs. campeão) por horizonte, reaproveitando
evaluation/statistical_tests.py:diebold_mariano.

Fica em silêncio (ready=False) até MIN_N_EXPERIMENTO observações validadas
por horizonte — mesma convenção não-espiar de evaluation/significance.py.
"""
import json
import logging
import numpy as np
import pandas as pd

from evaluation.statistical_tests import diebold_mariano
from config.settings import EXPERIMENTO_SIGNIFICANCE_FILE, MIN_N_EXPERIMENTO

logger = logging.getLogger(__name__)


def compute_experimento_significance(df_log: pd.DataFrame) -> dict:
    validated = df_log[df_log["validated"] == True].copy()

    results: dict[str, dict] = {}
    for day in [1, 2, 3]:
        subset = validated[validated["horizon"] == day]
        n      = len(subset)
        key    = f"d{day}"

        if n < MIN_N_EXPERIMENTO:
            results[key] = {"ready": False, "n": n, "needed": MIN_N_EXPERIMENTO - n}
            continue

        actual_ret     = subset["actual_ret"].values.astype(float)
        champion_ret   = subset["champion_pred_ret"].values.astype(float)
        challenger_ret = subset["challenger_pred_ret"].values.astype(float)

        e_champion   = actual_ret - champion_ret
        e_challenger = actual_ret - challenger_ret

        mae_champion   = float(np.mean(np.abs(e_champion)))
        mae_challenger = float(np.mean(np.abs(e_challenger)))

        # e1=desafiante, e2=campeão: dm<0 favorece desafiante (guarda a mesma
        # convenção de evaluation/statistical_tests.py:diebold_mariano).
        # h=day: previsões D+2/D+3 usam janelas sobrepostas em dias úteis
        # consecutivos, o que autocorrelaciona os erros — h=1 (default)
        # subestimaria a variância de longo prazo e inflaria falsos
        # positivos justamente nos horizontes mais longos (Diebold & Mariano
        # 1995 recomendam h = horizonte de previsão).
        dm = diebold_mariano(e_challenger, e_champion, h=day)

        # Guardrail: acurácia direcional do sinal do retorno previsto
        dir_actual     = np.sign(actual_ret)
        dir_champion   = (np.sign(champion_ret) == dir_actual).mean()
        dir_challenger = (np.sign(challenger_ret) == dir_actual).mean()

        results[key] = {
            "ready":            True,
            "n":                n,
            "mae_champion":     round(mae_champion, 6),
            "mae_challenger":   round(mae_challenger, 6),
            "dm_statistic":     round(dm["statistic"], 4),
            "dm_p":             round(dm["p_value"], 4),
            "sig":              bool(dm["p_value"] < 0.05),
            "challenger_wins":  bool(dm["p_value"] < 0.05 and dm["statistic"] < 0),
            "dir_acc_champion":   round(float(dir_champion), 4),
            "dir_acc_challenger": round(float(dir_challenger), 4),
            "guardrail_mae_ok":   bool(mae_challenger <= mae_champion * 1.20),
        }
        logger.info(
            "[Experimento] Significância D+%d: MAE campeão=%.4f desafiante=%.4f DM p=%.4f",
            day, mae_champion, mae_challenger, dm["p_value"],
        )

    return results


def save_significance_experimentos(sig: dict) -> None:
    EXPERIMENTO_SIGNIFICANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTO_SIGNIFICANCE_FILE.write_text(
        json.dumps(sig, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_significance_experimentos() -> dict:
    if not EXPERIMENTO_SIGNIFICANCE_FILE.exists():
        return {}
    try:
        return json.loads(EXPERIMENTO_SIGNIFICANCE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
