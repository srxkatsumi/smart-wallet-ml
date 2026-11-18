"""
Leituras honestas de acurácia para o email: intervalo de confiança, referência
"sempre ▲" e o histórico de cada faixa de confiança.

- IC por dia quando há pelo menos MIN_DIAS_IC dias: as previsões do mesmo dia
  reagem ao mesmo mercado, então o dia é a unidade (mesma lógica de
  evaluation/significance.py:daily_cluster_stats). Com menos dias, Wilson sobre
  as linhas, que é otimista (estreito demais) e por isso vem rotulado.
- Faixas de confiança: quando o modelo disse 70-80%, quantas vezes acertou de
  verdade. Usa a confiança BRUTA, que é a que o email mostra.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from data.storage import aligned_rows
from evaluation.significance import daily_cluster_stats

MIN_DIAS_IC   = 5
MIN_N_FAIXA   = 30
FAIXAS        = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centro - meia), min(1.0, centro + meia))


def ic_acuracia(correct: pd.Series, dates: pd.Series) -> dict:
    c = correct.astype(float)
    n = len(c)
    if n == 0:
        return {"acc": None, "n": 0, "n_dias": 0, "lo": None, "hi": None, "metodo": None}
    n_dias = int(pd.Series(dates.values).nunique())
    acc = float(c.mean())
    if n_dias >= MIN_DIAS_IC:
        s = daily_cluster_stats(c, dates)
        lo, hi, metodo = max(0.0, s["ci_lo"]), min(1.0, s["ci_hi"]), "por dia"
    else:
        lo, hi = wilson(int(c.sum()), n)
        metodo = "Wilson, otimista"
    return {"acc": acc, "n": n, "n_dias": n_dias, "lo": lo, "hi": hi, "metodo": metodo}


def p_binomial(k: int, n: int) -> float | None:
    """Bicaudal contra 50%. Sobre linhas, então otimista quando há muitos ativos por dia."""
    if n == 0:
        return None
    return float(stats.binomtest(k, n, 0.5).pvalue)


def acc_sempre_alta(rows: pd.DataFrame) -> float | None:
    """Quanto acertaria quem previsse sempre ▲ nas mesmas linhas (mesma regra do validador: >= 0)."""
    if rows.empty or "actual_change_pct" not in rows:
        return None
    chg = rows["actual_change_pct"].dropna().astype(float)
    return float((chg >= 0).mean()) if len(chg) else None


def faixas_confianca(df_log: pd.DataFrame) -> dict:
    """{h: [{"lo": .7, "hi": .8, "n": .., "acc": .., "ic": (lo, hi)}, ...]} sobre o universo alinhado."""
    v = aligned_rows(df_log[df_log["correct"].notna() & df_log["confidence"].notna()])
    out = {}
    for h in [1, 2, 3]:
        sub = v[v["horizon"] == h]
        conf = sub["confidence"].astype(float)
        linhas = []
        for lo, hi in FAIXAS:
            g = sub[(conf >= lo) & (conf < hi)]
            n = len(g)
            k = int(g["correct"].astype(float).sum())
            linhas.append({"lo": lo, "hi": min(hi, 1.0), "n": n,
                           "acc": (k / n) if n else None, "ic": wilson(k, n)})
        out[h] = linhas
    return out


def acerto_da_faixa(faixas: dict, h: int, conf: float) -> dict | None:
    for f in faixas.get(h, []):
        if f["lo"] <= conf < f["hi"] or (conf >= 1.0 and f["hi"] >= 1.0):
            return f if f["n"] >= MIN_N_FAIXA else None
    return None
