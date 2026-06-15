"""
Família: Drift Detection (fase 20 — Carteira apenas)
Modelos: ADWIN, Page-Hinkley

ADWIN (Adaptive Windowing, Bifet & Gavaldà 2007):
  Mantém uma janela adaptativa da série temporal. Para cada novo elemento,
  verifica candidatos de divisão com espaçamento logarítmico: se a diferença
  de médias entre a metade antiga e a metade recente exceder o limite de
  Hoeffding, a parte antiga é descartada e conta-se um evento de drift.
  Complexidade: O(n log n) por série.

Page-Hinkley (Page 1954):
  Teste sequencial para mudanças de média. Acumula desvios em relação à média
  de referência; dispara um alarme quando o valor acumulado cai significativa-
  mente abaixo do máximo histórico. Detecta tanto subidas como descidas (PHT+
  e PHT−). Após cada alarme, a referência é reiniciada no valor actual.

Integração com o ensemble:
  A família treina um Random Forest como base e usa ADWIN + Page-Hinkley
  para quantificar a instabilidade histórica da série (nº de eventos de drift
  no stream de targets e no stream de erros de treino). Um coeficiente de
  confiança drift_conf = exp(−0.3 × total_drifts) amorte as probabilidades
  em direcção a 0.5 quando há muito drift — sinalizando que o mercado está em
  regime instável e o modelo deve ser tratado com cautela.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

N_ESTIMATORS = 100
MAX_DEPTH    = 4

# ADWIN
ADWIN_DELTA  = 0.002   # confiança: P(falso alarme) ≤ delta

# Page-Hinkley
PH_DELTA     = 0.005   # tolerância para drift de média
PH_THRESHOLD = 10.0    # limiar de alarme (ajustado para séries binárias)

# Coeficiente de atenuação da confiança por evento de drift
DRIFT_DECAY  = 0.30


# ── ADWIN ─────────────────────────────────────────────────────────────────

def _adwin(series: np.ndarray, delta: float = ADWIN_DELTA) -> int:
    """
    Conta eventos de drift numa série com ADWIN.

    A janela cresce elemento a elemento. Em cada passo, candidatos de divisão
    com espaçamento geométrico (log n candidatos) são testados com o limite de
    Hoeffding com correcção de Bonferroni. Quando drift é detectado, a parte
    antiga da janela é descartada.
    """
    window: list[float] = []
    n_drifts = 0

    for x in series:
        window.append(float(x))
        n = len(window)
        if n < 8:
            continue

        w  = np.asarray(window, dtype=np.float64)
        cs = np.cumsum(w)

        # candidatos de divisão espaçados geometricamente
        n_cands  = min(30, n - 3)
        raw      = np.geomspace(2, n - 2, n_cands)
        splits   = np.unique(np.round(raw).astype(int))
        splits   = splits[(splits >= 2) & (splits <= n - 2)]

        drift_at = -1
        for m in splits:
            n0 = int(m)
            n1 = n - n0
            mu0 = cs[n0 - 1] / n0
            mu1 = (cs[-1] - cs[n0 - 1]) / n1
            # limite de Hoeffding com correcção de Bonferroni
            eps = np.sqrt(
                (0.5 / n0 + 0.5 / n1)
                * np.log(2.0 * n * max(1.0, np.log2(n + 1)) / delta)
            )
            if abs(mu0 - mu1) > eps:
                drift_at = n0
                break

        if drift_at >= 0:
            window = window[drift_at:]
            n_drifts += 1

    return n_drifts


# ── Page-Hinkley ──────────────────────────────────────────────────────────

def _page_hinkley(
    series: np.ndarray,
    delta: float     = PH_DELTA,
    threshold: float = PH_THRESHOLD,
) -> int:
    """
    Conta alarmes de drift de média com Page-Hinkley (PH+ e PH−).

    PH+ detecta subidas, PH− detecta descidas. Após cada alarme, o acumulador
    e a média de referência são reiniciados no valor actual.
    """
    n_init = max(5, len(series) // 10)
    if len(series) <= n_init:
        return 0

    mu   = float(np.mean(series[:n_init]))
    mt_p = mt_n = 0.0   # acumuladores PH+ e PH−
    Mt_p = Mt_n = 0.0   # máximos correntes
    n_alarms = 0

    for x in series[n_init:]:
        mt_p += x - mu - delta
        Mt_p  = max(Mt_p, mt_p)
        mt_n += mu - x - delta
        Mt_n  = max(Mt_n, mt_n)

        if (Mt_p - mt_p > threshold) or (Mt_n - mt_n > threshold):
            n_alarms += 1
            mt_p = mt_n = 0.0
            Mt_p = Mt_n = 0.0
            mu   = float(x)   # reiniciar referência

    return n_alarms


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X).astype(np.float32)
    y_int  = y.astype(int)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_sc, y_int)

    # Drift nos targets (regime de mercado)
    y_f        = y.astype(np.float64)
    n_adwin_y  = _adwin(y_f)
    n_ph_y     = _page_hinkley(y_f)

    # Drift nos erros de treino (degradação do modelo)
    errors      = (rf.predict(X_sc) != y_int).astype(np.float64)
    n_adwin_err = _adwin(errors)

    total_drifts = n_adwin_y + n_ph_y + n_adwin_err
    drift_conf   = float(np.exp(-DRIFT_DECAY * total_drifts))

    logger.debug(
        "drift: adwin_y=%d  ph_y=%d  adwin_err=%d  → total=%d  conf=%.3f",
        n_adwin_y, n_ph_y, n_adwin_err, total_drifts, drift_conf,
    )

    return {
        "scaler":      scaler,
        "rf":          rf,
        "y_mean":      float(y.mean()),
        "drift_conf":  drift_conf,
        "n_drifts":    total_drifts,
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc         = model_dict["scaler"]
    X_sc       = sc.transform(X).astype(np.float32)
    p_up       = model_dict["rf"].predict_proba(X_sc)[:, 1]
    drift_conf = model_dict["drift_conf"]

    # Amortece as probabilidades em direcção a 0.5 proporcional ao drift detectado
    return drift_conf * p_up + (1.0 - drift_conf) * 0.5
