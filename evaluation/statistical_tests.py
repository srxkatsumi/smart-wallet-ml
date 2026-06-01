"""
Fase 9 — Framework de Avaliação Estatística

Testes implementados:

Diebold-Mariano (Harvey, Leybourne & Newbold, 1997 — Journal of Business &
  Economic Statistics):
  Testa H0: dois modelos de previsão têm igual acurácia preditiva.
  Statistic DM = mean(d) / sqrt(var(d)/T), onde d_t = L(e1_t) - L(e2_t).
  Sob H0: DM ~ N(0, 1). p < 0.05 rejeita igualdade.

McNemar (McNemar, 1947 — Psychometrika):
  Testa H0: dois classificadores têm igual taxa de erro.
  Usa a tabela de contingência dos erros discordantes.
  Standard em comparação de classificadores binários.

Ljung-Box (Ljung & Box, 1978 — Biometrika):
  Testa H0: os resíduos não têm autocorrelação até lag k.
  p < 0.05 indica estrutura temporal nos resíduos — evidência de que o
  modelo não capturou toda a informação temporal disponível.

Métricas por domínio:
  Carteira: accuracy UP/DOWN, F1, AUC-ROC, Brier score
  Mega Sena: accuracy@k (acertos por sequência de 6), comparação vs baseline
  E-commerce (futuro): MAE, MAPE, RMSE, cobertura de intervalos
"""

import numpy as np
import warnings
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    brier_score_loss, precision_score, recall_score,
)

warnings.filterwarnings("ignore")


# ── Diebold-Mariano ───────────────────────────────────────────────────────

def diebold_mariano(e1: np.ndarray, e2: np.ndarray,
                    h: int = 1) -> dict:
    """
    Testa se dois modelos têm igual acurácia preditiva.

    Parameters
    ----------
    e1, e2 : erros de previsão dos dois modelos (arrays de igual comprimento)
    h      : horizonte de previsão (1 para D+1, 2 para D+2, etc.)

    Returns
    -------
    dict com: statistic, p_value, conclusion
    """
    d   = e1 ** 2 - e2 ** 2
    T   = len(d)
    mu  = d.mean()
    gamma_0 = np.var(d, ddof=1)

    # covariâncias de Harvey et al. (1997) para h > 1
    gamma = np.array([np.cov(d[:-j], d[j:])[0, 1] if j < T else 0
                      for j in range(1, h)])
    long_run_var = (gamma_0 + 2 * gamma.sum()) / T if h > 1 else gamma_0 / T

    if long_run_var <= 0:
        return {"statistic": 0.0, "p_value": 1.0,
                "conclusion": "Sem variância suficiente para o teste."}

    dm  = mu / np.sqrt(long_run_var)
    p   = 2 * (1 - stats.norm.cdf(abs(dm)))

    if p < 0.05:
        winner = "modelo 1" if dm < 0 else "modelo 2"
        conc = f"Diferença significativa (p={p:.4f}). {winner} é mais preciso."
    else:
        conc = f"Sem diferença significativa entre os modelos (p={p:.4f})."

    return {"statistic": float(dm), "p_value": float(p), "conclusion": conc}


# ── McNemar ───────────────────────────────────────────────────────────────

def mcnemar_test(correct1: np.ndarray, correct2: np.ndarray) -> dict:
    """
    Testa se dois classificadores têm igual taxa de erro.

    Parameters
    ----------
    correct1, correct2 : arrays booleanos (True = acertou)

    Returns
    -------
    dict com: n01, n10, statistic, p_value, conclusion
    """
    c1, c2 = correct1.astype(bool), correct2.astype(bool)
    n01 = (~c1 & c2).sum()   # modelo 1 errou, modelo 2 acertou
    n10 = (c1 & ~c2).sum()   # modelo 1 acertou, modelo 2 errou

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0,
                "conclusion": "Sem discordâncias entre os classificadores."}

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p    = 1 - stats.chi2.cdf(chi2, df=1)

    if p < 0.05:
        winner = "modelo 2" if n01 > n10 else "modelo 1"
        conc = f"Diferença significativa (p={p:.4f}). {winner} tem menor taxa de erro."
    else:
        conc = f"Sem diferença significativa na taxa de erro (p={p:.4f})."

    return {"n01": int(n01), "n10": int(n10),
            "statistic": float(chi2), "p_value": float(p), "conclusion": conc}


# ── Ljung-Box ─────────────────────────────────────────────────────────────

def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> dict:
    """
    Testa autocorrelação nos resíduos do modelo.

    Parameters
    ----------
    residuals : y_true - y_pred_proba (ou y_true - y_pred)
    lags      : número de lags a testar

    Returns
    -------
    dict com: statistic, p_value, has_autocorrelation, conclusion
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    stat   = float(result["lb_stat"].iloc[-1])
    p      = float(result["lb_pvalue"].iloc[-1])
    has_ac = p < 0.05

    if has_ac:
        conc = (f"Autocorrelação detectada nos resíduos até lag {lags} "
                f"(p={p:.4f}). O modelo não capturou toda a estrutura temporal.")
    else:
        conc = (f"Sem autocorrelação significativa nos resíduos até lag {lags} "
                f"(p={p:.4f}). Resíduos consistentes com ruído branco.")

    return {"statistic": stat, "p_value": p,
            "has_autocorrelation": has_ac, "conclusion": conc}


# ── Métricas por domínio ──────────────────────────────────────────────────

def metrics_carteira(y_true: np.ndarray, y_pred_proba: np.ndarray,
                     threshold: float = 0.5) -> dict:
    """Métricas para o domínio de acções (classificação binária UP/DOWN)."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_true = y_true.astype(int)

    result = {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1":              float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":       float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":          float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score":     float(brier_score_loss(y_true, y_pred_proba)),
        "baseline_random": 0.5,
    }
    try:
        result["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        result["auc_roc"] = None
    return result


def metrics_megasena(probs_60: np.ndarray, actual_balls: list,
                     top_k: int = 6) -> dict:
    """
    Métricas para o domínio da Mega Sena.

    Parameters
    ----------
    probs_60    : array de shape (60,) com P(aparece) para cada bola 1-60
    actual_balls: lista com as 6 bolas sorteadas (1-indexed)
    top_k       : quantas bolas o modelo "aposta" (default 6)
    """
    top_predicted = set(np.argsort(probs_60)[::-1][:top_k] + 1)
    actual_set    = set(actual_balls)
    matches       = len(top_predicted & actual_set)
    baseline      = top_k * top_k / 60   # E[acertos] aleatório = k²/60

    return {
        "matches":          matches,
        "accuracy_at_k":    matches / top_k,
        "baseline_random":  baseline / top_k,
        "above_baseline":   matches / top_k > baseline / top_k,
    }


def full_report(model_name: str, y_true: np.ndarray,
                y_pred_proba: np.ndarray,
                baseline_proba: np.ndarray | None = None) -> dict:
    """
    Relatório completo para um modelo no domínio da Carteira.
    Se baseline_proba for fornecido, corre também DM e McNemar.
    """
    metrics  = metrics_carteira(y_true, y_pred_proba)
    residuals = y_true.astype(float) - y_pred_proba
    lb        = ljung_box_test(residuals)

    report = {
        "model":    model_name,
        "metrics":  metrics,
        "ljung_box": lb,
    }

    if baseline_proba is not None:
        e_model    = (y_true - y_pred_proba) ** 2
        e_baseline = (y_true - baseline_proba) ** 2
        report["diebold_mariano"] = diebold_mariano(e_model, e_baseline)

        c_model    = (y_pred_proba >= 0.5).astype(bool) == y_true.astype(bool)
        c_baseline = (baseline_proba >= 0.5).astype(bool) == y_true.astype(bool)
        report["mcnemar"] = mcnemar_test(c_model, c_baseline)

    return report
