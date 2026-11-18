"""
Desafiante de regime: o mesmo trio da produção (RF + GB + SGD, média simples)
com três features de regime a mais. Registrado e conferido todo dia, sem mudar
as previsões do email, até passar pelo Jinshi com dados que o backtest não viu.

Origem: scripts/experimento_features_regime.py (18/11/2026), walk-forward de
2 anos nos ativos da carteira. Melhorou o Brier nos três horizontes, mas só o
D+3 passou no critério completo (p corrigido < 0,05 e vitória nas duas metades),
e o modelo continuou abaixo de "sempre ▲". Por isso entra como desafiante.

Features (shift(1), a mesma regra das features de contexto atuais):
  vix_term    = VIX / VIX3M          (acima de 1: medo de curto prazo)
  juro10_chg5 = variação de 5 dias do juro de 10 anos dos EUA, em p.p.
  usd_ret5    = retorno de 5 dias do índice do dólar (DX-Y.NYB)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler

from config.settings import (N_ESTIMATORS_RF, MAX_DEPTH_RF, N_ESTIMATORS_GB, MAX_DEPTH_GB,
                             LEARNING_RATE_GB, SGD_ALPHA, SGD_MAX_ITER)
from features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

NOVAS      = ["vix_term", "juro10_chg5", "usd_ret5"]
REGIME_LOG = Path("output/predictions_regime_log.csv")
_TICKERS   = ["^VIX", "^VIX3M", "^TNX", "DX-Y.NYB"]
_COLS      = ["pred_date", "ticker", "ref_date", "horizon", "direction", "prob_up",
              "actual_change_pct", "correct"]


def baixar_contexto_regime(period: str = "2y") -> dict[str, pd.Series]:
    import yfinance as yf
    out = {}
    for t in _TICKERS:
        d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[t] = s.sort_index()
    return out


def features_regime(ctx: dict[str, pd.Series], idx: pd.DatetimeIndex) -> pd.DataFrame:
    vix, vix3m, tnx, dxy = ctx["^VIX"], ctx["^VIX3M"], ctx["^TNX"], ctx["DX-Y.NYB"]
    f = pd.DataFrame({
        "vix_term":    (vix / vix3m).shift(1),
        "juro10_chg5": (tnx - tnx.shift(5)).shift(1),
        "usd_ret5":    dxy.pct_change(5, fill_method=None).shift(1),
    })
    original = pd.DatetimeIndex(idx)
    norm = (original.tz_localize(None) if original.tz is not None else original).normalize()
    # Séries com calendários diferentes (o ^TNX às vezes não tem o dia de hoje): propaga o último valor conhecido.
    out = f.sort_index().ffill().reindex(norm, method="ffill")
    out.index = original          # mesmo índice do DataFrame de features, para o join
    return out


def trio(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """RF + GB + SGD com RobustScaler, média simples da probabilidade de alta."""
    sc = RobustScaler().fit(X_tr)
    a, b = sc.transform(X_tr), sc.transform(X_te)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
                                random_state=42, n_jobs=-1).fit(a, y_tr)
    gb = GradientBoostingClassifier(n_estimators=N_ESTIMATORS_GB, max_depth=MAX_DEPTH_GB,
                                    learning_rate=LEARNING_RATE_GB, random_state=42).fit(a, y_tr)
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=SGD_ALPHA,
                        max_iter=SGD_MAX_ITER, random_state=42).fit(a, y_tr)
    return (rf.predict_proba(b)[:, 1] + gb.predict_proba(b)[:, 1] + sgd.predict_proba(b)[:, 1]) / 3


def prever_hoje(df: pd.DataFrame, ctx: dict[str, pd.Series]) -> dict[int, float]:
    """{h: prob_up} para a última barra, treinando com todo o histórico rotulado."""
    cols = FEATURE_COLS + NOVAS
    d = df.join(features_regime(ctx, df.index))
    ultima = d[cols].iloc[[-1]]
    if ultima.isna().any(axis=None):
        return {}
    out = {}
    for h in [1, 2, 3]:
        tgt = f"target_d{h}"
        treino = d.dropna(subset=cols + [tgt])
        if len(treino) < 200 or treino[tgt].nunique() < 2:
            continue
        out[h] = float(trio(treino[cols].values, treino[tgt].values.astype(int), ultima.values)[0])
    return out


def run_regime(resultados_ml: dict, carteira: list, prod_log: pd.DataFrame, pred_date: str,
               ctx: dict[str, pd.Series] | None = None) -> dict:
    """Confere o histórico, registra as previsões de hoje e devolve o placar contra a produção."""
    from research.ampliado import validate_ampliado, scoreboard
    log = pd.read_csv(REGIME_LOG, low_memory=False) if REGIME_LOG.exists() else pd.DataFrame(columns=_COLS)
    log = validate_ampliado(log, prod_log)
    ctx = ctx or baixar_contexto_regime()
    novos = []
    for t in carteira:
        res = resultados_ml.get(t) or {}
        if "df" not in res or "last_date" not in res:
            continue
        ref = str(pd.Timestamp(res["last_date"]).date())
        if not log.empty and ((log["ticker"] == t) & (log["ref_date"].astype(str) == ref)).any():
            continue
        for h, p in prever_hoje(res["df"], ctx).items():
            novos.append({"pred_date": pred_date, "ticker": t, "ref_date": ref, "horizon": h,
                          "direction": "up" if p >= 0.5 else "down", "prob_up": round(p, 4),
                          "actual_change_pct": np.nan, "correct": np.nan})
    if novos:
        df_novos = pd.DataFrame(novos, columns=_COLS)
        log = df_novos if log.empty else pd.concat([log, df_novos], ignore_index=True)
    REGIME_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(REGIME_LOG, index=False)
    logger.info("Desafiante de regime: %d previsões novas, %d no log", len(novos), len(log))
    return {"placar": scoreboard(log, prod_log)}
