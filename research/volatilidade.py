"""
Desafiantes de volatilidade ao vivo (etapa B do estudo de 18/11/2026).

Origem: scripts/estudo_volatilidade.py, walk-forward de ~2 anos na carteira.
  - Movimento grande amanhã (|retorno| > 1 ATR, ~20% dos dias): a regressão
    logística em 8 features de volatilidade teve BSS +1,4% contra a taxa de
    base, AUC 0,61, bem calibrada, venceu nas duas metades, mas p corrigido
    0,146. Entra como desafiante para confirmar com dados novos.
  - Volatilidade dos próximos 1, 3 e 5 pregões: a EWMA (λ=0,94) bateu a
    variância média histórica (QLIKE 5-10% menor, duas metades); GARCH e HAR
    não bateram a EWMA. Registrada para acompanhar se isso se mantém.

Cada dia, por ativo da carteira:
  p_grande        probabilidade de |r(t+1)| > ATR14/preço (logística)
  p_base          taxa de base do ativo no treino (a referência)
  atr_pct         o limiar usado
  var_ewma        variância diária prevista pela EWMA (vale para 1, 3 e 5 dias)
  var_const       variância média histórica (a referência da EWMA)
Conferência com os preços que chegam depois (do próprio DataFrame de features
das execuções seguintes): grande_real, rv1, rv3, rv5.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

VOL_LOG     = Path("output/predictions_volatilidade_log.csv")
K           = 1.0
LAMBDA_EWMA = 0.94
FEATS       = ["ATR14_pct", "vol_10d", "vol_ratio", "abs_ret_1d", "vix_level", "vix_change", "BB_width", "abs_ret_5d"]
_COLS = ["pred_date", "ticker", "ref_date", "p_grande", "p_base", "atr_pct", "var_ewma", "var_const",
         "grande_real", "rv1", "rv3", "rv5"]


def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    r = d["Close"].pct_change(fill_method=None)
    d["ret"] = r
    d["ATR14_pct"] = d["ATR14"] / d["Close"]
    d["abs_ret_1d"] = d["ret_1d"].abs()
    d["abs_ret_5d"] = d["ret_5d"].abs()
    d["grande"] = (r.shift(-1).abs() > K * d["ATR14_pct"]).astype(float)
    d.loc[r.shift(-1).isna(), "grande"] = np.nan
    return d


def ewma_var_ultima(ret: pd.Series, lam: float = LAMBDA_EWMA) -> float:
    r = ret.dropna().values
    if len(r) < 20:
        return float("nan")
    v = float(np.var(r[:20]))
    for x in r[1:]:
        v = lam * v + (1 - lam) * x * x
    return v


def prever_ativo(df: pd.DataFrame) -> dict | None:
    d = _preparar(df)
    ultima = d.iloc[[-1]]
    if ultima[FEATS].isna().any(axis=None):
        return None
    tr = d.dropna(subset=FEATS + ["grande"])
    if len(tr) < 200 or tr["grande"].nunique() < 2:
        return None
    sc = RobustScaler().fit(tr[FEATS].values)
    lg = LogisticRegression(max_iter=1000).fit(sc.transform(tr[FEATS].values), tr["grande"].astype(int))
    ret = d["ret"].dropna()
    return {
        "p_grande": float(lg.predict_proba(sc.transform(ultima[FEATS].values))[0, 1]),
        "p_base": float(tr["grande"].mean()),
        "atr_pct": float(ultima["ATR14_pct"].iloc[0]),
        "var_ewma": ewma_var_ultima(ret),
        "var_const": float((ret ** 2).mean()),
    }


def validar(log: pd.DataFrame, precos: dict[str, pd.Series]) -> pd.DataFrame:
    """Preenche o realizado quando os pregões seguintes à ref_date já existem."""
    if log.empty:
        return log
    for i, row in log.iterrows():
        if pd.notna(row.get("rv5")):
            continue
        c = precos.get(row["ticker"])
        if c is None:
            continue
        c = c.dropna()
        ref = pd.Timestamp(row["ref_date"])
        if ref not in c.index:
            continue
        pos = c.index.get_loc(ref)
        futuros = c.iloc[pos: pos + 6]
        r = futuros.pct_change(fill_method=None).dropna().values
        if len(r) >= 1 and pd.isna(row.get("grande_real")):
            log.at[i, "grande_real"] = float(abs(r[0]) > K * row["atr_pct"])
            log.at[i, "rv1"] = float(abs(r[0]))
        if len(r) >= 3 and pd.isna(row.get("rv3")):
            log.at[i, "rv3"] = float(np.sqrt(np.mean(r[:3] ** 2)))
        if len(r) >= 5:
            log.at[i, "rv5"] = float(np.sqrt(np.mean(r[:5] ** 2)))
    return log


def placar(log: pd.DataFrame) -> dict:
    out = {}
    v = log[log["grande_real"].notna()] if not log.empty else log
    if len(v):
        y = v["grande_real"].astype(float)
        br = float(((v["p_grande"] - y) ** 2).mean())
        bb = float(((v["p_base"] - y) ** 2).mean())
        faixas = pd.cut(v["p_grande"], [0, 0.1, 0.2, 0.3, 0.4, 1.0001], right=False)
        conf = [{"faixa": f"{iv.left:.0%}-{min(iv.right, 1):.0%}", "n": int(len(g)),
                 "prob_media": float(v.loc[g.index, "p_grande"].mean()), "real": float(g.mean())}
                for iv, g in y.groupby(faixas, observed=True)]
        out["grande"] = {"n": len(v), "n_dias": int(v["ref_date"].nunique()), "taxa_real": float(y.mean()),
                         "brier": br, "brier_base": bb, "bss": (1 - br / bb) if bb > 0 else None,
                         "confiabilidade": conf}
    for n in [1, 3, 5]:
        col = f"rv{n}"
        g = log[log[col].notna() & (log[col] > 0)] if not log.empty and col in log else log.iloc[0:0]
        if len(g):
            real = g[col].astype(float) ** 2
            ql = lambda f: (real / f - np.log(real / f) - 1).mean()  # noqa: E731
            out[f"vol{n}"] = {"n": len(g), "n_dias": int(g["ref_date"].nunique()),
                              "qlike_ewma": float(ql(g["var_ewma"].clip(lower=1e-12))),
                              "qlike_const": float(ql(g["var_const"].clip(lower=1e-12)))}
    return out


def run_volatilidade(resultados_ml: dict, carteira: list, pred_date: str) -> dict:
    log = pd.read_csv(VOL_LOG, low_memory=False) if VOL_LOG.exists() else pd.DataFrame(columns=_COLS)
    precos = {t: resultados_ml[t]["df"]["Close"] for t in carteira
              if t in resultados_ml and "df" in resultados_ml[t]}
    log = validar(log, precos)
    novos, hoje = [], {}
    for t in carteira:
        res = resultados_ml.get(t) or {}
        if "df" not in res or "last_date" not in res:
            continue
        ref = str(pd.Timestamp(res["last_date"]).date())
        ja = (not log.empty) and ((log["ticker"] == t) & (log["ref_date"].astype(str) == ref)).any()
        p = prever_ativo(res["df"])
        if p is None:
            continue
        hoje[t] = p
        if not ja:
            novos.append({"pred_date": pred_date, "ticker": t, "ref_date": ref, **{k: round(v, 6) for k, v in p.items()},
                          "grande_real": np.nan, "rv1": np.nan, "rv3": np.nan, "rv5": np.nan})
    if novos:
        dn = pd.DataFrame(novos, columns=_COLS)
        log = dn if log.empty else pd.concat([log, dn], ignore_index=True)
    VOL_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(VOL_LOG, index=False)
    logger.info("Desafiantes de volatilidade: %d novos, %d no log", len(novos), len(log))
    return {"hoje": hoje, "placar": placar(log)}
