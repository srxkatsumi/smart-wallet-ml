"""
Ensemble ampliado: produção (RF+GB+SGD) + as 13 famílias votando juntas.

É um DESAFIANTE: é registrado e conferido todos os dias, mas não substitui as
previsões de produção do email. Só pode virar a previsão principal depois de
passar pelo Jinshi contra a produção, com pelo menos MIN_DIAS dias alinhados.

Pesos com encolhimento (shrinkage): a acurácia de cada membro é puxada para
50% por um prior equivalente a PRIOR_N previsões,

    acc_enc = (acertos + PRIOR_N * 0.5) / (n + PRIOR_N)
    peso    = clip(1 + LAMBDA * (acc_enc - 0.5) / 0.5, PESO_MIN, PESO_MAX)

Com poucos dados todos pesam ~1 (voto igual). Um membro com 60% de acerto
sustentado em muitas previsões chega a peso ~2; um com 40% cai para ~0,25.
Assim 5 dias bons não bastam para um modelo dominar o voto.

Conferência: reaproveita a validação da produção (mesmo ticker, ref_date e
horizonte), que já usa o fechamento oficial; o acerto é a direção do ampliado
contra o sinal de actual_change_pct.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.storage import aligned_rows

logger = logging.getLogger(__name__)

AMPLIADO_LOG  = Path("output/predictions_ampliado_log.csv")
PRODUCAO      = "producao"
PRIOR_N       = 200    # ~25 dias × 8 ativos: abaixo disso o prior domina
LAMBDA        = 5.0
PESO_MIN      = 0.25
PESO_MAX      = 2.0
JANELA_DIAS   = 60     # últimas N datas de referência validadas por membro

_COLS = ["pred_date", "ticker", "ref_date", "horizon", "direction", "prob_up",
         "n_membros", "prob_up_producao", "prob_up_familias",
         "actual_change_pct", "correct"]


def peso_encolhido(acertos: float, n: int) -> float:
    acc = (acertos + PRIOR_N * 0.5) / (n + PRIOR_N)
    return float(np.clip(1 + LAMBDA * (acc - 0.5) / 0.5, PESO_MIN, PESO_MAX))


def _prob_up(direction, confidence) -> float | None:
    if pd.isna(confidence) or direction not in ("up", "down"):
        return None
    c = float(confidence)
    return c if direction == "up" else 1 - c


def _janela(df: pd.DataFrame) -> pd.DataFrame:
    datas = sorted(df["ref_date"].astype(str).unique())[-JANELA_DIAS:]
    return df[df["ref_date"].astype(str).isin(datas)]


def compute_member_weights(prod_log: pd.DataFrame, research_log: pd.DataFrame,
                           tickers: list) -> dict:
    """{ "d1": {"producao": w, "<familia>": w, ...}, ... } + n por membro, para o email."""
    out = {}
    prod = aligned_rows(prod_log[prod_log["ticker"].isin(tickers) & prod_log["correct"].notna()])
    res = research_log[research_log["ticker"].isin(tickers)] if not research_log.empty else research_log
    if not res.empty:
        res = res[res["ref_date"].notna()]
    for h in [1, 2, 3]:
        pesos, ns = {}, {}
        p = _janela(prod[prod["horizon"] == h]) if not prod.empty else prod
        pesos[PRODUCAO] = peso_encolhido(float(p["correct"].astype(float).sum()), len(p))
        ns[PRODUCAO] = len(p)
        col = f"correct_d{h}"
        if not res.empty and col in res.columns:
            for fam, g in res[res[col].notna()].groupby("family"):
                g = _janela(g)
                pesos[fam] = peso_encolhido(float(g[col].astype(float).sum()), len(g))
                ns[fam] = len(g)
        out[f"d{h}"] = {"pesos": {k: round(v, 3) for k, v in pesos.items()}, "n": ns}
    return out


def build_ampliado(prod_log: pd.DataFrame, research_log: pd.DataFrame, ref_dates: dict,
                   weights: dict, pred_date: str) -> list[dict]:
    """Uma linha por (ticker, horizonte) para a ref_date de hoje de cada ticker."""
    rows = []
    for ticker, ref_date in ref_dates.items():
        p_today = prod_log[(prod_log["ticker"] == ticker) & (prod_log["ref_date"].astype(str) == ref_date)]
        r_today = research_log[(research_log["ticker"] == ticker) &
                               (research_log["ref_date"].astype(str) == ref_date)] if not research_log.empty else research_log
        for h in [1, 2, 3]:
            w = weights.get(f"d{h}", {}).get("pesos", {})
            votos = []  # (peso, prob_up, é_produção)
            ph = p_today[p_today["horizon"] == h]
            p_prod = _prob_up(ph.iloc[-1]["direction"], ph.iloc[-1]["confidence"]) if not ph.empty else None
            if p_prod is not None:
                votos.append((w.get(PRODUCAO, 1.0), p_prod, True))
            col = f"prob_d{h}"
            if not r_today.empty and col in r_today.columns:
                for _, r in r_today.drop_duplicates("family", keep="last").iterrows():
                    if pd.notna(r[col]):
                        votos.append((w.get(r["family"], 1.0), float(r[col]), False))
            if p_prod is None or len(votos) < 2:
                continue  # sem produção ou sem nenhuma família, não há o que ampliar
            tot = sum(v[0] for v in votos)
            prob = sum(v[0] * v[1] for v in votos) / tot
            fam = [v for v in votos if not v[2]]
            prob_fam = sum(v[0] * v[1] for v in fam) / sum(v[0] for v in fam)
            rows.append({
                "pred_date": pred_date, "ticker": ticker, "ref_date": ref_date, "horizon": h,
                "direction": "up" if prob >= 0.5 else "down",
                "prob_up": round(prob, 4), "n_membros": len(votos),
                "prob_up_producao": round(p_prod, 4), "prob_up_familias": round(prob_fam, 4),
                "actual_change_pct": np.nan, "correct": np.nan,
            })
    return rows


def validate_ampliado(log: pd.DataFrame, prod_log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return log
    pend = log["correct"].isna()
    if not pend.any():
        return log
    val = prod_log[prod_log["actual_change_pct"].notna() & prod_log["ref_date"].notna()][
        ["ticker", "ref_date", "horizon", "actual_change_pct"]].copy()
    val["ref_date"] = val["ref_date"].astype(str)
    val = val.drop_duplicates(["ticker", "ref_date", "horizon"], keep="last")
    key = log.loc[pend, ["ticker", "ref_date", "horizon"]].astype({"ref_date": str}).reset_index()
    m = key.merge(val, on=["ticker", "ref_date", "horizon"], how="inner")
    for _, r in m.iterrows():
        chg = float(r["actual_change_pct"])
        d = log.at[r["index"], "direction"]
        log.at[r["index"], "actual_change_pct"] = chg
        log.at[r["index"], "correct"] = float((d == "up" and chg >= 0) or (d == "down" and chg <= 0))
    return log


def scoreboard(log: pd.DataFrame, prod_log: pd.DataFrame) -> dict:
    """Ampliado vs. produção nas MESMAS previsões (ticker, ref_date, horizonte) já conferidas."""
    out = {}
    if log.empty:
        return out
    v = log[log["correct"].notna()].copy()
    v["ref_date"] = v["ref_date"].astype(str)
    p = prod_log[prod_log["correct"].notna() & prod_log["ref_date"].notna()][
        ["ticker", "ref_date", "horizon", "correct"]].copy()
    p["ref_date"] = p["ref_date"].astype(str)
    p = p.drop_duplicates(["ticker", "ref_date", "horizon"], keep="last")
    m = v.merge(p, on=["ticker", "ref_date", "horizon"], suffixes=("", "_prod"))
    for h in [1, 2, 3]:
        g = m[m["horizon"] == h]
        out[f"d{h}"] = {
            "n": len(g),
            "n_dias": int(g["ref_date"].nunique()),
            "acc_ampliado": round(float(g["correct"].astype(float).mean()), 4) if len(g) else None,
            "acc_producao": round(float(g["correct_prod"].astype(float).mean()), 4) if len(g) else None,
        }
    return out


def run_ampliado(prod_log: pd.DataFrame, tickers: list, ref_dates: dict, pred_date: str) -> dict:
    """Valida o histórico, registra o ampliado de hoje e devolve o necessário para o email."""
    research_log = (pd.read_csv("output/predictions_research_log.csv", low_memory=False)
                    if Path("output/predictions_research_log.csv").exists() else pd.DataFrame())
    log = pd.read_csv(AMPLIADO_LOG, low_memory=False) if AMPLIADO_LOG.exists() else pd.DataFrame(columns=_COLS)
    log = validate_ampliado(log, prod_log)

    weights = compute_member_weights(prod_log, research_log, tickers)
    novos = build_ampliado(prod_log, research_log, ref_dates, weights, pred_date)
    if novos:
        novos_df = pd.DataFrame(novos, columns=_COLS)
        log = novos_df if log.empty else pd.concat([log, novos_df], ignore_index=True)
        log["ref_date"] = log["ref_date"].astype(str)
        log = log.drop_duplicates(["ticker", "ref_date", "horizon"], keep="first")
    AMPLIADO_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(AMPLIADO_LOG, index=False)
    logger.info("Ensemble ampliado: %d previsões novas, %d no log", len(novos), len(log))

    hoje = {}
    for r in novos:
        hoje.setdefault(r["ticker"], {})[r["horizon"]] = r
    # Reexecução no mesmo dia: as linhas de hoje já estavam no log
    for ticker, ref in ref_dates.items():
        if ticker in hoje:
            continue
        sub = log[(log["ticker"] == ticker) & (log["ref_date"].astype(str) == ref)]
        for _, r in sub.iterrows():
            hoje.setdefault(ticker, {})[int(r["horizon"])] = r.to_dict()
    return {"hoje": hoje, "pesos": weights, "placar": scoreboard(log, prod_log)}
