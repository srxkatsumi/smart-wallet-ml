"""
Onde a confiança do modelo tem vantagem?

Na média, a confiança não informa nada (λ = 0 no encolhimento, Brier pior que
dizer sempre 50%). Este estudo procura recortes em que ela funcione:
  - por ativo, por horizonte;
  - por regime do VIX (baixo < 15, médio 15-25, alto > 25);
  - por acordo entre RF, GB e SGD (os 3 na mesma direção ou não);
  - por faixa de confiança (a confiança alta acerta mais que a baixa?).

Dados: walk-forward causal de 2 anos nos ativos da carteira, mesmo trio da
produção (média simples, sem calibração), mesma montagem de
scripts/experimento_features_regime.py. O acordo com as 13 famílias não tem
histórico para backtest e fica para os dados ao vivo.

"Vantagem" num recorte exige as três coisas:
  1. acurácia acima da MELHOR regra constante do próprio recorte ("sempre ▲"
     ou "sempre ▼"), com o dia como unidade, p corrigido (BH por todos os
     recortes) < 0,05;
  2. vitória nas duas metades do período;
  3. pelo menos MIN_DIAS dias e MIN_N previsões.
Uma vantagem aqui é candidata ao Jinshi, não um sinal pronto.

Uso: python scripts/estudo_vantagem.py   (≈5-10 min)
Saída: output/estudo_vantagem.json (não vai para o git).
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (N_ESTIMATORS_RF, MAX_DEPTH_RF, N_ESTIMATORS_GB, MAX_DEPTH_GB,  # noqa: E402
                             LEARNING_RATE_GB, SGD_ALPHA, SGD_MAX_ITER)
from data.storage import load_portfolio_config, portfolio_tickers  # noqa: E402
from evaluation.significance import apply_bh_correction  # noqa: E402
from features.engineering import build_all_features, FEATURE_COLS  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("vantagem")

PERIODO, INICIO, RETREINO = "5y", 500, 63
MIN_N, MIN_DIAS = 150, 40
SAIDA = Path("output/estudo_vantagem.json")


def _baixar(tickers):
    out = {}
    for t in tickers:
        d = yf.download(t, period=PERIODO, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[t] = d.dropna(subset=["Close"]).sort_index()
    return out


def _tres(X_tr, y_tr, X_te):
    sc = RobustScaler().fit(X_tr)
    a, b = sc.transform(X_tr), sc.transform(X_te)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF, random_state=42, n_jobs=-1).fit(a, y_tr)
    gb = GradientBoostingClassifier(n_estimators=N_ESTIMATORS_GB, max_depth=MAX_DEPTH_GB,
                                    learning_rate=LEARNING_RATE_GB, random_state=42).fit(a, y_tr)
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=SGD_ALPHA, max_iter=SGD_MAX_ITER, random_state=42).fit(a, y_tr)
    return rf.predict_proba(b)[:, 1], gb.predict_proba(b)[:, 1], sgd.predict_proba(b)[:, 1]


def _walk_forward(df, h):
    tgt, fwd = f"target_d{h}", f"fwd_ret_d{h}"
    d = df.dropna(subset=FEATURE_COLS + [fwd])
    linhas = []
    for ini in range(INICIO, len(d), RETREINO):
        treino = d.iloc[: max(ini - h, 0)]
        treino = treino[treino[tgt].notna()]
        teste = d.iloc[ini: ini + RETREINO]
        if len(treino) < 200 or teste.empty or treino[tgt].nunique() < 2:
            continue
        p_rf, p_gb, p_sgd = _tres(treino[FEATURE_COLS].values, treino[tgt].values.astype(int), teste[FEATURE_COLS].values)
        for i, dt in enumerate(teste.index):
            linhas.append({"data": dt, "p_rf": p_rf[i], "p_gb": p_gb[i], "p_sgd": p_sgd[i],
                           "vix": float(teste["vix_level"].iloc[i]), "alta": float(teste[fwd].iloc[i] >= 0)})
    return pd.DataFrame(linhas)


def _avaliar(g: pd.DataFrame, nome: str) -> dict | None:
    if len(g) < MIN_N or g["data"].nunique() < MIN_DIAS:
        return None
    g = g.sort_values("data")
    acerto = g["acerto"].astype(float)
    taxa_alta = float(g["alta"].mean())
    constante = max(taxa_alta, 1 - taxa_alta)                     # melhor regra fixa do recorte
    acerto_const = g["alta"] if taxa_alta >= 0.5 else 1 - g["alta"]
    diff = (acerto - acerto_const).groupby(g["data"]).mean()       # dia como unidade
    n_d = len(diff)
    se = float(diff.std(ddof=1) / np.sqrt(n_d)) if n_d > 1 else 0.0
    t = float(diff.mean() / se) if se > 0 else 0.0
    p = float(stats.t.sf(t, n_d - 1)) if se > 0 else 1.0
    meio = n_d // 2
    metades = [bool(diff.iloc[:meio].mean() > 0), bool(diff.iloc[meio:].mean() > 0)]
    brier = float(((g["p_up"] - g["alta"]) ** 2).mean())
    return {"recorte": nome, "n": len(g), "n_dias": n_d, "acuracia": round(float(acerto.mean()), 4),
            "melhor_constante": round(constante, 4), "vantagem_pp": round((float(acerto.mean()) - constante) * 100, 2),
            "bss": round(1 - brier / 0.25, 3), "p": round(p, 4), "duas_metades": metades}


def main():
    carteira = portfolio_tickers(load_portfolio_config())
    log.info("Baixando dados (%s)...", PERIODO)
    raw = _baixar(carteira)
    c = _baixar(["^VIX", "SPY", "BTC-USD", "GLD"])
    ctx = {"vix": c["^VIX"]["Close"], "spy": c["SPY"]["Close"], "btc": c["BTC-USD"]["Close"], "gold": c["GLD"]["Close"]}
    feats = build_all_features(raw, ctx, 60)

    partes = []
    for t, df in feats.items():
        for h in [1, 2, 3]:
            log.info("%s D+%d", t, h)
            r = _walk_forward(df, h)
            r["ticker"], r["h"] = t, h
            partes.append(r)
    r = pd.concat(partes, ignore_index=True)
    r["p_up"] = (r["p_rf"] + r["p_gb"] + r["p_sgd"]) / 3
    r["conf"] = np.maximum(r["p_up"], 1 - r["p_up"])
    r["acerto"] = ((r["p_up"] >= 0.5) == (r["alta"] == 1)).astype(float)
    votos = (r[["p_rf", "p_gb", "p_sgd"]] >= 0.5).sum(axis=1)
    r["acordo"] = np.where(votos.isin([0, 3]), "os 3 concordam", "divergem")
    r["regime"] = pd.cut(r["vix"], [-np.inf, 15, 25, np.inf], labels=["VIX baixo", "VIX médio", "VIX alto"]).astype(str)
    r["faixa"] = pd.cut(r["conf"], [0.5, 0.55, 0.6, 0.7, 1.0001], right=False,
                        labels=["50-55%", "55-60%", "60-70%", "70%+"]).astype(str)

    recortes = []
    for h in [1, 2, 3]:
        g = r[r["h"] == h]
        recortes.append(_avaliar(g, f"D+{h} · todos"))
        for col in ["ticker", "regime", "acordo", "faixa"]:
            for val, gg in g.groupby(col):
                recortes.append(_avaliar(gg, f"D+{h} · {val}"))
        for (reg, ac), gg in g.groupby(["regime", "acordo"]):
            recortes.append(_avaliar(gg, f"D+{h} · {reg} · {ac}"))
    recortes = [x for x in recortes if x]
    apply_bh_correction(recortes)
    for x in recortes:
        x["p_corrigido"] = round(x.get("p_adj", x["p"]), 4)
        x["vantagem"] = bool(x["p_corrigido"] < 0.05 and all(x["duas_metades"]))

    # A confiança alta acerta mais que a baixa? (inclinação por horizonte)
    inclinacao = {}
    for h in [1, 2, 3]:
        g = r[r["h"] == h]
        rho, p_rho = stats.spearmanr(g["conf"], g["acerto"])
        inclinacao[f"D+{h}"] = {"spearman": round(float(rho), 4), "p": round(float(p_rho), 4),
                                "acc_por_faixa": g.groupby("faixa")["acerto"].agg(["mean", "size"]).round(4).to_dict("index")}

    saida = {"periodo_teste": [str(r["data"].min().date()), str(r["data"].max().date())],
             "n_total": len(r), "inclinacao_confianca": inclinacao,
             "recortes": sorted(recortes, key=lambda x: -x["vantagem_pp"])}
    SAIDA.write_text(json.dumps(saida, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({"inclinacao": inclinacao,
                      "com_vantagem": [x for x in recortes if x["vantagem"]],
                      "top10": sorted(recortes, key=lambda x: -x["vantagem_pp"])[:10]},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
