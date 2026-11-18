"""
Etapa A: o tamanho do movimento é previsível, onde a direção não é?

Dois alvos, walk-forward causal de ~2 anos nos ativos da carteira (mesma
montagem dos outros estudos: treino só com o passado, retreino a cada RETREINO
dias, intervalo = horizonte entre treino e teste):

1. MOVIMENTO GRANDE (binário): |retorno do próximo pregão| > K × ATR14/preço.
   Modelos:
     base     taxa de base do ativo até a véspera (a referência a vencer)
     ewma     probabilidade implícita de uma normal com σ = EWMA (λ=0,94)
     logit    regressão logística em features de volatilidade
     trio     RF + GB + SGD (como a produção) nas 33 features
   Métricas: Brier e BSS CONTRA A TAXA DE BASE (não contra 0,25), log loss,
   AUC, confiabilidade por faixa. DM (HLN) na média diária da diferença de
   Brier contra a base, unilateral; vitória nas duas metades; BH.

2. VOLATILIDADE REALIZADA nos próximos N pregões (N = 1, 3, 5):
   RV_N = sqrt(média dos retornos² de t+1 a t+N).
   Modelos:
     ewma     σ EWMA (λ=0,94), a referência a vencer
     garch    GARCH(1,1) gaussiano por máxima verossimilhança, previsão N passos
     har      HAR-RV (Corsi 2009): log RV ~ log RV de 1, 5 e 22 dias
   Métricas: QLIKE (perda padrão para volatilidade, robusta a ruído no realizado)
   e MSE da variância. DM (HLN, h=N) na média diária da diferença de QLIKE
   contra o EWMA, unilateral; vitória nas duas metades; BH.

Aprovação (candidato ao desafiante ao vivo): p corrigido < 0,05 E duas metades.

Uso: python scripts/estudo_volatilidade.py   (≈5-15 min)
Saídas (não vão para o git):
  output/estudo_volatilidade.json            resumo
  output/estudo_volatilidade_previsoes.csv   previsões linha a linha, para análise
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize, stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (N_ESTIMATORS_RF, MAX_DEPTH_RF, N_ESTIMATORS_GB, MAX_DEPTH_GB,  # noqa: E402
                             LEARNING_RATE_GB, SGD_ALPHA, SGD_MAX_ITER)
from data.storage import load_portfolio_config, portfolio_tickers  # noqa: E402
from evaluation.significance import apply_bh_correction  # noqa: E402
from evaluation.statistical_tests import dm_from_loss_diff  # noqa: E402
from features.engineering import build_all_features, FEATURE_COLS  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("volatilidade")

PERIODO, INICIO, RETREINO = "5y", 500, 63
K = 1.0                      # movimento grande = |retorno| > K × ATR14 (taxa de base perto de 20-30%)
LAMBDA_EWMA = 0.94
HORIZONTES_VOL = [1, 3, 5]
FEATS_VOL = ["ATR14_pct", "vol_10d", "vol_ratio", "abs_ret_1d", "vix_level", "vix_change", "BB_width", "abs_ret_5d"]
OUT_JSON = Path("output/estudo_volatilidade.json")
OUT_CSV = Path("output/estudo_volatilidade_previsoes.csv")


def _baixar(tickers):
    out = {}
    for t in tickers:
        d = yf.download(t, period=PERIODO, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[t] = d.dropna(subset=["Close"]).sort_index()
    return out


# ── modelos de volatilidade ──────────────────────────────────────────────

def ewma_var(r: np.ndarray, lam: float = LAMBDA_EWMA) -> np.ndarray:
    """Variância EWMA condicional: v[t] usa retornos até t (previsão para t+1)."""
    v = np.empty(len(r))
    v[0] = np.var(r[:20]) if len(r) >= 20 else r[0] ** 2
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t] ** 2
    return v


def garch_fit(r: np.ndarray) -> tuple[float, float, float]:
    """GARCH(1,1) gaussiano por máxima verossimilhança; devolve (ω, α, β)."""
    r = r - r.mean()
    var0 = np.var(r)

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = np.empty(len(r)); h[0] = var0
        for t in range(1, len(r)):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    res = optimize.minimize(nll, x0=[var0 * 0.05, 0.08, 0.9], method="Nelder-Mead",
                            options={"maxiter": 600, "xatol": 1e-8, "fatol": 1e-6})
    return tuple(res.x)


def garch_forecast(r_hist: np.ndarray, params, r_test: np.ndarray, n: int) -> np.ndarray:
    """Para cada dia de teste, variância média prevista nos próximos n pregões (usa só o passado)."""
    w, a, b = params
    mu = r_hist.mean()
    h = np.var(r_hist - mu)
    for x in (r_hist - mu)[1:]:
        h = w + a * x ** 2 + b * h
    longo = w / max(1 - a - b, 1e-6)
    out = []
    serie = r_test - mu
    for t in range(len(serie)):
        h1 = w + a * serie[t] ** 2 + b * h         # variância para t+1 com o retorno de t
        passos = [longo + (a + b) ** k * (h1 - longo) for k in range(n)]
        out.append(np.mean(passos))
        h = h1
    return np.array(out)


def _rv_futura(r: pd.Series, n: int) -> pd.Series:
    """RV dos próximos n pregões, alinhada ao dia t (retornos t+1..t+n)."""
    sq = r ** 2
    return np.sqrt(sum(sq.shift(-k) for k in range(1, n + 1)) / n)


# ── avaliação ────────────────────────────────────────────────────────────

def _teste_dm(perda_modelo: pd.Series, perda_ref: pd.Series, datas: pd.Series, h: int) -> dict:
    diff = (perda_modelo - perda_ref).groupby(datas.values).mean().sort_index()
    if len(diff) < 30:
        return {"p": 1.0, "dm": 0.0, "duas_metades": [False, False], "n_dias": len(diff)}
    dm = dm_from_loss_diff(diff.values, h=h)
    p = float(stats.t.cdf(dm["statistic"], dm["df"]))          # H1: modelo com perda menor
    meio = len(diff) // 2
    return {"p": round(p, 4), "dm": round(dm["statistic"], 3), "n_dias": len(diff),
            "duas_metades": [bool(diff.iloc[:meio].mean() < 0), bool(diff.iloc[meio:].mean() < 0)]}


def _confiabilidade(p: pd.Series, y: pd.Series) -> list[dict]:
    faixas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0001]
    g = pd.cut(p, faixas, right=False)
    out = []
    for iv, sub in y.groupby(g, observed=True):
        out.append({"faixa": f"{iv.left:.0%}-{min(iv.right, 1):.0%}", "n": int(len(sub)),
                    "prob_media": round(float(p[sub.index].mean()), 4), "frequencia_real": round(float(sub.mean()), 4)})
    return out


def main():
    carteira = portfolio_tickers(load_portfolio_config())
    log.info("Baixando dados (%s)...", PERIODO)
    raw = _baixar(carteira)
    c = _baixar(["^VIX", "SPY", "BTC-USD", "GLD"])
    ctx = {"vix": c["^VIX"]["Close"], "spy": c["SPY"]["Close"], "btc": c["BTC-USD"]["Close"], "gold": c["GLD"]["Close"]}
    feats = build_all_features(raw, ctx, 60)

    linhas_bin, linhas_vol = [], []
    for t, df in feats.items():
        d = df.copy()
        r = d["Close"].pct_change(fill_method=None)
        d["ret"] = r
        d["ATR14_pct"] = d["ATR14"] / d["Close"]
        d["abs_ret_1d"] = d["ret_1d"].abs()
        d["abs_ret_5d"] = d["ret_5d"].abs()
        d["grande"] = (r.shift(-1).abs() > K * d["ATR14_pct"]).astype(float)
        d.loc[r.shift(-1).isna(), "grande"] = np.nan
        for n in HORIZONTES_VOL:
            d[f"rv{n}"] = _rv_futura(r, n)
        d["rv_1"] = r.abs()
        d["rv_5"] = np.sqrt((r ** 2).rolling(5).mean())
        d["rv_22"] = np.sqrt((r ** 2).rolling(22).mean())
        d = d.dropna(subset=FEATURE_COLS + FEATS_VOL + ["ret", "rv_22"])
        log.info("%s: %d linhas", t, len(d))

        ev = ewma_var(d["ret"].values)
        d["ewma_var"] = ev

        for ini in range(INICIO, len(d), RETREINO):
            teste = d.iloc[ini: ini + RETREINO]
            if teste.empty:
                continue
            # ── binário: treino só com rótulos já conhecidos antes do teste
            tr = d.iloc[: ini - 1]
            tr = tr[tr["grande"].notna()]
            if len(tr) < 200 or tr["grande"].nunique() < 2:
                continue
            base = float(tr["grande"].mean())
            sigma = np.sqrt(teste["ewma_var"].values)
            p_ewma = 2 * (1 - stats.norm.cdf(K * teste["ATR14_pct"].values / np.maximum(sigma, 1e-9)))
            sc = RobustScaler().fit(tr[FEATS_VOL].values)
            lg = LogisticRegression(max_iter=1000).fit(sc.transform(tr[FEATS_VOL].values), tr["grande"].astype(int))
            p_logit = lg.predict_proba(sc.transform(teste[FEATS_VOL].values))[:, 1]
            sc2 = RobustScaler().fit(tr[FEATURE_COLS].values)
            a, b = sc2.transform(tr[FEATURE_COLS].values), sc2.transform(teste[FEATURE_COLS].values)
            y = tr["grande"].astype(int).values
            rf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF, random_state=42, n_jobs=-1).fit(a, y)
            gb = GradientBoostingClassifier(n_estimators=N_ESTIMATORS_GB, max_depth=MAX_DEPTH_GB,
                                            learning_rate=LEARNING_RATE_GB, random_state=42).fit(a, y)
            sgd = SGDClassifier(loss="log_loss", alpha=SGD_ALPHA, max_iter=SGD_MAX_ITER, random_state=42).fit(a, y)
            p_trio = (rf.predict_proba(b)[:, 1] + gb.predict_proba(b)[:, 1] + sgd.predict_proba(b)[:, 1]) / 3
            for i, dt in enumerate(teste.index):
                if pd.isna(teste["grande"].iloc[i]):
                    continue
                linhas_bin.append({"ticker": t, "data": dt, "realizado": teste["grande"].iloc[i],
                                   "p_base": base, "p_ewma": p_ewma[i], "p_logit": p_logit[i], "p_trio": p_trio[i],
                                   "vix": teste["vix_level"].iloc[i]})

            # ── volatilidade: GARCH ajustado só com o passado; HAR idem
            r_hist = d["ret"].iloc[:ini].values
            params = garch_fit(r_hist[-750:])
            har_tr = d.iloc[:ini]
            for n in HORIZONTES_VOL:
                g_var = garch_forecast(r_hist[-750:], params, teste["ret"].values, n)
                alvo = f"rv{n}"
                tr_h = har_tr.iloc[: max(len(har_tr) - n, 0)].dropna(subset=[alvo])
                Xh = np.log(tr_h[["rv_1", "rv_5", "rv_22"]].clip(lower=1e-6).values)
                lr = LinearRegression().fit(Xh, np.log(tr_h[alvo].clip(lower=1e-6).values))
                resid_var = float(np.var(np.log(tr_h[alvo].clip(lower=1e-6).values) - lr.predict(Xh)))
                har_sigma = np.exp(lr.predict(np.log(teste[["rv_1", "rv_5", "rv_22"]].clip(lower=1e-6).values)) + resid_var / 2)
                for i, dt in enumerate(teste.index):
                    real = teste[alvo].iloc[i]
                    if pd.isna(real) or real <= 0:
                        continue
                    linhas_vol.append({"ticker": t, "data": dt, "n": n, "rv_real": real,
                                       "var_ewma": teste["ewma_var"].iloc[i], "var_garch": g_var[i],
                                       "var_har": har_sigma[i] ** 2})

    rb, rv = pd.DataFrame(linhas_bin), pd.DataFrame(linhas_vol)
    rb.assign(tipo="movimento_grande").to_csv(OUT_CSV, index=False)
    rv.to_csv(OUT_CSV.with_name("estudo_volatilidade_previsoes_rv.csv"), index=False)

    resultado = {"K_atr": K, "periodo_teste": [str(rb["data"].min().date()), str(rb["data"].max().date())],
                 "movimento_grande": {}, "volatilidade": {}}
    testes = []

    # ── binário
    y = rb["realizado"]
    br_base = (rb["p_base"] - y) ** 2
    resultado["movimento_grande"]["taxa_real"] = round(float(y.mean()), 4)
    resultado["movimento_grande"]["n"] = len(rb)
    for m in ["base", "ewma", "logit", "trio"]:
        p = rb[f"p_{m}"].clip(1e-4, 1 - 1e-4)
        br = (p - y) ** 2
        item = {"brier": round(float(br.mean()), 5),
                "bss_vs_base": round(1 - float(br.mean()) / float(br_base.mean()), 4),
                "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 5),
                "auc": round(float(roc_auc_score(y, p)), 4) if m != "base" or rb["p_base"].nunique() > 1 else 0.5,
                "confiabilidade": _confiabilidade(p, y)}
        if m != "base":
            item.update(_teste_dm(br, br_base, rb["data"], 1))
            testes.append(("movimento_grande", m, item))
        resultado["movimento_grande"][m] = item

    # ── volatilidade
    for n in HORIZONTES_VOL:
        g = rv[rv["n"] == n]
        real_var = g["rv_real"] ** 2
        bloco = {"n": len(g)}
        ql = {}
        for m in ["ewma", "garch", "har"]:
            f = g[f"var_{m}"].clip(lower=1e-10)
            ql[m] = real_var / f - np.log(real_var / f) - 1
            bloco[m] = {"qlike": round(float(ql[m].mean()), 5), "mse_var": float(((real_var - f) ** 2).mean())}
        for m in ["garch", "har"]:
            bloco[m].update(_teste_dm(ql[m], ql["ewma"], g["data"], n))
            testes.append(("volatilidade", f"{m}_n{n}", bloco[m]))
        resultado["volatilidade"][f"N={n}"] = bloco

    itens = [x[2] for x in testes]
    apply_bh_correction(itens)
    for _, _, it in testes:
        it["p_corrigido"] = round(it.get("p_adj", it["p"]), 4)
        it["aprovado"] = bool(it["p_corrigido"] < 0.05 and all(it["duas_metades"]))

    OUT_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
