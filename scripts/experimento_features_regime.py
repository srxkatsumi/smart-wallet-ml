"""
Etapa 5: as features de regime melhoram a previsão de direção?

Candidatas (todas com shift(1), a mesma regra das features de contexto atuais):
  - vix_term:    VIX / VIX3M. Acima de 1 = curva invertida, medo de curto prazo.
  - juro10_chg5: variação em 5 dias do juro de 10 anos dos EUA (^TNX), em p.p.
  - usd_ret5:    retorno de 5 dias do índice do dólar (DX-Y.NYB).

Método: walk-forward causal nos ativos da carteira, 5 anos de dados. Treina com
tudo até o ponto (com intervalo = horizonte entre treino e teste, para o alvo
não vazar), retreina a cada RETREINO dias e prevê os dias seguintes. Mesmo trio
da produção (RF + GB + SGD, média simples das probabilidades, RobustScaler),
sem a calibração isotônica, que só reescala a probabilidade.

Comparação, por horizonte, nas MESMAS linhas:
  - acurácia (direção do retorno futuro, >= 0 conta como alta, igual ao validador);
  - Brier da probabilidade de alta;
  - Diebold-Mariano (com correção HLN) sobre a média DIÁRIA da diferença de
    Brier, unilateral: o novo erra menos?
  - robustez: o novo vence nas duas metades do período?
Veredito no espírito do Jinshi: aprova só com p corrigido (BH pelos 3
horizontes) < 0,05 E vitória nas duas metades. O critério de custo do Jinshi
não se aplica direto a uma probabilidade de direção.

Uso: python scripts/experimento_features_regime.py   (≈10-20 min)
Saída: output/experimento_features_regime.json (não vai para o git).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.storage import load_portfolio_config, portfolio_tickers  # noqa: E402
from evaluation.significance import apply_bh_correction  # noqa: E402
from evaluation.statistical_tests import dm_from_loss_diff  # noqa: E402
from features.engineering import build_all_features, FEATURE_COLS  # noqa: E402
from research.regime import NOVAS, features_regime, trio  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("etapa5")

PERIODO   = "5y"
INICIO    = 500     # linhas mínimas de treino antes do primeiro teste (~2 anos)
RETREINO  = 63      # dias entre retreinos (~trimestral)
SAIDA     = Path("output/experimento_features_regime.json")


def _baixar(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        d = yf.download(t, period=PERIODO, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[t] = d.dropna(subset=["Close"]).sort_index()
    return out


def _regime(ctx_extra: dict, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return features_regime({k: v["Close"] for k, v in ctx_extra.items()}, idx)


def _walk_forward(df: pd.DataFrame, h: int, cols: list[str]) -> pd.Series:
    tgt, fwd = f"target_d{h}", f"fwd_ret_d{h}"
    d = df.dropna(subset=cols + [fwd])
    probs = pd.Series(np.nan, index=d.index)
    for ini in range(INICIO, len(d), RETREINO):
        treino = d.iloc[: max(ini - h, 0)]            # intervalo = horizonte: o alvo do treino não toca o teste
        treino = treino[treino[tgt].notna()]
        teste = d.iloc[ini: ini + RETREINO]
        if len(treino) < 200 or teste.empty or treino[tgt].nunique() < 2:
            continue
        probs.loc[teste.index] = trio(treino[cols].values, treino[tgt].values.astype(int), teste[cols].values)
    return probs.dropna()


def main():
    carteira = portfolio_tickers(load_portfolio_config())
    log.info("Baixando %d ativos e contexto (%s)...", len(carteira), PERIODO)
    raw = _baixar(carteira)
    ctx_raw = _baixar(["^VIX", "SPY", "BTC-USD", "GLD"])
    context = {"vix": ctx_raw["^VIX"]["Close"], "spy": ctx_raw["SPY"]["Close"],
               "btc": ctx_raw["BTC-USD"]["Close"], "gold": ctx_raw["GLD"]["Close"]}
    extra = _baixar(["^VIX", "^VIX3M", "^TNX", "DX-Y.NYB"])
    feats = build_all_features(raw, context, 60)

    linhas = []
    for t, df in feats.items():
        df = df.join(_regime(extra, df.index))
        for h in [1, 2, 3]:
            log.info("%s D+%d", t, h)
            p_base = _walk_forward(df, h, FEATURE_COLS)
            p_novo = _walk_forward(df, h, FEATURE_COLS + NOVAS)
            comum = p_base.index.intersection(p_novo.index)
            alta = (df.loc[comum, f"fwd_ret_d{h}"] >= 0).astype(float)
            for dt in comum:
                linhas.append({"ticker": t, "h": h, "data": dt, "alta": alta[dt],
                               "p_base": p_base[dt], "p_novo": p_novo[dt]})

    r = pd.DataFrame(linhas)
    resultado = {}
    for h in [1, 2, 3]:
        g = r[r["h"] == h].sort_values("data")
        if g.empty:
            continue
        acc_b = float(((g["p_base"] >= 0.5) == (g["alta"] == 1)).mean())
        acc_n = float(((g["p_novo"] >= 0.5) == (g["alta"] == 1)).mean())
        br_b = (g["p_base"] - g["alta"]) ** 2
        br_n = (g["p_novo"] - g["alta"]) ** 2
        diff = (br_n - br_b).groupby(g["data"]).mean().sort_index()
        dm = dm_from_loss_diff(diff.values, h=h)
        p_uni = float(stats.t.cdf(dm["statistic"], dm["df"]))       # H1: novo com Brier menor
        meio = len(diff) // 2
        metades = [bool(diff.iloc[:meio].mean() < 0), bool(diff.iloc[meio:].mean() < 0)]
        resultado[f"d{h}"] = {
            "n": len(g), "n_dias": len(diff), "desde": str(g["data"].min().date()),
            "acc_base": round(acc_b, 4), "acc_novo": round(acc_n, 4),
            "brier_base": round(float(br_b.mean()), 5), "brier_novo": round(float(br_n.mean()), 5),
            "sempre_alta": round(float(g["alta"].mean()), 4),
            "dm": round(dm["statistic"], 3), "p": round(p_uni, 4), "venceu_nas_duas_metades": metades,
        }
    lista = [dict(v, h=k) for k, v in resultado.items()]
    apply_bh_correction(lista)
    for item in lista:
        k = item["h"]
        resultado[k]["p_corrigido"] = round(item.get("p_adj", item["p"]), 4)
        resultado[k]["aprovado"] = bool(item.get("p_adj", 1) < 0.05 and all(resultado[k]["venceu_nas_duas_metades"]))
    SAIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
