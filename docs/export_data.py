"""
Gera docs/assets/data.js a partir dos arquivos reais em output/.

Script manual por enquanto (rodar depois de cada execução do pipeline, ou
sempre que quiser atualizar o site local com dados mais recentes). Reaproveita
exatamente a mesma lógica de reports/email_report.py e research/runner.py
(tendência, correlação, comparação de famílias, consenso) para não duplicar
regras de negócio com significados diferentes.

O card de Alertas da Home lê docs/assets/alerts_data.js (gerado por
export_alerts.py), não output/anomalia.json diretamente, pra não ter duas
fontes divergentes do mesmo dado. Rode export_alerts.py antes deste script.

Uso: python docs/export_alerts.py && python docs/export_data.py   (raiz do repositório)
"""
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import GBP_PENCE_TICKERS
from data.storage import aligned_rows, portfolio_tickers
from research.runner import _build_comparison

MIN_VAL = 15  # mesmo limiar de reports/email_report.py:_calcular_tendencia
CHART_WINDOW_DAYS = 90   # janela exibida nos gráficos (dias úteis)
# Acima disto um preço de GBP_PENCE_TICKERS no log só pode estar em pence (linhas antigas).
PENCE_LOG_THRESHOLD = 500


def is_usd_ticker(ticker: str) -> bool:
    """Tickers americanos não têm sufixo de bolsa (AFRM); os europeus têm (.DE, .MI, .L)."""
    return "." not in ticker


def log_price_to_eur(price: float, ticker: str, gbp_eur_rate: float,
                     usd_eur_rate: float | None = None) -> float:
    """
    Os logs guardam GBP_PENCE_TICKERS em EUR desde que o download passou a
    converter; só as linhas mais antigas ainda estão em pence. Ações em dólar
    ficam em USD no log (o modelo prevê o preço da bolsa onde são negociadas)
    e só são convertidas aqui, para exibir em euro, quando `usd_eur_rate` vem.
    """
    if ticker in GBP_PENCE_TICKERS and price > PENCE_LOG_THRESHOLD:
        return price / 100 * gbp_eur_rate
    if usd_eur_rate is not None and is_usd_ticker(ticker):
        return price * usd_eur_rate
    return price


def latest_rows_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Ticker sem barra nova não ganha previsão no dia, mas continua no site com a última que tem."""
    return df[df["pred_date"] == df.groupby("ticker")["pred_date"].transform("max")]


def _load_json(path, default=None):
    p = ROOT / path
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def calcular_tendencia(df_log: pd.DataFrame, tickers: list) -> dict:
    """Espelha reports/email_report.py:_calcular_tendencia: 30 dias úteis vs. os 30 anteriores."""
    validadas = aligned_rows(df_log[
        (df_log["correct"].notna()) & (df_log["ticker"].isin(tickers)) & (df_log["horizon"] == 1)
    ]).copy()
    if validadas.empty:
        return {"estado": "treino", "n": 0}
    validadas["target_date"] = pd.to_datetime(validadas["target_date"])
    validadas = validadas.sort_values("target_date")
    hoje = pd.Timestamp.today()
    corte30 = hoje - pd.offsets.BDay(30)
    corte60 = hoje - pd.offsets.BDay(60)
    recentes = validadas[validadas["target_date"] >= corte30]
    anteriores = validadas[(validadas["target_date"] >= corte60) & (validadas["target_date"] < corte30)]

    if len(recentes) < MIN_VAL:
        return {"estado": "treino", "n": len(recentes), "faltam": MIN_VAL - len(recentes)}

    acc_rec = float(recentes["correct"].astype(float).mean())
    acc_ant = float(anteriores["correct"].astype(float).mean()) if len(anteriores) >= MIN_VAL else None
    delta = round(acc_rec - acc_ant, 4) if acc_ant is not None else None
    return {"estado": "ok", "n": len(recentes), "acc_recente": round(acc_rec, 4), "delta_vs_periodo_anterior": delta}


def correlacao_favoritos(df_log: pd.DataFrame, tickers: list, gbp_eur_rate: float) -> dict:
    """
    Aproxima reports/email_report.py:_build_correlation_html, mas a partir do
    ref_price já registado em predictions_log.csv (não temos o histórico de
    preços completo persistido, só o que já foi usado em previsões). Janela
    menor que os 120 dias do email (varia por ticker, alguns só entraram na
    carteira em julho). Resultado é direcional, não tão robusto quanto o do
    email. Fica mais preciso assim que ligarmos isto ao pipeline principal.
    """
    d1 = df_log[(df_log["ticker"].isin(tickers)) & (df_log["horizon"] == 1)].copy()
    d1["pred_date"] = pd.to_datetime(d1["pred_date"])
    closes = {}
    for t in tickers:
        s = d1[d1["ticker"] == t].sort_values("pred_date").set_index("pred_date")["ref_price"]
        s = s.map(lambda p: log_price_to_eur(p, t, gbp_eur_rate))
        s = s[~s.index.duplicated(keep="last")]
        if len(s) >= 10:
            closes[t] = s
    if len(closes) < 2:
        return {"tickers": [], "matrix": [], "n_dias": {}}

    df_c = pd.DataFrame(closes).pct_change().dropna(how="all")
    corr = df_c.corr()
    ticks = list(corr.columns)
    matrix = [[round(float(corr.values[i, j]), 2) for j in range(len(ticks))] for i in range(len(ticks))]
    n_dias = {t: int(closes[t].shape[0]) for t in ticks}
    return {"tickers": ticks, "matrix": matrix, "n_dias": n_dias}


_FAMILY_LABELS = {
    "classico_avancado": "Classical advanced",
    "estado_oculto":     "Hidden state (HMM)",
    "series_temporais":  "Time series",
    "neural_recorrente": "Recurrent neural",
    "neural_atencao":    "Attention-based neural",
    "bayesiano":         "Bayesian (GP+BNN)",
    "generativo":        "Generative (VAE+GAN)",
    "reinforcement":     "Reinforcement (DQN+PPO)",
    "contrarian":        "Contrarian (CB+EWI+PEL)",
    "eficiente":         "Efficient (TCN/DLinear/PatchTST)",
    "foundation":        "Foundation (Chronos/TimesFM/Moirai)",
    "conformal":         "Conformal (calibration)",
    "drift":             "Drift detection (ADWIN/PH)",
}


def comparacao_familias(log: pd.DataFrame) -> list:
    """Mesma tabela de research/runner.py:_build_comparison, com os rótulos do site."""
    results = _build_comparison(log, None)
    for r in results:
        r["label"] = _FAMILY_LABELS.get(r["family"], r["family"])
    return results


def consenso_familias(log: pd.DataFrame, tickers: list, research_weights: dict) -> list:
    """Espelha research/runner.py:_build_consensus, usando a última semana registada."""
    last_week = log["week_date"].max()
    rows = log[(log["week_date"] == last_week) & log["direction_d1"].notna()]
    consensus = []
    for ticker in tickers:
        ticker_rows = rows[rows["ticker"] == ticker]
        if ticker_rows.empty:
            continue
        w = research_weights.get("d1", {})
        weight_up = sum(w.get(r["family"], 1.0) for _, r in ticker_rows.iterrows() if r.get("direction_d1") == "up")
        weight_total = sum(w.get(r["family"], 1.0) for _, r in ticker_rows.iterrows())
        pct = weight_up / max(weight_total, 1e-9)
        consensus.append({
            "ticker": ticker,
            "direction": "UP" if pct >= 0.5 else "DOWN",
            "strength": "strong" if abs(pct - 0.5) >= 0.25 else "weak",
            "up_count": int(sum(1 for _, r in ticker_rows.iterrows() if r.get("direction_d1") == "up")),
            "total": len(ticker_rows),
            "pct_up": round(pct, 4),
        })
    consensus.sort(key=lambda x: x["pct_up"], reverse=True)
    return {"consenso": consensus, "semana": str(last_week)}


def acertou_ontem(ticker: str, df_log: pd.DataFrame, ontem_str: str):
    """Espelha reports/email_report.py:_acertou_ontem."""
    mask = (
        (df_log["ticker"] == ticker) & (df_log["target_date"] == ontem_str) &
        (df_log["horizon"] == 1) & (df_log["correct"].notna())
    )
    rows = df_log[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if not bool(row["correct"]):
        return "wrong"
    return "up_correct" if row.get("direction") == "up" else "down_correct"


def fetch_gbp_eur_rate() -> float:
    """Mesma chamada de data/downloader.py:_fetch_fx('EURGBP=X', ...), invertida pra GBP→EUR."""
    try:
        eur_gbp = yf.Ticker("EURGBP=X").fast_info["last_price"]
        if eur_gbp and eur_gbp > 0:
            return 1 / float(eur_gbp)
    except Exception:
        pass
    return 1 / 0.85  # fallback, mesma ordem de grandeza de EUR_GBP_FALLBACK em config/settings.py


def fetch_usd_eur_history(period: str = "2y") -> pd.Series:
    """USD→EUR de cada dia (1 / EURUSD=X), para converter cada preço com o câmbio do próprio dia."""
    try:
        fx = yf.download("EURUSD=X", period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        s = fx["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        if not s.empty:
            return (1 / s).sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)


def usd_eur_on(usd_eur_hist: pd.Series, date) -> float:
    """Câmbio do dia (ou do último dia com cotação antes dele); fallback na mesma ordem de EUR_USD_FALLBACK."""
    if usd_eur_hist is not None and not usd_eur_hist.empty:
        rate = usd_eur_hist.asof(pd.Timestamp(date).normalize())
        if pd.notna(rate):
            return float(rate)
        return float(usd_eur_hist.iloc[0])
    return 1 / 1.08


def technical_indicators(close: pd.Series) -> pd.DataFrame:
    """Espelha exatamente features/engineering.py:build_features: SMA20/50, Bollinger, RSI14, MACD."""
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()

    std20 = close.rolling(20).std()
    out["bb_upper"] = out["sma20"] + 2 * std20
    out["bb_lower"] = out["sma20"] - 2 * std20

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]
    return out


def chart_data_for_ticker(ticker: str, gbp_eur_rate: float, usd_eur_hist: pd.Series | None = None) -> dict | None:
    """Baixa OHLC ao vivo (yfinance, mesma fonte do pipeline) e monta as séries dos 4 painéis."""
    try:
        raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None
    except Exception:
        return None

    close = raw["Close"]
    is_pence = ticker in GBP_PENCE_TICKERS
    if is_pence:
        close = close / 100 * gbp_eur_rate
    elif is_usd_ticker(ticker):
        idx = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close * pd.Series([usd_eur_on(usd_eur_hist, d) for d in idx], index=close.index)

    ind = technical_indicators(close).tail(CHART_WINDOW_DAYS)
    dates = [d.strftime("%Y-%m-%d") for d in ind.index]

    def series(col):
        return [
            {"date": d, "value": round(float(v), 4)}
            for d, v in zip(dates, ind[col])
            if not pd.isna(v)
        ]

    return {
        "currency": "EUR",
        "price": series("close"),
        "sma20": series("sma20"),
        "sma50": series("sma50"),
        "bb_upper": series("bb_upper"),
        "bb_lower": series("bb_lower"),
        "rsi14": series("rsi14"),
        "macd": series("macd"),
        "macd_sig": series("macd_sig"),
        "macd_hist": series("macd_hist"),
    }


def accuracy_curve(df_log: pd.DataFrame, ticker: str) -> list:
    """
    Acurácia D+1 em janela deslizante de MIN_VAL previsões, pro painel de
    acurácia, mesma janela (15) do card "D+1 accuracy (last 15)", pra que
    o último ponto do gráfico bata com o número mostrado ao lado (antes
    usava expanding().mean(), uma média cumulativa desde o início da janela
    exibida: número diferente por desenho, o que gerava a confusão).
    """
    all_sub = aligned_rows(df_log[
        (df_log["ticker"] == ticker) & (df_log["horizon"] == 1) & (df_log["correct"].notna())
    ]).sort_values("target_date")
    # busca um pouco mais de histórico do que o exibido, pra a janela
    # deslizante já vir "aquecida" (com 15 amostras) no primeiro ponto mostrado
    sub = all_sub.tail(CHART_WINDOW_DAYS + MIN_VAL)
    if sub.empty:
        return []
    rolling_acc = sub["correct"].astype(float).rolling(MIN_VAL, min_periods=5).mean()
    curve = [
        {"date": pd.to_datetime(d).strftime("%Y-%m-%d"), "value": round(float(v) * 100, 1)}
        for d, v in zip(sub["target_date"], rolling_acc)
        if not pd.isna(v)
    ]
    return curve[-CHART_WINDOW_DAYS:]


def experimento_predictions(tickers: list, gbp_eur_rate: float, usd_eur_hist: pd.Series | None = None) -> dict:
    """
    Previsões de hoje do desafiante/campeão, por ticker/horizonte, mesma fonte
    do email de Experimentos. predictions_experimentos_log.csv vem da mesma
    fonte crua que predictions_log.csv, então tem a mesma mistura de moeda
    nas linhas antigas do SGLN.L, em GBp (ver log_price_to_eur).
    """
    path = ROOT / "output/predictions_experimentos_log.csv"
    if not path.exists():
        return {}
    exp_log = pd.read_csv(path)
    exp_log = exp_log[exp_log["ticker"].isin(tickers)]
    if exp_log.empty:
        return {}
    today = latest_rows_per_ticker(exp_log)

    out = {}
    for t in sorted(tickers):
        rows = today[today["ticker"] == t]
        if rows.empty:
            continue
        rate = usd_eur_on(usd_eur_hist, rows.iloc[0]["pred_date"])
        to_eur = lambda p: round(log_price_to_eur(float(p), t, gbp_eur_rate, rate), 4)  # noqa: E731
        horizons = {}
        for _, r in rows.iterrows():
            horizons[f"d{int(r['horizon'])}"] = {
                "champion_price": to_eur(r["champion_pred_price"]),
                "challenger_price": to_eur(r["challenger_pred_price"]),
                "challenger_ret": round(float(r["challenger_pred_ret"]), 4),
                "interval_lo": to_eur(r["challenger_interval_lo"]),
                "interval_hi": to_eur(r["challenger_interval_hi"]),
            }
        out[t] = {"ref_price": to_eur(rows.iloc[0]["ref_price"]), "horizons": horizons}
    return out


def _load_alerts_data() -> dict:
    """
    Reads docs/assets/alerts_data.js (written by export_alerts.py) instead of
    output/anomalia.json. The two used to be separate sources for the same
    concept: this page's card read the production file (empty until main.py
    actually runs the anomaly block), while alerts.html read its own live
    fetch, so they could silently disagree. Both now read the one file.
    Run export_alerts.py before this script to refresh it.
    """
    path = Path(__file__).resolve().parent / "assets" / "alerts_data.js"
    if not path.exists():
        return {"alerta": False, "date": None, "motivos": [], "market": {}, "tickers": {}}
    text = path.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].rsplit(";", 1)[0].strip())


def anomaly_reasons_en(anomalia: dict) -> list:
    """
    Rebuilds the alert's reasons in English from the raw fields in
    alerts_data.js, instead of reusing evaluation/anomaly.py's Portuguese
    `motivos` strings (those feed the email, which stays PT).
    """
    reasons = []
    market = anomalia.get("market", {})
    if market.get("transicao"):
        reasons.append(f"VIX moved into the high-volatility regime ({market['vix_ontem']} → {market['vix_hoje']})")
    elif market.get("spike"):
        verb = "rose" if market["variacao_1d"] > 0 else "fell"
        reasons.append(f"VIX {verb} {abs(market['variacao_1d'])*100:.0f}% in a day ({market['vix_ontem']} → {market['vix_hoje']})")
    for ticker, r in anomalia.get("tickers", {}).items():
        if r.get("alerta"):
            reasons.append(f"{ticker}: move was {r['multiplo_atr']:.1f}x its normal ATR range")
    return reasons


def main():
    portfolio = _load_json("config/portfolio.json")
    tickers = portfolio_tickers(portfolio)

    log = pd.read_csv(ROOT / "output/predictions_log.csv")
    log["pred_date"] = pd.to_datetime(log["pred_date"])
    last_date = log[log["ticker"].isin(tickers)]["pred_date"].max()
    ontem_bday = (last_date - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    today_rows = latest_rows_per_ticker(log[log["ticker"].isin(tickers)]).sort_values(["ticker", "horizon"])
    val = aligned_rows(log[(log["ticker"].isin(tickers)) & (log["horizon"] == 1) & (log["correct"].notna())])

    print("Fetching live GBP/EUR rate...")
    gbp_eur_rate = fetch_gbp_eur_rate()
    usd_eur_hist = fetch_usd_eur_history()

    favoritos = {}
    for t in sorted(tickers):
        rows = today_rows[today_rows["ticker"] == t]
        if rows.empty:
            continue
        price = log_price_to_eur(float(rows.iloc[0]["ref_price"]), t, gbp_eur_rate,
                                 usd_eur_on(usd_eur_hist, rows.iloc[0]["pred_date"]))
        horizons = {
            f"d{int(r['horizon'])}": {"direction": r["direction"], "confidence": round(float(r["confidence"]), 4)}
            for _, r in rows.iterrows()
        }
        sub = val[val["ticker"] == t].tail(15)
        acc = round(float(sub["correct"].mean()) * 100, 1) if len(sub) else None

        hist = log[(log["ticker"] == t) & (log["horizon"] == 1) & log["ref_price"].notna()].sort_values("pred_date")
        hist = hist.drop_duplicates(subset="pred_date", keep="last")
        price_history = [
            {"date": d.strftime("%Y-%m-%d"),
             "price": round(log_price_to_eur(float(p), t, gbp_eur_rate, usd_eur_on(usd_eur_hist, d)), 4)}
            for d, p in zip(hist["pred_date"], hist["ref_price"])
        ]

        print(f"Fetching OHLC + indicators for {t}...")
        favoritos[t] = {
            "price": round(price, 4),
            "currency": "EUR",
            "horizons": horizons,
            "acc_recent": acc,
            "n_recent": len(sub),
            "acertou_ontem": acertou_ontem(t, log, ontem_bday),
            "price_history": price_history,
            "chart": chart_data_for_ticker(t, gbp_eur_rate, usd_eur_hist),
            "accuracy_curve": accuracy_curve(log, t),
        }

    sig = _load_json("output/significance.json")
    sig_exp = _load_json("output/significance_experimentos.json")
    backtest_exp = _load_json("output/backtest_experimentos.json", default={"tickers": {}})
    exp_predictions = experimento_predictions(tickers, gbp_eur_rate, usd_eur_hist)
    anomalia = _load_alerts_data()
    anomalia = {**anomalia, "reasons_en": anomaly_reasons_en(anomalia)}

    n_total = int(log["correct"].notna().sum())  # trilha de auditoria completa, inclui linhas antigas
    tendencia = calcular_tendencia(log, tickers)
    correlacao = correlacao_favoritos(log, tickers, gbp_eur_rate)

    research_log_path = ROOT / "output/predictions_research_log.csv"
    familias, consenso = [], {"consenso": [], "semana": None}
    if research_log_path.exists():
        rlog = pd.read_csv(research_log_path)
        familias = comparacao_familias(rlog)
        research_weights = _load_json("output/research_weights.json")
        consenso = consenso_familias(rlog, tickers, research_weights)

    out = {
        "last_update": str(last_date.date()),
        "accuracy": {
            "d1": sig["d1"].get("acc"), "d2": sig["d2"].get("acc"), "d3": sig["d3"].get("acc"),
            "n_total": n_total,
        },
        "tendencia": tendencia,
        "favoritos": favoritos,
        "correlacao": correlacao,
        "familias": familias,
        "consenso": consenso,
        "experimentos": sig_exp,
        "backtest_experimentos": backtest_exp.get("tickers", {}),
        "exp_predictions": exp_predictions,
        "anomalia": anomalia,
    }

    out_path = Path(__file__).resolve().parent / "assets" / "data.js"
    out_path.write_text(
        "// Gerado por docs/export_data.py a partir dos dados reais em output/.\n"
        "// Rodar de novo (python docs/export_data.py) sempre que quiser atualizar o site local.\n"
        f"const SITE_DATA = {json.dumps(out, indent=2, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    print(f"OK - {out_path} atualizado (dados de {out['last_update']})")


if __name__ == "__main__":
    main()
