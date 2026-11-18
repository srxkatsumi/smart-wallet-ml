"""
Gera docs/assets/watchlist_data.js, the full ~543-ticker research universe
(config/watchlist.json's sectors), separate from the main data.js because:
  1. It's big (hundreds of tickers) and only watchlist.html needs it.
  2. Dividend data (rate/yield/asset type) and dividend frequency
     (Monthly/Quarterly/Semi-annual/Annual, inferred from trailing-12-month
     payment history) each require a slow per-ticker yfinance call
     (~0.5s each, several minutes total), both cached to disk separately
     so re-running the main export_data.py doesn't refetch them every time.

Prices and dividend amounts are converted to EUR at export time using live FX
rates (EURUSD=X, EURGBP=X, EURCHF=X), same as the Favorites page, unlike
Favorites' 7 EUR-native tickers, most of this universe trades in USD, so this
is a live conversion rather than a single hardcoded GBp fix.

Usage:
  python docs/export_watchlist.py             # uses the caches if present
  python docs/export_watchlist.py --refresh    # refetches everything from Yahoo
"""
import json
import sys
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import GBP_PENCE_TICKERS
from data.storage import aligned_rows

DIV_CACHE = Path(__file__).resolve().parent / "assets" / "dividends_cache.json"
FREQ_CACHE = Path(__file__).resolve().parent / "assets" / "dividend_frequency_cache.json"
DIV_BATCH_SLEEP = 0.15  # be polite to Yahoo across ~540 sequential requests


def load_sector_map() -> dict:
    cfg = json.loads((ROOT / "config/watchlist.json").read_text())
    sector_map = {}
    for sector, tickers in cfg.get("sectors", {}).items():
        for t in tickers:
            sector_map[t] = sector
    return sector_map


def _moeda_pelo_sufixo(ticker: str) -> str:
    """Moeda provável quando o Yahoo não responde, pelo sufixo da bolsa; sem sufixo, USD."""
    if ticker.endswith((".DE", ".PA", ".AS", ".MC", ".MI")):
        return "EUR"
    return {".SW": "CHF", ".T": "JPY", ".HK": "HKD"}.get("." + ticker.rsplit(".", 1)[-1], "USD") if "." in ticker else "USD"


def fetch_dividends(tickers: list, refresh: bool = False) -> dict:
    # Com cache e sem --refresh, busca só os tickers que faltam: um ticker novo
    # não pode esperar a segunda-feira para ter a moeda certa (sem isso, um
    # ticker em euro aparecia como dólar e era convertido errado).
    out = {}
    if DIV_CACHE.exists() and not refresh:
        out = json.loads(DIV_CACHE.read_text())
        faltam = [t for t in tickers if t not in out]
        if not faltam:
            print(f"Using cached dividend data ({DIV_CACHE.name}), pass --refresh to refetch.")
            return out
        print(f"Cache de dividendos sem {len(faltam)} ticker(s), buscando só esses: {faltam}")
        tickers = faltam

    print(f"Fetching dividend data for {len(tickers)} tickers from Yahoo (~{len(tickers)*DIV_BATCH_SLEEP/60:.1f} min)...")
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).get_info()
            yield_pct = info.get("dividendYield")
            rate = info.get("trailingAnnualDividendRate")
            currency = info.get("currency") or "USD"
            # yfinance reports LSE-listed tickers in pence (GBp), normalize the
            # currency label to GBP (and the per-share rate accordingly) regardless
            # of whether this ticker happens to pay a dividend, since the price
            # itself is also quoted in pence and needs the same /100 downstream.
            is_pence = currency == "GBp"
            if is_pence:
                currency = "GBP"
                if rate:
                    rate = float(rate) / 100
            pays_dividends = bool(yield_pct and yield_pct > 0)
            quote_type = (info.get("quoteType") or "").upper()
            asset_type = "ETF" if quote_type in ("ETF", "MUTUALFUND") else ("Stock" if quote_type == "EQUITY" else "Other")
            # No explicit "accumulating vs distributing" field exists on yfinance,
            # for ETFs this is inferred from whether they currently pay a dividend
            # (accumulating share classes reinvest instead of distributing).
            # Doesn't apply to individual stocks/other asset types.
            distribution = None
            if asset_type == "ETF":
                distribution = "Distributing" if pays_dividends else "Accumulating"
            out[t] = {
                "pays_dividends": pays_dividends,
                "yield_pct": round(float(yield_pct), 2) if yield_pct else 0.0,
                "rate": round(float(rate), 4) if rate else 0.0,
                "currency": currency,
                "is_pence": is_pence,
                "asset_type": asset_type,
                "distribution": distribution,
            }
        except Exception:
            out[t] = {"pays_dividends": False, "yield_pct": 0.0, "rate": 0.0, "currency": _moeda_pelo_sufixo(t),
                       "is_pence": False, "asset_type": "Other", "distribution": None}
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}...")
        time.sleep(DIV_BATCH_SLEEP)

    DIV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DIV_CACHE.write_text(json.dumps(out, indent=2))
    print(f"Dividend data cached to {DIV_CACHE}")
    return out


def classify_frequency(n_payments: int) -> str | None:
    if n_payments >= 10:
        return "Monthly"
    if n_payments >= 3:
        return "Quarterly"
    if n_payments == 2:
        return "Semi-annual"
    if n_payments == 1:
        return "Annual"
    return None


def fetch_dividend_frequency(tickers: list, refresh: bool = False) -> dict:
    out = {}
    if FREQ_CACHE.exists() and not refresh:
        out = json.loads(FREQ_CACHE.read_text())
        faltam = [t for t in tickers if t not in out]
        if not faltam:
            print(f"Using cached dividend frequency ({FREQ_CACHE.name}), pass --refresh to refetch.")
            return out
        print(f"Cache de frequência sem {len(faltam)} ticker(s), buscando só esses.")
        tickers = faltam

    print(f"Fetching dividend frequency for {len(tickers)} dividend-paying tickers "
          f"(~{len(tickers)*DIV_BATCH_SLEEP/60:.1f} min)...")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    for i, t in enumerate(tickers, 1):
        try:
            divs = yf.Ticker(t).dividends
            n_recent = int((divs.index >= cutoff).sum()) if divs is not None and len(divs) else 0
            out[t] = classify_frequency(n_recent)
        except Exception:
            out[t] = None
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}...")
        time.sleep(DIV_BATCH_SLEEP)

    FREQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FREQ_CACHE.write_text(json.dumps(out, indent=2))
    print(f"Dividend frequency cached to {FREQ_CACHE}")
    return out


def fetch_fx_rates() -> dict:
    """Live EUR→ccy rates (units of ccy per 1 EUR) for every non-EUR currency seen
    in the watchlist. Same source/pattern as export_data.py's fetch_gbp_eur_rate()."""
    pairs = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X", "JPY": "EURJPY=X", "HKD": "EURHKD=X"}
    fallbacks = {"USD": 1.08, "GBP": 0.85, "CHF": 0.94, "JPY": 165.0, "HKD": 8.5}
    rates = {"EUR": 1.0}
    for ccy, yf_ticker in pairs.items():
        try:
            v = yf.Ticker(yf_ticker).fast_info["last_price"]
            rates[ccy] = float(v) if v and v > 0 else fallbacks[ccy]
        except Exception:
            rates[ccy] = fallbacks[ccy]
    return rates


SERIES_PATH = Path(__file__).resolve().parent / "assets" / "watchlist_series.json"
SERIES_DIAS = 170   # 90 exibidos + aquecimento da SMA50, do MACD e do RSI


def fetch_fx_history() -> dict:
    """EUR→moeda de cada dia, último ano. Câmbio de cada dia (como nos Favoritos), não o de hoje para tudo."""
    pares = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X", "JPY": "EURJPY=X", "HKD": "EURHKD=X"}
    try:
        fxd = yf.download(list(pares.values()), period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]
        fxd.index = pd.to_datetime(fxd.index).tz_localize(None).normalize()
        return {ccy: fxd[par].dropna() for ccy, par in pares.items() if par in fxd and fxd[par].notna().any()}
    except Exception as e:
        print(f"Aviso: histórico de câmbio indisponível ({e}), usando o câmbio de hoje")
        return {}


def taxa_no_dia(fx_hist: dict, fx_rates: dict, moeda: str, dia) -> float:
    if moeda == "EUR":
        return 1.0
    s = fx_hist.get(moeda)
    if s is not None and not s.empty:
        v = s.asof(pd.Timestamp(dia).normalize())
        if pd.notna(v):
            return float(v)
    return float(fx_rates.get(moeda, 1.0))


def export_series(tickers: list, log: pd.DataFrame, dividends: dict, fx_rates: dict,
                  fx_hist: dict | None = None) -> None:
    """
    Série compacta para o detalhe de cada ativo na página Watchlist (os mesmos
    gráficos dos Favoritos): fechos em EUR, previsões D+1..D+3 de hoje e curva de
    acurácia D+1. Os indicadores (SMA, Bollinger, RSI, MACD) são calculados no
    navegador a partir dos fechos, com as mesmas fórmulas de export_data.py.
    Arquivo separado e carregado só no primeiro clique, para não pesar a página.
    Datas como deslocamento em dias a partir de "base", para caber em ~1 MB.
    """
    from export_data import accuracy_curve
    print(f"Baixando {len(tickers)} séries de preço para o detalhe da Watchlist...")
    raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    base = closes.index[-SERIES_DIAS] if len(closes) > SERIES_DIAS else closes.index[0]

    fx_hist = fx_hist if fx_hist is not None else fetch_fx_history()

    ult = log[log["pred_date"] == log.groupby("ticker")["pred_date"].transform("max")]
    out = {}
    for t in tickers:
        if t not in closes:
            continue
        c = closes[t].dropna().tail(SERIES_DIAS)
        if len(c) < 60:
            continue
        div = dividends.get(t, {})
        pence = 0.01 if (div.get("is_pence") or t in GBP_PENCE_TICKERS) else 1.0
        moeda = "GBP" if t in GBP_PENCE_TICKERS else div.get("currency", "USD")
        if moeda == "EUR":
            taxa = pd.Series(1.0, index=c.index)
        elif moeda in fx_hist:
            taxa = fx_hist[moeda].reindex(c.index, method="ffill").bfill().fillna(fx_rates.get(moeda, 1.0))
        else:
            taxa = pd.Series(fx_rates.get(moeda, 1.0), index=c.index)
        c_eur = c * pence / taxa
        h = {}
        for _, r in ult[ult["ticker"] == t].iterrows():
            h[f"d{int(r['horizon'])}"] = [r["direction"], round(float(r["confidence"]), 4)]
        acc = accuracy_curve(log, t)
        out[t] = {
            "o": [int((d - base).days) for d in c.index],
            "c": [round(float(v), 4) for v in c_eur.values],
            "h": h,
            "a": [[int((pd.Timestamp(x["date"]) - base).days), x["value"]] for x in acc],
        }
    SERIES_PATH.write_text(json.dumps({"base": str(base.date()), "t": out}, separators=(",", ":")), encoding="utf-8")
    print(f"OK - {SERIES_PATH} ({len(out)} tickers, {SERIES_PATH.stat().st_size / 1e6:.2f} MB)")


def main():
    refresh = "--refresh" in sys.argv
    sector_map = load_sector_map()
    tickers = sorted(sector_map.keys())

    log = pd.read_csv(ROOT / "output/predictions_log.csv")
    log = log[log["ticker"].isin(tickers)].copy()
    log["pred_date"] = pd.to_datetime(log["pred_date"])
    last_date = log["pred_date"].max()

    # Ticker sem barra nova não ganha previsão no dia, mas continua na página com a última que tem.
    d1 = log[log["horizon"] == 1]
    today = d1[d1["pred_date"] == d1.groupby("ticker")["pred_date"].transform("max")]
    d1_log = log[log["horizon"] == 1].sort_values("pred_date")

    dividends = fetch_dividends(tickers, refresh=refresh)
    payers = sorted(t for t in tickers if dividends.get(t, {}).get("pays_dividends"))
    frequencies = fetch_dividend_frequency(payers, refresh=refresh)
    fx_rates = fetch_fx_rates()
    fx_hist = fetch_fx_history()
    print(f"FX rates (per 1 EUR): {fx_rates}")

    rows = []
    for t in tickers:
        today_row = today[today["ticker"] == t]
        if today_row.empty:
            continue
        r = today_row.iloc[0]

        hist = d1_log[(d1_log["ticker"] == t) & d1_log["ref_price"].notna()]
        var_1d = None
        if len(hist) >= 2:
            last_two = hist["ref_price"].tail(2).values
            if last_two[0]:
                var_1d = round((last_two[1] / last_two[0] - 1) * 100, 2)

        val = aligned_rows(hist[hist["correct"].notna()]).tail(15)
        acc = round(float(val["correct"].mean()) * 100, 1) if len(val) else None

        div = dividends.get(t, {"pays_dividends": False, "yield_pct": 0.0, "rate": 0.0, "currency": _moeda_pelo_sufixo(t),
                                 "is_pence": False, "asset_type": "Other", "distribution": None})
        price_native = round(float(r["ref_price"]), 4) if pd.notna(r["ref_price"]) else None
        currency = div.get("currency", "USD")
        if t in GBP_PENCE_TICKERS:
            # data/downloader.py already converted these to EUR before the log was written.
            currency = "EUR"
            price_ccy = price_native
        else:
            # Other LSE-listed tickers reach the log in pence, like the raw dividend rate above.
            price_ccy = (price_native / 100) if (price_native is not None and div.get("is_pence")) else price_native
        price = round(price_ccy / fx_rates.get(currency, 1.0), 4) if price_ccy is not None else None

        # Var% em euro: a variação do preço e a do câmbio entre os dois dias.
        if var_1d is not None and currency != "EUR" and len(hist) >= 2:
            ult2 = hist.tail(2)
            dias = (ult2["ref_date"].fillna(ult2["pred_date"].dt.strftime("%Y-%m-%d"))
                    if "ref_date" in ult2 else ult2["pred_date"]).tolist()
            p0, p1 = ult2["ref_price"].values
            r0, r1 = (taxa_no_dia(fx_hist, fx_rates, currency, d) for d in dias)
            if p0 and r1:
                var_1d = round(((p1 / p0) * (r0 / r1) - 1) * 100, 2)

        # Yahoo's trailingAnnualDividendRate is unreliable for foreign ADRs (e.g. TM,
        # MUFG, HMC): it sometimes reports the unadjusted home-market dividend
        # (wrong scale/currency) while dividendYield is still computed correctly
        # against the real ADR price. Re-deriving the per-share rate from
        # native price × yield keeps it self-consistent regardless of that bug,
        # then converts to EUR like everything else on this page.
        dividend_rate = None
        if div["pays_dividends"] and price_ccy:
            dividend_rate = round((price_ccy * div["yield_pct"] / 100) / fx_rates.get(currency, 1.0), 4)

        rows.append({
            "ticker": t,
            "sector": sector_map[t],
            "price": price,
            "var_1d": var_1d,
            "direction": r["direction"],
            "confidence": round(float(r["confidence"]), 4),
            "acc_recent": acc,
            "n_recent": len(val),
            "pays_dividends": div["pays_dividends"],
            "dividend_yield_pct": div["yield_pct"],
            "dividend_rate": dividend_rate,
            "asset_type": div.get("asset_type", "Other"),
            "distribution": div.get("distribution"),
            "dividend_frequency": frequencies.get(t) if div["pays_dividends"] else None,
        })

    out = {
        "last_update": str(last_date.date()),
        "sectors": sorted(set(sector_map.values())),
        "tickers": rows,
    }

    out_path = Path(__file__).resolve().parent / "assets" / "watchlist_data.js"
    out_path.write_text(
        "// Generated by docs/export_watchlist.py, the full research universe (config/watchlist.json).\n"
        f"const WATCHLIST_DATA = {json.dumps(out, indent=2, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    print(f"OK - {out_path} updated ({len(rows)} tickers, data as of {out['last_update']})")

    try:
        export_series([r["ticker"] for r in rows], log, dividends, fx_rates, fx_hist)
    except Exception as e:
        print(f"Aviso: séries do detalhe da Watchlist não geradas ({e})")


if __name__ == "__main__":
    main()
