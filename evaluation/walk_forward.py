"""
Walk-Forward Validation (WFV): honest out-of-sample accuracy.

For each of the last n_steps trading days where outcomes are known:
  - Train only on data strictly before that day (expanding window)
  - Predict D+1 / D+2 / D+3 direction with RF(50 trees)
  - Compare prediction to actual outcome

Runs every market day, appends to output/wfv_log.csv,
and trims entries older than 30 days.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from features.engineering import FEATURE_COLS
from config.settings import MAX_DEPTH_RF

logger = logging.getLogger(__name__)

WFV_LOG_FILE  = Path("output/wfv_log.csv")
WFV_JSON_FILE = Path("output/wfv_results.json")
WFV_N_STEPS   = 30   # trading days lookback window
WFV_MIN_TRAIN = 252  # minimum rows required before first WFV step

_LOG_COLS = ["date", "ticker", "acc_d1", "acc_d2", "acc_d3", "n_d1", "n_d2", "n_d3"]


# ── Core walk-forward per ticker ──────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, n_steps: int = WFV_N_STEPS) -> dict:
    """
    Walk-forward simulation for a single ticker.

    Returns dict with acc_d1/acc_d2/acc_d3 and n_d1/n_d2/n_d3,
    or {} if there isn't enough historical data.
    """
    valid_mask = df["target_d1"].notna()
    valid_df   = df[valid_mask]

    if len(valid_df) < WFV_MIN_TRAIN + n_steps:
        return {}

    test_rows = valid_df.iloc[-n_steps:]
    results   = {1: [], 2: [], 3: []}

    for _, test_row in test_rows.iterrows():
        pos      = df.index.get_loc(test_row.name)
        train_df = df.iloc[:pos]

        X_te = np.array(test_row[FEATURE_COLS].values, dtype=float).reshape(1, -1)
        if np.any(np.isnan(X_te)):
            X_te = np.nan_to_num(X_te, nan=0.0)

        for day in [1, 2, 3]:
            target_col = f"target_d{day}"
            actual     = test_row.get(target_col)
            if pd.isna(actual):
                continue

            train_valid = train_df.dropna(subset=[target_col])
            if len(train_valid) < WFV_MIN_TRAIN:
                continue

            X_tr = train_valid[FEATURE_COLS].values
            y_tr = train_valid[target_col].values.astype(int)

            scaler  = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=MAX_DEPTH_RF,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_tr_sc, y_tr)
            pred = int(rf.predict(X_te_sc)[0])
            results[day].append(int(pred == int(actual)))

    out = {}
    for day, hits in results.items():
        out[f"acc_d{day}"] = round(float(np.mean(hits)), 4) if hits else None
        out[f"n_d{day}"]   = len(hits)
    return out


# ── Log management ────────────────────────────────────────────────────────

def _load_log() -> pd.DataFrame:
    if WFV_LOG_FILE.exists():
        df = pd.read_csv(WFV_LOG_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=_LOG_COLS)


def _save_log(df: pd.DataFrame) -> None:
    WFV_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WFV_LOG_FILE, index=False)


# ── Summary JSON ──────────────────────────────────────────────────────────

def _build_summary(log: pd.DataFrame, today: date, n_steps: int) -> dict:
    """Summarise the full log into wfv_results.json."""
    # Latest run (today)
    today_rows = log[log["date"] == today]

    def _mean_latest(col: str) -> float | None:
        vals = today_rows[col].dropna().tolist()
        return round(float(np.mean(vals)), 4) if vals else None

    # 30-day trend: mean acc_d1 per day, for plotting
    daily_trend = (
        log.groupby("date")["acc_d1"]
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"acc_d1": "mean_acc_d1"})
    )
    daily_trend["date"] = daily_trend["date"].astype(str)
    trend_list = daily_trend.to_dict(orient="records")

    return {
        "date":           str(today),
        "n_steps":        n_steps,
        "portfolio_mean": {
            "acc_d1": _mean_latest("acc_d1"),
            "acc_d2": _mean_latest("acc_d2"),
            "acc_d3": _mean_latest("acc_d3"),
        },
        "n_days_in_log":  int(log["date"].nunique()),
        "daily_trend":    trend_list,
        "tickers":        {
            row["ticker"]: {
                "acc_d1": row.get("acc_d1"),
                "acc_d2": row.get("acc_d2"),
                "acc_d3": row.get("acc_d3"),
                "n_d1":   int(row.get("n_d1", 0)),
            }
            for _, row in today_rows.iterrows()
        },
    }


# ── Public entry point ────────────────────────────────────────────────────

def run_portfolio_wfv(
    featured_data: dict,
    my_tickers: list[str],
    n_steps: int = WFV_N_STEPS,
    run_date: date | None = None,
) -> dict:
    """
    Runs WFV for all portfolio tickers, appends to wfv_log.csv,
    trims entries older than WFV_RETENTION days, and saves wfv_results.json.
    Returns the summary dict.
    """
    today = run_date or date.today()

    log = _load_log()

    # Skip if already ran today
    if not log.empty and (log["date"] == today).any():
        logger.info("WFV: já executou hoje (%s) — a saltar", today)
        return _build_summary(log, today, n_steps)

    # Run WFV per ticker
    new_rows: list[dict] = []
    for ticker in my_tickers:
        df = featured_data.get(ticker)
        if df is None:
            continue
        try:
            r = run_walk_forward(df, n_steps=n_steps)
            if not r:
                continue
            row = {"date": today, "ticker": ticker, **r}
            new_rows.append(row)
            logger.info(
                "WFV %-12s D+1=%.0f%% D+2=%.0f%% D+3=%.0f%% (n=%d)",
                ticker,
                (r.get("acc_d1") or 0) * 100,
                (r.get("acc_d2") or 0) * 100,
                (r.get("acc_d3") or 0) * 100,
                r.get("n_d1", 0),
            )
        except Exception as e:
            logger.warning("WFV %s falhou: %s", ticker, e)

    if not new_rows:
        logger.warning("WFV: sem dados suficientes para nenhum ticker")
        return {}

    # Append — log acumula para sempre, nunca apaga linhas
    new_df = pd.DataFrame(new_rows, columns=_LOG_COLS)
    log = new_df if log.empty else pd.concat([log, new_df], ignore_index=True)
    _save_log(log)

    # Build and save summary
    summary = _build_summary(log, today, n_steps)
    WFV_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    WFV_JSON_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    mean_d1 = summary["portfolio_mean"]["acc_d1"] or 0
    logger.info(
        "WFV completo: %d tickers | D+1=%.1f%% | log: %d dias acumulados",
        len(new_rows),
        mean_d1 * 100,
        summary["n_days_in_log"],
    )
    return summary
