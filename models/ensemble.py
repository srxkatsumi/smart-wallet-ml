import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from features.engineering import FEATURE_COLS
from config.settings import (
    N_ESTIMATORS_RF, MAX_DEPTH_RF,
    N_ESTIMATORS_GB, MAX_DEPTH_GB, LEARNING_RATE_GB,
    N_SPLITS_CV, CV_GAP, SGD_ALPHA, SGD_MAX_ITER,
    SGD_PARTIAL_FIT_DAYS, HORIZONS,
    METADATA_FILE, RECALIB_FILE, RECALIBRATION_DAYS,
)

logger = logging.getLogger(__name__)


# ── SGD persistence ───────────────────────────────────────────────────────

def _sgd_paths(ticker: str, day: int, models_dir) -> tuple:
    safe = ticker.replace(".", "_").replace("-", "_")
    return (
        models_dir / f"sgd_{safe}_d{day}.pkl",
        models_dir / f"scaler_{safe}_d{day}.json",
    )


def _load_sgd(ticker: str, day: int, models_dir):
    sgd_path, scaler_path = _sgd_paths(ticker, day, models_dir)
    if not (sgd_path.exists() and scaler_path.exists()):
        return None, None, True
    model = joblib.load(sgd_path)
    with open(scaler_path) as f:
        sc_data = json.load(f)
    if len(sc_data["center"]) != len(FEATURE_COLS):
        logger.info("SGD %s D+%s: feature count changed (%d→%d), reinitializing",
                    sgd_path.stem, day, len(sc_data["center"]), len(FEATURE_COLS))
        return None, None, True
    scaler = RobustScaler()
    scaler.center_ = np.array(sc_data["center"])
    scaler.scale_  = np.array(sc_data["scale"])
    scaler.n_features_in_ = len(sc_data["center"])
    return model, scaler, False


def _save_sgd(ticker: str, day: int, model, scaler, models_dir):
    sgd_path, scaler_path = _sgd_paths(ticker, day, models_dir)
    joblib.dump(model, sgd_path)
    with open(scaler_path, "w") as f:
        json.dump({
            "center":   scaler.center_.tolist(),
            "scale":    scaler.scale_.tolist(),
            "features": FEATURE_COLS,
        }, f)


# ── Single horizon training ───────────────────────────────────────────────

def _train_horizon(df: pd.DataFrame, target_col: str, weights: dict,
                   ticker: str, models_dir) -> dict:
    day      = int(target_col[-1])
    df_train = df.iloc[:-day].copy() if day > 0 else df.copy()
    X        = df_train[FEATURE_COLS].values
    y        = df_train[target_col].values
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    tscv      = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=CV_GAP)
    acuracias = {"rf": [], "gb": [], "sgd": []}

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
        random_state=42, n_jobs=-1,
    )
    gb_model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS_GB, max_depth=MAX_DEPTH_GB,
        learning_rate=LEARNING_RATE_GB, random_state=42,
    )

    sgd_model, scaler_sgd, needs_init = _load_sgd(ticker, day, models_dir)
    if needs_init:
        sgd_model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=SGD_ALPHA,
            max_iter=SGD_MAX_ITER, random_state=42,
        )
        sgd_model.fit(X_scaled, y)
        scaler_sgd = scaler
        _save_sgd(ticker, day, sgd_model, scaler_sgd, models_dir)
        logger.info("SGD %s D+%d: treino inicial (%d exemplos)", ticker, day, len(y))
    else:
        X_novo = scaler_sgd.transform(df_train[FEATURE_COLS].values[-SGD_PARTIAL_FIT_DAYS:])
        y_novo = df_train[target_col].values[-SGD_PARTIAL_FIT_DAYS:]
        sgd_model.partial_fit(X_novo, y_novo, classes=[0, 1])
        _save_sgd(ticker, day, sgd_model, scaler_sgd, models_dir)

    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        rf_model.fit(X_tr, y_tr)
        gb_model.fit(X_tr, y_tr)
        acuracias["rf"].append(accuracy_score(y_val, rf_model.predict(X_val)))
        acuracias["gb"].append(accuracy_score(y_val, gb_model.predict(X_val)))

    for train_idx, val_idx in tscv.split(X_scaled):
        X_val_sgd = scaler_sgd.transform(df_train[FEATURE_COLS].values[val_idx])
        acuracias["sgd"].append(accuracy_score(y[val_idx], sgd_model.predict(X_val_sgd)))

    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)

    acc_media = {k: float(np.mean(v)) for k, v in acuracias.items()}

    X_last     = scaler.transform(df[FEATURE_COLS].iloc[[-1]].values)
    X_last_sgd = scaler_sgd.transform(df[FEATURE_COLS].iloc[[-1]].values)

    probs = {
        "rf":  float(rf_model.predict_proba(X_last)[0][1]),
        "gb":  float(gb_model.predict_proba(X_last)[0][1]),
        "sgd": float(sgd_model.predict_proba(X_last_sgd)[0][1]),
    }
    preds_ind = {name: "up" if p > 0.5 else "down" for name, p in probs.items()}

    total_w   = sum(weights.values())
    prob_ens  = sum(probs[k] * weights.get(k, 1.0) for k in probs) / total_w
    direction = "up" if prob_ens > 0.5 else "down"
    confidence= max(prob_ens, 1 - prob_ens)

    return {
        "direction":  direction,
        "confidence": confidence,
        "prob":       prob_ens,
        "acc_media":  acc_media,
        "preds_ind":  preds_ind,
        "probs_ind":  probs,
        "rf_model":   rf_model,
        "gb_model":   gb_model,
    }


# ── Train all tickers ─────────────────────────────────────────────────────

def train_all(featured_data: dict, ensemble_weights: dict,
              models_dir) -> dict:
    resultados_ml = {}
    logger.info("Treinando modelos (D+1, D+2, D+3)...")

    for ticker, df in featured_data.items():
        try:
            horizons = {}
            for day in HORIZONS:
                target_col = f"target_d{day}"
                day_key    = f"d{day}"
                w          = ensemble_weights.get(day_key, {"rf": 1.0, "gb": 1.0, "sgd": 1.0})
                horizons[day] = _train_horizon(df, target_col, w, ticker, models_dir)

            close_now  = float(df["Close"].iloc[-1])
            close_prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else close_now
            var_1d     = round((close_now / close_prev - 1) * 100, 2)
            atr        = float(df["ATR14"].iloc[-1])
            preds_dict = {}
            for day in HORIZONS:
                h     = horizons[day]
                mag   = atr * 0.5 * np.sqrt(day)
                price = (close_now + mag) if h["direction"] == "up" else (close_now - mag)
                preds_dict[day] = (h["direction"], price, h["confidence"])

            resultados_ml[ticker] = {
                "close_now":  close_now,
                "var_1d":     var_1d,
                "last_date":  df.index[-1],
                "direction":  horizons[1]["direction"],
                "confidence": horizons[1]["confidence"],
                "prob":       horizons[1]["prob"],
                "preds_dict": preds_dict,
                "horizons":   horizons,
                "df":         df,
                "atr":        atr,
            }
            logger.info(
                "%s | D+1: %s %.0f%% | D+2: %s %.0f%% | D+3: %s %.0f%%",
                ticker,
                horizons[1]["direction"], horizons[1]["confidence"] * 100,
                horizons[2]["direction"], horizons[2]["confidence"] * 100,
                horizons[3]["direction"], horizons[3]["confidence"] * 100,
            )
        except Exception as e:
            logger.error("%s: erro no treino: %s", ticker, e)

    logger.info("Modelos treinados: %d ativos", len(resultados_ml))
    return resultados_ml


# ── Model metadata ────────────────────────────────────────────────────────

def save_model_metadata(resultados_ml: dict, my_tickers: list[str]):
    from config.settings import METADATA_FILE
    METADATA_COLS = (
        ["date", "ticker", "horizon", "model", "n_samples", "acc_cv_mean", "acc_cv_std"]
        + [f"feat_{f}" for f in FEATURE_COLS]
    )

    if not METADATA_FILE.exists():
        pd.DataFrame(columns=METADATA_COLS).to_csv(METADATA_FILE, index=False)

    hoje_str     = pd.Timestamp.now().strftime("%Y-%m-%d")
    novas_linhas = []

    for ticker, res in resultados_ml.items():
        if ticker not in my_tickers:
            continue
        n = len(res.get("df", []))
        for day in HORIZONS:
            h = res["horizons"][day]
            for model_name, model_key in [("rf", "rf_model"), ("gb", "gb_model")]:
                m = h.get(model_key)
                if m is None or not hasattr(m, "feature_importances_"):
                    continue
                linha = {
                    "date":        hoje_str,
                    "ticker":      ticker,
                    "horizon":     day,
                    "model":       model_name,
                    "n_samples":   n,
                    "acc_cv_mean": round(h["acc_media"].get(model_name, 0), 4),
                    "acc_cv_std":  0.0,
                }
                for fname, imp in zip(FEATURE_COLS, m.feature_importances_):
                    linha[f"feat_{fname}"] = round(float(imp), 6)
                novas_linhas.append(linha)

    if novas_linhas:
        df_meta  = pd.read_csv(METADATA_FILE)
        df_novas = pd.DataFrame(novas_linhas)
        df_meta  = pd.concat([df_meta, df_novas], ignore_index=True)
        df_meta.to_csv(METADATA_FILE, index=False)
        logger.info("model_metadata: %d linhas adicionadas", len(novas_linhas))


# ── Monthly SGD recalibration ─────────────────────────────────────────────

def monthly_recalibration(featured_data: dict, my_tickers: list[str],
                          models_dir):
    if not RECALIB_FILE.exists():
        needs = True
    else:
        with open(RECALIB_FILE) as f:
            data = json.load(f)
        ultima = pd.Timestamp(data.get("data", "2000-01-01"))
        needs  = (pd.Timestamp.today() - ultima).days >= RECALIBRATION_DAYS

    if not needs:
        with open(RECALIB_FILE) as f:
            data = json.load(f)
        days_left = RECALIBRATION_DAYS - (pd.Timestamp.today() - pd.Timestamp(data["data"])).days
        logger.info("Recalibração: próxima em %d dias", days_left)
        return

    logger.info("Recalibração mensal do SGD...")
    for ticker in my_tickers:
        if ticker not in featured_data:
            continue
        df_t = featured_data[ticker]
        for day in HORIZONS:
            target_col = f"target_d{day}"
            df_train   = df_t.iloc[:-day].copy()
            X          = df_train[FEATURE_COLS].values
            y          = df_train[target_col].values
            scaler_new = RobustScaler()
            X_sc       = scaler_new.fit_transform(X)
            model_new  = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=SGD_ALPHA,
                max_iter=SGD_MAX_ITER, random_state=42,
            )
            model_new.fit(X_sc, y)
            _save_sgd(ticker, day, model_new, scaler_new, models_dir)
            logger.info("%s D+%d: SGD recalibrado", ticker, day)

    with open(RECALIB_FILE, "w") as f:
        json.dump({"data": pd.Timestamp.today().strftime("%Y-%m-%d")}, f)
    logger.info("Recalibração concluída — próxima em %d dias", RECALIBRATION_DAYS)
