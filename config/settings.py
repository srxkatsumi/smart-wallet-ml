import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output")
CHARTS_DIR   = OUTPUT_DIR / "charts"
MODELS_DIR   = OUTPUT_DIR / "models"
PRED_LOG     = OUTPUT_DIR / "predictions_log.csv"
PUBLIC_LOG   = OUTPUT_DIR / "predictions_log_public.csv"
WEIGHTS_FILE = OUTPUT_DIR / "ensemble_weights.json"
METADATA_FILE= OUTPUT_DIR / "model_metadata.csv"
RECALIB_FILE = OUTPUT_DIR / "ultima_recalibracao.json"
HTML_REPORT  = OUTPUT_DIR / "resumo_diario.html"

# ── Model hyperparameters ─────────────────────────────────────────────────
N_ESTIMATORS_RF  = 100
MAX_DEPTH_RF     = 5
N_ESTIMATORS_GB  = 100
MAX_DEPTH_GB     = 3
LEARNING_RATE_GB = 0.05
N_SPLITS_CV      = 5
CV_GAP           = 1          # gap between train and val to prevent same-day lookahead
SGD_ALPHA        = 0.0001
SGD_MAX_ITER     = 1000

# ── Pipeline settings ─────────────────────────────────────────────────────
HORIZONS              = [1, 2, 3]
PRICE_PERIOD          = "2y"
CHARTS_RETENTION_DAYS = 30
DOWNLOAD_BATCH_SIZE        = 20
DOWNLOAD_BATCH_SLEEP       = 2
SPLIT_DETECTION_THRESHOLD  = 0.40
WEIGHT_DECAY_FACTOR   = 0.1
RECALIBRATION_DAYS    = 30
MIN_SAMPLES_FEATURES  = 60
SGD_PARTIAL_FIT_DAYS  = 5
MIN_VALIDATIONS_WEIGHT= 5

# ── Portfolio settings ────────────────────────────────────────────────────
HORIZONTE_ANOS         = [1, 3, 5, 10]
TAXA_CRESCIMENTO_BASE  = {"pessimista": 0.03, "base": 0.08, "optimista": 0.15}

# ── FX fallbacks ─────────────────────────────────────────────────────────
EUR_USD_FALLBACK = 1.12
EUR_GBP_FALLBACK = 0.85

# ── CSV schema ────────────────────────────────────────────────────────────
PRED_COLS = [
    "ticker", "pred_date", "target_date", "horizon",
    "direction", "ref_price", "pred_price", "confidence",
    "actual_price", "actual_change_pct", "correct",
    "atr_at_prediction", "predicted_price",
    "model_rf", "model_gb", "model_sgd",
]

DEFAULT_WEIGHTS = {
    "d1": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d2": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d3": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
}

# ── Email timezone (Barcelona) ────────────────────────────────────────────
BARCELONA_UTC_OFFSET = 2  # CEST (verão); alterar para 1 em outubro (CET)

# ── Asset class per ticker ────────────────────────────────────────────────
# 0 = individual stock  1 = equity ETF  2 = crypto  3 = commodity ETF
ASSET_CLASSES: dict[str, int] = {
    "ALV.DE":  0, "BABA":    0, "BTC-USD": 2, "DHER.DE": 0,
    "LLY":     0, "NVDA":    0, "SIE.DE":  0, "BMW.DE":  0,
    "BAS.DE":  0, "NESN.SW": 0, "NOVN.SW": 0, "ROG.SW":  0,
    "EMIM.AS": 1, "EXUS.MI": 1, "IWDA.AS": 1, "MEUD.PA": 1,
    "CSPX.L":  1, "VWCE.DE": 1, "SJPA.MI": 1, "ICGA.DE": 1,
    "SGLN.L":  3,
}

# ── Market calendar mapping (pandas-market-calendars) ────────────────────
# Tickers not listed here default to NYSE.
TICKER_CALENDAR: dict[str, str] = {
    # London Stock Exchange
    "EXUS.MI": "XMIL",   # Borsa Italiana (EUR)
    "SGLN.L":  "LSE",
    "CSPX.L":  "LSE",
    # Euronext (per-exchange MIC codes)
    "EMIM.AS": "XAMS",   # Amsterdam
    "IWDA.AS": "XAMS",   # Amsterdam
    "MEUD.PA": "XPAR",   # Paris
    "SJPA.MI": "XMIL",   # Milan (Borsa Italiana)
    # Xetra (Frankfurt)
    "ALV.DE":  "XETR",
    "SIE.DE":  "XETR",
    "BMW.DE":  "XETR",
    "BAS.DE":  "XETR",
    "DHER.DE": "XETR",
    "VWCE.DE": "XETR",
    "ICGA.DE": "XETR",
    # Swiss Exchange
    "NESN.SW": "SIX",
    "NOVN.SW": "SIX",
    "ROG.SW":  "SIX",
}
