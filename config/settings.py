import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output")
CHARTS_DIR   = OUTPUT_DIR / "charts"
MODELS_DIR   = OUTPUT_DIR / "models"
PRED_LOG     = OUTPUT_DIR / "predictions_log.csv"
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
SGD_ALPHA        = 0.0001
SGD_MAX_ITER     = 1000

# ── Pipeline settings ─────────────────────────────────────────────────────
HORIZONS              = [1, 2, 3]
PRICE_PERIOD          = "2y"
CHARTS_RETENTION_DAYS = 30
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
    "direction", "pred_price", "confidence",
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
