from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
OUTPUT_DIR      = BASE_DIR / "output"
RESULTS_FILE    = OUTPUT_DIR / "mega_sena_results.csv"
PREDICTIONS_FILE= OUTPUT_DIR / "predictions_log.csv"
WEIGHTS_FILE    = OUTPUT_DIR / "ensemble_weights.json"
OUTPUT_MD       = OUTPUT_DIR / "previsoes.md"

# ── Mega Sena settings ────────────────────────────────────────────────────
N_BALLS         = 60        # numbers in pool
BALLS_PER_DRAW  = 6         # drawn per game
N_SEQUENCES     = 5         # predicted sequences per draw day
DRAW_DAYS       = ["Tuesday", "Thursday", "Saturday"]

# ── Feature windows ───────────────────────────────────────────────────────
FREQ_WINDOWS    = [5, 10, 20, 50]   # last N draws for frequency features
MIN_DRAWS_TRAIN  = 30               # minimum draws needed to train
DAILY_BATCH_SIZE = 300              # max new historical draws to process per daily run
RETRAIN_INTERVAL = 50               # retrain every N draws during backfill

# ── Model hyperparameters (same family as stock system) ───────────────────
N_ESTIMATORS_RF  = 100
MAX_DEPTH_RF     = 4
N_ESTIMATORS_GB  = 100
MAX_DEPTH_GB     = 3
LEARNING_RATE_GB = 0.05
N_SPLITS_CV      = 3
SGD_ALPHA        = 0.001

# ── Ensemble weights ──────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {"rf": 1.0, "gb": 1.0, "sgd": 1.0}
MIN_DRAWS_WEIGHT= 10        # minimum validated draws to update weights
WEIGHT_DECAY    = 0.1

# ── Prize thresholds (for reporting) ─────────────────────────────────────
PRIZE_TIERS = {6: "Sena", 5: "Quina", 4: "Quadra", 3: "Terno", 2: "Duque"}

# ── Random baseline (theoretical) ────────────────────────────────────────
# E[matches per sequence] = 6 × (6/60) = 0.6
RANDOM_EXPECTED_MATCHES = BALLS_PER_DRAW * (BALLS_PER_DRAW / N_BALLS)
