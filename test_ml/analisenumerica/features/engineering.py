import numpy as np
import pandas as pd
from config import N_BALLS, BALLS_PER_DRAW, FREQ_WINDOWS

# Features computed per number (1-60) per draw
FEATURE_COLS = (
    [f"freq_{w}d" for w in FREQ_WINDOWS]      # frequency in last N draws
    + ["draws_since_last"]                      # how many draws since last appeared
    + ["freq_trend"]                            # freq_5 / freq_20 — acceleration
    + ["deviation"]                             # freq vs expected 10%
    + ["decade"]                                # 0-5 group of 10
    + ["is_even", "is_prime"]                   # numeric properties
    + ["prev_sum", "prev_mean", "prev_spread"]  # previous draw stats
    + ["day_mon", "day_thu", "day_sat"]         # draw day one-hot
)

_PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}


def _number_features(number: int, history: pd.DataFrame, draw_day: str) -> dict:
    """Compute features for a single number given the draw history up to (not including) target draw."""
    n_draws = len(history)
    ball_cols = ["b1", "b2", "b3", "b4", "b5", "b6"]

    def appeared_in(row):
        return int(number in row[ball_cols].values)

    appeared = history.apply(appeared_in, axis=1).values  # 1 if appeared, 0 if not

    feats = {}
    for w in FREQ_WINDOWS:
        feats[f"freq_{w}d"] = appeared[-w:].mean() if n_draws >= w else appeared.mean()

    # Draws since last appearance
    last_idx = np.where(appeared[::-1] == 1)[0]
    feats["draws_since_last"] = int(last_idx[0]) if len(last_idx) else n_draws

    # Trend: recent frequency vs medium-term
    f5  = feats["freq_5d"]
    f20 = feats["freq_20d"]
    feats["freq_trend"] = f5 / f20 if f20 > 0 else 1.0

    # Deviation from expected frequency
    feats["deviation"] = feats["freq_20d"] - (BALLS_PER_DRAW / N_BALLS)

    # Number properties
    feats["decade"]   = (number - 1) // 10          # 0,1,2,3,4,5
    feats["is_even"]  = int(number % 2 == 0)
    feats["is_prime"] = int(number in _PRIMES)

    # Previous draw stats
    last = history.iloc[-1]
    prev_numbers = [int(last[c]) for c in ball_cols]
    feats["prev_sum"]    = sum(prev_numbers)
    feats["prev_mean"]   = np.mean(prev_numbers)
    feats["prev_spread"] = max(prev_numbers) - min(prev_numbers)

    # Draw day one-hot
    feats["day_mon"] = int(draw_day == "Monday")
    feats["day_thu"] = int(draw_day == "Thursday")
    feats["day_sat"] = int(draw_day == "Saturday")

    return feats


def build_training_data(results: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X, y for training.

    For each draw (row i in results), for each number 1-60:
    - X = features computed from history[:i]
    - y = 1 if number was drawn in draw i, else 0
    """
    X_rows, y_rows = [], []
    ball_cols = ["b1", "b2", "b3", "b4", "b5", "b6"]

    for i in range(10, len(results)):   # need at least 10 draws for history
        history = results.iloc[:i]
        row     = results.iloc[i]
        drawn   = set(int(row[c]) for c in ball_cols)
        day     = pd.Timestamp(row["data"]).day_name()

        for number in range(1, N_BALLS + 1):
            feats = _number_features(number, history, day)
            X_rows.append([feats[f] for f in FEATURE_COLS])
            y_rows.append(1 if number in drawn else 0)

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)


def build_prediction_features(results: pd.DataFrame, draw_day: str) -> np.ndarray:
    """Build X for inference: features for each number 1-60 using all available history."""
    X = []
    for number in range(1, N_BALLS + 1):
        feats = _number_features(number, results, draw_day)
        X.append([feats[f] for f in FEATURE_COLS])
    return np.array(X, dtype=float)
