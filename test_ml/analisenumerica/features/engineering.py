import numpy as np
import pandas as pd
from config import N_BALLS, BALLS_PER_DRAW, FREQ_WINDOWS

FEATURE_COLS = (
    [f"freq_{w}d" for w in FREQ_WINDOWS]
    + ["draws_since_last"]
    + ["freq_trend"]
    + ["deviation"]
    + ["decade"]
    + ["is_even", "is_prime"]
    + ["prev_sum", "prev_mean", "prev_spread"]
    + ["day_mon", "day_thu", "day_sat"]
)

_PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}
_BALL_COLS = ["b1", "b2", "b3", "b4", "b5", "b6"]

# Precomputed static number properties (indexed 0..59)
_DECADE    = np.array([(n - 1) // 10 for n in range(1, N_BALLS + 1)], dtype=float)
_IS_EVEN   = np.array([n % 2 == 0    for n in range(1, N_BALLS + 1)], dtype=float)
_IS_PRIME  = np.array([n in _PRIMES  for n in range(1, N_BALLS + 1)], dtype=float)


def _build_appeared_matrix(results: pd.DataFrame) -> np.ndarray:
    """Return bool matrix of shape (n_draws, N_BALLS). appeared[i, j] = 1 if ball j+1 drawn in row i."""
    balls = results[_BALL_COLS].values.astype(int)  # (n, 6)
    mat = np.zeros((len(results), N_BALLS), dtype=np.int8)
    for col in range(6):
        mat[np.arange(len(results)), balls[:, col] - 1] = 1
    return mat


def _draw_features(appeared_mat: np.ndarray, i: int, draw_day: str,
                   prev_numbers: np.ndarray) -> np.ndarray:
    """
    Build feature matrix of shape (N_BALLS, len(FEATURE_COLS)) for draw i
    using appeared_mat[:i] as history. Fully vectorized over all 60 numbers.
    """
    hist = appeared_mat[:i].astype(float)  # (i, 60)
    n_hist = i

    # Frequency windows — shape (N_BALLS,) each
    freq = {}
    for w in FREQ_WINDOWS:
        if n_hist >= w:
            freq[w] = hist[-w:].mean(axis=0)
        else:
            freq[w] = hist.mean(axis=0)

    # Draws since last appearance — vectorized
    # Reverse the history and find first occurrence for each number
    rev = hist[::-1]  # (i, 60)
    # For each number: argmax of first 1 (or n_hist if never appeared)
    has_appeared = hist.sum(axis=0) > 0           # (60,) bool
    draws_since  = np.where(
        has_appeared,
        np.argmax(rev, axis=0).astype(float),
        float(n_hist),
    )

    f5  = freq[5]
    f20 = freq[20]
    trend     = np.where(f20 > 0, f5 / np.maximum(f20, 1e-10), 1.0)
    deviation = f20 - (BALLS_PER_DRAW / N_BALLS)

    # Draw-level scalars (same for all 60 numbers)
    prev_sum    = float(prev_numbers.sum())
    prev_mean   = float(prev_numbers.mean())
    prev_spread = float(prev_numbers.max() - prev_numbers.min())
    day_mon = float(draw_day == "Monday")
    day_thu = float(draw_day == "Thursday")
    day_sat = float(draw_day == "Saturday")

    # Stack into (N_BALLS, n_features)
    X = np.column_stack([
        freq[5], freq[10], freq[20], freq[50],
        draws_since,
        trend,
        deviation,
        _DECADE, _IS_EVEN, _IS_PRIME,
        np.full(N_BALLS, prev_sum),
        np.full(N_BALLS, prev_mean),
        np.full(N_BALLS, prev_spread),
        np.full(N_BALLS, day_mon),
        np.full(N_BALLS, day_thu),
        np.full(N_BALLS, day_sat),
    ])
    return X


def build_training_data(results: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build X, y for training. Vectorized over all 60 numbers per draw."""
    appeared_mat = _build_appeared_matrix(results)
    n = len(results)

    X_list, y_list = [], []
    balls_vals = results[_BALL_COLS].values.astype(int)

    for i in range(10, n):
        draw_day     = pd.Timestamp(results.iloc[i]["data"]).day_name()
        prev_numbers = balls_vals[i - 1]
        X_draw       = _draw_features(appeared_mat, i, draw_day, prev_numbers)  # (60, F)

        drawn = set(balls_vals[i])
        y_draw = np.array([1 if (j + 1) in drawn else 0 for j in range(N_BALLS)], dtype=int)

        X_list.append(X_draw)
        y_list.append(y_draw)

    return np.vstack(X_list), np.concatenate(y_list)


def build_prediction_features(results: pd.DataFrame, draw_day: str) -> np.ndarray:
    """Build X for inference: features for each number 1-60 using all available history."""
    appeared_mat = _build_appeared_matrix(results)
    prev_numbers = results[_BALL_COLS].values[-1].astype(int)
    return _draw_features(appeared_mat, len(results), draw_day, prev_numbers)
