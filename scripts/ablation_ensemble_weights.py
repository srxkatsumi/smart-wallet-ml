"""
Ablation: adaptive ensemble weights (models/validator.py:update_ensemble_weights)
vs a fixed equal-weight baseline, evaluated on output/predictions_log.csv.

Motivation: an earlier external review assumed WEIGHT_DECAY_FACTOR made the
adaptive scheme collapse onto yesterday's single validation (~90% of the
weight). Recomputing the actual formula (exp(WEIGHT_DECAY_FACTOR * arange(n)),
normalized) shows yesterday holds only ~10% of the weight over a 30-entry
window, so that specific claim was wrong. Running the ablation anyway found a
real, separate bug: update_ensemble_weights windowed by the last 30 ROWS, and
with 500+ tickers validated on the same target_date, that collapsed the
intended ~30-day window into a single day of noise (see git history for the
fix in models/validator.py and research/runner.py). This script now compares
three schemes: the original buggy adaptive weights (as actually logged), a
fixed equal-weight baseline, and the corrected day-windowed adaptive weights,
replayed causally over history.

Method: for every logged prediction we already have model_rf/model_gb/model_sgd
(each model's individual directional vote) and the ground-truth outcome.
- equal-weight: majority vote of the three models (ties impossible with 3).
- old adaptive: the direction actually logged at the time (pre-fix weights).
- corrected adaptive: models/validator.py's post-fix logic replayed day by
  day, using only data from target_dates strictly before the one being
  scored (no lookahead), applied as a weighted vote since raw probabilities
  aren't logged (see caveat below).
McNemar's test is the right test here: same trials, paired binary outcomes
per pair of classifiers, not independent binomial samples.

Caveat: predictions_log.csv only stores each model's thresholded direction,
not its raw probability, so "weighted vote" is a proxy for the real weighted-
probability blend in models/ensemble.py:_train_horizon, not an exact replica.
Equal-weighted probability averaging (mean(p_rf, p_gb, p_sgd) > 0.5) can
disagree with vote-based schemes in edge cases where one model's probability
is far from 0.5 and the other two are close to it on the other side. The two
should agree on the vast majority of rows; treat exact effect sizes as
approximate, the direction and significance of the results as the reliable
part.

Usage: python scripts/ablation_ensemble_weights.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import WEIGHT_DECAY_FACTOR, MIN_VALIDATIONS_WEIGHT

LOG = ROOT / "output" / "predictions_log.csv"
_MODEL_COLS = {"rf": "model_rf", "gb": "model_gb", "sgd": "model_sgd"}


def _replay_corrected_weights(sub: pd.DataFrame) -> pd.Series:
    """
    Causal replay of the corrected (day-windowed) weight update for one
    horizon. For each target_date, scores that date's predictions using
    weights derived only from dates strictly before it, then folds that
    date's outcomes into the window before moving to the next date.
    """
    sub   = sub.sort_values("target_date")
    dates = sorted(sub["target_date"].unique())
    pred  = pd.Series(index=sub.index, dtype=object)
    weights = {"rf": 1.0, "gb": 1.0, "sgd": 1.0}

    for date in dates:
        today = sub[sub["target_date"] == date]
        up_w    = sum(weights[m] * (today[col] == "up") for m, col in _MODEL_COLS.items())
        total_w = sum(weights.values())
        pred.loc[today.index] = np.where(up_w > total_w / 2, "up", "down")

        win_dates = [d for d in dates if d <= date][-30:]
        hist_win  = sub[sub["target_date"].isin(win_dates)]
        if len(hist_win) < MIN_VALIDATIONS_WEIGHT:
            continue

        n_dias = len(win_dates)
        decay  = np.exp(WEIGHT_DECAY_FACTOR * np.arange(n_dias))
        decay  = decay / decay.sum()
        peso_por_data = dict(zip(win_dates, decay))

        new_weights = {}
        for m, col in _MODEL_COLS.items():
            correto     = (hist_win[col] == hist_win["truth"]).astype(float)
            acc_por_dia = correto.groupby(hist_win["target_date"]).mean().reindex(win_dates)
            acc_weighted = sum(acc_por_dia[d] * peso_por_data[d] for d in win_dates)
            new_weights[m] = max(0.1, acc_weighted)
        tot = sum(new_weights.values())
        weights = {k: v * 3.0 / tot for k, v in new_weights.items()}

    return pred


def _mcnemar(a_correct: pd.Series, b_correct: pd.Series) -> tuple[int, int, float]:
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    discordant = a_only + b_only
    p = binomtest(a_only, discordant, p=0.5, alternative="two-sided").pvalue if discordant else 1.0
    return a_only, b_only, p


def _ground_truth(row) -> str | None:
    """Actual direction, derived from actual_change_pct sign (robust to how
    `correct` handles edge cases like flat closes or split-NaN rows)."""
    chg = row["actual_change_pct"]
    if pd.isna(chg):
        return None
    return "up" if chg > 0 else "down"


def run():
    df = pd.read_csv(LOG)
    df = df[df["correct"].notna()].copy()
    df["truth"] = df.apply(_ground_truth, axis=1)
    df = df.dropna(subset=["truth", "model_rf", "model_gb", "model_sgd"])

    votes = df[["model_rf", "model_gb", "model_sgd"]]
    up_votes = (votes == "up").sum(axis=1)
    df["equal_dir"] = pd.Series(
        pd.cut(up_votes, bins=[-1, 1, 3], labels=["down", "up"]).astype(str),
        index=df.index,
    )
    df["old_correct"]   = (df["direction"] == df["truth"])
    df["equal_correct"] = (df["equal_dir"]  == df["truth"])

    for horizon in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == horizon].copy()
        sub["corrected_dir"]     = _replay_corrected_weights(sub)
        sub["corrected_correct"] = (sub["corrected_dir"] == sub["truth"])

        n     = len(sub)
        accs  = {k: sub[f"{k}_correct"].mean() for k in ["old", "equal", "corrected"]}
        print(f"\n=== D+{horizon}  (n={n}) ===")
        for k in ["old", "equal", "corrected"]:
            print(f"  {k:<12}{accs[k]:>7.1%}")

        for a, b in [("corrected", "old"), ("corrected", "equal")]:
            a_only, b_only, p = _mcnemar(sub[f"{a}_correct"], sub[f"{b}_correct"])
            delta = accs[a] - accs[b]
            print(f"  {a} vs {b}: delta={delta:+.1%}  p(McNemar)={p:.4f}  "
                  f"({a}-only-right={a_only}, {b}-only-right={b_only})")


if __name__ == "__main__":
    run()
