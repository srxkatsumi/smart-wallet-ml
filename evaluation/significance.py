"""
Statistical significance of directional forecast accuracy.

One-sided binomial test: H0 = accuracy ≤ 0.50 (no better than random).
Wilson score 95 % confidence interval for the accuracy proportion.

Stays silent (returns empty dict) until MIN_N validated predictions exist
per horizon — below that threshold any accuracy figure is meaningless.
"""
import json
import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

SIGNIFICANCE_FILE = Path("output/significance.json")
MIN_N = 100   # minimum validated predictions per horizon before reporting


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI for a proportion."""
    centre = (k + z ** 2 / 2) / (n + z ** 2)
    margin = z * math.sqrt(k * (n - k) / n + z ** 2 / 4) / (n + z ** 2)
    return round(centre - margin, 4), round(centre + margin, 4)


def compute_significance(df_log: pd.DataFrame) -> dict:
    """
    Computes per-horizon accuracy, p-value, and CI from predictions_log.

    Returns a dict with keys d1, d2, d3. Each value has:
      ready    – True when n >= MIN_N (safe to interpret)
      n        – number of validated predictions
      k        – number of correct predictions
      acc      – accuracy (k/n)
      p        – p-value (one-sided binomial, H1: acc > 0.50)
      ci_lo/hi – 95 % Wilson CI
      sig      – True if p < 0.05
    """
    validated = df_log[df_log["correct"].notna()].copy()
    validated["correct"] = validated["correct"].astype(float)

    results: dict[str, dict] = {}
    for day in [1, 2, 3]:
        subset = validated[validated["horizon"] == day]["correct"]
        n      = len(subset)
        key    = f"d{day}"

        if n < MIN_N:
            results[key] = {"ready": False, "n": n, "needed": MIN_N - n}
            continue

        k     = int(subset.sum())
        acc   = k / n
        p_val = binomtest(k, n, p=0.5, alternative="greater").pvalue
        ci_lo, ci_hi = _wilson_ci(k, n)

        results[key] = {
            "ready": True,
            "n":     n,
            "k":     k,
            "acc":   round(acc, 4),
            "p":     round(float(p_val), 4),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "sig":   bool(p_val < 0.05),
        }
        logger.info(
            "Significância D+%d: %.1f%% (n=%d) p=%.4f %s",
            day, acc * 100, n, p_val,
            "✅ significativo" if p_val < 0.05 else ("⏳ tendência" if p_val < 0.10 else "—"),
        )

    return results


def save_significance(sig: dict) -> None:
    SIGNIFICANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIGNIFICANCE_FILE.write_text(
        json.dumps(sig, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_significance() -> dict:
    if not SIGNIFICANCE_FILE.exists():
        return {}
    try:
        return json.loads(SIGNIFICANCE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_telegram_line(sig: dict) -> str | None:
    """
    Returns a significance block for Telegram, or None if no horizon is ready.
    Keeps the message concise: one line per ready horizon.
    """
    if not sig:
        return None

    ready = [k for k in ["d1", "d2", "d3"] if sig.get(k, {}).get("ready")]
    if not ready:
        # Show progress towards MIN_N only once (D+1)
        d1 = sig.get("d1", {})
        if d1 and "needed" in d1:
            return f"\n📐 Significância: aguarda +{d1['needed']} previsões validadas (D+1)"
        return None

    lines = ["\n📐 Significância (p-val binomial, H1: acc > 50%):"]
    for key in ["d1", "d2", "d3"]:
        r = sig.get(key, {})
        if not r.get("ready"):
            lines.append(f"  D+{key[1]}: —  (n={r.get('n', 0)}/{MIN_N})")
            continue
        if r["sig"]:
            icon = "✅"
        elif r["p"] < 0.10:
            icon = "⏳"
        else:
            icon = "—"
        ci = f"[{r['ci_lo']*100:.1f}–{r['ci_hi']*100:.1f}%]"
        lines.append(
            f"  D+{key[1]}: {r['acc']*100:.1f}% {ci}  p={r['p']:.3f} {icon}"
        )
    return "\n".join(lines)
