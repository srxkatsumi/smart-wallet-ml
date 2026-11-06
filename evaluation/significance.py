"""
Statistical significance of directional forecast accuracy.

One-sided t-test on daily accuracy: H0 = accuracy ≤ 0.50 (no better than random).

Stays silent until MIN_N validated predictions over MIN_DIAS days exist per
horizon — below that threshold any accuracy figure is meaningless.
"""
import json
import logging
import math
import pandas as pd
from pathlib import Path
from scipy.stats import t as student_t

from data.storage import aligned_rows

logger = logging.getLogger(__name__)

SIGNIFICANCE_FILE = Path("output/significance.json")
MIN_N    = 100   # minimum validated predictions per horizon before reporting
MIN_DIAS = 20    # minimum distinct target dates: below this the daily t-test is unstable


def apply_bh_correction(results: list[dict], alpha: float = 0.05, p_key: str = "p") -> None:
    """
    Benjamini-Hochberg FDR correction across several simultaneous p-values,
    applied in place. Adds 'p_adj' (adjusted p-value) and 'sig' (p_adj < alpha)
    to each dict in `results`.

    Use whenever a table ranks several models/families side by side (e.g. the
    13-family research comparison): with m comparisons at alpha=0.05, the
    chance of at least one spurious p<0.05 by chance alone is 1 - (1-alpha)^m
    (about 49 % already at m=13) if left uncorrected. BH controls the false
    discovery rate instead of treating every low raw p-value as a real find.
    """
    m = len(results)
    if m == 0:
        return
    ranked = sorted(results, key=lambda r: r[p_key])
    prev_adj = 1.0
    for i in range(m - 1, -1, -1):
        r    = ranked[i]
        rank = i + 1
        adj  = min(r[p_key] * m / rank, 1.0)
        adj  = min(adj, prev_adj)
        prev_adj = adj
        r["p_adj"] = round(float(adj), 4)
        r["sig"]   = bool(adj < alpha)


def daily_cluster_stats(correct: pd.Series, dates: pd.Series) -> dict:
    """
    Teste unilateral (H1: acurácia > 50%) com o DIA como unidade. Tickers no mesmo
    dia reagem ao mesmo mercado, então ~530 previsões de um dia não são 530
    observações independentes, e um binomial sobre as linhas encolhe o p-valor.
    """
    daily = correct.astype(float).groupby(dates.values).mean()
    n     = len(daily)
    mean  = float(daily.mean()) if n else 0.5
    se    = float(daily.std(ddof=1) / math.sqrt(n)) if n >= 2 else 0.0
    if se == 0:
        return {"p": 1.0, "ci_lo": round(mean, 4), "ci_hi": round(mean, 4), "n_dias": n}
    t_crit = student_t.ppf(0.975, n - 1)
    return {
        "p":      float(student_t.sf((mean - 0.5) / se, n - 1)),
        "ci_lo":  round(mean - t_crit * se, 4),
        "ci_hi":  round(mean + t_crit * se, 4),
        "n_dias": n,
    }


def _relevant_move_accuracy(subset: pd.DataFrame, day: int) -> tuple[float | None, int]:
    """Acurácia só nos movimentos acima de 0.3 ATR, o mesmo limiar usado nos rótulos de treino."""
    cols = {"actual_price", "ref_price", "atr_at_prediction"}
    if not cols.issubset(subset.columns):
        return None, 0
    threshold = subset["atr_at_prediction"].astype(float) * 0.3 * math.sqrt(day)
    relevant  = subset[(subset["actual_price"] - subset["ref_price"]).abs() > threshold]
    if relevant.empty:
        return None, 0
    return round(float(relevant["correct"].astype(float).mean()), 4), len(relevant)


def compute_significance(df_log: pd.DataFrame) -> dict:
    """
    Por horizonte, sobre as previsões com horizonte alinhado (ver aligned_rows).

    Cada valor tem ready, n, k, acc, n_dias, p (t unilateral sobre a acurácia
    diária), ci_lo/ci_hi (IC 95% por dia), sig, e a acurácia só nos movimentos
    acima do limiar de treino (acc_mov_relevante, n_mov_relevante). Fica em
    silêncio (ready=False) até MIN_N linhas em MIN_DIAS dias.
    """
    validated = aligned_rows(df_log[df_log["correct"].notna()]).copy()
    validated["correct"] = validated["correct"].astype(float)

    results: dict[str, dict] = {}
    for day in [1, 2, 3]:
        subset = validated[validated["horizon"] == day]
        n      = len(subset)
        key    = f"d{day}"
        dias   = subset["target_date"].nunique() if n else 0

        if n < MIN_N or dias < MIN_DIAS:
            results[key] = {"ready": False, "n": n, "n_dias": dias,
                            "needed": max(MIN_N - n, MIN_DIAS - dias, 0)}
            continue

        k     = int(subset["correct"].sum())
        acc   = k / n
        stats = daily_cluster_stats(subset["correct"], subset["target_date"])
        acc_rel, n_rel = _relevant_move_accuracy(subset, day)

        results[key] = {
            "ready":  True,
            "n":      n,
            "k":      k,
            "acc":    round(acc, 4),
            "n_dias": stats["n_dias"],
            "p":      round(stats["p"], 4),
            "ci_lo":  stats["ci_lo"],
            "ci_hi":  stats["ci_hi"],
            "sig":    bool(stats["p"] < 0.05),
            "acc_mov_relevante": acc_rel,
            "n_mov_relevante":   n_rel,
        }
        logger.info(
            "Significância D+%d: %.1f%% (n=%d, %d dias) p=%.4f %s",
            day, acc * 100, n, stats["n_dias"], stats["p"],
            "✅ significativo" if stats["p"] < 0.05 else ("⏳ tendência" if stats["p"] < 0.10 else "—"),
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

    lines = ["\n📐 Significância (p por dia, H1: acc > 50%):"]
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
