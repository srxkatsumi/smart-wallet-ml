import pandas as pd
import numpy as np
from config import PRIZE_TIERS, RANDOM_EXPECTED_MATCHES, BALLS_PER_DRAW, N_BALLS


def last_week_table(results: pd.DataFrame, n: int = 7) -> str:
    """Markdown table of the last N draws."""
    recent = results.tail(n).copy()
    recent["data"] = pd.to_datetime(recent["data"]).dt.strftime("%d/%m/%Y")
    ball_cols = ["b1", "b2", "b3", "b4", "b5", "b6"]
    lines = ["| Concurso | Data | Dezenas |", "|----------|------|---------|"]
    for _, row in recent.iloc[::-1].iterrows():
        balls = " · ".join(f"**{int(row[c]):02d}**" for c in ball_cols)
        lines.append(f"| {int(row['concurso'])} | {row['data']} | {balls} |")
    return "\n".join(lines)


def accuracy_summary(pred_df: pd.DataFrame) -> dict:
    """Compute aggregate statistics over all validated predictions."""
    v = pred_df[pred_df["validated"] == True].copy()
    if v.empty:
        return {"n_draws": 0, "n_predictions": 0}

    n_draws = v["target_concurso"].nunique()
    n_preds = len(v)
    avg_matches  = v["matches"].mean()
    best_matches = v["matches"].max()

    prize_counts = {tier: 0 for tier in PRIZE_TIERS}
    for _, row in v.iterrows():
        m = int(row["matches"])
        if m in PRIZE_TIERS:
            prize_counts[m] += 1

    return {
        "n_draws":       n_draws,
        "n_predictions": n_preds,
        "avg_matches":   round(avg_matches, 3),
        "best_matches":  int(best_matches),
        "random_baseline": round(RANDOM_EXPECTED_MATCHES, 3),
        "vs_random":     round(avg_matches - RANDOM_EXPECTED_MATCHES, 3),
        "prize_counts":  prize_counts,
    }


def next_predictions_table(pred_df: pd.DataFrame) -> str:
    """Markdown table of the most recent unvalidated predictions."""
    pending = pred_df[pred_df["validated"] != True].copy()
    if pending.empty:
        return "_Nenhuma previsão pendente._"

    pending = pending.sort_values(["target_date", "seq_num"])
    pending["target_date"] = pd.to_datetime(pending["target_date"]).dt.strftime("%d/%m/%Y")

    lines = ["| Concurso | Data | Dia | Seq | Dezenas |",
             "|----------|------|-----|-----|---------|"]
    for _, row in pending.iterrows():
        balls = " · ".join(
            f"{int(row[f'n{i}']):02d}" for i in range(1, 7)
            if pd.notna(row.get(f"n{i}"))
        )
        lines.append(
            f"| {int(row['target_concurso'])} | {row['target_date']} "
            f"| {row['draw_day'][:3]} | {int(row['seq_num'])} | {balls} |"
        )
    return "\n".join(lines)


def stats_markdown(stats: dict) -> str:
    if stats["n_draws"] == 0:
        return "_Ainda sem validações — aguardando primeiros sorteios._"

    lines = [
        f"- **Sorteios analisados:** {stats['n_draws']}",
        f"- **Sequências avaliadas:** {stats['n_predictions']}",
        f"- **Média de acertos por sequência:** {stats['avg_matches']:.3f}",
        f"- **Baseline aleatório:** {stats['random_baseline']:.3f}",
        f"- **vs Aleatório:** {stats['vs_random']:+.3f}",
        f"- **Melhor resultado:** {stats['best_matches']} acertos",
    ]
    if stats["prize_counts"]:
        for n_match, name in sorted(PRIZE_TIERS.items(), reverse=True):
            count = stats["prize_counts"].get(n_match, 0)
            if count > 0:
                lines.append(f"- **{name} ({n_match} acertos):** {count}×")
    return "\n".join(lines)
