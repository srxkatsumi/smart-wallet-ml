"""
Mega Sena ML Experiment — weekly runner.

Purpose: apply adaptive ML to lottery prediction and measure whether
         it can beat random chance. Spoiler: it cannot. That is the point.

Usage:
    python main.py            # normal weekly run
    python main.py --force    # force re-download of results
"""
import sys
import logging
import argparse
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

# Allow running from this directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

np.random.seed(42)


def _next_draw_dates() -> list[dict]:
    """Return the next 3 draw dates (Mon/Thu/Sat) from today."""
    today = pd.Timestamp.today().normalize()
    day_map = {"Monday": 0, "Thursday": 3, "Saturday": 5}
    draws = []
    for day_name, weekday in day_map.items():
        delta = (weekday - today.weekday()) % 7
        if delta == 0:
            delta = 7   # if today IS the draw day, predict next week's
        draws.append({"day": day_name, "date": today + pd.Timedelta(days=delta)})
    return sorted(draws, key=lambda x: x["date"])


def _validate_predictions(pred_df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Fill in actual results for predictions whose target_concurso has been drawn."""
    from models.ensemble import count_matches

    ball_cols_pred   = ["n1", "n2", "n3", "n4", "n5", "n6"]
    ball_cols_actual = ["b1", "b2", "b3", "b4", "b5", "b6"]
    known_concursos  = set(results["concurso"].values)

    updated = 0
    for idx, row in pred_df.iterrows():
        if row.get("validated") == True:
            continue
        if int(row["target_concurso"]) not in known_concursos:
            continue

        actual_row = results[results["concurso"] == int(row["target_concurso"])].iloc[0]
        actual     = [int(actual_row[c]) for c in ball_cols_actual]
        predicted  = [int(row[c]) for c in ball_cols_pred]
        matches    = count_matches(predicted, actual)

        from config import PRIZE_TIERS
        best_prize = next((PRIZE_TIERS[m] for m in sorted(PRIZE_TIERS.keys(), reverse=True)
                           if m <= matches), "—")

        for i, n in enumerate(actual, 1):
            pred_df.at[idx, f"actual_n{i}"] = n
        pred_df.at[idx, "matches"]    = matches
        pred_df.at[idx, "best_prize"] = best_prize
        pred_df.at[idx, "validated"]  = True
        updated += 1

    if updated:
        logger.info("%d previsões validadas", updated)
    return pred_df


def _update_readme(pred_df: pd.DataFrame, results: pd.DataFrame):
    """Regenerate READMEs with latest results and predictions."""
    from reports.summary import (
        last_week_table, accuracy_summary, next_predictions_table, stats_markdown
    )
    from config import BASE_DIR

    stats       = accuracy_summary(pred_df)
    last_week   = last_week_table(results, n=7)
    next_preds  = next_predictions_table(pred_df)
    stats_md    = stats_markdown(stats)
    last_concurso = int(results["concurso"].iloc[-1])
    last_date     = pd.to_datetime(results["data"].iloc[-1]).strftime("%d/%m/%Y")

    for lang in ["en", "pt"]:
        path = BASE_DIR / ("README.md" if lang == "en" else "README_pt.md")
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")

        def replace_block(marker_start, marker_end, new_content, t):
            start = t.find(marker_start)
            end   = t.find(marker_end)
            if start == -1 or end == -1:
                return t
            return t[:start + len(marker_start)] + "\n" + new_content + "\n" + t[end:]

        text = replace_block(
            "<!-- LAST_WEEK_START -->", "<!-- LAST_WEEK_END -->", last_week, text
        )
        text = replace_block(
            "<!-- NEXT_PREDS_START -->", "<!-- NEXT_PREDS_END -->", next_preds, text
        )
        text = replace_block(
            "<!-- STATS_START -->", "<!-- STATS_END -->", stats_md, text
        )
        text = text.replace("{{LAST_CONCURSO}}", str(last_concurso))
        text = text.replace("{{LAST_DATE}}", last_date)
        path.write_text(text, encoding="utf-8")
    logger.info("READMEs atualizados")


def main(force_download: bool = False):
    logger.info("=== Mega Sena ML Experiment — %s ===", date.today())

    from data.storage  import ensure_dirs, load_predictions, save_predictions, load_weights, save_weights
    from data.downloader import download_results
    from features.engineering import build_training_data
    from models.ensemble import train, predict_sequences, update_weights
    from config import RESULTS_FILE, MIN_DRAWS_TRAIN, DRAW_DAYS

    ensure_dirs()

    # 1 — Download / refresh results
    results = download_results(RESULTS_FILE, force=force_download)
    if len(results) < MIN_DRAWS_TRAIN:
        logger.error("Dados insuficientes para treino (%d < %d)", len(results), MIN_DRAWS_TRAIN)
        return

    # 2 — Load predictions and validate past ones
    pred_df  = load_predictions()
    pred_df  = _validate_predictions(pred_df, results)

    # 3 — Update ensemble weights based on validated predictions
    weights  = load_weights()
    weights  = update_weights(pred_df, weights)
    save_weights(weights)

    # 4 — Train models on full history
    logger.info("Treinando modelos sobre %d sorteios...", len(results))
    X, y = build_training_data(results)
    models = train(X, y)
    logger.info("Treino concluído — %d exemplos (%d positivos)", len(y), y.sum())

    # 5 — Predict for next Mon / Thu / Sat
    today = pd.Timestamp.today().normalize()
    new_rows = []
    for draw in _next_draw_dates():
        draw_day  = draw["day"]
        draw_date = draw["date"]

        # Estimate concurso number (Mega Sena started on concurso 1 in 1996)
        last_concurso = int(results["concurso"].iloc[-1])
        last_date     = pd.Timestamp(results["data"].iloc[-1])
        draws_per_week = 3
        days_gap  = (draw_date - last_date).days
        est_concurso = last_concurso + max(1, round(days_gap / 7 * draws_per_week))

        seqs = predict_sequences(results, draw_day, models, weights)

        for seq_num, seq in enumerate(seqs, 1):
            new_rows.append({
                "prediction_date":  today.strftime("%Y-%m-%d"),
                "target_concurso":  est_concurso,
                "target_date":      draw_date.strftime("%Y-%m-%d"),
                "draw_day":         draw_day,
                "seq_num":          seq_num,
                "n1": seq[0], "n2": seq[1], "n3": seq[2],
                "n4": seq[3], "n5": seq[4], "n6": seq[5],
                "matches":    None,
                "best_prize": None,
                "validated":  False,
                "actual_n1": None, "actual_n2": None, "actual_n3": None,
                "actual_n4": None, "actual_n5": None, "actual_n6": None,
            })

        seqs_str = " | ".join(f"[{' '.join(f'{n:02d}' for n in s)}]" for s in seqs)
        logger.info("Previsões %s %s: %s", draw_day, draw_date.strftime("%d/%m/%Y"), seqs_str)

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        pred_df = pd.concat([pred_df, new_df], ignore_index=True)
        save_predictions(pred_df)

    # 6 — Update READMEs
    _update_readme(pred_df, results)

    logger.info("=== Concluído ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    main(force_download=args.force)
