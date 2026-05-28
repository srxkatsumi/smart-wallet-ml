"""
Mega Sena ML Experiment — weekly runner.

First run: backfills predictions for the last BACKFILL_WINDOW historical draws
           (using only past data for each prediction), then predicts Saturday.
Next runs: validates last predictions, retrains, predicts next draw.

Usage:
    python main.py            # normal run
    python main.py --force    # force re-download of results
"""
import sys
import logging
import argparse
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

np.random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────────

def _actual_balls(row) -> list[int]:
    return [int(row[f"b{i}"]) for i in range(1, 7)]


def _count_matches(predicted: list[int], actual: list[int]) -> int:
    return len(set(predicted) & set(actual))


def _prize(matches: int) -> str:
    from config import PRIZE_TIERS
    return PRIZE_TIERS.get(matches, "—")


def _next_saturday() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    days_until_sat = (5 - today.weekday()) % 7
    if days_until_sat == 0:
        days_until_sat = 7
    return today + pd.Timedelta(days=days_until_sat)


def _estimate_concurso(results: pd.DataFrame, target_date: pd.Timestamp) -> int:
    """Estimate the concurso number for a future date."""
    last_concurso = int(results["concurso"].iloc[-1])
    last_date     = pd.Timestamp(results["data"].iloc[-1])
    days_gap      = (target_date - last_date).days
    # Mega Sena: ~3 draws/week → 1 draw every ~2.33 days
    estimated     = last_concurso + max(1, round(days_gap / 2.33))
    return estimated


# ── Backfill ─────────────────────────────────────────────────────────────

def backfill_historical(results: pd.DataFrame, pred_df: pd.DataFrame,
                        weights: dict) -> pd.DataFrame:
    """
    For each historical draw in the backfill window:
      1. Train on all draws BEFORE that draw (no lookahead)
      2. Generate 5 sequences
      3. Validate immediately against the actual result

    Retrains every RETRAIN_INTERVAL draws to balance accuracy vs speed.
    """
    from features.engineering import build_training_data, build_prediction_features
    from models.ensemble     import train, predict_sequences
    from config              import MIN_DRAWS_TRAIN, BACKFILL_WINDOW, RETRAIN_INTERVAL

    known_concursos = set(pred_df["target_concurso"].dropna().astype(int).values)

    # Window: backfill the last BACKFILL_WINDOW draws
    start_idx = max(MIN_DRAWS_TRAIN, len(results) - BACKFILL_WINDOW)
    new_rows  = []
    models    = None

    logger.info("Backfill: concursos %d → %d (%d sorteios)",
                int(results.iloc[start_idx]["concurso"]),
                int(results.iloc[-1]["concurso"]),
                len(results) - start_idx)

    for i in range(start_idx, len(results)):
        concurso = int(results.iloc[i]["concurso"])
        if concurso in known_concursos:
            continue

        # Retrain every RETRAIN_INTERVAL draws
        if models is None or (i - start_idx) % RETRAIN_INTERVAL == 0:
            history_slice = results.iloc[:i]
            X, y = build_training_data(history_slice)
            models = train(X, y)
            logger.info("  Retreinado em concurso %d (treino: %d draws)", concurso, i)

        draw_row = results.iloc[i]
        draw_day = pd.Timestamp(draw_row["data"]).day_name()
        actual   = _actual_balls(draw_row)

        seqs = predict_sequences(results.iloc[:i], draw_day, models, weights)

        pred_date = pd.Timestamp(draw_row["data"]) - pd.Timedelta(days=1)

        for seq_num, seq in enumerate(seqs, 1):
            matches = _count_matches(seq, actual)
            new_rows.append({
                "prediction_date":  pred_date.strftime("%Y-%m-%d"),
                "target_concurso":  concurso,
                "target_date":      draw_row["data"].strftime("%Y-%m-%d")
                                    if hasattr(draw_row["data"], "strftime")
                                    else str(draw_row["data"])[:10],
                "draw_day":         draw_day,
                "seq_num":          seq_num,
                "n1": seq[0], "n2": seq[1], "n3": seq[2],
                "n4": seq[3], "n5": seq[4], "n6": seq[5],
                "matches":          matches,
                "prize":            _prize(matches),
                "acertou":          "Sim" if matches >= 1 else "Não",
                "acuracia":         round(matches / 6 * 100, 1),
                "validated":        True,
                "actual_n1": actual[0], "actual_n2": actual[1],
                "actual_n3": actual[2], "actual_n4": actual[3],
                "actual_n5": actual[4], "actual_n6": actual[5],
            })

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        pred_df = pd.concat([pred_df, new_df], ignore_index=True)
        logger.info("Backfill: %d novas linhas adicionadas", len(new_rows))

    return pred_df


# ── Validate pending ──────────────────────────────────────────────────────

def validate_pending(pred_df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    known = {int(r["concurso"]): r for _, r in results.iterrows()}
    updated = 0
    for idx, row in pred_df.iterrows():
        if row.get("validated") == True:
            continue
        c = int(row["target_concurso"]) if pd.notna(row.get("target_concurso")) else None
        if c not in known:
            continue
        actual  = _actual_balls(known[c])
        pred    = [int(row[f"n{i}"]) for i in range(1, 7)]
        matches = _count_matches(pred, actual)
        pred_df.at[idx, "matches"]  = matches
        pred_df.at[idx, "prize"]    = _prize(matches)
        pred_df.at[idx, "acertou"]  = "Sim" if matches >= 1 else "Não"
        pred_df.at[idx, "acuracia"] = round(matches / 6 * 100, 1)
        pred_df.at[idx, "validated"] = True
        for i, n in enumerate(actual, 1):
            pred_df.at[idx, f"actual_n{i}"] = n
        updated += 1
    if updated:
        logger.info("%d previsões validadas", updated)
    return pred_df


# ── Predict next Saturday ─────────────────────────────────────────────────

def predict_saturday(results: pd.DataFrame, models, weights: dict,
                     pred_df: pd.DataFrame) -> pd.DataFrame:
    from models.ensemble import predict_sequences
    from config          import N_SEQUENCES

    saturday    = _next_saturday()
    concurso    = _estimate_concurso(results, saturday)
    draw_day    = "Saturday"
    today_str   = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Don't duplicate if we already predicted this concurso
    if not pred_df.empty and concurso in pred_df["target_concurso"].values:
        logger.info("Previsão para concurso %d já existe — ignorando", concurso)
        return pred_df

    seqs = predict_sequences(results, draw_day, models, weights)
    new_rows = []
    for seq_num, seq in enumerate(seqs, 1):
        new_rows.append({
            "prediction_date":  today_str,
            "target_concurso":  concurso,
            "target_date":      saturday.strftime("%Y-%m-%d"),
            "draw_day":         draw_day,
            "seq_num":          seq_num,
            "n1": seq[0], "n2": seq[1], "n3": seq[2],
            "n4": seq[3], "n5": seq[4], "n6": seq[5],
            "matches":   None, "prize": "⏳", "acertou": "⏳ Pendente",
            "acuracia":  None, "validated": False,
            "actual_n1": None, "actual_n2": None, "actual_n3": None,
            "actual_n4": None, "actual_n5": None, "actual_n6": None,
        })

    seqs_str = " | ".join(f"[{' '.join(f'{n:02d}' for n in s)}]" for s in seqs)
    logger.info("Previsão Sábado %s (concurso ~%d): %s",
                saturday.strftime("%d/%m/%Y"), concurso, seqs_str)

    pred_df = pd.concat([pred_df, pd.DataFrame(new_rows)], ignore_index=True)
    return pred_df


# ── Main ─────────────────────────────────────────────────────────────────

def main(force_download: bool = False):
    logger.info("=== Mega Sena ML Experiment — %s ===", date.today())

    from data.storage    import ensure_dirs, load_predictions, save_predictions
    from data.storage    import load_weights, save_weights
    from data.downloader import download_results
    from features.engineering import build_training_data
    from models.ensemble import train, update_weights
    from reports.generate_md import save_md
    from config import RESULTS_FILE, MIN_DRAWS_TRAIN

    ensure_dirs()

    # 1 — Download results
    results = download_results(RESULTS_FILE, force=force_download)
    if len(results) < MIN_DRAWS_TRAIN:
        logger.error("Dados insuficientes (%d draws)", len(results))
        return

    # 2 — Load predictions + validate any pending
    pred_df = load_predictions()
    pred_df = validate_pending(pred_df, results)

    # 3 — Backfill historical predictions (only runs for new draws)
    weights = load_weights()
    pred_df = backfill_historical(results, pred_df, weights)
    save_predictions(pred_df)

    # 4 — Update weights from validated history
    weights = update_weights(pred_df, weights)
    save_weights(weights)

    # 5 — Train on full history and predict next Saturday
    logger.info("Treinando modelo final sobre %d sorteios...", len(results))
    X, y = build_training_data(results)
    models = train(X, y)

    pred_df = predict_saturday(results, models, weights, pred_df)
    save_predictions(pred_df)

    # 6 — Generate output .md
    save_md(pred_df, results)

    logger.info("=== Concluído ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force_download=args.force)
