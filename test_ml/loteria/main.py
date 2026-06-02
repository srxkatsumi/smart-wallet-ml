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


_DRAW_WEEKDAYS = {0: "Monday", 3: "Thursday", 5: "Saturday"}
_RNG = np.random.RandomState(42)


def _compute_baselines(appeared_mat: np.ndarray, i: int,
                       prev_numbers: np.ndarray) -> dict:
    """Return {strategy_name: [6 numbers 1-indexed]} for hot, cold, random."""
    hist = appeared_mat[:i].astype(float)
    freq = hist.mean(axis=0)  # (60,) long-term frequency per number

    hot_idx  = np.argsort(freq)[-6:]
    cold_idx = np.argsort(freq)[:6]
    rand_idx = _RNG.choice(60, size=6, replace=False)

    return {
        "hot":    sorted((hot_idx  + 1).tolist()),
        "cold":   sorted((cold_idx + 1).tolist()),
        "random": sorted((rand_idx + 1).tolist()),
    }


def _next_draw_dates(n: int = 3) -> list[tuple[pd.Timestamp, str]]:
    """Return the next n upcoming Mega Sena draw dates as (date, day_name) tuples."""
    today = pd.Timestamp.today().normalize()
    result = []
    delta = 1
    while len(result) < n:
        candidate = today + pd.Timedelta(days=delta)
        if candidate.weekday() in _DRAW_WEEKDAYS:
            result.append((candidate, _DRAW_WEEKDAYS[candidate.weekday()]))
        delta += 1
    return result


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
                        weights: dict) -> tuple[pd.DataFrame, bool, pd.DataFrame]:
    """
    Walk-forward backfill of all historical draws not yet in pred_df.
    Processes at most DAILY_BATCH_SIZE new draws per run so each daily
    execution adds one batch and the full history fills in over ~10 days.

    Returns (updated_pred_df, backfill_complete, new_baselines_df).
    """
    from features.engineering import build_training_data, build_prediction_features, _build_appeared_matrix
    from models.ensemble     import train, predict_sequences
    from config              import MIN_DRAWS_TRAIN, DAILY_BATCH_SIZE, RETRAIN_INTERVAL
    from data.storage        import BASELINE_COLS

    known_concursos = set(pred_df["target_concurso"].dropna().astype(int).values)

    appeared_mat = _build_appeared_matrix(results)
    balls_vals   = results[["b1", "b2", "b3", "b4", "b5", "b6"]].values.astype(int)

    # Find first draw not yet processed (chronological order, skip early ones)
    start_idx = MIN_DRAWS_TRAIN
    new_rows      = []
    baseline_rows = []
    models    = None
    processed = 0
    last_train_idx = -RETRAIN_INTERVAL  # force first retrain

    total_pending = sum(
        1 for i in range(start_idx, len(results))
        if int(results.iloc[i]["concurso"]) not in known_concursos
    )
    if total_pending == 0:
        logger.info("Backfill completo — nenhum sorteio histórico pendente")
        return pred_df, True

    logger.info("Backfill: %d sorteios históricos pendentes (batch=%d)",
                total_pending, DAILY_BATCH_SIZE)

    for i in range(start_idx, len(results)):
        if processed >= DAILY_BATCH_SIZE:
            break

        concurso = int(results.iloc[i]["concurso"])
        if concurso in known_concursos:
            continue

        # Retrain every RETRAIN_INTERVAL new draws processed
        if models is None or (i - last_train_idx) >= RETRAIN_INTERVAL:
            history_slice = results.iloc[:i]
            X, y = build_training_data(history_slice)
            models = train(X, y)
            last_train_idx = i
            logger.info("  Retreinado em concurso %d (treino: %d draws)", concurso, i)

        draw_row  = results.iloc[i]
        draw_day  = pd.Timestamp(draw_row["data"]).day_name()
        actual    = _actual_balls(draw_row)
        seqs      = predict_sequences(results.iloc[:i], draw_day, models, weights)
        pred_date = pd.Timestamp(draw_row["data"]) - pd.Timedelta(days=1)
        target_date_str = (draw_row["data"].strftime("%Y-%m-%d")
                           if hasattr(draw_row["data"], "strftime")
                           else str(draw_row["data"])[:10])

        for seq_num, seq in enumerate(seqs, 1):
            matches = _count_matches(seq, actual)
            new_rows.append({
                "prediction_date":  pred_date.strftime("%Y-%m-%d"),
                "target_concurso":  concurso,
                "target_date":      target_date_str,
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

        # Naive baselines for the same draw
        prev_numbers = balls_vals[i - 1]
        baselines = _compute_baselines(appeared_mat, i, prev_numbers)
        for strategy, seq in baselines.items():
            matches = _count_matches(seq, actual)
            baseline_rows.append({
                "prediction_date":  pred_date.strftime("%Y-%m-%d"),
                "target_concurso":  concurso,
                "target_date":      target_date_str,
                "draw_day":         draw_day,
                "seq_num":          1,
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
                "strategy":         strategy,
            })
        processed += 1

    remaining    = total_pending - processed
    new_baselines = pd.DataFrame(baseline_rows, columns=BASELINE_COLS) if baseline_rows else pd.DataFrame(columns=BASELINE_COLS)
    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        pred_df = pd.concat([pred_df, new_df], ignore_index=True)
        logger.info("Backfill: +%d linhas | restam ~%d sorteios históricos",
                    len(new_rows), remaining)

    return pred_df, remaining == 0, new_baselines


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

def predict_upcoming_draws(results: pd.DataFrame, models, weights: dict,
                           pred_df: pd.DataFrame) -> pd.DataFrame:
    """Predict the next 3 upcoming draws (Mon/Thu/Sat) if not already predicted."""
    from models.ensemble import predict_sequences

    today_str  = pd.Timestamp.today().strftime("%Y-%m-%d")
    new_rows   = []
    added      = 0

    for target_date, draw_day in _next_draw_dates(n=3):
        concurso = _estimate_concurso(results, target_date)

        if not pred_df.empty and concurso in pred_df["target_concurso"].values:
            logger.info("Concurso ~%d (%s) já previsto — ignorando",
                        concurso, target_date.strftime("%d/%m/%Y"))
            continue

        seqs = predict_sequences(results, draw_day, models, weights)
        for seq_num, seq in enumerate(seqs, 1):
            new_rows.append({
                "prediction_date":  today_str,
                "target_concurso":  concurso,
                "target_date":      target_date.strftime("%Y-%m-%d"),
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
        logger.info("Previsão %s %s (concurso ~%d): %s",
                    draw_day, target_date.strftime("%d/%m/%Y"), concurso, seqs_str)
        added += 1

    if new_rows:
        pred_df = pd.concat([pred_df, pd.DataFrame(new_rows)], ignore_index=True)
        logger.info("%d sorteios futuros previstos", added)
    return pred_df


# ── Main ─────────────────────────────────────────────────────────────────

def main(force_download: bool = False):
    logger.info("=== Mega Sena ML Experiment — %s ===", date.today())

    from data.storage    import ensure_dirs, load_predictions, save_predictions
    from data.storage    import load_weights, save_weights, load_baselines, save_baselines
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

    # 3 — Backfill histórico: 300 draws/dia até cobrir todo o histórico
    weights = load_weights()
    baselines_df = load_baselines()
    pred_df, backfill_done, new_baselines = backfill_historical(results, pred_df, weights)
    save_predictions(pred_df)
    if not new_baselines.empty:
        baselines_df = pd.concat([baselines_df, new_baselines], ignore_index=True)
        save_baselines(baselines_df)

    # 4 — Actualizar pesos com base no histórico validado (todos os dias)
    weights = update_weights(pred_df, weights)
    save_weights(weights)

    # 5 — Previsões futuras: APENAS às segundas-feiras
    is_monday = pd.Timestamp.today().weekday() == 0
    if is_monday:
        logger.info("Segunda-feira — treinando modelo final e prevendo próximos sorteios...")
        X, y = build_training_data(results)
        models = train(X, y)
        pred_df = predict_upcoming_draws(results, models, weights, pred_df)
        save_predictions(pred_df)
    else:
        logger.info("Não é segunda-feira — só backfill hoje (previsões futuras apenas às segundas)")

    if not backfill_done:
        logger.info("Backfill em progresso — retoma amanhã")

    # 6 — Generate output .md
    save_md(pred_df, results)

    logger.info("=== Concluído ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force_download=args.force)
