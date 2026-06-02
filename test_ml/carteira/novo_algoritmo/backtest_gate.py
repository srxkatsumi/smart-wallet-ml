"""
Backtest Walk-Forward — ContextGate vs Static Weights vs Random Baseline
Mega Sena Experiment

Split:
  Treino base   : draws 30 → 2699  (2670 draws)
  Gate train    : draws 2700 → 2799 (100 draws — extrai features + treina gate)
  Teste         : draws 2800 → 2999 (200 draws — avaliação final)

Comparação:
  1. Baseline Aleatório — E[matches] = 6 × (6/60) = 0.6
  2. Hot Numbers — top-6 bolas mais frequentes nos últimos 50 sorteios
  3. Ensemble Estático — pesos iguais 1/3, 1/3, 1/3
  4. ContextGate — pesos adaptativos por sorteio

Execução:
  python backtest_gate.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
PROJ    = ROOT.parent.parent / "loteria"
sys.path.insert(0, str(PROJ))

from features.engineering import build_training_data, _draw_features, _build_appeared_matrix
from context_gate import (
    extract_gate_features, train_gate, predict_weights, GATE_MIN_SAMPLES
)

RESULTS_FILE = PROJ / "output" / "mega_sena_results.csv"
N_BALLS      = 60
BALLS        = 6
TRAIN_END    = 2700
GATE_END     = 2800
TEST_END     = 3000
BALL_COLS    = ["b1", "b2", "b3", "b4", "b5", "b6"]

np.random.seed(42)


def _top6(probs: np.ndarray) -> set:
    return set(np.argsort(probs)[::-1][:BALLS] + 1)


def _matches(pred: set, actual: set) -> int:
    return len(pred & actual)


def _hot_numbers(appeared_mat: np.ndarray, i: int, window: int = 50) -> set:
    hist = appeared_mat[max(0, i - window):i]
    freq = hist.mean(axis=0)
    return set(np.argsort(freq)[::-1][:BALLS] + 1)


def run_backtest():
    print("=== Backtest ContextGate vs Baselines — Mega Sena ===\n")

    results = pd.read_csv(RESULTS_FILE)
    results["data"] = pd.to_datetime(results["data"])
    print(f"Sorteios carregados: {len(results)}")

    # ── 1. Treinar modelos base no split de treino ────────────────────────
    train_slice = results.iloc[:TRAIN_END]
    print(f"\nTreinando modelos base em {len(train_slice)} sorteios...", end=" ")
    X_tr, y_tr = build_training_data(train_slice)

    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X_tr)

    rf  = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, class_weight="balanced")
    gb  = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.001, max_iter=1000, random_state=42, class_weight="balanced")

    rf.fit(X_sc, y_tr)
    gb.fit(X_sc, y_tr)
    sgd.fit(X_sc, y_tr)
    print("OK")

    # ── 2. Extrair features e targets para treino do gate ─────────────────
    appeared_mat = _build_appeared_matrix(results)
    balls_vals   = results[BALL_COLS].values.astype(int)

    print(f"\nColetando dados de treino do gate ({TRAIN_END}→{GATE_END})...", end=" ")
    gate_X_list, gate_y_list = [], []
    recent_matches_buffer    = []

    for i in range(TRAIN_END, GATE_END):
        draw_day    = results.iloc[i]["data"].day_name()
        prev_nums   = balls_vals[i - 1]
        X_draw      = _draw_features(appeared_mat, i, draw_day, prev_nums)  # (60, F)
        X_draw_sc   = scaler.transform(X_draw)

        p_rf  = rf.predict_proba(X_draw_sc)[:, 1]
        p_gb  = gb.predict_proba(X_draw_sc)[:, 1]
        p_sgd = sgd.predict_proba(X_draw_sc)[:, 1]

        feat = extract_gate_features(p_rf, p_gb, p_sgd, recent_matches_buffer, draw_day)
        gate_X_list.append(feat)

        actual  = set(balls_vals[i])
        p_ens   = (p_rf + p_gb + p_sgd) / 3.0
        m       = _matches(_top6(p_ens), actual)
        gate_y_list.append(m / BALLS)
        recent_matches_buffer.append(m)

    gate_X = np.vstack(gate_X_list)
    gate_y = np.array(gate_y_list)
    print(f"OK ({len(gate_X)} amostras, acc_media={gate_y.mean():.3f})")

    # ── 3. Treinar o ContextGate ──────────────────────────────────────────
    if len(gate_X) >= GATE_MIN_SAMPLES:
        gate_model = train_gate(gate_X, gate_y)
        print(f"ContextGate treinado — Ridge coeficientes: {gate_model['model'].coef_.round(3)}")
    else:
        gate_model = None
        print("AVISO: amostras insuficientes para treino do gate")

    # ── 4. Backtest no split de teste ─────────────────────────────────────
    print(f"\nBacktest ({GATE_END}→{min(TEST_END, len(results))})...")

    results_list = []
    weights_log  = []

    for i in range(GATE_END, min(TEST_END, len(results))):
        draw_day  = results.iloc[i]["data"].day_name()
        prev_nums = balls_vals[i - 1]
        actual    = set(balls_vals[i])

        X_draw    = _draw_features(appeared_mat, i, draw_day, prev_nums)
        X_draw_sc = scaler.transform(X_draw)

        p_rf  = rf.predict_proba(X_draw_sc)[:, 1]
        p_gb  = gb.predict_proba(X_draw_sc)[:, 1]
        p_sgd = sgd.predict_proba(X_draw_sc)[:, 1]

        # Static ensemble (pesos iguais)
        p_static = (p_rf + p_gb + p_sgd) / 3.0
        m_static = _matches(_top6(p_static), actual)

        # ContextGate
        w = predict_weights(gate_model, p_rf, p_gb, p_sgd, recent_matches_buffer, draw_day)
        p_gate = p_rf * w["rf"] + p_gb * w["gb"] + p_sgd * w["sgd"]
        m_gate = _matches(_top6(p_gate), actual)

        # Hot numbers baseline
        m_hot  = _matches(_hot_numbers(appeared_mat, i), actual)

        results_list.append({
            "draw":     i,
            "day":      draw_day,
            "m_static": m_static,
            "m_gate":   m_gate,
            "m_hot":    m_hot,
            "w_rf":     w["rf"],
            "w_gb":     w["gb"],
            "w_sgd":    w["sgd"],
        })
        recent_matches_buffer.append(m_gate)

    df = pd.DataFrame(results_list)

    # ── 5. Resultados ─────────────────────────────────────────────────────
    random_baseline = BALLS * (BALLS / N_BALLS)

    print("\n" + "="*55)
    print(f"{'Método':<22} {'Média matches':>14} {'≥1 match %':>12} {'≥2 matches %':>12}")
    print("-"*55)

    for name, col in [
        ("Aleatório (teórico)", None),
        ("Hot Numbers",        "m_hot"),
        ("Ensemble Estático",  "m_static"),
        ("ContextGate",        "m_gate"),
    ]:
        if col is None:
            mean_m = random_baseline
            pct1   = (1 - ((54/60)*(53/59)*(52/58)*(51/57)*(50/56)*(49/55))) * 100
            pct2   = float("nan")
            print(f"{name:<22} {mean_m:>14.4f} {pct1:>11.1f}% {'—':>12}")
        else:
            mean_m = df[col].mean()
            pct1   = (df[col] >= 1).mean() * 100
            pct2   = (df[col] >= 2).mean() * 100
            print(f"{name:<22} {mean_m:>14.4f} {pct1:>11.1f}% {pct2:>11.1f}%")

    print("="*55)

    delta = df["m_gate"].mean() - df["m_static"].mean()
    print(f"\nContextGate vs Estático: {delta:+.4f} matches/sorteio")
    print(f"Meta da Fase 3: +0.10 — {'ATINGIDA ✓' if delta >= 0.10 else f'não atingida (delta={delta:.4f})'}")

    # ── 6. Verificar se os pesos realmente variam ─────────────────────────
    w_std = df[["w_rf", "w_gb", "w_sgd"]].std()
    print(f"\nVariação dos pesos ao longo do tempo (std):")
    print(f"  RF:  {w_std['w_rf']:.5f}  GB: {w_std['w_gb']:.5f}  SGD: {w_std['w_sgd']:.5f}")
    print(f"  Gate adaptou pesos? {'SIM ✓' if w_std.max() > 0.05 else 'NÃO — pesos quase estáticos'}")

    print(f"\nPesos médios no período de teste:")
    print(f"  RF={df['w_rf'].mean():.3f}  GB={df['w_gb'].mean():.3f}  SGD={df['w_sgd'].mean():.3f}")

    print(f"\nDistribuição de matches — ContextGate vs Estático:")
    for m in range(4):
        n_gate   = (df["m_gate"]   == m).sum()
        n_static = (df["m_static"] == m).sum()
        print(f"  {m} acertos: Gate={n_gate:3d}  Estático={n_static:3d}")

    return df


if __name__ == "__main__":
    df = run_backtest()
