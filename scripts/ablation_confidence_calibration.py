"""
Ablation: a "confidence" bruta do ensemble carrega informação real sobre
acerto, em produção? (ERR-013)

Motivação: models/ensemble.py já calibra RF e GB individualmente com
CalibratedClassifierCV(method="isotonic"), mas o SGD não é calibrado, e a
confiança final reportada é a média ponderada dos três — média de
probabilidades já calibradas não preserva calibração do conjunto. Este script
mede, sobre todo o histórico já validado em output/predictions_log.csv, se
isso importa na prática: calcula o Brier score real por horizonte e a
acurácia por decil de confiança reportada.

Resultado (2026-11-05, n=48802/48786/47718 para D+1/D+2/D+3): Brier score de
0.2855-0.2921 nos três horizontes — pior do que simplesmente prever 0.5
constante (Brier=0.25) — e acurácia por decil oscila entre 47.6% e 52.5% sem
tendência crescente. A confiança reportada hoje não separa certo de errado.
models/calibration.py implementa a correção: uma curva isotônica pós-hoc
ajustada contra este mesmo histórico, reportada numa coluna nova
(confidence_calibrated) lado a lado com a bruta.

Usage: python scripts/ablation_confidence_calibration.py
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics import brier_score_loss

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PRED_LOG


def main():
    df = pd.read_csv(PRED_LOG)
    validated = df[df["correct"].notna()].copy()
    validated["correct"] = validated["correct"].astype(float)

    print("=== Brier score por horizonte (0.25 = equivalente a prever sempre 50%) ===")
    for day in [1, 2, 3]:
        sub = validated[validated["horizon"] == day]
        if sub.empty:
            continue
        brier = brier_score_loss(sub["correct"], sub["confidence"])
        pior_que_baseline = "pior que 0.5 constante" if brier > 0.25 else "melhor que 0.5 constante"
        print(f"D+{day}: n={len(sub):>6}  Brier={brier:.4f}  ({pior_que_baseline})")

    print("\n=== Acurácia real por decil de confiança reportada ===")
    for day in [1, 2, 3]:
        sub = validated[validated["horizon"] == day].copy()
        if sub.empty:
            continue
        sub["decil"] = pd.qcut(sub["confidence"], 10, duplicates="drop")
        rel = sub.groupby("decil", observed=True)["correct"].agg(["mean", "count"])
        print(f"\n--- D+{day} ---")
        print(rel.round(3).to_string())


if __name__ == "__main__":
    main()
