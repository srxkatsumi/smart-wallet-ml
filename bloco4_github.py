# ══════════════════════════════════════════════════════════════
# BLOCO 4 — Caminhos (GitHub Actions / Local)
#
# Quando corre no GitHub Actions, os ficheiros estão
# directamente na pasta do repositório clonado.
# Quando corres localmente, usa o mesmo caminho relativo.
# ══════════════════════════════════════════════════════════════

import os
from pathlib import Path

# Detecta se está a correr no GitHub Actions ou localmente
GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
COLAB          = False   # Não usa Drive — ficheiros ficam no repositório

# Caminho base: pasta AnaliseV5/ dentro do repositório
# No GitHub Actions: /home/runner/work/<repo>/<repo>/AnaliseV5/
# Localmente:        ./AnaliseV5/
LOG_DIR      = Path("AnaliseV5")
GRAFICOS_DIR = LOG_DIR / "AnaliseGraficos"

LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

PRED_LOG  = LOG_DIR / "predictions_log.csv"
WEIGHTS_F = LOG_DIR / "ensemble_weights.json"

PRED_COLS = [
    "ticker", "pred_date", "target_date", "horizon",
    "direction", "pred_price", "confidence",
    "actual_price", "correct",
    "model_rf", "model_gb", "model_lr"
]

if not PRED_LOG.exists():
    pd.DataFrame(columns=PRED_COLS).to_csv(PRED_LOG, index=False)
    print(f"✅ CSV criado: {PRED_LOG}")
else:
    n = len(pd.read_csv(PRED_LOG))
    print(f"✅ CSV carregado: {n} registos existentes")

# Pesos por horizonte
DEFAULT_WEIGHTS = {
    "d1": {"rf": 1.0, "gb": 1.0, "lr": 1.0},
    "d2": {"rf": 1.0, "gb": 1.0, "lr": 1.0},
    "d3": {"rf": 1.0, "gb": 1.0, "lr": 1.0},
}

if WEIGHTS_F.exists():
    with open(WEIGHTS_F) as f:
        ensemble_weights = json.load(f)
    if "d1" not in ensemble_weights:
        ensemble_weights = DEFAULT_WEIGHTS.copy()
    print(f"✅ Pesos carregados:")
    for dk, dw in ensemble_weights.items():
        print(f"   {dk}: RF={dw['rf']:.2f}  GB={dw['gb']:.2f}  LR={dw['lr']:.2f}")
else:
    ensemble_weights = DEFAULT_WEIGHTS.copy()
    with open(WEIGHTS_F, 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    print(f"✅ Pesos inicializados (iguais por horizonte)")

env = "GitHub Actions" if GITHUB_ACTIONS else "Local"
print(f"\n✅ Ambiente: {env}")
print(f"   Dados:    {LOG_DIR.resolve()}")
print(f"   Gráficos: {GRAFICOS_DIR.resolve()}")
