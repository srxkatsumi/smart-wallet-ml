"""
ContextGate — Porteiro Contextual de Ensemble
Fase 4: implementação MVP para validação experimental na Mega Sena.

Lógica:
  Para cada sorteio, extrai 8 features de contexto (discordância entre
  modelos, confiança do ensemble, acurácia recente e dia da semana).
  Um Ridge regressor aprende a prever a qualidade esperada da previsão
  (matches/6). Na inferência, converte a qualidade prevista em pesos
  adaptativos para os modelos base.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

GATE_MIN_SAMPLES  = 30
GATE_RECENT_WIN   = 10
BALLS_PER_DRAW    = 6


def extract_gate_features(
    p_rf: np.ndarray,
    p_gb: np.ndarray,
    p_sgd: np.ndarray,
    recent_matches: list,
    draw_day: str,
) -> np.ndarray:
    """
    Constrói o vetor de 8 features de contexto para o ContextGate.

    Features:
      d_mean     — discordância média entre os 3 modelos (std das probabilidades)
      d_max      — pior discordância (bola onde os modelos mais discordam)
      conf_max   — confiança média do modelo mais confiante por bola
      acc_rec    — acurácia normalizada do ensemble nos últimos N sorteios
      acc_trend  — tendência: segunda metade da janela vs primeira metade
      day_mon, day_thu, day_sat — dummies do dia do sorteio
    """
    stacked = np.vstack([p_rf, p_gb, p_sgd])   # (3, 60)
    d_per_ball = stacked.std(axis=0)            # (60,)

    d_mean   = float(d_per_ball.mean())
    d_max    = float(d_per_ball.max())
    conf_max = float(stacked.max(axis=0).mean())

    if len(recent_matches) == 0:
        acc_rec   = BALLS_PER_DRAW / 60.0  # baseline aleatório teórico
        acc_trend = 0.0
    else:
        arr = np.array(recent_matches[-GATE_RECENT_WIN:], dtype=float)
        acc_rec = float(arr.mean()) / BALLS_PER_DRAW
        if len(arr) >= 4:
            half = len(arr) // 2
            acc_trend = float(arr[half:].mean() - arr[:half].mean()) / BALLS_PER_DRAW
        else:
            acc_trend = 0.0

    day_mon = float(draw_day == "Monday")
    day_thu = float(draw_day == "Thursday")
    day_sat = float(draw_day == "Saturday")

    return np.array([d_mean, d_max, conf_max, acc_rec, acc_trend,
                     day_mon, day_thu, day_sat], dtype=float)


def train_gate(
    gate_X: np.ndarray,
    gate_y: np.ndarray,
) -> dict:
    """
    Treina o ContextGate como Ridge regressor.

    gate_X : (N, 8)  — features de contexto por sorteio
    gate_y : (N,)    — matches / BALLS_PER_DRAW, normalizado em [0, 1]

    Retorna dict com modelo e scaler para serialização.
    """
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(gate_X)
    model  = Ridge(alpha=1.0)
    model.fit(X_sc, gate_y)
    return {"model": model, "scaler": scaler}


def predict_weights(
    gate_model: dict,
    p_rf: np.ndarray,
    p_gb: np.ndarray,
    p_sgd: np.ndarray,
    recent_matches: list,
    draw_day: str,
) -> dict:
    """
    Usa o ContextGate para calcular pesos contextuais.

    Estratégia de conversão confiança → pesos:
      - Confiança alta (prevê qualidade acima do random baseline):
          RF recebe mais peso (modelo mais expressivo, ganha mais em contextos ricos)
      - Confiança baixa (contexto ambíguo):
          Equaliza os pesos (hedging — 1/3 cada)

    Fallback: se gate_model is None → pesos iguais.
    """
    if gate_model is None:
        return {"rf": 1/3, "gb": 1/3, "sgd": 1/3}

    feat    = extract_gate_features(p_rf, p_gb, p_sgd, recent_matches, draw_day)
    feat_sc = gate_model["scaler"].transform(feat.reshape(1, -1))
    conf    = float(gate_model["model"].predict(feat_sc)[0])

    # Normaliza confiança: 0.0 = baseline aleatório, 1.0 = perfeito
    random_baseline = BALLS_PER_DRAW / 60.0  # ≈ 0.1
    alpha = np.clip((conf - random_baseline) / (1.0 - random_baseline), 0.0, 1.0)

    # alpha=0 → pesos iguais; alpha=1 → RF dominante
    w_rf  = 1/3 + alpha * (2/3 - 1/3)   # varia de 0.33 a 0.67
    w_gb  = 1/3 - alpha * (1/3 - 0.20)  # varia de 0.33 a 0.20
    w_sgd = 1/3 - alpha * (1/3 - 0.13)  # varia de 0.33 a 0.13

    total = w_rf + w_gb + w_sgd
    return {
        "rf":  round(w_rf  / total, 4),
        "gb":  round(w_gb  / total, 4),
        "sgd": round(w_sgd / total, 4),
    }
