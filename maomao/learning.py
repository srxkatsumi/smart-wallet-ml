"""
Orquestra a recompensa de três fatores do brain.Brain: quando um pool
plástico (FEARFUL ou EUPHORIC) dispara forte num dia, isso entra numa fila
de "apostas" pendentes. Só quando dias reais suficientes já passaram (e o
retorno real deles já foi validado pelo pipeline principal) é que a aposta
é resolvida e vira uma atualização de peso — nunca na hora, porque não há
como saber "valeu a pena ter medo" no mesmo dia em que o medo aconteceu.

A recompensa nunca vem de acertar a direção prevista pelo ensemble (ver
README_Errors.md ERR-013 — isso é ruído). Vem do retorno real já realizado
nos dias seguintes: FEARFUL é reforçado quando o medo "se confirma" (a queda
continua); EUPHORIC é reforçado quando a euforia "se confirma" (a alta
continua). Isso testa clustering de retorno/volatilidade, um fenômeno real,
não a previsão de direção do próprio ensemble.
"""
import numpy as np

from maomao.brain import Brain, PLASTIC_POOLS

FIRE_THRESHOLD = 0.5   # taxa do pool acima da qual vale registar a aposta
HORIZON_DAYS   = 3     # dias reais seguintes usados para julgar se "valeu a pena"


def maybe_queue_bets(pending: list, date: str, pool_rates: dict, sensors: dict) -> list:
    """Regista uma aposta pendente para cada pool plástico que disparou forte hoje."""
    pending = list(pending)
    for pool in PLASTIC_POOLS:
        if pool_rates.get(pool, 0.0) >= FIRE_THRESHOLD:
            pending.append({
                "date": date,
                "pool": pool,
                "eligibility": {k: v for k, v in sensors.items() if k != "date"},
            })
    return pending


def resolve_due_bets(pending: list, by_day_returns, brain: Brain) -> tuple[list, list]:
    """
    Resolve apostas cuja janela de HORIZON_DAYS já tem retorno real
    suficiente em by_day_returns (pd.Series indexada por target_date, mesma
    que maomao.sensors.compute_sensors usa). Devolve (pending_restante, resolvidas).
    """
    dates = list(by_day_returns.index)
    still_pending, resolved = [], []

    for bet in pending:
        if bet["date"] not in dates:
            still_pending.append(bet)  # data ainda não validada por nenhum motivo — mantém na fila
            continue
        idx = dates.index(bet["date"])
        future = dates[idx + 1 : idx + 1 + HORIZON_DAYS]
        if len(future) < HORIZON_DAYS:
            still_pending.append(bet)  # ainda não passaram dias suficientes
            continue

        avg_next = float(np.mean([by_day_returns[d] for d in future]))
        if bet["pool"] == "FEARFUL":
            reward = float(np.clip(-avg_next / 2.0, -1.0, 1.0))
        else:  # EUPHORIC
            reward = float(np.clip(avg_next / 2.0, -1.0, 1.0))

        brain.learn(bet["pool"], bet["eligibility"], reward)
        resolved.append({**bet, "avg_next_return": avg_next, "reward": reward})

    return still_pending, resolved
