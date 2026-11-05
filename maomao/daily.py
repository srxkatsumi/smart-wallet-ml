"""
Ponto de entrada diário do MaoMao, chamado uma vez por execução do pipeline
principal (ver main.py). Duas redes independentes, mesma lógica de
orquestração: `run_daily_step` (humor de carteira, só os 7 ETFs) e
`run_market_step` (humor de mercado, watchlist inteiro) — ver
maomao/README.md para a diferença entre os dois.

Idempotente por data: se este dia já foi processado (outra reexecução
redundante do mesmo dia — ver README_Errors.md ERR-010/ERR-013), devolve o
último humor já calculado em vez de reprocessar.
"""
import logging
from pathlib import Path

from maomao.brain import Brain, POOLS
from maomao.sensors import compute_sensors, compute_market_sensors, real_returns_by_day
from maomao.learning import maybe_queue_bets, resolve_due_bets
import maomao.state as maomao_state

logger = logging.getLogger(__name__)


def _run_step(compute_sensors_fn, df_log, anomaly: dict, tickers: list[str],
             state_path: Path, label: str) -> dict:
    # state_path é lido do módulo (maomao_state.PORTFOLIO_STATE_FILE /
    # MARKET_STATE_FILE) no site de chamada, não copiado num `from ... import` —
    # ver nota em models/calibration.py sobre por que isso importa para testes
    # com monkeypatch.
    st = maomao_state.load_state(state_path)
    sensors = compute_sensors_fn(df_log, anomaly, tickers, st["recent_alertas"])
    by_day  = real_returns_by_day(df_log, tickers)

    if sensors["date"] is None:
        logger.info("MaoMao (%s): sem histórico validado suficiente ainda — nada a fazer", label)
        return {"mood": {p: 0.0 for p in POOLS}, "date": None}

    if sensors["date"] == st["last_processed_date"]:
        logger.info("MaoMao (%s): %s já processado nesta execução do dia — a saltar", label, sensors["date"])
        last = st["mood_history"][-1]["mood"] if st["mood_history"] else {p: 0.0 for p in POOLS}
        return {"mood": last, "date": sensors["date"]}

    brain = Brain(learned=st["learned"])

    st["pending"], resolved = resolve_due_bets(st["pending"], by_day, brain)
    for r in resolved:
        logger.info(
            "MaoMao (%s): aposta de %s (%s) resolvida — retorno médio seguinte=%.2f%%, reward=%.3f",
            label, r["date"], r["pool"], r["avg_next_return"], r["reward"],
        )

    mood = brain.step(sensors)
    st["pending"] = maybe_queue_bets(st["pending"], sensors["date"], mood, sensors)
    st["learned"] = brain.snapshot_learned()
    st["last_processed_date"] = sensors["date"]
    alerta = sensors.get("ANOMALY", 0.0) >= 1.0
    maomao_state.push_history(st, sensors["date"], mood, alerta)

    maomao_state.save_state(st, state_path)
    logger.info("MaoMao (%s) %s: %s", label, sensors["date"], {k: round(v, 3) for k, v in mood.items()})

    return {"mood": mood, "date": sensors["date"]}


def run_daily_step(df_log, anomaly: dict, portfolio: list[str]) -> dict:
    """Humor de carteira — só os 7 ETFs investidos de verdade."""
    return _run_step(compute_sensors, df_log, anomaly, portfolio,
                     maomao_state.PORTFOLIO_STATE_FILE, "carteira")


def run_market_step(df_log, anomaly: dict, watchlist: list[str]) -> dict:
    """Humor de mercado — watchlist inteiro, não só a carteira."""
    return _run_step(compute_market_sensors, df_log, anomaly, watchlist,
                     maomao_state.MARKET_STATE_FILE, "mercado")
