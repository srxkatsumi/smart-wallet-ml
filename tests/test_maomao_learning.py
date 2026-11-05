import pandas as pd
import pytest

from maomao.brain import Brain
from maomao.learning import maybe_queue_bets, resolve_due_bets, FIRE_THRESHOLD, HORIZON_DAYS


def test_queue_bets_only_for_pools_above_threshold():
    pending = maybe_queue_bets([], "2026-01-05",
                                {"FEARFUL": FIRE_THRESHOLD + 0.1, "EUPHORIC": 0.0, "CALM": 0.9, "CONFUSED": 0.0},
                                {"VOL_SPIKE": 0.8})
    assert len(pending) == 1
    assert pending[0]["pool"] == "FEARFUL"
    assert pending[0]["eligibility"] == {"VOL_SPIKE": 0.8}


def test_queue_bets_can_register_both_plastic_pools_same_day():
    pending = maybe_queue_bets([], "2026-01-05",
                                {"FEARFUL": 0.9, "EUPHORIC": 0.9, "CALM": 0.0, "CONFUSED": 0.0},
                                {"VOL_SPIKE": 0.5})
    assert {b["pool"] for b in pending} == {"FEARFUL", "EUPHORIC"}


def test_bet_stays_pending_until_horizon_days_have_real_returns():
    by_day = pd.Series(
        {f"2026-01-0{i}": 1.0 for i in range(1, 1 + HORIZON_DAYS)},  # só a data da aposta + menos que o horizonte
    )
    pending = [{"date": "2026-01-01", "pool": "FEARFUL", "eligibility": {"VOL_SPIKE": 1.0}}]
    still_pending, resolved = resolve_due_bets(pending, by_day, Brain())
    assert resolved == []
    assert still_pending == pending


def test_bet_resolves_once_horizon_days_exist_after_it():
    dates = [f"2026-01-{d:02d}" for d in range(1, 2 + HORIZON_DAYS)]
    by_day = pd.Series({d: -1.0 for d in dates})  # queda contínua depois da aposta
    pending = [{"date": dates[0], "pool": "FEARFUL", "eligibility": {"VOL_SPIKE": 1.0}}]
    brain = Brain()
    still_pending, resolved = resolve_due_bets(pending, by_day, brain)
    assert still_pending == []
    assert len(resolved) == 1
    assert resolved[0]["reward"] > 0  # medo seguido de queda real -> recompensa positiva
    assert brain.snapshot_learned()["FEARFUL"]["VOL_SPIKE"] > 0


def test_fearful_bet_punished_when_market_bounces_back():
    dates = [f"2026-01-{d:02d}" for d in range(1, 2 + HORIZON_DAYS)]
    by_day = pd.Series({d: 2.0 for d in dates})  # recuperação, não queda
    pending = [{"date": dates[0], "pool": "FEARFUL", "eligibility": {"VOL_SPIKE": 1.0}}]
    brain = Brain()
    _, resolved = resolve_due_bets(pending, by_day, brain)
    assert resolved[0]["reward"] < 0
    assert brain.snapshot_learned()["FEARFUL"]["VOL_SPIKE"] < 0


def test_euphoric_bet_rewarded_when_gains_continue():
    dates = [f"2026-01-{d:02d}" for d in range(1, 2 + HORIZON_DAYS)]
    by_day = pd.Series({d: 1.5 for d in dates})
    pending = [{"date": dates[0], "pool": "EUPHORIC", "eligibility": {"STREAK_UP": 1.0}}]
    brain = Brain()
    _, resolved = resolve_due_bets(pending, by_day, brain)
    assert resolved[0]["reward"] > 0


def test_bet_for_date_not_yet_in_by_day_stays_pending():
    pending = [{"date": "2099-01-01", "pool": "FEARFUL", "eligibility": {}}]
    still_pending, resolved = resolve_due_bets(pending, pd.Series(dtype=float), Brain())
    assert still_pending == pending
    assert resolved == []
