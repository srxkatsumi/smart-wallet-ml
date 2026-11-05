import numpy as np
import pytest

from maomao.brain import Brain, CHANNELS, POOLS, PLASTIC_POOLS


def _zeros():
    return {c: 0.0 for c in CHANNELS}


def test_step_returns_all_pools_in_range():
    brain = Brain(rng=np.random.default_rng(0))
    mood = brain.step(_zeros())
    assert set(mood.keys()) == set(POOLS)
    for v in mood.values():
        assert 0.0 <= v <= 1.0


def test_calm_sensors_drive_calm_pool_higher_than_fearful():
    brain = Brain(rng=np.random.default_rng(1))
    intensities = {**_zeros(), "CALM_STREAK": 1.0}
    mood = brain.step(intensities)
    assert mood["CALM"] > mood["FEARFUL"]


def test_fearful_sensors_drive_fearful_pool_higher_than_calm():
    brain = Brain(rng=np.random.default_rng(1))
    intensities = {**_zeros(), "VOL_SPIKE": 1.0, "VIX_SPIKE": 1.0, "ANOMALY": 1.0, "STREAK_DOWN": 1.0}
    mood = brain.step(intensities)
    assert mood["FEARFUL"] > mood["CALM"]


def test_disagreement_drives_confused_pool():
    brain = Brain(rng=np.random.default_rng(2))
    baseline = brain.step(_zeros())
    intensities = {**_zeros(), "DISAGREEMENT": 1.0}
    mood = brain.step(intensities)
    assert mood["CONFUSED"] > baseline["CONFUSED"]


def test_no_sensory_input_settles_near_zero_for_all_pools():
    brain = Brain(rng=np.random.default_rng(3))
    mood = brain.step(_zeros())
    for v in mood.values():
        assert v < 0.1


# ── learn() / three-factor rule ──────────────────────────────────────────

def test_learn_ignored_for_non_plastic_pool():
    brain = Brain()
    brain.learn("CALM", {"VOL_SPIKE": 1.0}, reward=1.0)
    assert brain.snapshot_learned().get("CALM", {}) == {}


def test_learn_ignored_when_reward_is_zero():
    brain = Brain()
    brain.learn("FEARFUL", {"VOL_SPIKE": 1.0}, reward=0.0)
    assert brain.snapshot_learned()["FEARFUL"] == {}


def test_positive_reward_increases_weight_for_active_channel():
    brain = Brain()
    brain.learn("FEARFUL", {"VOL_SPIKE": 1.0, "RET_UP": 0.0}, reward=1.0)
    learned = brain.snapshot_learned()["FEARFUL"]
    assert learned["VOL_SPIKE"] > 0
    assert learned.get("RET_UP", 0.0) == 0.0  # canal inativo não aprende nada


def test_negative_reward_decreases_weight():
    brain = Brain()
    brain.learn("FEARFUL", {"VOL_SPIKE": 1.0}, reward=-1.0)
    assert brain.snapshot_learned()["FEARFUL"]["VOL_SPIKE"] < 0


def test_learned_weight_is_clipped():
    brain = Brain()
    for _ in range(1000):
        brain.learn("FEARFUL", {"VOL_SPIKE": 1.0}, reward=1.0, rate=1.0)
    from maomao.brain import LEARN_MAX
    assert brain.snapshot_learned()["FEARFUL"]["VOL_SPIKE"] == pytest.approx(LEARN_MAX)


def test_learning_actually_shifts_the_steady_state_mood():
    """
    Regression guard for the actual point of this module: after enough
    positive reward on VOL_SPIKE -> FEARFUL, the same sensory snapshot
    should produce a visibly higher FEARFUL rate than an untrained brain.
    """
    naive  = Brain(rng=np.random.default_rng(7))
    trained = Brain(rng=np.random.default_rng(7))
    for _ in range(50):
        trained.learn("FEARFUL", {"VOL_SPIKE": 1.0}, reward=1.0, rate=0.05)

    probe = {**_zeros(), "VOL_SPIKE": 0.6}
    assert trained.step(probe)["FEARFUL"] > naive.step(probe)["FEARFUL"]


def test_brain_round_trips_learned_weights_through_constructor():
    brain = Brain()
    brain.learn("EUPHORIC", {"STREAK_UP": 1.0}, reward=1.0)
    snap = brain.snapshot_learned()
    reloaded = Brain(learned=snap)
    assert reloaded.snapshot_learned() == snap
