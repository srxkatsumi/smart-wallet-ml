import maomao.state as state


def test_load_state_defaults_when_file_missing(tmp_path):
    s = state.load_state(tmp_path / "maomao_state.json")
    assert s["last_processed_date"] is None
    assert s["learned"] == {}
    assert s["pending"] == []
    assert s["mood_history"] == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "maomao_state.json"
    original = {
        "last_processed_date": "2026-01-05",
        "learned": {"FEARFUL": {"VOL_SPIKE": 0.3}},
        "pending": [{"date": "2026-01-05", "pool": "FEARFUL", "eligibility": {}}],
        "recent_alertas": [False, True],
        "mood_history": [{"date": "2026-01-05", "mood": {"CALM": 0.1}}],
    }
    state.save_state(original, path)
    assert path.exists()
    reloaded = state.load_state(path)
    assert reloaded == original


def test_save_state_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "maomao_state.json"
    state.save_state(state._fresh_default(), path)
    assert not path.with_suffix(".tmp").exists()


def test_load_state_falls_back_to_default_on_corrupted_json(tmp_path):
    path = tmp_path / "maomao_state.json"
    path.write_text("{not valid json")
    s = state.load_state(path)
    assert s["learned"] == {}


def test_portfolio_and_market_state_files_are_independent(tmp_path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    state.save_state({**state._fresh_default(), "last_processed_date": "2026-01-01"}, p1)
    state.save_state({**state._fresh_default(), "last_processed_date": "2026-02-02"}, p2)
    assert state.load_state(p1)["last_processed_date"] == "2026-01-01"
    assert state.load_state(p2)["last_processed_date"] == "2026-02-02"


def test_push_history_trims_to_max_length():
    s = state._fresh_default()
    for i in range(state.MAX_HISTORY + 5):
        state.push_history(s, f"day-{i}", {"CALM": 0.5}, alerta=(i % 2 == 0))
    assert len(s["mood_history"]) == state.MAX_HISTORY
    assert len(s["recent_alertas"]) == state.MAX_RECENT_ALERTAS
    assert s["mood_history"][-1]["date"] == f"day-{state.MAX_HISTORY + 4}"
