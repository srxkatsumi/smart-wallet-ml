import pytest
import tempfile
import os
from pathlib import Path
from evaluation.tracking import (
    init_experiment,
    log_run,
    log_run_from_report,
    list_runs,
)


@pytest.fixture
def mlflow_tmpdir(tmp_path, monkeypatch):
    """Redireciona MLflow para base SQLite temporária."""
    import mlflow
    db_path = tmp_path / "test_mlflow.db"
    uri     = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("evaluation.tracking._DEFAULT_DB", uri)
    return tmp_path


def test_init_experiment(mlflow_tmpdir):
    exp_id = init_experiment("test_carteira_rf")
    assert exp_id is not None
    assert isinstance(exp_id, str)


def test_log_run_returns_id(mlflow_tmpdir):
    init_experiment("test_carteira_rf")
    run_id = log_run(
        "test_carteira_rf",
        "random_forest",
        params={"n_estimators": 100, "max_depth": 5},
        metrics={"accuracy": 0.55, "f1": 0.54},
    )
    assert run_id is not None
    assert len(run_id) > 0


def test_log_run_from_report(mlflow_tmpdir):
    init_experiment("test_megasena_markov")
    report = {
        "metrics": {"accuracy": 0.10, "f1": 0.09, "brier_score": 0.09},
        "ljung_box": {"p_value": 0.42, "has_autocorrelation": False},
        "diebold_mariano": {"p_value": 0.87, "statistic": 0.2},
    }
    run_id = log_run_from_report("test_megasena_markov", "markov", report)
    assert run_id is not None


def test_list_runs_returns_dataframe(mlflow_tmpdir):
    init_experiment("test_list_runs")
    log_run("test_list_runs", "model_a",
            params={"x": 1}, metrics={"acc": 0.5})
    df = list_runs("test_list_runs")
    assert df is not None
    assert len(df) >= 1


def test_list_runs_nonexistent_returns_none(mlflow_tmpdir):
    result = list_runs("this_experiment_does_not_exist")
    assert result is None
