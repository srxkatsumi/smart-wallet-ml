"""
Fase 12 — Rastreamento de Experimentos

MLflow (Zaharia et al., 2018 — "Accelerating the Machine Learning Lifecycle
  with MLflow", VLDB 2018):
  Plataforma de rastreamento de experimentos ML. Regista automaticamente:
  parâmetros, métricas, artefactos (modelos, gráficos) e metadados de cada
  run. Permite comparar experimentos lado a lado e reproduzir qualquer run.
  Crítico para teses com múltiplos modelos: sem tracking, comparar 25 modelos
  × 3 domínios (75 experimentos) torna-se inviável.

DVC — Data Version Control (Kuprieiev et al., 2020 — Iterative):
  Versiona ficheiros de dados grandes (CSVs, modelos) com o mesmo fluxo do
  git sem guardar os binários no repositório. Cada hash DVC regista o estado
  exacto dos dados associado a cada commit — garante reprodutibilidade total.

Convenção de nomes para este projecto:
  Experimento MLflow: "{domínio}_{família}_{modelo}"
  Ex: "carteira_neural_lstm", "megasena_classico_xgboost"

Interface:
  init_experiment(name)                   -> experiment_id
  log_run(experiment, model_name, params, metrics, artifacts) -> run_id
  log_run_from_report(experiment, model_name, report_dict)    -> run_id
  list_runs(experiment_name)              -> DataFrame
  dvc_track(file_paths)                   -> list de .dvc paths criados
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── MLflow ────────────────────────────────────────────────────────────────

_DEFAULT_DB = "sqlite:///mlflow.db"


def init_experiment(name: str, tracking_uri: str | None = None) -> str:
    """
    Cria ou recupera um experimento MLflow.

    Parameters
    ----------
    name         : nome do experimento (convenção: domínio_família_modelo)
    tracking_uri : URI do servidor MLflow; None usa SQLite local (mlflow.db)

    Returns
    -------
    experiment_id como string
    """
    import mlflow
    uri = tracking_uri or _DEFAULT_DB
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id


def log_run(experiment_name: str, model_name: str,
            params: dict, metrics: dict,
            artifacts: dict | None = None,
            tags: dict | None = None) -> str:
    """
    Regista um run MLflow com parâmetros, métricas e artefactos.

    Parameters
    ----------
    experiment_name : nome do experimento
    model_name      : nome do modelo (aparece como tag)
    params          : hiperparâmetros do modelo
    metrics         : métricas de avaliação (accuracy, f1, etc.)
    artifacts       : dict {nome: caminho_ficheiro} a guardar como artefactos
    tags            : tags MLflow adicionais

    Returns
    -------
    run_id como string
    """
    import mlflow
    if not mlflow.get_tracking_uri() or "mlflow.db" not in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri(_DEFAULT_DB)
    mlflow.set_experiment(experiment_name)

    _tags = {"model": model_name, "project": "smart_wallet_phd"}
    if tags:
        _tags.update(tags)

    with mlflow.start_run(tags=_tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=name)
        return run.info.run_id


def log_run_from_report(experiment_name: str, model_name: str,
                        report: dict) -> str:
    """
    Regista um run directamente a partir do dict gerado por full_report()
    em evaluation/statistical_tests.py.
    """
    metrics = {}
    if "metrics" in report:
        for k, v in report["metrics"].items():
            if isinstance(v, (int, float)) and v is not None:
                metrics[k] = float(v)
    if "ljung_box" in report:
        lb = report["ljung_box"]
        metrics["ljung_box_pvalue"] = float(lb.get("p_value", -1))
    if "diebold_mariano" in report:
        dm = report["diebold_mariano"]
        metrics["dm_pvalue"]    = float(dm.get("p_value", -1))
        metrics["dm_statistic"] = float(dm.get("statistic", 0))

    return log_run(experiment_name, model_name,
                   params={"model_name": model_name},
                   metrics=metrics)


def list_runs(experiment_name: str) -> object:
    """
    Retorna DataFrame com todos os runs de um experimento.
    Colunas: run_id, status, start_time, metrics.*, params.*
    """
    import mlflow
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


# ── DVC ───────────────────────────────────────────────────────────────────

def dvc_track(file_paths: list[str],
              repo_root: str | None = None) -> list[str]:
    """
    Adiciona ficheiros ao tracking DVC.

    Requer que o repositório já tenha DVC inicializado (dvc init).
    Para cada ficheiro, cria um .dvc pointer e adiciona ao .gitignore.

    Parameters
    ----------
    file_paths : lista de caminhos absolutos ou relativos ao repo_root
    repo_root  : directório raiz do repositório; None usa cwd

    Returns
    -------
    lista de caminhos dos ficheiros .dvc criados
    """
    import subprocess
    root     = Path(repo_root) if repo_root else Path.cwd()
    dvc_files = []

    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue
        result = subprocess.run(
            ["dvc", "add", str(p)],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            dvc_path = str(p) + ".dvc"
            if Path(dvc_path).exists():
                dvc_files.append(dvc_path)

    return dvc_files


def dvc_status(repo_root: str | None = None) -> dict:
    """
    Verifica o estado dos ficheiros rastreados pelo DVC.

    Returns
    -------
    dict com: tracked_files, changed_files, message
    """
    import subprocess
    root   = Path(repo_root) if repo_root else Path.cwd()
    result = subprocess.run(
        ["dvc", "status"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"tracked_files": [], "changed_files": [],
                "message": result.stderr or "DVC não inicializado."}

    output = result.stdout.strip()
    if not output or "Data and pipelines are up to date." in output:
        return {"tracked_files": [], "changed_files": [],
                "message": "Todos os ficheiros DVC estão actualizados."}

    return {"tracked_files": [], "changed_files": output.splitlines(),
            "message": output}
