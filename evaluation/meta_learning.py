"""
Fase 11 — Meta-learning

Stacking (Wolpert, 1992 — "Stacked Generalization", Neural Networks):
  Em vez de combinar modelos com pesos fixos (ensemble ponderado), o stacking
  aprende a combinação óptima a partir dos dados. As previsões dos modelos
  base são usadas como features de um meta-learner (aqui LogisticRegression).
  A validação cruzada out-of-fold garante que o meta-learner não aprende
  simplesmente a confiar em quem decorou os dados de treino.

Optuna (Akiba et al., 2019 — "Optuna: A Next-generation Hyperparameter
  Optimization Framework", KDD 2019):
  Optimização bayesiana de hiperparâmetros com pruning adaptativo (TPE sampler
  + Median pruner). Para cada modelo, define-se o espaço de busca e a função
  objectivo; o Optuna encontra os hiperparâmetros que maximizam a acurácia
  em validação cruzada com um orçamento fixo de trials.

Interface:
  train_stacking(predict_fns, X, y)      -> meta-model + scaler
  predict_stacking(meta_dict, predict_fns, X) -> np.ndarray probabilidades

  optimize_rf(X, y, n_trials)      -> best_params dict
  optimize_xgb(X, y, n_trials)     -> best_params dict
  optimize_lgbm(X, y, n_trials)    -> best_params dict
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ── Stacking ──────────────────────────────────────────────────────────────

def train_stacking(predict_fns: list, X: np.ndarray,
                   y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Treina um meta-learner sobre as previsões out-of-fold dos modelos base.

    Parameters
    ----------
    predict_fns : lista de callables — cada um aceita X e retorna probs (N,)
    X, y        : dados de treino
    n_splits    : folds para out-of-fold stacking

    Returns
    -------
    dict com: meta_model, scaler, n_base_models
    """
    n = len(X)
    n_base = len(predict_fns)
    oof    = np.zeros((n, n_base))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y.astype(int))):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val      = X[val_idx]
        for j, fn in enumerate(predict_fns):
            try:
                oof[val_idx, j] = fn(X_tr, y_tr, X_val)
            except Exception:
                oof[val_idx, j] = 0.5

    scaler     = RobustScaler()
    oof_sc     = scaler.fit_transform(oof)
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(oof_sc, y.astype(int))

    return {"meta_model": meta_model, "scaler": scaler,
            "n_base_models": n_base}


def predict_stacking(meta_dict: dict, predict_fns: list,
                     X: np.ndarray) -> np.ndarray:
    """
    Gera previsões do ensemble empilhado para novos dados.

    Cada predict_fn é chamada com apenas X (sem y), por isso deve encapsular
    um modelo já treinado: predict_fn(X) -> probs.
    """
    n_base = meta_dict["n_base_models"]
    stack  = np.zeros((len(X), n_base))
    for j, fn in enumerate(predict_fns):
        try:
            stack[:, j] = fn(X)
        except Exception:
            stack[:, j] = 0.5

    stack_sc = meta_dict["scaler"].transform(stack)
    return meta_dict["meta_model"].predict_proba(stack_sc)[:, 1]


# ── Optuna ────────────────────────────────────────────────────────────────

def _cv_accuracy(model_cls, params: dict, X: np.ndarray,
                 y: np.ndarray, n_splits: int = 3) -> float:
    from sklearn.model_selection import cross_val_score
    model = model_cls(**params, random_state=42)
    sc    = RobustScaler()
    X_sc  = sc.fit_transform(X)
    scores = cross_val_score(model, X_sc, y.astype(int),
                             cv=n_splits, scoring="accuracy")
    return float(scores.mean())


def optimize_rf(X: np.ndarray, y: np.ndarray,
                n_trials: int = 20) -> dict:
    """Optimiza hiperparâmetros do RandomForest com Optuna."""
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth":    trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "class_weight": "balanced",
            "n_jobs":       -1,
        }
        return _cv_accuracy(RandomForestClassifier, params, X, y)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"model": "random_forest", "best_params": study.best_params,
            "best_accuracy": study.best_value, "n_trials": n_trials}


def optimize_xgb(X: np.ndarray, y: np.ndarray,
                 n_trials: int = 20) -> dict:
    """Optimiza hiperparâmetros do XGBoost com Optuna."""
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "eval_metric":      "logloss",
            "verbosity":        0,
            "use_label_encoder": False,
        }
        return _cv_accuracy(XGBClassifier, params, X, y)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"model": "xgboost", "best_params": study.best_params,
            "best_accuracy": study.best_value, "n_trials": n_trials}


def optimize_lgbm(X: np.ndarray, y: np.ndarray,
                  n_trials: int = 20) -> dict:
    """Optimiza hiperparâmetros do LightGBM com Optuna."""
    import optuna
    from lightgbm import LGBMClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "verbose":          -1,
        }
        return _cv_accuracy(LGBMClassifier, params, X, y)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"model": "lightgbm", "best_params": study.best_params,
            "best_accuracy": study.best_value, "n_trials": n_trials}
