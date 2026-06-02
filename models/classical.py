"""
Família: Clássico Avançado
Modelos: XGBoost, LightGBM, CatBoost, SVM

Todos implementam a API scikit-learn e são escalados com RobustScaler antes
do treino. SVM usa kernel RBF com probability=True para retornar probabilidades
calibradas. XGBoost, LightGBM e CatBoost têm regularização L1/L2 explícita
para reduzir sobreajuste em séries financeiras ruidosas.

Interface (compatível com ensemble.py):
  train(X, y)           -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]

Funções individuais também expostas para uso isolado e comparação académica:
  train_xgb / predict_xgb
  train_lgbm / predict_lgbm
  train_cat / predict_cat
  train_svm / predict_svm
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ── XGBoost ───────────────────────────────────────────────────────────────

def train_xgb(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_xgb(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


# ── LightGBM ──────────────────────────────────────────────────────────────

def train_lgbm(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_lgbm(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


# ── CatBoost ──────────────────────────────────────────────────────────────

def train_cat(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_cat(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


# ── SVM ───────────────────────────────────────────────────────────────────

def train_svm(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_svm(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    return {
        "scaler": scaler,
        "xgb":   train_xgb(X, y, scaler),
        "lgbm":  train_lgbm(X, y, scaler),
        "cat":   train_cat(X, y, scaler),
        "svm":   train_svm(X, y, scaler),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc = model_dict["scaler"]
    probs = np.stack([
        predict_xgb(model_dict["xgb"],  X, sc),
        predict_lgbm(model_dict["lgbm"], X, sc),
        predict_cat(model_dict["cat"],  X, sc),
        predict_svm(model_dict["svm"],  X, sc),
    ])
    return probs.mean(axis=0)
