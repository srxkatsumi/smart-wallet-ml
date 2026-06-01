"""
Família: Clássico Avançado
Modelos: XGBoost, LightGBM, CatBoost, SVM
Projecto: Mega Sena ML Experiment

Mesma família do projecto Carteira, aplicada a features de bolas da Mega Sena.
O objectivo académico é documentar que estes modelos, mesmo sendo mais
poderosos que RF/GB/SGD, não conseguem extrair padrões de um processo
declaradamente aleatório — confirmando o resultado do baseline.

Interface (compatível com ensemble.py da Mega Sena):
  train(X, y)           -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1], shape (60,)
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def train_xgb(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_xgb(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def train_lgbm(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=3,
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


def train_cat(X: np.ndarray, y: np.ndarray, scaler: RobustScaler) -> object:
    model = CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    model.fit(scaler.transform(X), y)
    return model


def predict_cat(model: object, X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


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
