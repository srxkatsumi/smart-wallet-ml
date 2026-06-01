"""
Família: Modelos Bayesianos
Modelos: Gaussian Process (GP), BNN via MC Dropout
Projecto: Mega Sena ML Experiment

Referências:
  GP  — Williams & Rasmussen, "Gaussian Processes for Machine Learning", 1996
  BNN — Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016

Objectivo académico:
  O GP fornece probabilidades calibradas teoricamente. Se as bolas fossem
  dependentes, o kernel RBF capturaria similaridade entre estados de features.
  O resultado esperado é que GP e BNN convirjam para P ≈ 0.10, confirmando
  ausência de estrutura exploitável no processo aleatório da Mega Sena.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import RobustScaler

GP_MAX_SAMPLES = 300
MC_SAMPLES     = 20
HIDDEN         = 32
EPOCHS         = 30
LR             = 1e-3
BATCH_SIZE     = 128
DROPOUT_RATE   = 0.3


def _train_gp(X_sc: np.ndarray, y: np.ndarray) -> GaussianProcessClassifier:
    if len(X_sc) > GP_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_sc), GP_MAX_SAMPLES, replace=False)
        X_tr, y_tr = X_sc[idx], y[idx]
    else:
        X_tr, y_tr = X_sc, y
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
    gp.fit(X_tr, y_tr.astype(int))
    return gp


def _predict_gp(gp: GaussianProcessClassifier, X_sc: np.ndarray) -> np.ndarray:
    return gp.predict_proba(X_sc)[:, 1]


class _BNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _train_bnn(X_sc: np.ndarray, y: np.ndarray) -> _BNNClassifier:
    model   = _BNNClassifier(X_sc.shape[1], HIDDEN, DROPOUT_RATE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_sc.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
    )
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def _predict_bnn(model: _BNNClassifier, X_sc: np.ndarray) -> np.ndarray:
    X_t = torch.from_numpy(X_sc.astype(np.float32))
    model.train()
    with torch.no_grad():
        samples = torch.stack([model(X_t).squeeze(1) for _ in range(MC_SAMPLES)])
    return samples.mean(dim=0).numpy()


def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    X_sc = scaler.transform(X)
    return {
        "scaler": scaler,
        "gp":     _train_gp(X_sc, y),
        "bnn":    _train_bnn(X_sc, y),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    X_sc  = model_dict["scaler"].transform(X)
    p_gp  = _predict_gp(model_dict["gp"],   X_sc)
    p_bnn = _predict_bnn(model_dict["bnn"], X_sc)
    return (p_gp + p_bnn) / 2.0
