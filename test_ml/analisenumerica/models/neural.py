"""
Família: Redes Neurais Recorrentes
Modelos: LSTM, GRU
Projecto: Mega Sena ML Experiment

Cada bola tem uma sequência histórica de features (frequências, tendências).
LSTM e GRU aprendem se existe alguma dependência temporal entre sorteios
consecutivos — o resultado esperado num processo i.i.d. é que os modelos
não consigam superar o baseline aleatório.

Contexto académico:
  LSTM (Hochreiter & Schmidhuber, 1997) e GRU (Cho et al., 2014) são os
  modelos neurais recorrentes mais citados em previsão de séries temporais.
  Testar estes modelos na Mega Sena documenta formalmente que nem mesmo
  arquitecturas com memória de longo alcance detectam padrões num processo
  declaradamente aleatório.

Interface (compatível com ensemble.py da Mega Sena):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

SEQ_LEN    = 5
HIDDEN     = 32
EPOCHS     = 30
LR         = 1e-3
BATCH_SIZE = 128


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        seqs.append(X[i - seq_len:i])
        labels.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


class _RNN(nn.Module):
    def __init__(self, input_size: int, hidden: int, rnn_type: str):
        super().__init__()
        cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = cls(input_size, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


def _train_rnn(X: np.ndarray, y: np.ndarray, scaler: RobustScaler,
               rnn_type: str) -> nn.Module:
    X_sc = scaler.transform(X).astype(np.float32)
    seqs, labels = _make_sequences(X_sc, y, SEQ_LEN)
    if len(seqs) == 0:
        return None

    model   = _RNN(X_sc.shape[1], HIDDEN, rnn_type)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(seqs),
        torch.from_numpy(labels).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    return model


def _predict_rnn(model: nn.Module, X: np.ndarray, scaler: RobustScaler,
                 y_mean: float) -> np.ndarray:
    if model is None:
        return np.full(len(X), y_mean)

    X_sc = scaler.transform(X).astype(np.float32)
    seqs, _ = _make_sequences(X_sc, np.zeros(len(X_sc)), SEQ_LEN)
    if len(seqs) == 0:
        return np.full(len(X), y_mean)

    model.eval()
    with torch.no_grad():
        probs_seq = model(torch.from_numpy(seqs)).squeeze(1).numpy()

    probs = np.full(len(X), y_mean)
    probs[SEQ_LEN:] = probs_seq
    return probs


def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    return {
        "scaler": scaler,
        "y_mean": float(y.mean()),
        "lstm":   _train_rnn(X, y, scaler, "lstm"),
        "gru":    _train_rnn(X, y, scaler, "gru"),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc     = model_dict["scaler"]
    y_mean = model_dict["y_mean"]
    p_lstm = _predict_rnn(model_dict["lstm"], X, sc, y_mean)
    p_gru  = _predict_rnn(model_dict["gru"],  X, sc, y_mean)
    return (p_lstm + p_gru) / 2.0
