"""
Família: Redes Neurais Recorrentes
Modelos: LSTM, GRU

Ambos os modelos usam arquitectura de 1 camada recorrente seguida de uma
camada linear com saída sigmoid. A entrada é uma janela temporal de
seq_len=20 dias de features, produzindo uma probabilidade de UP para o
dia seguinte.

Contexto académico:
  LSTM (Hochreiter & Schmidhuber, 1997) e GRU (Cho et al., 2014) capturam
  dependências de longo alcance em séries temporais que os modelos clássicos
  não conseguem modelar. A hipótese é que a sequência de indicadores técnicos
  contém informação temporal que vai além de um único snapshot.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]

Nota: as primeiras seq_len posições do output recebem o valor médio de y
(sem sequência suficiente para prever). Isso é documentado nas análises.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

SEQ_LEN    = 20
HIDDEN     = 64
EPOCHS     = 50
LR         = 1e-3
BATCH_SIZE = 64


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
    X_sc  = scaler.transform(X).astype(np.float32)
    seqs, labels = _make_sequences(X_sc, y, SEQ_LEN)
    if len(seqs) == 0:
        return None

    device = torch.device("cpu")
    model  = _RNN(X_sc.shape[1], HIDDEN, rnn_type).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
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
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
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

    # as primeiras SEQ_LEN posições não têm sequência — usa média histórica
    probs = np.full(len(X), y_mean)
    probs[SEQ_LEN:] = probs_seq
    return probs


# ── Interface unificada ───────────────────────────────────────────────────

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
