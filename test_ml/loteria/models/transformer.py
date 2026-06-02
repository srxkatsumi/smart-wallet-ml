"""
Família: Redes Neurais com Atenção
Modelos: Transformer, TFT (simplificado), N-BEATS (simplificado)
Projecto: Mega Sena ML Experiment

Referências académicas:
  Transformer — Vaswani et al., 2017, "Attention Is All You Need"
  TFT — Lim et al., 2021, Google, "Temporal Fusion Transformers for Interpretable
         Multi-horizon Time Series Forecasting"
  N-BEATS — Oreshkin et al., 2020, Mila/Element AI, "N-BEATS: Neural basis
              expansion analysis for interpretable time series forecasting"

Objectivo académico:
  Testar se mecanismos de atenção conseguem detectar padrões temporais na
  sequência de sorteios. O resultado esperado é convergência para o baseline
  aleatório (P ≈ 0.10), documentando que nem modelos state-of-the-art de
  atenção temporal encontram estrutura num processo i.i.d.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

SEQ_LEN    = 5
D_MODEL    = 16
NHEAD      = 2
HIDDEN     = 16
EPOCHS     = 30
LR         = 1e-3
BATCH_SIZE = 128
N_BLOCKS   = 3


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        seqs.append(X[i - seq_len:i])
        labels.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


class _TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class _SimplifiedTFT(nn.Module):
    def __init__(self, input_size: int, hidden: int, nhead: int):
        super().__init__()
        self.vsn  = nn.Sequential(nn.Linear(input_size, input_size), nn.Softmax(dim=-1))
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x * self.vsn(x)
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        h    = self.norm(h + a)
        return torch.sigmoid(self.fc(h[:, -1, :]))


class _NBEATSBlock(nn.Module):
    def __init__(self, flat_size: int, hidden: int):
        super().__init__()
        self.fc  = nn.Sequential(
            nn.Linear(flat_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x_flat):
        return torch.sigmoid(self.out(self.fc(x_flat)))


class _NBEATS(nn.Module):
    def __init__(self, seq_len: int, input_size: int, hidden: int, n_blocks: int):
        super().__init__()
        flat = seq_len * input_size
        self.blocks = nn.ModuleList([_NBEATSBlock(flat, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        preds  = torch.stack([b(x_flat) for b in self.blocks], dim=1)
        return preds.mean(dim=1)


def _train_model(model: nn.Module, seqs: np.ndarray, labels: np.ndarray) -> nn.Module:
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(seqs),
        torch.from_numpy(labels).unsqueeze(1),
    )
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def _predict_model(model: nn.Module, X_sc: np.ndarray, y_mean: float) -> np.ndarray:
    seqs, _ = _make_sequences(X_sc, np.zeros(len(X_sc)), SEQ_LEN)
    if len(seqs) == 0:
        return np.full(len(X_sc), y_mean)
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(seqs)).squeeze(1).numpy()
    probs = np.full(len(X_sc), y_mean)
    probs[SEQ_LEN:] = p
    return probs


def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    X_sc   = scaler.transform(X).astype(np.float32)
    seqs, labels = _make_sequences(X_sc, y, SEQ_LEN)

    n_feat = X_sc.shape[1]
    transformer = _train_model(_TransformerClassifier(n_feat, D_MODEL, NHEAD), seqs, labels)
    tft         = _train_model(_SimplifiedTFT(n_feat, HIDDEN, NHEAD), seqs, labels)
    nbeats      = _train_model(_NBEATS(SEQ_LEN, n_feat, HIDDEN, N_BLOCKS), seqs, labels)

    return {
        "scaler":      scaler,
        "y_mean":      float(y.mean()),
        "transformer": transformer,
        "tft":         tft,
        "nbeats":      nbeats,
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc     = model_dict["scaler"]
    y_mean = model_dict["y_mean"]
    X_sc   = sc.transform(X).astype(np.float32)
    probs  = np.stack([
        _predict_model(model_dict["transformer"], X_sc, y_mean),
        _predict_model(model_dict["tft"],         X_sc, y_mean),
        _predict_model(model_dict["nbeats"],      X_sc, y_mean),
    ])
    return probs.mean(axis=0)
