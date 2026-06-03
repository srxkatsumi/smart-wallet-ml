"""
Família: Arquiteturas Eficientes (pós-2022)
Modelos: TCN, DLinear, NLinear, PatchTST

TCN (Bai et al., 2018 — "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling"):
  Convoluções dilatadas causais com conexões residuais e weight normalization.
  A dilatação exponencial (1, 2, 4, 8) garante receptive field de 30 passos
  sem custo quadrático da atenção.

DLinear (Zeng et al., 2023 — "Are Transformers Effective for Time Series Forecasting?"):
  Decompõe a série em tendência (média móvel) e resíduo. Aplica projecções
  lineares independentes a cada componente. Surpreendentemente competitivo com
  Transformers em benchmarks de longo prazo.

NLinear (Zeng et al., 2023 — mesmo artigo que DLinear):
  Subtrai o último valor antes da projecção linear (normalização de instância
  simplificada). Robusto a distribution shift entre treino e inferência.

PatchTST (Nie et al., 2023 — "A Time Series is Worth 64 Words"):
  Divide a série em patches (análogo a ViT). Cada patch é um token do
  Transformer — reduz custo quadrático da atenção e captura dependências locais.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

SEQ_LEN      = 20
EPOCHS       = 50
LR           = 1e-3
BATCH_SIZE   = 64
TCN_CHANNELS = 32
TCN_KERNEL   = 3
TCN_DILATIONS= [1, 2, 4, 8]
PATCH_SIZE   = 4
PATCH_STRIDE = 2
D_MODEL      = 32
NHEAD        = 4


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        seqs.append(X[i - seq_len:i])
        labels.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── TCN ───────────────────────────────────────────────────────────────────

class _TCNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation))
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        nn.init.normal_(self.conv.weight_v, 0, 0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.drop(self.act(self.conv(F.pad(x, (self.pad, 0)))))
        res = x if self.res is None else self.res(x)
        return F.relu(out + res)


class _TCN(nn.Module):
    def __init__(self, n_feat: int, channels: int, kernel: int, dilations: list):
        super().__init__()
        layers, in_ch = [], n_feat
        for d in dilations:
            layers.append(_TCNLayer(in_ch, channels, kernel, d))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(channels, 1)

    def forward(self, x):
        out = self.net(x.transpose(1, 2))   # (B, channels, seq)
        return torch.sigmoid(self.fc(out[:, :, -1]))


# ── DLinear ───────────────────────────────────────────────────────────────

class _DLinear(nn.Module):
    def __init__(self, seq_len: int, n_feat: int):
        super().__init__()
        k              = max(3, seq_len // 4)
        self.kernel    = k + (1 - k % 2)       # força ímpar
        self.padding   = self.kernel // 2
        flat           = seq_len * n_feat
        self.trend_fc  = nn.Linear(flat, 1)
        self.resid_fc  = nn.Linear(flat, 1)

    def forward(self, x):
        xt    = x.transpose(1, 2)              # (B, feat, seq)
        trend = F.avg_pool1d(xt, self.kernel, stride=1, padding=self.padding)
        trend = trend[:, :, :x.size(1)].transpose(1, 2)
        resid = x - trend
        b     = x.size(0)
        return torch.sigmoid(
            self.trend_fc(trend.reshape(b, -1)) + self.resid_fc(resid.reshape(b, -1))
        )


# ── NLinear ───────────────────────────────────────────────────────────────

class _NLinear(nn.Module):
    def __init__(self, seq_len: int, n_feat: int):
        super().__init__()
        self.fc = nn.Linear(seq_len * n_feat, 1)

    def forward(self, x):
        last = x[:, -1:, :]
        return torch.sigmoid(self.fc((x - last).reshape(x.size(0), -1)))


# ── PatchTST ──────────────────────────────────────────────────────────────

class _PatchTST(nn.Module):
    def __init__(self, seq_len: int, n_feat: int, patch_size: int,
                 stride: int, d_model: int, nhead: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        n_p             = (seq_len - patch_size) // stride + 1
        self.proj       = nn.Linear(patch_size * n_feat, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, n_p, d_model) * 0.02)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        b, s, f = x.shape
        patches = torch.stack([
            x[:, i:i + self.patch_size, :].reshape(b, -1)
            for i in range(0, s - self.patch_size + 1, self.stride)
        ], dim=1)
        tokens = self.proj(patches) + self.pos_enc
        out    = self.encoder(tokens)
        return torch.sigmoid(self.fc(out[:, -1, :]))


# ── Treino e predição genéricos ───────────────────────────────────────────

def _train_model(model: nn.Module, seqs: np.ndarray, labels: np.ndarray) -> nn.Module:
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(seqs),
        torch.from_numpy(labels).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
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
    probs         = np.full(len(X_sc), y_mean)
    probs[SEQ_LEN:] = p
    return probs


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    torch.manual_seed(42)
    scaler       = RobustScaler()
    X_sc         = scaler.fit_transform(X).astype(np.float32)
    seqs, labels = _make_sequences(X_sc, y, SEQ_LEN)
    n_feat       = X_sc.shape[1]

    tcn      = _train_model(_TCN(n_feat, TCN_CHANNELS, TCN_KERNEL, TCN_DILATIONS), seqs, labels)
    dlinear  = _train_model(_DLinear(SEQ_LEN, n_feat), seqs, labels)
    nlinear  = _train_model(_NLinear(SEQ_LEN, n_feat), seqs, labels)
    patchtst = _train_model(
        _PatchTST(SEQ_LEN, n_feat, PATCH_SIZE, PATCH_STRIDE, D_MODEL, NHEAD),
        seqs, labels,
    )

    return {
        "scaler":   scaler,
        "y_mean":   float(y.mean()),
        "tcn":      tcn,
        "dlinear":  dlinear,
        "nlinear":  nlinear,
        "patchtst": patchtst,
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc     = model_dict["scaler"]
    y_mean = model_dict["y_mean"]
    X_sc   = sc.transform(X).astype(np.float32)
    probs  = np.stack([
        _predict_model(model_dict["tcn"],      X_sc, y_mean),
        _predict_model(model_dict["dlinear"],  X_sc, y_mean),
        _predict_model(model_dict["nlinear"],  X_sc, y_mean),
        _predict_model(model_dict["patchtst"], X_sc, y_mean),
    ])
    return probs.mean(axis=0)
