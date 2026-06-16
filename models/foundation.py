"""
Família: Foundation Models (2023-2024)
Modelos: Chronos (Amazon), TimesFM-inspired (Google), Moirai-inspired (Salesforce)

Chronos (Ansari et al., 2024 — Amazon Science):
  Modelo T5 pré-treinado em 84B pontos de séries temporais. Zero-shot:
  o contexto histórico é fornecido e o modelo gera amostras da distribuição
  futura sem fine-tuning. Usa chronos-t5-tiny (~8M params) via HuggingFace
  com fallback gracioso para y_mean quando o pacote não está disponível.

TimesFM (Das et al., 2024 — Google DeepMind):
  Decoder-only Transformer pré-treinado em 100B pontos. Inovação central:
  tokenização por patches reduz custo da atenção de O(T²) → O((T/P)²) e
  captura padrões locais. A implementação JAX original não é prática em CI;
  reproduzimos a ideia com um Transformer causal em PyTorch com patch encoding.

Moirai (Woo et al., 2024 — Salesforce Research):
  "Any-variate" foundation model: atenção conjunta entre variáveis e tempo.
  Aqui implementamos atenção variate-level por time-step seguida de atenção
  temporal, preservando a intenção de multi-variate mixing sem dependências JAX.

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

try:
    from chronos import ChronosPipeline as _ChronosPipeline
    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False
    logger.info("chronos package not installed — Foundation uses TimesFM + Moirai only")

SEQ_LEN    = 20
EPOCHS     = 30
LR         = 5e-4
BATCH_SIZE = 32
D_MODEL    = 32
NHEAD      = 4
PATCH_SIZE = 4


# ── TimesFM-inspired (causal patch Transformer) ───────────────────────────

class _TimesFM(nn.Module):
    """Patch-based causal Transformer (simplified TimesFM)."""

    def __init__(self, seq_len: int, n_feat: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        n_patches       = seq_len // patch_size
        self.proj       = nn.Linear(patch_size * n_feat, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.1)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model, nhead=NHEAD, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, V)
        B, T, V = x.shape
        n       = T // self.patch_size
        patches = x[:, :n * self.patch_size, :].reshape(B, n, self.patch_size * V)
        tokens  = self.proj(patches) + self.pos_enc[:, :n, :]
        mask    = torch.triu(torch.full((n, n), float("-inf"), device=x.device), diagonal=1)
        out     = self.encoder(tokens, mask=mask)
        return torch.sigmoid(self.fc(out[:, -1, :]))


# ── Moirai-inspired (dual-axis: variate + temporal attention) ─────────────

class _Moirai(nn.Module):
    """Variate-then-temporal dual-axis attention (simplified Moirai)."""

    def __init__(self, seq_len: int, n_feat: int, d_model: int, nhead: int):
        super().__init__()
        half          = d_model // 2
        self.var_proj = nn.Linear(1, half)
        self.var_pos  = nn.Parameter(torch.randn(1, n_feat, half) * 0.1)
        var_layer     = nn.TransformerEncoderLayer(
            half, nhead=max(1, nhead // 2), dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.var_attn  = nn.TransformerEncoder(var_layer, num_layers=1)
        self.time_proj = nn.Linear(half, d_model)
        self.time_pos  = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        time_layer     = nn.TransformerEncoderLayer(
            d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.time_attn = nn.TransformerEncoder(time_layer, num_layers=1)
        self.fc        = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, V)
        B, T, V = x.shape
        xv      = x.reshape(B * T, V, 1)
        var_tok = self.var_attn(self.var_proj(xv) + self.var_pos).mean(dim=1)
        t_tok   = self.time_proj(var_tok).reshape(B, T, -1) + self.time_pos[:, :T, :]
        out     = self.time_attn(t_tok)
        return torch.sigmoid(self.fc(out[:, -1, :]))


# ── Chronos zero-shot wrapper ─────────────────────────────────────────────

def _chronos_prob(context: np.ndarray, y_mean: float) -> float:
    """Zero-shot probability of next value > last via Chronos-T5-Tiny."""
    if not _CHRONOS_AVAILABLE:
        return y_mean
    try:
        pipeline = _ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        series   = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(series, prediction_length=1, num_samples=20)
        samples  = forecast[0, :, 0].numpy()
        return float((samples > float(context[-1])).mean())
    except Exception as exc:
        logger.warning("Chronos falhou: %s", exc)
        return y_mean


# ── Treino e predição genéricos ───────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)
    seqs   = np.stack([X[i - seq_len:i] for i in range(seq_len, len(X))])
    labels = y[seq_len:].astype(np.float32)
    return seqs.astype(np.float32), labels


def _train_model(model: nn.Module, seqs: np.ndarray, labels: np.ndarray) -> nn.Module:
    if len(seqs) == 0:
        return model
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    ds      = torch.utils.data.TensorDataset(
        torch.from_numpy(seqs), torch.from_numpy(labels).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
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
    probs           = np.full(len(X_sc), y_mean)
    probs[SEQ_LEN:] = p
    return probs


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    torch.manual_seed(42)
    scaler       = RobustScaler()
    X_sc         = scaler.fit_transform(X).astype(np.float32)
    seqs, labels = _make_sequences(X_sc, y, SEQ_LEN)
    n_feat       = X_sc.shape[1]

    timesfm = _train_model(_TimesFM(SEQ_LEN, n_feat, PATCH_SIZE, D_MODEL), seqs, labels)
    moirai  = _train_model(_Moirai(SEQ_LEN, n_feat, D_MODEL, NHEAD),       seqs, labels)

    return {
        "scaler":          scaler,
        "y_mean":          float(y.mean()),
        "timesfm":         timesfm,
        "moirai":          moirai,
        "chronos_context": X_sc.mean(axis=1),
    }


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    sc     = model_dict["scaler"]
    y_mean = model_dict["y_mean"]
    X_sc   = sc.transform(X).astype(np.float32)

    ctx       = np.concatenate([model_dict["chronos_context"], X_sc.mean(axis=1)])
    p_chr     = _chronos_prob(ctx, y_mean)

    probs = np.stack([
        np.full(len(X_sc), p_chr),
        _predict_model(model_dict["timesfm"], X_sc, y_mean),
        _predict_model(model_dict["moirai"],  X_sc, y_mean),
    ])
    return probs.mean(axis=0)
