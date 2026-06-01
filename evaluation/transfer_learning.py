"""
Fase 14 — Transfer Learning entre Domínios

Pergunta central: o conhecimento aprendido num domínio transfere para outro?
Esta é a contribuição original do Capítulo 14 da tese — a ponte entre os
três domínios.

Maximum Mean Discrepancy (Gretton et al., 2012 — JMLR):
  Mede a distância entre duas distribuições P e Q no espaço de Hilbert.
  MMD = 0: distribuições idênticas (transferência perfeita esperada).
  MMD > 0: distribuições diferentes (domain shift, transferência difícil).

Feature Alignment — CORAL (Sun & Saenko, 2016 — ECCV):
  "Return of Frustratingly Easy Domain Adaptation".
  Alinha a covariância das features do domínio fonte para o domínio alvo.
  Transformação: X_aligned = X_source @ W onde W = C_s^{-1/2} @ C_t^{1/2}.
  Simples, eficaz, sem treino adicional.

Fine-tuning Transfer:
  Pega num modelo PyTorch já treinado, congela as camadas base,
  e retreina apenas a cabeça (última camada linear) no novo domínio.
  Abordagem: feature extractor fixo + novo classificador.

Cross-domain Evaluation:
  Avalia sistematicamente: train no domínio A, test no domínio B.
  Compara com: train e test no mesmo domínio (upper bound).
  A diferença mede o domain gap.
"""

import warnings
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# ── MMD ───────────────────────────────────────────────────────────────────

def maximum_mean_discrepancy(X_source: np.ndarray,
                              X_target: np.ndarray,
                              kernel: str = "rbf",
                              sigma: float | None = None) -> float:
    """
    Calcula MMD entre duas amostras (Gretton et al., 2012).

    Parameters
    ----------
    X_source, X_target : arrays (N, D) das duas distribuições
    kernel             : "rbf" ou "linear"
    sigma              : bandwidth do kernel RBF; None usa mediana das distâncias

    Returns
    -------
    MMD² (não-negativo; 0 = distribuições idênticas)
    """
    def _rbf(A, B, s):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * s ** 2))

    def _linear(A, B):
        return A @ B.T

    ns = min(len(X_source), 200)
    nt = min(len(X_target), 200)
    Xs = X_source[np.random.default_rng(42).choice(len(X_source), ns, replace=False)]
    Xt = X_target[np.random.default_rng(42).choice(len(X_target), nt, replace=False)]

    if sigma is None:
        all_pts = np.vstack([Xs, Xt])
        dists   = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=-1)
        sigma   = float(np.median(dists[dists > 0])) or 1.0

    if kernel == "rbf":
        Kss = _rbf(Xs, Xs, sigma).mean()
        Ktt = _rbf(Xt, Xt, sigma).mean()
        Kst = _rbf(Xs, Xt, sigma).mean()
    else:
        Kss = _linear(Xs, Xs).mean()
        Ktt = _linear(Xt, Xt).mean()
        Kst = _linear(Xs, Xt).mean()

    mmd2 = float(Kss + Ktt - 2 * Kst)
    return max(mmd2, 0.0)


# ── CORAL Feature Alignment ───────────────────────────────────────────────

def coral_align(X_source: np.ndarray,
                X_target: np.ndarray) -> np.ndarray:
    """
    Alinha features do domínio fonte para o domínio alvo via CORAL.

    Parameters
    ----------
    X_source : features do domínio fonte (N_s, D)
    X_target : features do domínio alvo  (N_t, D)

    Returns
    -------
    X_source_aligned : (N_s, D) alinhado com a distribuição alvo
    """
    eps = 1e-6
    Cs  = np.cov(X_source, rowvar=False) + eps * np.eye(X_source.shape[1])
    Ct  = np.cov(X_target,  rowvar=False) + eps * np.eye(X_target.shape[1])

    # decomposição eigenvalue para matriz simétrica
    vals_s, vecs_s = np.linalg.eigh(Cs)
    vals_t, vecs_t = np.linalg.eigh(Ct)

    Cs_inv_sqrt = vecs_s @ np.diag(1.0 / np.sqrt(np.maximum(vals_s, eps))) @ vecs_s.T
    Ct_sqrt     = vecs_t @ np.diag(np.sqrt(np.maximum(vals_t, eps)))         @ vecs_t.T

    W = Cs_inv_sqrt @ Ct_sqrt
    return (X_source - X_source.mean(axis=0)) @ W + X_target.mean(axis=0)


# ── Fine-tuning Transfer ──────────────────────────────────────────────────

def finetune_transfer(model, X_target: np.ndarray,
                      y_target: np.ndarray,
                      epochs: int = 20,
                      lr: float = 1e-3) -> object:
    """
    Fine-tuning de um modelo PyTorch no domínio alvo.

    Congela todas as camadas excepto a última (classificador linear),
    e retreina apenas essa camada no novo domínio.

    Parameters
    ----------
    model    : modelo PyTorch com atributo .fc ou última camada linear
    X_target : features do domínio alvo
    y_target : labels do domínio alvo

    Returns
    -------
    modelo fine-tuned (in-place + retornado)
    """
    import torch
    import torch.nn as nn

    # congelar todos os parâmetros
    for param in model.parameters():
        param.requires_grad = False

    # descongelar apenas a última camada linear
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        for param in last_linear.parameters():
            param.requires_grad = True

    opt     = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loss_fn = nn.BCELoss()
    X_t     = torch.from_numpy(X_target.astype(np.float32))
    y_t     = torch.from_numpy(y_target.astype(np.float32)).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out  = model(X_t)
        if out.shape[-1] == 2:
            out = torch.softmax(out, dim=-1)[:, 1:2]
        loss = loss_fn(out.clamp(1e-6, 1 - 1e-6), y_t)
        loss.backward()
        opt.step()

    return model


# ── Cross-domain Evaluation ───────────────────────────────────────────────

def cross_domain_evaluation(predict_fn_source,
                              X_source: np.ndarray, y_source: np.ndarray,
                              X_target: np.ndarray, y_target: np.ndarray,
                              use_coral: bool = True) -> dict:
    """
    Avalia transferência de conhecimento entre dois domínios.

    Parameters
    ----------
    predict_fn_source : callable que aceita X e retorna probs (N,)
                        (modelo treinado no domínio fonte)
    X_source, y_source: dados do domínio fonte
    X_target, y_target: dados do domínio alvo
    use_coral         : se True, alinha features com CORAL antes de avaliar

    Returns
    -------
    dict com:
      mmd              : distância entre domínios
      acc_source       : acurácia no domínio fonte (upper bound)
      acc_transfer     : acurácia no alvo sem alinhamento
      acc_coral        : acurácia no alvo com CORAL (se use_coral=True)
      transfer_gap     : acc_source - acc_transfer
      coral_improvement: acc_coral - acc_transfer (se use_coral=True)
    """
    sc = RobustScaler()
    sc.fit(X_source)
    Xs_sc = sc.transform(X_source)
    Xt_sc = sc.transform(X_target)

    mmd = maximum_mean_discrepancy(Xs_sc, Xt_sc)

    p_source   = predict_fn_source(Xs_sc)
    p_transfer = predict_fn_source(Xt_sc)

    acc_source   = accuracy_score(y_source.astype(int),
                                  (p_source >= 0.5).astype(int))
    acc_transfer = accuracy_score(y_target.astype(int),
                                  (p_transfer >= 0.5).astype(int))

    result = {
        "mmd":           mmd,
        "acc_source":    float(acc_source),
        "acc_transfer":  float(acc_transfer),
        "transfer_gap":  float(acc_source - acc_transfer),
    }

    if use_coral:
        Xt_aligned  = coral_align(Xt_sc, Xs_sc)
        p_coral     = predict_fn_source(Xt_aligned)
        acc_coral   = accuracy_score(y_target.astype(int),
                                     (p_coral >= 0.5).astype(int))
        result["acc_coral"]         = float(acc_coral)
        result["coral_improvement"] = float(acc_coral - acc_transfer)

    return result
