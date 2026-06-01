"""
Fase 10 — Explicabilidade (XAI)

SHAP — SHapley Additive exPlanations (Lundberg & Lee, 2017 — NeurIPS):
  Baseia-se na teoria dos jogos cooperativos (Shapley, 1953). Cada feature
  recebe um valor que representa a sua contribuição marginal média para a
  previsão, em relação ao valor base (média do modelo). A soma dos SHAP
  values de todas as features iguala sempre a diferença entre a previsão
  e o valor base — propriedade de consistência única.

  TreeExplainer: O(T·D) exact computation para árvores.
  DeepExplainer: backpropagation-based para redes neurais (PyTorch).

Attention Weights (Vaswani et al., 2017):
  Extraí os pesos de atenção dos cabeçotes do Transformer. Cada peso
  A[i,j] representa quanta atenção o passo i presta ao passo j.
  Alta atenção num passo passado = aquele momento é relevante para a
  previsão actual.

LIME — Local Interpretable Model-agnostic Explanations
  (Ribeiro, Singh & Guestrin, 2016 — KDD):
  Aproxima o modelo com um modelo linear local em torno de cada previsão.
  Pertuba a vizinhança de x e treina uma regressão linear sobre as
  perturbações ponderadas pela proximidade.
  Model-agnostic: funciona com qualquer modelo via predict_fn.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ── SHAP para modelos de árvore ───────────────────────────────────────────

def shap_tree(model, X: np.ndarray,
              feature_names: list | None = None) -> dict:
    """
    Calcula SHAP values para modelos de árvore (RF, GB, XGBoost, LightGBM,
    CatBoost).

    Returns
    -------
    dict com:
      shap_values      : array (N, n_features) de valores SHAP
      expected_value   : valor base do modelo
      mean_abs_shap    : importância global de cada feature (média |SHAP|)
      top5_features    : top 5 features com maior importância global
    """
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    # SHAP 0.41+ pode retornar (N, features, classes) — colapsar para (N, features)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs    = np.abs(sv).mean(axis=0)
    top5_idx    = np.argsort(mean_abs)[::-1][:5].tolist()
    top5_names  = ([feature_names[i] for i in top5_idx]
                   if feature_names else [str(i) for i in top5_idx])

    return {
        "shap_values":    sv,
        "expected_value": float(np.array(explainer.expected_value).flat[-1]),
        "mean_abs_shap":  mean_abs,
        "top5_features":  list(zip(top5_names, mean_abs[top5_idx].tolist())),
    }


# ── SHAP para redes neurais (PyTorch) ─────────────────────────────────────

def shap_deep(model, X: np.ndarray,
              background: np.ndarray | None = None,
              feature_names: list | None = None) -> dict:
    """
    Calcula SHAP values para modelos PyTorch via DeepExplainer.

    Parameters
    ----------
    model      : modelo PyTorch em modo eval()
    X          : dados de teste (numpy array)
    background : dados de fundo para o explainer (subset de X; default 50 amostras)
    """
    import shap
    import torch

    if background is None:
        n_bg = min(50, len(X))
        background = X[:n_bg]

    bg_t  = torch.from_numpy(background.astype(np.float32))
    X_t   = torch.from_numpy(X.astype(np.float32))

    model.eval()
    explainer   = shap.DeepExplainer(model, bg_t)
    shap_values = explainer.shap_values(X_t)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    if hasattr(sv, "numpy"):
        sv = sv.numpy()

    mean_abs   = np.abs(sv).mean(axis=0)
    top5_idx   = np.argsort(mean_abs)[::-1][:5]
    top5_names = ([feature_names[i] for i in top5_idx]
                  if feature_names else [str(i) for i in top5_idx])

    return {
        "shap_values":   sv,
        "mean_abs_shap": mean_abs,
        "top5_features": list(zip(top5_names, mean_abs[top5_idx].tolist())),
    }


# ── Attention weights (Transformer) ──────────────────────────────────────

def attention_weights(transformer_model, X: np.ndarray) -> dict:
    """
    Extrai os pesos de atenção de um modelo Transformer PyTorch.

    Regista os attention weights de cada cabeçote usando forward hooks.

    Returns
    -------
    dict com:
      weights          : lista de tensores de atenção por camada
      mean_over_heads  : média sobre cabeçotes, shape (N, seq_len, seq_len)
      most_attended    : posição temporal mais atendida em média
    """
    import torch

    captured = []

    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            captured.append(output[1].detach())

    handles = []
    for module in transformer_model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            handles.append(module.register_forward_hook(hook))

    X_t = torch.from_numpy(X.astype(np.float32))
    transformer_model.eval()
    with torch.no_grad():
        transformer_model(X_t)

    for h in handles:
        h.remove()

    if not captured:
        return {"weights": [], "mean_over_heads": None, "most_attended": None}

    attn = torch.stack(captured, dim=0).numpy()   # (layers, N, heads, seq, seq)
    mean = attn.mean(axis=(0, 2))                  # (N, seq, seq)
    most_attended = int(mean.mean(axis=(0, 1)).argmax())

    return {
        "weights":         captured,
        "mean_over_heads": mean,
        "most_attended":   most_attended,
    }


# ── LIME ─────────────────────────────────────────────────────────────────

def lime_explain(predict_fn, X: np.ndarray,
                 feature_names: list | None = None,
                 n_samples: int = 100,
                 n_features_display: int = 5) -> dict:
    """
    Explica previsões individuais com LIME (model-agnostic).

    Parameters
    ----------
    predict_fn         : função que aceita array (N, n_features) e retorna
                         probabilidades (N, n_classes) ou (N,)
    X                  : dados a explicar
    n_samples          : amostras de perturbação por explicação
    n_features_display : features a mostrar por explicação

    Returns
    -------
    dict com:
      explanations  : lista de explicações LIME (uma por amostra)
      global_importance : importância global média (média |peso| por feature)
      top5_features : top 5 features mais importantes globalmente
    """
    from lime.lime_tabular import LimeTabularExplainer

    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    exp   = LimeTabularExplainer(
        X,
        feature_names=names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def _predict(arr):
        p = predict_fn(arr)
        if p.ndim == 1:
            return np.column_stack([1 - p, p])
        return p

    explanations    = []
    importances_all = np.zeros(X.shape[1])

    for i, x in enumerate(X):
        try:
            e = exp.explain_instance(
                x, _predict,
                num_features=n_features_display,
                num_samples=n_samples,
            )
            explanations.append(e)
            for feat_name, weight in e.as_list():
                for j, fn in enumerate(names):
                    if fn in feat_name:
                        importances_all[j] += abs(weight)
                        break
        except Exception:
            continue

    if len(explanations) > 0:
        importances_all /= len(explanations)

    top5_idx  = np.argsort(importances_all)[::-1][:5]
    top5      = [(names[i], float(importances_all[i])) for i in top5_idx]

    return {
        "explanations":       explanations,
        "global_importance":  importances_all,
        "top5_features":      top5,
    }


# ── Relatório completo ────────────────────────────────────────────────────

def feature_importance_report(model_type: str, model,
                               X: np.ndarray,
                               feature_names: list | None = None) -> dict:
    """
    Gera relatório de importância de features para o modelo dado.

    model_type : "tree" | "neural" | "transformer"
    """
    if model_type == "tree":
        return shap_tree(model, X, feature_names)
    elif model_type == "neural":
        return shap_deep(model, X, feature_names=feature_names)
    elif model_type == "transformer":
        return attention_weights(model, X)
    else:
        raise ValueError(f"model_type desconhecido: {model_type}")
