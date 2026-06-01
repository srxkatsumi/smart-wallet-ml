"""
Fase 13 — Teoria da Informação

Esta fase responde à pergunta fundamental antes de modelar: "quanta informação
previsível existe neste domínio?". É o enquadramento teórico que une os três
domínios da tese.

Shannon Entropy (Shannon, 1948 — "A Mathematical Theory of Communication",
  Bell System Technical Journal):
  H(X) = -Σ p(x) log₂ p(x)
  Máxima: H = log₂(n) bits para n estados equiprováveis (loteria perfeita).
  Mínima: H = 0 bits quando o resultado é determinístico.

Mutual Information (Shannon, 1948 / Cover & Thomas, 2006):
  I(X;Y) = H(Y) - H(Y|X)
  Mede quanto X reduz a incerteza de Y. I=0: independência total.
  sklearn.feature_selection.mutual_info_classif estima MI via k-NN.

Permutation Entropy (Bandt & Pompe, 2002 — Physical Review Letters):
  Mede a complexidade de uma série temporal via ordenação de padrões
  de comprimento m. PE próxima de 1 = série máxima-aleatória.
  PE próxima de 0 = série altamente ordenada/previsível.

Transfer Entropy (Schreiber, 2000 — Physical Review Letters):
  TE(X→Y) = I(Y_{t+1}; X_t | Y_t)
  Mede a informação que X transmite sobre o futuro de Y para além do
  que Y já sabe sobre si próprio. TE > 0 indica causalidade de Granger
  informacional.

Resultado esperado por domínio:
  Mega Sena  → H máxima, MI ≈ 0, PE ≈ 1, TE ≈ 0
  Carteira   → H moderada, MI > 0 em algumas features, PE < 1
  E-commerce → H baixa, MI alto para features sazonais, PE baixo
"""

import math
import warnings
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ── Shannon Entropy ───────────────────────────────────────────────────────

def shannon_entropy(series: np.ndarray, n_bins: int = 10,
                    base: int = 2) -> float:
    """
    Estima a entropia de Shannon de uma série contínua via histograma.

    Parameters
    ----------
    series : array 1D de observações
    n_bins : número de bins do histograma
    base   : base do logaritmo (2 = bits, e = nats)

    Returns
    -------
    H em bits (ou nats se base=e)
    """
    counts, _ = np.histogram(series, bins=n_bins)
    probs     = counts / counts.sum()
    probs     = probs[probs > 0]
    return float(stats.entropy(probs, base=base))


def max_entropy(n_states: int, base: int = 2) -> float:
    """Entropia máxima para n_states estados equiprováveis."""
    return float(np.log(n_states) / np.log(base))


def normalized_entropy(series: np.ndarray, n_bins: int = 10) -> float:
    """
    Entropia normalizada em [0, 1]. 1 = máximo aleatório; 0 = determinístico.
    """
    h      = shannon_entropy(series, n_bins)
    h_max  = max_entropy(n_bins)
    return float(h / h_max) if h_max > 0 else 0.0


# ── Mutual Information ────────────────────────────────────────────────────

def mutual_information_features(X: np.ndarray, y: np.ndarray,
                                 feature_names: list | None = None,
                                 n_neighbors: int = 3) -> dict:
    """
    Calcula Mutual Information entre cada feature e o label y.

    Returns
    -------
    dict com:
      mi_scores     : array de MI por feature
      top5_features : 5 features com maior MI
      mean_mi       : MI médio (proxy de previsibilidade)
    """
    y_int  = y.astype(int)
    mi     = mutual_info_classif(X, y_int, n_neighbors=n_neighbors,
                                 random_state=42)
    top5   = np.argsort(mi)[::-1][:5].tolist()
    names  = feature_names or [f"f{i}" for i in range(X.shape[1])]
    top5_named = [(names[i], float(mi[i])) for i in top5]

    return {
        "mi_scores":     mi,
        "top5_features": top5_named,
        "mean_mi":       float(mi.mean()),
        "max_mi":        float(mi.max()),
    }


# ── Permutation Entropy ───────────────────────────────────────────────────

def permutation_entropy(series: np.ndarray, m: int = 3,
                        tau: int = 1) -> float:
    """
    Entropia de permutação (Bandt & Pompe, 2002).

    Parameters
    ----------
    series : série temporal 1D
    m      : comprimento do padrão (embedding dimension)
    tau    : atraso temporal

    Returns
    -------
    PE normalizada em [0, 1]. 1 = máximo aleatório.
    """
    n         = len(series) - (m - 1) * tau
    patterns  = {}
    for i in range(n):
        window = tuple(np.argsort(series[i: i + m * tau: tau]))
        patterns[window] = patterns.get(window, 0) + 1

    total  = sum(patterns.values())
    probs  = np.array([v / total for v in patterns.values()])
    h      = stats.entropy(probs, base=2)
    h_max  = np.log2(math.factorial(m))
    return float(h / h_max) if h_max > 0 else 0.0


# ── Transfer Entropy ──────────────────────────────────────────────────────

def transfer_entropy(source: np.ndarray, target: np.ndarray,
                     lag: int = 1, n_bins: int = 5) -> float:
    """
    Transfer Entropy de source para target (Schreiber, 2000).

    TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)

    Estimativa discreta via histograma 2D/3D.
    TE > 0: source transmite informação sobre o futuro de target.
    TE ≈ 0: independência (esperado na Mega Sena).
    """
    def _discretize(s):
        bins  = np.percentile(s, np.linspace(0, 100, n_bins + 1))
        bins  = np.unique(bins)
        return np.digitize(s, bins[1:-1])

    s = _discretize(source)
    t = _discretize(target)
    n = min(len(s), len(t)) - lag

    yt1 = t[lag:lag + n]
    yt  = t[:n]
    xt  = s[:n]

    def _h2(a, b):
        cnt = np.zeros((n_bins, n_bins))
        for i in range(n):
            ai = min(int(a[i]) - 1, n_bins - 1)
            bi = min(int(b[i]) - 1, n_bins - 1)
            cnt[ai, bi] += 1
        p = cnt / cnt.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _h3(a, b, c):
        cnt = np.zeros((n_bins, n_bins, n_bins))
        for i in range(n):
            ai = min(int(a[i]) - 1, n_bins - 1)
            bi = min(int(b[i]) - 1, n_bins - 1)
            ci = min(int(c[i]) - 1, n_bins - 1)
            cnt[ai, bi, ci] += 1
        p = cnt / cnt.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    te = _h2(yt1, yt) - _h3(yt1, yt, xt) + _h2(yt1, xt) - shannon_entropy(yt1)
    return float(max(te, 0.0))


# ── Relatório completo por domínio ────────────────────────────────────────

def domain_predictability_report(X: np.ndarray, y: np.ndarray,
                                  series: np.ndarray | None = None,
                                  feature_names: list | None = None,
                                  domain_name: str = "unknown") -> dict:
    """
    Relatório completo de previsibilidade de um domínio.

    Parameters
    ----------
    X           : matriz de features
    y           : labels (0/1)
    series      : série temporal univariada para PE (opcional; usa y se None)
    feature_names: nomes das features
    domain_name : "carteira" | "megasena" | "ecommerce"
    """
    s = series if series is not None else y.astype(float)

    sc      = RobustScaler()
    X_sc    = sc.fit_transform(X)

    h_y     = normalized_entropy(y.astype(float))
    mi      = mutual_information_features(X_sc, y, feature_names)
    pe      = permutation_entropy(s, m=3, tau=1)
    te      = transfer_entropy(s[:-1], s[1:]) if len(s) > 10 else 0.0

    verdict = (
        "Alta previsibilidade" if h_y < 0.5 and mi["mean_mi"] > 0.01
        else "Previsibilidade moderada" if h_y < 0.8
        else "Processo próximo de aleatório"
    )

    return {
        "domain":              domain_name,
        "normalized_entropy":  h_y,
        "mean_mutual_info":    mi["mean_mi"],
        "max_mutual_info":     mi["max_mi"],
        "top5_features":       mi["top5_features"],
        "permutation_entropy": pe,
        "transfer_entropy":    te,
        "verdict":             verdict,
    }
