"""
O portão que qualquer sinal candidato do Jinshi precisa atravessar antes de
ser considerado real. Critérios combinados, nenhum isolado basta:

  1. O candidato é MELHOR que o baseline, com significância unicaudal que
     sobrevive à correção de múltiplas comparações (Benjamini-Hochberg).
  2. A melhora do erro supera o custo de operar (spread + taxa). É uma condição
     mínima, não prova de lucro: uma previsão melhor que o custo ainda pode não
     render uma estratégia lucrativa, mas uma melhora menor que o custo não paga
     a operação.
  3. Robustez: o candidato vence nas duas metades cronológicas do período
     testado, não só no total (senão pode ser um evento isolado).
  4. Razão econômica declarada: não verificável por código, mas exigida.

Os três critérios numéricos usam a mesma medida: erro absoluto em % do preço de
referência de cada observação, comparável entre ativos e entre épocas.

validate_signal() avalia um candidato. Só validate_many() emite veredito, porque
"múltiplas comparações" só existe no agregado.
"""
import numpy as np
from scipy import stats

from evaluation.significance import apply_bh_correction
from evaluation.statistical_tests import diebold_mariano

ROUND_TRIP_COST_PCT = 0.15


def _halves(a: np.ndarray, b: np.ndarray) -> tuple:
    mid = len(a) // 2
    return (a[:mid], b[:mid]), (a[mid:], b[mid:])


def validate_signal(name: str, errors_candidate: np.ndarray, errors_baseline: np.ndarray,
                    economic_rationale: str, reference_price) -> dict:
    """
    `errors_*`: erros de previsão com sinal (actual - previsto), pareados, em
    ordem cronológica. `reference_price`: preço de referência, um valor único ou
    um por observação (preferível quando o preço muda muito no período).
    """
    if not economic_rationale or len(economic_rationale.strip()) < 20:
        raise ValueError(
            "economic_rationale é obrigatório e precisa explicar POR QUE esse "
            "sinal deveria existir, não só que ele apareceu nos dados."
        )
    errors_candidate = np.asarray(errors_candidate, dtype=float)
    errors_baseline  = np.asarray(errors_baseline, dtype=float)
    if len(errors_candidate) != len(errors_baseline) or len(errors_candidate) < 10:
        raise ValueError("errors_candidate e errors_baseline precisam ter o mesmo "
                         "comprimento (pareados) e pelo menos 10 observações.")
    reference = np.asarray(reference_price, dtype=float)
    if np.any(reference <= 0):
        raise ValueError("reference_price precisa ser positivo.")

    candidate_pct = errors_candidate / reference * 100
    baseline_pct  = errors_baseline / reference * 100

    mae_candidate = float(np.mean(np.abs(candidate_pct)))
    mae_baseline  = float(np.mean(np.abs(baseline_pct)))
    effect_pct    = mae_baseline - mae_candidate

    # Estatística negativa = candidato com erro menor.
    dm = diebold_mariano(candidate_pct, baseline_pct, loss="absolute")
    p_one_sided = float(stats.norm.cdf(dm["statistic"]))

    (a1, b1), (a2, b2) = _halves(candidate_pct, baseline_pct)
    half1_better = np.mean(np.abs(a1)) < np.mean(np.abs(b1))
    half2_better = np.mean(np.abs(a2)) < np.mean(np.abs(b2))

    return {
        "name": name,
        "n": len(errors_candidate),
        "mae_candidate_pct": round(mae_candidate, 4),
        "mae_baseline_pct": round(mae_baseline, 4),
        "1_p_value": round(p_one_sided, 4),
        "2_efeito_pct": round(effect_pct, 4),
        "2_supera_custo": bool(effect_pct > ROUND_TRIP_COST_PCT),
        "3_robusto_nas_duas_metades": bool(half1_better and half2_better),
        "4_razao_economica": economic_rationale,
    }


def validate_many(results: list[dict]) -> list[dict]:
    """Aplica BH entre os p-values de vários validate_signal() e decide a aprovação final."""
    for r in results:
        r["p"] = r["1_p_value"]
    apply_bh_correction(results, p_key="p")
    for r in results:
        r["1_significativo_corrigido"] = r.pop("sig")
        del r["p"]
        r["aprovado"] = bool(
            r["1_significativo_corrigido"]
            and r["2_supera_custo"]
            and r["3_robusto_nas_duas_metades"]
        )
    return results
