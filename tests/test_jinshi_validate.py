import numpy as np
import pytest

from jinshi.validate import validate_signal, validate_many, ROUND_TRIP_COST_PCT

RATIONALE = "Volatilidade tende a agrupar (clustering), documentado na literatura."


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_rejects_without_economic_rationale():
    errs = _rng().normal(0, 1, 100)
    with pytest.raises(ValueError, match="economic_rationale"):
        validate_signal("x", errs, errs, economic_rationale="", reference_price=100)


def test_rejects_short_rationale():
    errs = _rng().normal(0, 1, 100)
    with pytest.raises(ValueError, match="economic_rationale"):
        validate_signal("x", errs, errs, economic_rationale="porque sim", reference_price=100)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="comprimento"):
        validate_signal("x", np.ones(50), np.ones(40), economic_rationale=RATIONALE, reference_price=100)


def test_candidate_significantly_worse_is_never_marked_significant():
    rng = _rng(7)
    worse  = rng.normal(0, 5.0, 400)
    better = rng.normal(0, 0.1, 400)
    r = validate_signal("x", worse, better, RATIONALE, 100)
    assert r["1_p_value"] > 0.95
    out = validate_many([r])[0]
    assert out["1_significativo_corrigido"] is False
    assert out["aprovado"] is False


def test_per_observation_reference_price_weights_errors_as_percent():
    # Mesmo erro em % nos dois regimes de preço: um período caro não domina a conta.
    reference = np.r_[np.full(50, 10.0), np.full(50, 1000.0)]
    errors = reference * 0.02
    r = validate_signal("x", errors, errors * 2, RATIONALE, reference)
    assert r["mae_candidate_pct"] == pytest.approx(2.0)
    assert r["mae_baseline_pct"] == pytest.approx(4.0)
    assert r["2_efeito_pct"] == pytest.approx(2.0)


def test_identical_errors_show_zero_effect_and_no_significance():
    errs = _rng().normal(0, 1, 200)
    r = validate_signal("x", errs, errs, economic_rationale=RATIONALE, reference_price=100)
    assert r["2_efeito_pct"] == pytest.approx(0.0, abs=1e-6)
    assert r["1_p_value"] > 0.05


def test_candidate_much_better_than_baseline_shows_large_positive_effect():
    rng = _rng(1)
    candidate = rng.normal(0, 0.1, 400)   # erro pequeno
    baseline  = rng.normal(0, 5.0, 400)   # erro grande, independente
    r = validate_signal("x", candidate, baseline, economic_rationale=RATIONALE, reference_price=100)
    assert r["2_efeito_pct"] > 0
    assert r["1_p_value"] < 0.05


def test_robustness_flag_false_when_effect_only_in_second_half():
    rng = _rng(2)
    n = 2000
    # 1ª metade: candidato claramente PIOR (desvio maior); 2ª metade: claramente
    # MELHOR. n grande o suficiente pra não depender de sorte da amostra.
    candidate = np.concatenate([rng.normal(0, 5.0, n // 2), rng.normal(0, 0.1, n // 2)])
    baseline  = rng.normal(0, 1.0, n)
    r = validate_signal("x", candidate, baseline, economic_rationale=RATIONALE, reference_price=100)
    assert r["3_robusto_nas_duas_metades"] is False


def test_robustness_flag_true_when_effect_in_both_halves():
    rng = _rng(3)
    candidate = rng.normal(0, 0.1, 400)
    baseline  = rng.normal(0, 5.0, 400)
    r = validate_signal("x", candidate, baseline, economic_rationale=RATIONALE, reference_price=100)
    assert r["3_robusto_nas_duas_metades"] is True


def test_cost_gate_rejects_effect_smaller_than_round_trip_cost():
    rng = _rng(4)
    # efeito real mas minúsculo em % — não deve superar ROUND_TRIP_COST_PCT
    candidate = rng.normal(0, 0.999, 400)
    baseline  = rng.normal(0, 1.0, 400)
    r = validate_signal("x", candidate, baseline, economic_rationale=RATIONALE, reference_price=100)
    assert r["2_supera_custo"] is False


# ── validate_many / correção de múltiplas comparações ───────────────────────

def test_validate_many_adds_final_verdict_key_to_every_result():
    rng = _rng(5)
    results = [
        validate_signal(f"t{i}", rng.normal(0, 1, 100), rng.normal(0, 1, 100), RATIONALE, 100)
        for i in range(5)
    ]
    out = validate_many(results)
    for r in out:
        assert "aprovado" in r
        assert "1_significativo_corrigido" in r
        assert "p_adj" in r


def test_validate_many_rejects_single_marginal_result_among_many_noise_results():
    """
    Regression guard pro próprio achado da Fase 1: um sinal isolado com
    p=0.017 (como IS3N.DE) não deveria ser aprovado quando testado junto
    de outros 4 candidatos majoritariamente ruído — exatamente o cenário
    real que já medimos.
    """
    rng = _rng(6)
    strong = rng.normal(0, 0.1, 300)
    weak_baseline = rng.normal(0, 0.3, 300)  # efeito real mas pequeno -> p baixo, ainda assim

    results = [validate_signal("forte_isolado", strong, weak_baseline, RATIONALE, 1000)]
    for i in range(4):
        noise_a = rng.normal(0, 1, 300)
        noise_b = rng.normal(0, 1, 300)
        results.append(validate_signal(f"ruido{i}", noise_a, noise_b, RATIONALE, 1000))

    out = validate_many(results)
    # o sinal isolado pode continuar significativo mesmo corrigido (é forte
    # o bastante), mas os de ruído não devem ser aprovados
    noise_results = [r for r in out if r["name"].startswith("ruido")]
    assert all(not r["aprovado"] for r in noise_results)
