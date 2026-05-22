import pytest
from portfolio.pnl import calculate_etoro_pnl


EUR_USD = 1.10

RESULTADOS_ML = {
    "AAPL": {"close_now": 200.0},
    "NVDA": {"close_now": 130.0},
}


def _make_ativo(ticker: str, moeda: str, pa: float, uni: float, fee: float) -> dict:
    return {
        "ticker":         ticker,
        "nome":           ticker,
        "moeda":          moeda,
        "preco_abertura": pa,
        "unidades":       uni,
        "fee_euro":       fee,
    }


def test_breakeven_higher_with_fees():
    """Com fees > 0, o breakeven deve ser superior ao preço de abertura em EUR."""
    etoro = [_make_ativo("AAPL", "USD", pa=150.0, uni=5.0, fee=10.0)]
    resumo, _ = calculate_etoro_pnl(etoro, RESULTADOS_ML, EUR_USD)
    r = resumo[0]
    pa_eur = 150.0 / EUR_USD
    assert r["breakeven_eur"] > pa_eur, (
        f"Breakeven ({r['breakeven_eur']:.4f}) devia ser > preço abertura EUR ({pa_eur:.4f})"
    )


def test_breakeven_equals_pa_without_fees():
    """Sem fees, o breakeven deve ser igual ao preço de abertura em EUR."""
    etoro = [_make_ativo("AAPL", "USD", pa=150.0, uni=5.0, fee=0.0)]
    resumo, _ = calculate_etoro_pnl(etoro, RESULTADOS_ML, EUR_USD)
    r = resumo[0]
    pa_eur = 150.0 / EUR_USD
    assert abs(r["breakeven_eur"] - pa_eur) < 1e-9, (
        f"Sem fees, breakeven ({r['breakeven_eur']:.6f}) devia ser == pa_eur ({pa_eur:.6f})"
    )


def test_breakeven_with_fees_greater_than_without():
    """Breakeven com fees é sempre maior do que sem fees."""
    ativo_com = _make_ativo("AAPL", "USD", pa=150.0, uni=5.0, fee=10.0)
    ativo_sem = _make_ativo("AAPL", "USD", pa=150.0, uni=5.0, fee=0.0)

    resumo_com, _ = calculate_etoro_pnl([ativo_com], RESULTADOS_ML, EUR_USD)
    resumo_sem, _ = calculate_etoro_pnl([ativo_sem], RESULTADOS_ML, EUR_USD)

    assert resumo_com[0]["breakeven_eur"] > resumo_sem[0]["breakeven_eur"]


def test_pnl_usd_conversion():
    """P&L em USD deve ser convertido para EUR correctamente."""
    etoro = [_make_ativo("AAPL", "USD", pa=110.0, uni=1.0, fee=0.0)]
    resumo, _ = calculate_etoro_pnl(etoro, RESULTADOS_ML, EUR_USD)
    r = resumo[0]
    expected_close_eur = 200.0 / EUR_USD
    assert abs(r["close_eur"] - expected_close_eur) < 1e-6
