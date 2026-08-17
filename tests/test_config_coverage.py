"""
Garante que todo ticker da carteira tem entrada em ASSET_CLASSES e
TICKER_CALENDAR — sem isso, o ticker cai num default silencioso (classe de
ativo errada / calendário NYSE) em vez de falhar alto. Já aconteceu com o
PPFB.DE (iShares Physical Gold ETC, Xetra) ficar de fora dos dois dicts.
"""
import json
from pathlib import Path

from config.settings import ASSET_CLASSES, TICKER_CALENDAR

PORTFOLIO_FILE = Path(__file__).resolve().parent.parent / "config" / "portfolio.json"


def _portfolio_tickers() -> list[str]:
    cfg = json.loads(PORTFOLIO_FILE.read_text())
    return cfg.get("etoro", []) + cfg.get("etf_acumulacao", [])


def test_portfolio_tickers_have_asset_class():
    faltando = [t for t in _portfolio_tickers() if t not in ASSET_CLASSES]
    assert not faltando, f"Tickers da carteira sem ASSET_CLASSES: {faltando}"


def test_portfolio_tickers_have_calendar():
    faltando = [t for t in _portfolio_tickers() if t not in TICKER_CALENDAR]
    assert not faltando, f"Tickers da carteira sem TICKER_CALENDAR: {faltando}"
