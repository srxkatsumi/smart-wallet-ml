"""Tickers novos herdam o calendário da bolsa pelo sufixo; o cache de dividendos busca só o que falta."""
import json
import sys
from pathlib import Path

import pandas as pd

from config.settings import calendar_for
from data.calendars import target_dates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))


def test_calendario_pelo_sufixo_e_dict_explicito_tem_prioridade():
    assert calendar_for("ITX.MC") == "XMAD"
    assert calendar_for("MC.PA") == "XPAR"
    assert calendar_for("ADYEN.AS") == "XAMS"
    assert calendar_for("EXUS.MI") == "XMIL"       # do dict explícito
    assert calendar_for("KLAR") == "NYSE"


def test_madrid_pula_natal_e_ano_novo():
    t = target_dates(pd.Timestamp("2026-12-23"), "ITX.MC")
    assert [d.strftime("%Y-%m-%d") for d in (t[1], t[2], t[3])] == ["2026-12-24", "2026-12-28", "2026-12-29"]


def test_cache_de_dividendos_busca_so_os_que_faltam(monkeypatch, tmp_path):
    import export_watchlist as ew
    cache = tmp_path / "div.json"
    cache.write_text(json.dumps({"AAA": {"pays_dividends": False, "currency": "USD"}}))
    monkeypatch.setattr(ew, "DIV_CACHE", cache)
    monkeypatch.setattr(ew, "DIV_BATCH_SLEEP", 0)
    pedidos = []

    class _T:
        def __init__(self, t):
            pedidos.append(t)

        def get_info(self):
            raise RuntimeError("sem rede")
    monkeypatch.setattr(ew.yf, "Ticker", _T)
    out = ew.fetch_dividends(["AAA", "ITX.MC"])
    assert pedidos == ["ITX.MC"]
    assert out["ITX.MC"]["currency"] == "EUR" and "AAA" in out


def test_toquio_e_hong_kong():
    assert calendar_for("8035.T") == "XTKS" and calendar_for("3690.HK") == "XHKG"
    # Tóquio fecha de 31/12 a 3/1; Hong Kong fecha no Natal e no dia seguinte
    t = target_dates(pd.Timestamp("2026-12-30"), "7974.T")
    assert t[1].strftime("%Y-%m-%d") == "2027-01-04"
    h = target_dates(pd.Timestamp("2026-12-24"), "3690.HK")
    assert h[1].strftime("%Y-%m-%d") == "2026-12-28"


def test_moeda_pelo_sufixo_asia_e_suica():
    import export_watchlist as ew
    assert ew._moeda_pelo_sufixo("7974.T") == "JPY"
    assert ew._moeda_pelo_sufixo("1810.HK") == "HKD"
    assert ew._moeda_pelo_sufixo("UBSG.SW") == "CHF"
    assert ew._moeda_pelo_sufixo("HSBC") == "USD"
