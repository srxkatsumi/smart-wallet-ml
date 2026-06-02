import logging
import io
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_CAIXA_URL = "https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Column mapping for Caixa Excel format
_EXCEL_COLS = {
    "Concurso":        "concurso",
    "Data do Sorteio": "data",
    "Bola1": "b1", "Bola2": "b2", "Bola3": "b3",
    "Bola4": "b4", "Bola5": "b5", "Bola6": "b6",
}

# Column mapping for Caixa HTML format
_HTML_COLS = {
    "Concurso":     "concurso",
    "Data Sorteio": "data",
    "1ª Dezena": "b1", "2ª Dezena": "b2", "3ª Dezena": "b3",
    "4ª Dezena": "b4", "5ª Dezena": "b5", "6ª Dezena": "b6",
}


def _parse_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns=_EXCEL_COLS)
    keep = list(_EXCEL_COLS.values())
    df = df[[c for c in keep if c in df.columns]]
    df["data"]     = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    for c in ["b1","b2","b3","b4","b5","b6"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df.dropna(subset=["concurso","data","b1"]).sort_values("concurso").reset_index(drop=True)
    logger.info("Excel: %d sorteios (concurso %d → %d)",
                len(df), int(df["concurso"].iloc[0]), int(df["concurso"].iloc[-1]))
    return df


def _parse_caixa_html(content: bytes) -> pd.DataFrame:
    tables = pd.read_html(io.BytesIO(content), encoding="utf-8", thousands=".", decimal=",")
    for t in tables:
        if "Concurso" in t.columns and "1ª Dezena" in t.columns:
            df = t.rename(columns=_HTML_COLS)
            df = df[[c for c in _HTML_COLS.values() if c in df.columns]]
            df["data"]     = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
            for c in ["b1","b2","b3","b4","b5","b6"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["concurso","data","b1"]).sort_values("concurso").reset_index(drop=True)
            logger.info("HTML: %d sorteios", len(df))
            return df
    raise ValueError("Tabela não encontrada no HTML da Caixa")


_API_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"


def _fetch_api_latest() -> pd.DataFrame | None:
    """Fetch the latest available draw from the Caixa public API."""
    try:
        resp = requests.get(_API_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        d     = resp.json()
        num   = int(d.get("numero", 0))
        data  = d.get("dataApuracao", "")
        bolas = [int(b) for b in d.get("listaDezenas", [])]
        if num > 0 and len(bolas) == 6:
            return pd.DataFrame([{
                "concurso": num,
                "data":     pd.to_datetime(data, dayfirst=True),
                "b1": bolas[0], "b2": bolas[1], "b3": bolas[2],
                "b4": bolas[3], "b5": bolas[4], "b6": bolas[5],
            }])
    except Exception as e:
        logger.warning("API Caixa falhou: %s", e)
    return None


def _supplement_with_api(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Append any draw newer than the last in df using the Caixa API."""
    latest_api = _fetch_api_latest()
    if latest_api is None:
        return df
    last_known = int(df["concurso"].max())
    new_rows   = latest_api[latest_api["concurso"] > last_known]
    if new_rows.empty:
        return df
    logger.info("API: %d novo(s) sorteio(s) encontrado(s) (concurso %d)",
                len(new_rows), int(new_rows["concurso"].iloc[-1]))
    df = pd.concat([df, new_rows], ignore_index=True).sort_values("concurso").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def download_results(cache_path: Path, force: bool = False) -> pd.DataFrame:
    """Load Mega Sena results and supplement with latest from API.

    Priority: cached CSV (if fresher than Excel) → Excel → API only.
    Always calls API at the end to fill in draws newer than the base source.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1 — Cached CSV: use if it exists and is not being forced
    if not force and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["data"])
            logger.info("Cache CSV: %d sorteios", len(df))
            return _supplement_with_api(df, cache_path)
        except Exception:
            logger.warning("Cache corrompido — tentando Excel")

    # 2 — Excel file as historical base
    search_dirs = [
        cache_path.parent.parent.parent,   # test_ml/
        cache_path.parent,                  # output/
        cache_path.parent.parent,           # loteria/
    ]
    for d in search_dirs:
        for xlsx in d.glob("*.xlsx"):
            logger.info("Excel encontrado: %s", xlsx)
            df = _parse_excel(xlsx)
            df.to_csv(cache_path, index=False)
            return _supplement_with_api(df, cache_path)

    # 3 — Manual HTML file
    manual = cache_path.parent / "mega_sena_manual.html"
    if manual.exists():
        logger.info("HTML manual: %s", manual)
        df = _parse_caixa_html(manual.read_bytes())
        df.to_csv(cache_path, index=False)
        return _supplement_with_api(df, cache_path)

    # 4 — API only (no local file at all)
    latest = _fetch_api_latest()
    if latest is not None:
        latest.to_csv(cache_path, index=False)
        logger.info("API: apenas último sorteio disponível (%d)", int(latest["concurso"].iloc[-1]))
        return latest

    raise RuntimeError(
        "Não foi possível carregar os dados.\n"
        "Coloca o ficheiro Mega-Sena.xlsx em test_ml/loteria/ e volta a correr."
    )
