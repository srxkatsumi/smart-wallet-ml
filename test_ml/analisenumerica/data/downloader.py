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


def download_results(cache_path: Path, force: bool = False) -> pd.DataFrame:
    """Load Mega Sena results. Priority: xlsx → cached csv → html → download."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1 — Look for Excel file placed in test_ml/ or output/
    search_dirs = [
        cache_path.parent.parent.parent,   # test_ml/
        cache_path.parent,                  # output/
        cache_path.parent.parent,           # analisenumerica/
    ]
    for d in search_dirs:
        for xlsx in d.glob("*.xlsx"):
            logger.info("Excel encontrado: %s", xlsx)
            df = _parse_excel(xlsx)
            df.to_csv(cache_path, index=False)
            return df

    # 2 — Cached CSV (if not forcing)
    if not force and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["data"])
            logger.info("Cache CSV: %d sorteios", len(df))
            return df
        except Exception:
            logger.warning("Cache corrompido — tentando outras fontes")

    # 3 — Manual HTML file
    manual = cache_path.parent / "mega_sena_manual.html"
    if manual.exists():
        logger.info("HTML manual: %s", manual)
        df = _parse_caixa_html(manual.read_bytes())
        df.to_csv(cache_path, index=False)
        return df

    # 4 — Try downloading from Caixa
    logger.info("A tentar download da Caixa...")
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(_CAIXA_URL, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        btn  = soup.find(id="btnResultados")
        if btn and btn.get("href"):
            from urllib.parse import urljoin
            url  = urljoin(_CAIXA_URL, btn["href"])
            resp2 = requests.get(url, headers=_HEADERS, timeout=30)
            resp2.raise_for_status()
            df = _parse_caixa_html(resp2.content)
            df.to_csv(cache_path, index=False)
            return df
    except Exception as e:
        logger.warning("Download automático falhou: %s", e)

    raise RuntimeError(
        "Não foi possível carregar os dados.\n"
        "Coloca o ficheiro Mega-Sena.xlsx em test_ml/ e volta a correr."
    )
