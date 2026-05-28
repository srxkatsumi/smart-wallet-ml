import logging
import io
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Caixa results file — link extracted from the "Resultados por ordem crescente" button
_CAIXA_URL = (
    "https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx"
)
_RESULTS_API = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena/"
)

# Column names in the Caixa download file
_CAIXA_COLS = {
    "Concurso":      "concurso",
    "Data Sorteio":  "data",
    "1ª Dezena":     "b1",
    "2ª Dezena":     "b2",
    "3ª Dezena":     "b3",
    "4ª Dezena":     "b4",
    "5ª Dezena":     "b5",
    "6ª Dezena":     "b6",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _parse_caixa_html(content: bytes) -> pd.DataFrame:
    tables = pd.read_html(io.BytesIO(content), encoding="utf-8", thousands=".", decimal=",")
    for t in tables:
        if "Concurso" in t.columns and "1ª Dezena" in t.columns:
            df = t.rename(columns=_CAIXA_COLS)
            df = df[[c for c in _CAIXA_COLS.values() if c in df.columns]]
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            for col in ["b1", "b2", "b3", "b4", "b5", "b6"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["concurso", "data", "b1"])
            df["concurso"] = df["concurso"].astype(int)
            df = df.sort_values("concurso").reset_index(drop=True)
            logger.info("Caixa: %d sorteios carregados (último: concurso %d)",
                        len(df), df["concurso"].iloc[-1])
            return df
    raise ValueError("Tabela de resultados não encontrada no arquivo da Caixa")


def _try_download_file() -> bytes:
    """Try to find and download the results file from the Caixa page."""
    try:
        # The Caixa page has a link that generates the results HTML file
        # We request the page and look for the download button target
        from bs4 import BeautifulSoup
        resp = requests.get(_CAIXA_URL, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        btn = soup.find("a", {"id": "btnResultados"}) or soup.find("button", {"id": "btnResultados"})
        if btn and btn.get("href"):
            file_url = btn["href"]
            if not file_url.startswith("http"):
                from urllib.parse import urljoin
                file_url = urljoin(_CAIXA_URL, file_url)
            resp2 = requests.get(file_url, headers=_HEADERS, timeout=30)
            resp2.raise_for_status()
            return resp2.content
    except Exception as e:
        logger.debug("Tentativa de download via BeautifulSoup falhou: %s", e)

    # Direct known path (Caixa serves the file via an aspx handler)
    try:
        direct = _CAIXA_URL.replace(".aspx", "_pesquisa_new.asp?tipoOrdenacao=0")
        resp = requests.get(direct, headers=_HEADERS, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
    except Exception as e:
        logger.debug("Tentativa de download direto falhou: %s", e)

    raise RuntimeError(
        "Não foi possível fazer download automático dos resultados da Caixa.\n"
        "Acede a https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx, "
        "clica em 'Resultados por ordem crescente' e guarda o ficheiro em "
        "test_ml/analisenumerica/output/mega_sena_manual.html"
    )


def download_results(cache_path: Path, force: bool = False) -> pd.DataFrame:
    """Download or load Mega Sena results. Uses cache if recent enough."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try loading from cache first (if not forcing refresh)
    if not force and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["data"])
            logger.info("Resultados carregados do cache: %d sorteios", len(df))
            return df
        except Exception:
            logger.warning("Cache corrompido — a fazer novo download")

    # Try manual file first
    manual_path = cache_path.parent / "mega_sena_manual.html"
    if manual_path.exists():
        logger.info("A usar ficheiro manual: %s", manual_path)
        with open(manual_path, "rb") as f:
            content = f.read()
        df = _parse_caixa_html(content)
        df.to_csv(cache_path, index=False)
        return df

    # Try automatic download
    logger.info("A fazer download dos resultados da Mega Sena...")
    content = _try_download_file()
    df = _parse_caixa_html(content)
    df.to_csv(cache_path, index=False)
    logger.info("Resultados guardados em %s", cache_path)
    return df
