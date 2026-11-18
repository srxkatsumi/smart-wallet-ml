from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output")
CHARTS_DIR   = OUTPUT_DIR / "charts"
MODELS_DIR   = OUTPUT_DIR / "models"
PRED_LOG     = OUTPUT_DIR / "predictions_log.csv"
PUBLIC_LOG   = OUTPUT_DIR / "predictions_log_public.csv"
WEIGHTS_FILE          = OUTPUT_DIR / "ensemble_weights.json"
RESEARCH_WEIGHTS_FILE = OUTPUT_DIR / "research_weights.json"
METADATA_FILE= OUTPUT_DIR / "model_metadata.csv"
RECALIB_FILE = OUTPUT_DIR / "ultima_recalibracao.json"
HTML_REPORT  = OUTPUT_DIR / "resumo_diario.html"

# ── Calibração de confiança (ERR-013) ───────────────────────────────────────
CONFIDENCE_CALIB_FILE = OUTPUT_DIR / "confidence_calibration.json"
CONFIDENCE_CALIB_DAYS = 30   # mesma cadência do RECALIBRATION_DAYS do SGD
MIN_N_CALIBRATION     = 200  # por horizonte; abaixo disso o isotonic fit é ruído
MIN_DAYS_CALIBRATION  = 20   # um único dia traz ~530 linhas correlacionadas, não é amostra

# ── Experimento de regressão (email separado, não afeta a carteira) ────────
EXPERIMENTO_MODELS_DIR       = OUTPUT_DIR / "models" / "experimento"
EXPERIMENTO_PRED_LOG         = OUTPUT_DIR / "predictions_experimentos_log.csv"
EXPERIMENTO_HTML_REPORT      = OUTPUT_DIR / "resumo_experimentos.html"
EXPERIMENTO_SIGNIFICANCE_FILE= OUTPUT_DIR / "significance_experimentos.json"
EXPERIMENTO_BACKTEST_FILE    = OUTPUT_DIR / "backtest_experimentos.json"
MIN_N_EXPERIMENTO            = 100

# ── Alerta de anomalia (email separado, só dispara em evento fora do normal) ─
ALERTA_FILE        = OUTPUT_DIR / "anomalia.json"
ALERTA_HTML_REPORT = OUTPUT_DIR / "resumo_alerta.html"
ALERTA_ATR_MULTIPLICADOR = 2.5   # retorno diário >= 2.5x o ATR normal do ativo
ALERTA_VIX_SPIKE_PCT     = 0.15  # variação diária do VIX >= 15%

# ── Model hyperparameters ─────────────────────────────────────────────────
N_ESTIMATORS_RF  = 100
MAX_DEPTH_RF     = 5
N_ESTIMATORS_GB  = 100
MAX_DEPTH_GB     = 3
LEARNING_RATE_GB = 0.05
N_SPLITS_CV      = 5
CV_GAP           = 1          # gap between train and val to prevent same-day lookahead
SGD_ALPHA        = 0.0001
SGD_MAX_ITER     = 1000

# ── Pipeline settings ─────────────────────────────────────────────────────
HORIZONS              = [1, 2, 3]
PRICE_PERIOD          = "2y"
CHARTS_RETENTION_DAYS = 30
DOWNLOAD_BATCH_SIZE        = 20
DOWNLOAD_BATCH_SLEEP       = 2
SPLIT_DETECTION_THRESHOLD  = 0.40
WEIGHT_DECAY_FACTOR   = 0.1
RECALIBRATION_DAYS    = 30
MIN_SAMPLES_FEATURES  = 60
SGD_PARTIAL_FIT_DAYS  = 5
MIN_VALIDATIONS_WEIGHT= 5
MAX_BAR_AGE_DAYS      = 5    # última barra mais velha que isto: ticker sem dados frescos

# Tickers cotados em GBp (pence) na LSE — convertidos para EUR logo no
# download, para que features, modelos e o log de previsões fiquem numa
# única moeda, com o câmbio de cada dia (ver data/downloader.py).
GBP_PENCE_TICKERS = {"SGLN.L"}

# ── Portfolio settings ────────────────────────────────────────────────────
HORIZONTE_ANOS         = [1, 3, 5, 10]
TAXA_CRESCIMENTO_BASE  = {"pessimista": 0.03, "base": 0.08, "optimista": 0.15}

# ── FX fallbacks ─────────────────────────────────────────────────────────
EUR_USD_FALLBACK = 1.12
EUR_GBP_FALLBACK = 0.85

# ── CSV schema ────────────────────────────────────────────────────────────
PRED_COLS = [
    "ticker", "pred_date", "target_date", "horizon",
    "direction", "ref_price", "pred_price", "confidence",
    "actual_price", "actual_change_pct", "correct",
    "atr_at_prediction", "predicted_price",
    "model_rf", "model_gb", "model_sgd",
    "confidence_calibrated", "ref_date",
]

EXPERIMENTO_PRED_COLS = [
    "ticker", "pred_date", "target_date", "horizon",
    "champion_pred_price", "champion_pred_ret",
    "challenger_pred_price", "challenger_pred_ret",
    "challenger_interval_lo", "challenger_interval_hi",
    "ref_price", "actual_price", "actual_ret",
    "champion_sq_err", "challenger_sq_err", "validated", "ref_date",
]

DEFAULT_WEIGHTS = {
    "d1": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d2": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
    "d3": {"rf": 1.0, "gb": 1.0, "sgd": 1.0},
}

_RESEARCH_FAMILIES = [
    "classico_avancado", "estado_oculto", "series_temporais",
    "neural_recorrente", "neural_atencao", "bayesiano",
    "generativo", "reinforcement", "contrarian", "eficiente",
    "foundation", "conformal", "drift",
]

DEFAULT_RESEARCH_WEIGHTS = {
    f"d{d}": {fam: 1.0 for fam in _RESEARCH_FAMILIES}
    for d in [1, 2, 3]
}

# ── Email timezone (Barcelona) ────────────────────────────────────────────
# Fuso com horário de verão automático (CET no inverno, CEST no verão).
# Antes era um offset fixo (+2) que precisava ser trocado à mão em outubro e março.
from zoneinfo import ZoneInfo
BARCELONA_TZ = ZoneInfo("Europe/Madrid")

# ── Asset class per ticker ────────────────────────────────────────────────
# 0 = individual stock  1 = equity ETF  2 = crypto  3 = commodity ETF
ASSET_CLASSES: dict[str, int] = {
    "ALV.DE":  0, "BABA":    0, "BTC-USD": 2, "DHER.DE": 0,
    "LLY":     0, "NVDA":    0, "SIE.DE":  0, "BMW.DE":  0,
    "BAS.DE":  0, "NESN.SW": 0, "NOVN.SW": 0, "ROG.SW":  0,
    "EMIM.AS": 1, "EXUS.MI": 1, "IWDA.AS": 1, "MEUD.PA": 1,
    "CSPX.L":  1, "VWCE.DE": 1, "SJPA.MI": 1, "ICGA.DE": 1,
    "LYP6.DE": 1, "IS3N.DE": 1, "EUNN.DE": 1,
    "SGLN.L":  3, "PPFB.DE": 3,  # iShares Physical Gold ETC: PPFB.DE (Xetra, EUR) é a da carteira; SGLN.L (Londres, GBp) é a mesma, só na watchlist
    "AFRM":    0,  # Affirm Holdings (Nasdaq), ação da carteira
    # ETFs UCITS de acumulação (Xetra) e ETFs de mineradoras, adicionados em 18/11/2026
    "SXR8.DE": 1, "SXRV.DE": 1, "SXRT.DE": 1, "IUSN.DE": 1, "ZPRV.DE": 1, "IS3Q.DE": 1,
    "IS3R.DE": 1, "QDV5.DE": 1, "VVSM.DE": 1, "IQQH.DE": 1, "IBCI.DE": 1,
    "GDX":     1, "GDXJ":    1, "SIL":     1,
}

# ── Market calendar mapping (pandas-market-calendars) ────────────────────
# Tickers not listed here default to NYSE.
# Sufixo do Yahoo → calendário da bolsa, para tickers que não estão no dict abaixo.
# Sem isto, um ticker novo de Madri ou Paris cairia no calendário da NYSE e os
# alvos D+1..D+3 seriam contados com os feriados americanos.
CALENDAR_BY_SUFFIX: dict[str, str] = {
    ".DE": "XETR", ".PA": "XPAR", ".AS": "XAMS", ".MC": "XMAD", ".MI": "XMIL",
    ".L": "LSE", ".SW": "SIX", ".CO": "XCSE", ".T": "XTKS", ".HK": "XHKG",
}


def calendar_for(ticker: str) -> str:
    """Calendário da bolsa do ticker: o dict explícito, depois o sufixo, por fim a NYSE."""
    if ticker in TICKER_CALENDAR:
        return TICKER_CALENDAR[ticker]
    for suf, cal in CALENDAR_BY_SUFFIX.items():
        if ticker.endswith(suf):
            return cal
    return "NYSE"


TICKER_CALENDAR: dict[str, str] = {
    # Nasdaq segue o calendário da NYSE
    "AFRM":    "NYSE",
    # London Stock Exchange
    "EXUS.MI": "XMIL",   # Borsa Italiana (EUR)
    "SGLN.L":  "LSE",
    "CSPX.L":  "LSE",
    # Euronext (per-exchange MIC codes)
    "EMIM.AS": "XAMS",   # Amsterdam
    "IWDA.AS": "XAMS",   # Amsterdam
    "MEUD.PA": "XPAR",   # Paris
    "SJPA.MI": "XMIL",   # Milan (Borsa Italiana)
    # Xetra (Frankfurt)
    "ALV.DE":  "XETR",
    "SIE.DE":  "XETR",
    "BMW.DE":  "XETR",
    "BAS.DE":  "XETR",
    "DHER.DE": "XETR",
    "VWCE.DE": "XETR",
    "ICGA.DE": "XETR",
    "LYP6.DE": "XETR",
    "IS3N.DE": "XETR",
    "EUNN.DE": "XETR",
    "PPFB.DE": "XETR",
    # Swiss Exchange
    "NESN.SW": "SIX",
    "NOVN.SW": "SIX",
    "ROG.SW":  "SIX",
}
