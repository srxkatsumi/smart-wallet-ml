"""
Calibração pós-hoc da confiança do ensemble contra resultados reais validados.

Contexto (ERR-013): o ensemble reporta uma "confidence" bruta = max(p, 1-p),
construída a partir de RF/GB já passados por CalibratedClassifierCV (isotonic,
em models/ensemble.py) combinados com o SGD (não calibrado) por média
ponderada. Medido contra os ~48 mil registos já validados em
predictions_log.csv, esse número combinado não carrega informação em
produção: Brier score pior que prever sempre 0.5 constante, e acurácia real
praticamente plana entre o decil mais baixo e o mais alto de confiança
reportada — ver scripts/ablation_confidence_calibration.py.

Este módulo ajusta uma curva isotônica por horizonte (confiança bruta →
acurácia empírica), usando o histórico validado já existente, e guarda os
pontos da curva como grade simples em JSON (sem objetos sklearn pickled,
mesmo espírito do scaler_*.json do SGD). A confiança calibrada é gravada
numa coluna própria (confidence_calibrated), lado a lado com a bruta — nunca
sobrescrevendo-a, porque a bruta é o que alimenta o próximo refit.

A calibração é ajustada de forma agregada (todos os tickers do watchlist
juntos, por horizonte), não por ticker: com ~66-150 observações validadas
por ticker/horizonte, um isotonic fit por ticker seria ruído. Agregado,
há amostra (ver MIN_N_CALIBRATION) para um fit honesto.
"""
import json
import logging
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.settings import (
    HORIZONS, CONFIDENCE_CALIB_FILE, CONFIDENCE_CALIB_DAYS, MIN_N_CALIBRATION, MIN_DAYS_CALIBRATION,
)
from data.storage import aligned_rows

logger = logging.getLogger(__name__)

_GRID = np.linspace(0.5, 1.0, 51)  # confidence só existe em [0.5, 1.0] (é max(p, 1-p))


def _fit_curve(subset: pd.DataFrame) -> np.ndarray | None:
    # Agrupa em bins de frequência igual antes do fit isotônico. Ajustar
    # diretamente sobre ~n floats quase todos únicos deixa uma única
    # observação extrema definir o valor da curva na borda (um acerto ou
    # erro isolado na confiança mais alta observada "fixa" esse extremo em
    # 0 ou 1). Agrupar primeiro e pesar pelo tamanho do bin é a prática
    # padrão de reliability diagram — suaviza exatamente esse efeito.
    n_bins = max(3, min(20, len(subset) // 20))
    subset = subset.copy()
    subset["bin"] = pd.qcut(subset["confidence"], n_bins, duplicates="drop")
    grouped = subset.groupby("bin", observed=True).agg(
        conf_mean=("confidence", "mean"),
        acc_mean=("correct", "mean"),
        n=("correct", "size"),
    )
    if len(grouped) < 3:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(grouped["conf_mean"].values, grouped["acc_mean"].values, sample_weight=grouped["n"].values)
    return iso.predict(_GRID)


def _holdout_split(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Período mais antigo para ajustar, 30% mais recentes para medir."""
    if "target_date" in subset.columns:
        dates = sorted(subset["target_date"].unique())
        cut   = dates[int(len(dates) * 0.7)]
        return subset[subset["target_date"] < cut], subset[subset["target_date"] >= cut]
    cut = int(len(subset) * 0.7)
    return subset.iloc[:cut], subset.iloc[cut:]


def fit_calibration(df_log: pd.DataFrame) -> dict:
    """
    Ajusta uma curva isotônica por horizonte (confidence -> P(correct) empírica).

    Horizontes com menos de MIN_N_CALIBRATION linhas validadas, ou de
    MIN_DAYS_CALIBRATION dias, são omitidos do resultado; apply_calibration cai
    de volta para o valor bruto nesse caso. O Brier é medido fora da amostra, com
    uma curva ajustada só no período anterior; a curva guardada usa todo o histórico.
    """
    validated = aligned_rows(df_log[df_log["correct"].notna()]).copy()
    validated["correct"]    = validated["correct"].astype(float)
    validated["confidence"] = validated["confidence"].astype(float)

    curves: dict[str, dict] = {}
    for day in HORIZONS:
        key    = f"d{day}"
        subset = validated[validated["horizon"] == day]
        n      = len(subset)
        if n < MIN_N_CALIBRATION:
            logger.info("Calibração D+%d: histórico insuficiente (%d/%d)", day, n, MIN_N_CALIBRATION)
            continue
        if "target_date" in subset.columns and subset["target_date"].nunique() < MIN_DAYS_CALIBRATION:
            logger.info("Calibração D+%d: poucos dias (%d/%d)", day, subset["target_date"].nunique(), MIN_DAYS_CALIBRATION)
            continue

        y = _fit_curve(subset)
        fit_part, test_part = _holdout_split(subset)
        y_holdout = _fit_curve(fit_part) if len(fit_part) else None
        if y is None or y_holdout is None or test_part.empty:
            logger.info("Calibração D+%d: poucos bins com dados suficientes — ignorado", day)
            continue

        calibrated_test = np.interp(test_part["confidence"].values, _GRID, y_holdout)
        brier_raw = float(np.mean((test_part["confidence"].values - test_part["correct"].values) ** 2))
        brier_cal = float(np.mean((calibrated_test - test_part["correct"].values) ** 2))

        curves[key] = {
            "x": _GRID.tolist(),
            "y": [round(float(v), 4) for v in y],
            "n": n,
            "brier_raw":        round(brier_raw, 4),
            "brier_calibrated": round(brier_cal, 4),
        }
        logger.info(
            "Calibração D+%d: n=%d Brier fora da amostra %.4f -> %.4f",
            day, n, brier_raw, brier_cal,
        )

    return curves


def save_calibration(curves: dict) -> None:
    # Lê CONFIDENCE_CALIB_FILE do módulo a cada chamada (não como default de
    # parâmetro) de propósito: um default binda o valor uma vez na definição
    # da função, e testes que fazem monkeypatch do atributo do módulo nunca
    # veriam o path de teste — o fallback cairia sempre no ficheiro real.
    path = CONFIDENCE_CALIB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(curves)
    payload["updated"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration() -> dict:
    path = CONFIDENCE_CALIB_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def needs_refit(curves: dict, days: int = CONFIDENCE_CALIB_DAYS) -> bool:
    updated = curves.get("updated")
    if not updated:
        return True
    return (pd.Timestamp.today() - pd.Timestamp(updated)).days >= days


def apply_calibration(confidence: float, day: int, curves: dict) -> float:
    """
    Mapeia uma confiança bruta pela curva ajustada daquele horizonte.
    Sem curva disponível (cold start / histórico insuficiente) devolve o
    valor bruto sem alterar — nunca inventa uma calibração sem dados.
    """
    if not curves:
        return confidence
    curve = curves.get(f"d{day}")
    if not curve:
        return confidence
    return float(np.interp(confidence, curve["x"], curve["y"]))


def apply_to_log(df_log: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Recalcula confidence_calibrated em todas as linhas do log pela curva atual."""
    df_log = df_log.copy()
    if "confidence_calibrated" not in df_log.columns:
        df_log["confidence_calibrated"] = np.nan
    for day in HORIZONS:
        curve = curves.get(f"d{day}")
        if not curve:
            continue
        mask = df_log["horizon"] == day
        if not mask.any():
            continue
        df_log.loc[mask, "confidence_calibrated"] = np.round(
            np.interp(df_log.loc[mask, "confidence"].astype(float), curve["x"], curve["y"]), 4
        )
    return df_log


def monthly_calibration_refresh(df_log: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reajusta as curvas no máximo uma vez a cada CONFIDENCE_CALIB_DAYS (mesmo
    padrão de guarda do models.ensemble.monthly_recalibration). Quando reajusta,
    também reaplica a curva nova a todo o histórico já validado (não só às
    previsões novas de hoje) e persiste — a coluna calibrada deve sempre
    refletir a curva mais recente, não a curva vigente no dia em que cada
    linha foi escrita.
    """
    from data.storage import save_predictions_log

    curves = load_calibration()
    if curves and not needs_refit(curves):
        return df_log, curves

    new_curves = fit_calibration(df_log)
    if not new_curves:
        logger.info("Calibração: sem horizonte com amostra suficiente ainda — mantém estado anterior")
        return df_log, curves

    save_calibration(new_curves)
    df_log = apply_to_log(df_log, new_curves)
    save_predictions_log(df_log)
    logger.info("Calibração de confiança atualizada e reaplicada a %d linhas", len(df_log))

    return df_log, load_calibration()
