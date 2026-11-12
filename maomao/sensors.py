"""
Deriva os canais sensoriais do MaoMao. `compute_sensors_core` é a lógica
central (intensidade 0..1 por canal, a partir de um retorno real médio por
dia, um dict de anomalia, e a taxa de desacordo do ensemble) — compartilhada
por quem alimenta com dados já no disco (`compute_sensors`,
`compute_market_sensors`, ambos lendo output/anomalia.json e
output/predictions_log.csv) e por quem alimenta com histórico baixado na
hora (`maomao/backfill.py`, replay causal contra anos de yfinance real).

Cada canal é uma intensidade em [0, 1]. Nenhum deles usa a direção prevista
(direction/confidence) como entrada — ver maomao/README.md, "O que o MaoMao
nunca aprende".

v1 não implementa habituação por canal (a depressão sináptica do Pestinha
que faz um estímulo repetido perder força). É um corte deliberado de âmbito,
não um esquecimento — ver maomao/README.md, "O que falta (v2)".
"""
import pandas as pd

from maomao.brain import CHANNELS


def _streak(values: list[bool]) -> int:
    """Tamanho da sequência atual (mais recente primeiro) de True consecutivos."""
    n = 0
    for v in reversed(values):
        if not v:
            break
        n += 1
    return n


def real_returns_by_day(df_log: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Retorno real médio por target_date, entre os `tickers` dados (não o erro
    de previsão — ver nota em README_Errors.md sobre actual_change_pct ser
    (actual/pred - 1), não (actual/ref - 1)). Única fonte de verdade de
    retorno real para maomao.sensors e maomao.learning, pra nunca haver duas
    contas divergentes. `tickers` pode ser a carteira (ETFs e ações) ou o
    watchlist inteiro — a função não assume qual.
    """
    sub = df_log[
        (df_log["ticker"].isin(tickers)) &
        (df_log["horizon"] == 1) &
        df_log["actual_price"].notna()
    ].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["real_ret"] = (sub["actual_price"] / sub["ref_price"] - 1) * 100
    # Mediana: splits e dados ruins chegam a +2114% e um só distorceria a média.
    grouped = sub.groupby("target_date")["real_ret"]
    # Antes do fecho das bolsas, um dia pode ter só cripto validado (2 de ~530 tickers).
    counts = grouped.size()
    return grouped.median()[counts >= counts.median() / 2].sort_index()


def _alert_is_only_vix_falling(anomaly: dict) -> bool:
    market = anomaly.get("market", {})
    return (
        float(market.get("variacao_1d", 0.0)) < 0
        and not market.get("transicao", False)
        and not any(t.get("alerta", False) for t in anomaly.get("tickers", {}).values())
    )


def compute_sensors_core(by_day: pd.Series, anomaly: dict,
                         recent_alertas: list[bool] | None, disagreement: float) -> dict:
    """
    Núcleo puro: dado o retorno médio real por dia já calculado (qualquer
    universo de tickers), o dict de anomalia do dia, o histórico curto de
    alertas, e a taxa de desacordo já calculada, devolve a intensidade (0..1)
    de cada canal sensorial. Não sabe nem precisa saber de onde `by_day` ou
    `disagreement` vieram — isso é responsabilidade de quem chama.
    """
    if by_day.empty:
        return {c: 0.0 for c in CHANNELS} | {"date": None}

    last_date = by_day.index[-1]
    last_ret  = float(by_day.iloc[-1])

    ret_up   = max(0.0, min(1.0, last_ret / 2.0))        # +2%/dia já satura o canal
    ret_down = max(0.0, min(1.0, -last_ret / 2.0))

    up_days   = (by_day > 0).tolist()
    down_days = (by_day < 0).tolist()
    streak_up   = min(1.0, _streak(up_days) / 5.0)       # 5 dias seguidos já satura
    streak_down = min(1.0, _streak(down_days) / 5.0)

    # VIX caindo é medo recuando: não pode alimentar FEARFUL pelos canais de stress.
    alerta  = bool(anomaly.get("alerta", False)) and not _alert_is_only_vix_falling(anomaly)
    market  = anomaly.get("market", {})
    vix_var = max(0.0, float(market.get("variacao_1d", 0.0)))
    vol_spike = max(
        (float(t.get("multiplo_atr", 0.0)) for t in anomaly.get("tickers", {}).values()),
        default=0.0,
    )

    calm_streak = min(1.0, _streak([not a for a in (recent_alertas or [])]) / 10.0)

    return {
        "date":          str(last_date),
        "RET_UP":        ret_up,
        "RET_DOWN":      ret_down,
        "VOL_SPIKE":     max(0.0, min(1.0, vol_spike / 2.5)),   # 2.5x ATR satura
        "VIX_SPIKE":     max(0.0, min(1.0, vix_var / 0.30)),    # 30% num dia satura
        "STREAK_UP":     streak_up,
        "STREAK_DOWN":   streak_down,
        "ANOMALY":       1.0 if alerta else 0.0,
        "DISAGREEMENT":  disagreement,
        "CALM_STREAK":   calm_streak,
    }


def _disagreement_from_log(df_log: pd.DataFrame, tickers: list[str], last_date) -> float:
    rows = df_log[
        (df_log["ticker"].isin(tickers)) &
        (df_log["horizon"] == 1) &
        (df_log["target_date"] == last_date)
    ]
    model_cols = ["model_rf", "model_gb", "model_sgd"]
    if not all(c in rows.columns for c in model_cols) or rows.empty:
        return 0.0
    agree = (rows["model_rf"] == rows["model_gb"]) & (rows["model_gb"] == rows["model_sgd"])
    return float(1.0 - agree.mean())


def compute_sensors(df_log: pd.DataFrame, anomaly: dict, portfolio: list[str],
                    recent_alertas: list[bool] | None = None) -> dict:
    """
    Humor de CARTEIRA: intensidade de cada canal pros ativos da carteira que você
    realmente tem investidos. Lê output/anomalia.json (que já roda só sobre
    esses 7) e output/predictions_log.csv filtrado a eles.
    """
    by_day = real_returns_by_day(df_log, portfolio)
    if by_day.empty:
        return {c: 0.0 for c in CHANNELS} | {"date": None}
    disagreement = _disagreement_from_log(df_log, portfolio, by_day.index[-1])
    return compute_sensors_core(by_day, anomaly, recent_alertas, disagreement)


def compute_market_sensors(df_log: pd.DataFrame, anomaly: dict, watchlist: list[str],
                           recent_alertas: list[bool] | None = None) -> dict:
    """
    Humor de MERCADO: mesma lógica, mas sobre o watchlist inteiro (~500+
    tickers), não só a carteira. Duas diferenças deliberadas em relação a
    `compute_sensors`:

    - ANOMALY vem só do VIX (anomaly["market"]), não de anomaly["alerta"] —
      esse último mistura em rupturas de ATR de tickers específicos da
      carteira, que não são "o mercado", são "a sua carteira".
    - VOL_SPIKE continua a usar o múltiplo de ATR da carteira (anomaly não
      tem isso calculado pro watchlist inteiro) — é uma aproximação honesta,
      documentada em maomao/README.md, não um cálculo de amplitude de
      mercado de verdade. Dá pra melhorar depois sem mudar a interface.
    """
    by_day = real_returns_by_day(df_log, watchlist)
    if by_day.empty:
        return {c: 0.0 for c in CHANNELS} | {"date": None}
    market = anomaly.get("market", {})
    vix_subiu_forte = market.get("spike", False) and float(market.get("variacao_1d", 0.0)) > 0
    market_anomaly = {
        **anomaly,
        "alerta": bool(vix_subiu_forte or market.get("transicao", False)),
    }
    disagreement = _disagreement_from_log(df_log, watchlist, by_day.index[-1])
    return compute_sensors_core(by_day, market_anomaly, recent_alertas, disagreement)
