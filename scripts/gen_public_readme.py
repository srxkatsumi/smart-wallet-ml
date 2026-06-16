#!/usr/bin/env python3
"""Generate the README for the public smart-wallet-ml repo.

Usage:
    python3 scripts/gen_public_readme.py TODAY DATE_MIN DATE_MAX COUNT OUTPUT_PATH
"""
import sys

today    = sys.argv[1]
date_min = sys.argv[2]
date_max = sys.argv[3]
count    = sys.argv[4]
out_path = sys.argv[5]

readme = f"""\
# smart-wallet-ml

> **Last updated:** {today} &nbsp;·&nbsp; Charts from {date_min} to {date_max} ({count} charts)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![GitHub Actions](https://img.shields.io/badge/Automated-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart-wallet-ml/actions)
[![Last sync](https://img.shields.io/github/last-commit/srxkatsumi/smart-wallet-ml?label=last%20sync&color=4c8f6b&logo=github)](https://github.com/srxkatsumi/smart-wallet-ml/commits/main)

---

## What is this?

An automated machine learning system that runs every weekday and generates daily directional forecasts for a personal investment portfolio.

Every night after market close it downloads fresh prices for **543 assets** across 70 sectors, builds 33 engineered features per asset, trains **16 independent model families**, and blends their predictions into a final ensemble. Each forecast is validated against actual closing prices and the models are reweighted accordingly.

Charts are published here with a **10-day delay**. No prices, positions, or portfolio holdings are disclosed.

---

## How it works

```mermaid
flowchart TD
    A["📥 Market Data\\n543 tickers · 70 sectors\\nVIX · SPY · BTC · GLD · FX"] --> B["⚙️ Feature Engineering\\n33 features in 5 groups\\n(Technical · Momentum · Calendar\\n Cross-asset · Extremes)"]
    B --> C["🏭 Production Ensemble\\nper asset × D+1 · D+2 · D+3"]
    C --> P1["🌲 Random Forest"]
    C --> P2["📈 Gradient Boosting"]
    C --> P3["📐 SGD Classifier"]
    B --> R["🔬 Research Pipeline\\n13 model families\\nMonday: retrain · Tue–Fri: predict"]
    R --> R1["HMM · Bayesian · VAE"]
    R --> R2["Transformer · Foundation\\nConformal · Drift"]
    P1 & P2 & P3 --> E["⚖️ Blend\\n50% Production + 50% Research\\nadaptive weights per family"]
    R1 & R2 --> E
    E --> F["📊 Final Forecast\\nUP / DOWN + confidence"]
    F --> G["✅ Daily Validation\\nactual close vs prediction"]
    G --> H["🔄 Weight Update\\nexponential decay"]
    H --> E
    F --> I["📧 Email + Telegram\\nCharts → public repo (D-10)"]
```

---

## Two-layer ensemble

### Production layer (3 classifiers)

| Model | Config | Why |
|-------|--------|-----|
| **Random Forest** | 100 trees, max depth 5 | Robust generaliser — bootstrapped trees resist overfitting on noisy market data. |
| **Gradient Boosting** | 100 estimators, lr 0.05 | Captures momentum patterns RF misses via sequential residual learning. |
| **SGD Classifier** | log loss, L2, monthly recal. | Linear dissenting vote — penalises when the other two agree on noise. |

### Research layer (13 model families)

| Family | Technique |
|--------|-----------|
| `classico_avancado` | Extended classical ML (SVM, ExtraTrees, stacking) |
| `estado_oculto` | Hidden Markov Model — regime detection |
| `series_temporais` | ARIMA, Prophet, SARIMA |
| `neural_recorrente` | LSTM / GRU |
| `neural_atencao` | Transformer encoder |
| `bayesiano` | Bayesian classifier with uncertainty |
| `generativo` | VAE / GAN-based features |
| `reinforcement` | Q-learning signal |
| `contrarian` | Mean-reversion counter-trend |
| `eficiente` | TCN / PatchTST |
| `foundation` | Chronos-T5 (zero-shot), TimesFM, Moirai |
| `conformal` | Conformal prediction intervals |
| `drift` | Concept drift detector |

Each family's vote is weighted by its recent validated accuracy (exponential decay). The final prediction blends production and research 50/50.

---

## Features (what the model sees)

33 features grouped in 5 categories:

| Group | Features |
|-------|---------|
| **Technical** | `SMA20_dist`, `SMA50_dist`, `sma_cross`, `RSI14`, `MACD`, `MACD_sig`, `MACD_hist`, `BB_width`, `BB_pos`, `ATR14`, `ret_1d`, `ret_5d`, `vol_10d`, `vol_ratio`, `obv_trend` |
| **Context** | `spy_ret_1d`, `vix_level`, `vix_change`, `vix_regime`, `asset_class` |
| **Momentum** | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` |
| **Calendar** | `day_of_week`, `month`, `is_options_expiry` |
| **Cross-asset** | `btc_ret_1d`, `gold_ret_1d`, `corr_spy_20d`, `vwap_dist` |
| **Extremes** | `high52w_dist`, `low52w_dist` |

All external features use T-1 values to prevent data leakage.

---

## Sample daily output

This is what the system produces each weekday evening (Telegram summary):

```
🤖 Modelos — 14/06/2026 (⚡ predict)

D+1 Acurácia (últ. 10 dias):
🟢 Foundation         68%  ↑ +5pp
🟢 Clássico           65%  →
⚪ Transformer        58%  →
⚪ HMM                55%  ↓ -3pp
🔴 VAE/GAN            47%  ↓ -8pp

🏆 Melhor: Foundation (68%)
⚠️  Pior:   VAE/GAN (47%)

📊 WFV 2026-06-09 — D+1 real: ⚪ 53.2%
```

And the anonymised predictions table (`predictions_log_public.csv`):

| asset_type | pred_date | target_date | direction | confidence | correct | model_rf | model_gb | model_sgd |
|-----------|-----------|-------------|-----------|------------|---------|----------|----------|-----------|
| portfolio | 2026-06-13 | 2026-06-14 | up | 0.61 | True | up | up | up |
| portfolio | 2026-06-13 | 2026-06-14 | down | 0.54 | False | down | up | down |
| watchlist | 2026-06-13 | 2026-06-14 | up | 0.57 | True | up | up | down |

---

## Walk-Forward Validation

The system runs a daily walk-forward backtest over the last 30 trading days: train only on data available before each day, predict that day, compare to actual outcome. Results are logged in `output/wfv_log.csv` (cumulative, never deleted).

This gives an honest out-of-sample accuracy benchmark — no look-ahead bias.

---

## Accuracy context

| Benchmark | Value |
|-----------|-------|
| Random directional forecast | 50% |
| System target | 55–65% |
| Degradation signal | < 52% over 30+ validations |
| Walk-Forward Validation | updated daily in `wfv_log.csv` |

---

## Tech stack

```
Python 3.11
├── yfinance       — market data (543 tickers, VIX, SPY, BTC, GLD, FX)
├── scikit-learn   — RandomForest, GradientBoosting, SGDClassifier
├── torch          — Transformer, LSTM, Foundation model wrappers
├── pandas/numpy   — feature engineering (33 features)
├── joblib         — model serialisation
└── matplotlib     — chart generation

GitHub Actions     — free daily automation (22h00 UTC Mon–Fri)
```

---

## Verifiable accuracy

`predictions_log_public.csv` is published in this repository.
It contains every forecast made since the system went live, with the actual outcome filled in once the target date passes.

| Column | Content |
|--------|---------|
| `asset_type` | `portfolio` or `watchlist` — no ticker names |
| `pred_date` | Date the forecast was made |
| `target_date` | Date the forecast refers to |
| `direction` | `up` or `down` |
| `confidence` | Blended ensemble probability |
| `correct` | `True` / `False` / `NaN` (NaN = stock split detected) |
| `model_rf` | Individual Random Forest vote |
| `model_gb` | Individual Gradient Boosting vote |
| `model_sgd` | Individual SGD Classifier vote |

No prices, tickers, or portfolio weights are included.

---

*Charts contain no financial advice, no portfolio positions, and no entry prices.
This is a personal data science project — not an investment product.*

Built by **Vicky Costa** — Data Analyst & Data Science student
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
"""

with open(out_path, "w") as f:
    f.write(readme)
