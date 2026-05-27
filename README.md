# smart-wallet-ml

> **Last updated:** 2026-05-27 &nbsp;·&nbsp; Charts from 2026-05-08 to 2026-05-17 (23 charts)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![GitHub Actions](https://img.shields.io/badge/Automated-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart-wallet-ml/actions)
[![Last sync](https://img.shields.io/github/last-commit/srxkatsumi/smart-wallet-ml?label=last%20sync&color=4c8f6b&logo=github)](https://github.com/srxkatsumi/smart-wallet-ml/commits/main)

---

## What is this?

An automated machine learning system that runs every weekday and generates daily directional forecasts for a personal investment portfolio using ensemble models.

The pipeline downloads fresh market data for 90+ assets, engineers technical indicators, trains three independent classifiers per asset per time horizon, validates each forecast against real closing prices, and reweights the models based on recent accuracy.

Charts are published here with a **10-day delay**. No prices, positions, or portfolio holdings are disclosed.

---

## How it works

```mermaid
flowchart TD
    A["📥 Market Data\nyfinance · 90+ tickers\nVIX · SPY · FX rates"] --> B["⚙️ Feature Engineering\nRSI · MACD · Bollinger Bands\nATR · SMA · 1d/5d returns"]
    B --> C["🤖 Independent Ensemble\nper asset × per horizon\nD+1 · D+2 · D+3"]
    C --> D1["🌲 Random Forest\n300 trees · depth 6"]
    C --> D2["📈 Gradient Boosting\n200 est. · lr 0.05"]
    C --> D3["📐 SGD Classifier\nlog loss · monthly recal."]
    D1 & D2 & D3 --> E["⚖️ Weighted Vote\nadaptive weights\nexponential decay"]
    E --> F["📊 Forecast\nUP / DOWN + confidence"]
    F --> G["✅ Validation\nactual close vs prediction"]
    G --> H["🔄 Weight Update\nmore recent = more weight"]
    H --> E
    F --> I["📧 Daily email\n+ charts → public repo"]
```

---

## AI Models

Three classifiers are combined in a weighted ensemble. Each model was chosen for a specific reason:

| Model | Config | Why |
|-------|--------|-----|
| **Random Forest** | 300 trees, max depth 6 | Robust generaliser — bootstrapped trees resist overfitting on noisy market data. Acts as the ensemble's stability anchor. |
| **Gradient Boosting** | 200 estimators, lr 0.05 | Sequential residual learner — captures patterns RF misses, especially short-term momentum signals. Low learning rate prevents memorising noise. |
| **SGD Classifier** | log loss, L2 | Linear counterweight — cannot model non-linear interactions, so it acts as a dissenting vote when the other two agree on noise. Fully recalibrated monthly. |

### Why independent ensembles per horizon?

D+1, D+2, and D+3 are trained as **completely separate ensembles**. The patterns that predict tomorrow's price differ structurally from those that predict a 3-day move — conflating them into a single model produces weaker forecasts for all horizons.

### Adaptive weights with exponential decay

```
weight(model) ∝ accuracy(model) × Σ decay^(days_ago)
```

Recent correct predictions count more than older ones. If a model starts underperforming after a market regime change, the ensemble automatically reduces its vote share — no manual intervention needed.

---

## Features (what the model sees)

| Feature | Description |
|---------|-------------|
| `sma_20`, `sma_50` | Short and medium-term trend |
| `rsi_14` | Overbought / oversold signal |
| `macd`, `macd_signal` | Momentum and crossovers |
| `bb_upper`, `bb_lower`, `bb_width` | Volatility regime and price extremity |
| `atr_14` | Expected daily move magnitude |
| `ret_1d`, `ret_5d` | Recent momentum |
| `spy_ret_1d` | S&P 500 return (T-1) — global market context |
| `vix_level` | CBOE VIX close (T-1) — market fear level |
| `vix_change` | VIX daily change (T-1) — fear acceleration |

All external context features use T-1 values to prevent data leakage.

---

## Tech stack

```
Python 3.11
├── yfinance       — market data (prices, FX, VIX, SPY)
├── scikit-learn   — RandomForest, GradientBoosting, SGDClassifier
├── pandas/numpy   — data processing and feature computation
├── joblib         — model serialisation
└── matplotlib     — chart generation

GitHub Actions     — free daily automation
```

---

## Accuracy context

| Benchmark | Target |
|-----------|--------|
| Random directional forecast | 50% |
| System target | 55–65% |
| Degradation signal | < 52% over 30+ validations |

---

*Charts contain no financial advice, no portfolio positions, and no entry prices.
This is a personal data science project — not an investment product.*

Built by **Vicky Costa** — Data Analyst & Data Science student
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
