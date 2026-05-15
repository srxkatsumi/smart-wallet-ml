# Carteira Inteligente — ML Portfolio Forecasting System

> A self-learning investment portfolio analysis and forecasting system built from scratch.
> Runs automatically every weekday at 15:35 (Barcelona / CET+1) via GitHub Actions.

[![GitHub Actions](https://img.shields.io/badge/Automated-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart_wallet/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

---

## Overview

This system tracks a real investment portfolio (eToro equities + long-term accumulation ETFs) and generates daily directional forecasts for the next 3 trading days using an ensemble of machine learning classifiers. Every prediction is stored, validated when the target date arrives, and used to re-weight the ensemble — the system continuously improves on its own mistakes.

The output is an HTML email report delivered daily with:
- Portfolio P&L in EUR (including fees and breakeven)
- D+1 / D+2 / D+3 directional forecasts per asset
- Long-term ETF projections (1 / 3 / 5 / 10 years)
- Model accuracy tracking (last 30 business days)

---

## Machine Learning Architecture

### Problem formulation

Binary classification: will the closing price be **higher or lower** than today's close in N trading days?

- `target_d1` = `1` if `Close[t+1] > Close[t]` else `0`
- `target_d2` = `1` if `Close[t+2] > Close[t]` else `0`
- `target_d3` = `1` if `Close[t+3] > Close[t]` else `0`

### Independent ensembles per horizon

Three separate ensembles are trained — one for D+1, one for D+2, one for D+3. Each learns its own temporal pattern. There is no extrapolation between horizons.

```
D+1 Ensemble ──► RF_d1 · GB_d1 · SGD_d1  ──► weights_d1
D+2 Ensemble ──► RF_d2 · GB_d2 · SGD_d2  ──► weights_d2
D+3 Ensemble ──► RF_d3 · GB_d3 · SGD_d3  ──► weights_d3
```

### Models

| Model | Role |
|-------|------|
| **Random Forest** (n=300, max_depth=6) | Robust baseline, handles non-linearity, low overfitting risk |
| **Gradient Boosting** (n=200, lr=0.05) | Captures residual patterns RF misses, especially momentum |
| **SGD Classifier** (log_loss) | Stable linear anchor — prevents the ensemble from overfitting on short-term noise |

The SGD model undergoes **full monthly recalibration**: scaler refit + retrain from scratch. This prevents the standardisation baseline from drifting as market conditions change.

### Feature engineering

| Feature | Description |
|---------|-------------|
| `sma_20`, `sma_50` | Simple moving averages — short and medium trend |
| `rsi_14` | Relative Strength Index — overbought / oversold |
| `macd`, `macd_signal` | Momentum and signal line crossovers |
| `bb_upper`, `bb_lower`, `bb_width` | Bollinger Bands — volatility and price position |
| `atr_14` | Average True Range — expected daily move magnitude |
| `ret_1d`, `ret_5d` | Recent return (1-day, 5-day) |
| `spy_ret_1d` | S&P 500 return (T-1) — global market context |
| `vix_level` | CBOE VIX close (T-1) — market fear level |
| `vix_change` | VIX daily change (T-1) — fear acceleration |

**Why VIX and SPY?** NVDA in a global panic session behaves differently from NVDA in a neutral session. Adding market context allows each model to condition its forecast on the macro environment.

**Data leakage prevention:** all market context features use T-1 values. When European markets open (15:35 Barcelona), only yesterday's US close is available. Using T-0 would be leakage.

### Adaptive ensemble weights with temporal decay

After each validation cycle, model weights are updated using accuracy over a rolling window with exponential decay:

```
weight(model) ∝ accuracy(model) · Σ decay^(days_ago)
```

More recent correct predictions carry more weight than older ones. If a model starts underperforming systematically, the ensemble lowers its vote share automatically.

### Validation and audit trail

Every forecast written to `predictions_log.csv` includes:

| Column | Description |
|--------|-------------|
| `pred_date` | Date the forecast was made |
| `target_date` | Date the forecast refers to |
| `horizon` | 1, 2, or 3 |
| `direction` | `up` or `down` |
| `confidence` | Ensemble probability |
| `actual_price` | Filled on validation day (initially `NaN`) |
| `correct` | `True` / `False` (filled on validation day) |

Nothing is deleted or overwritten. The full audit trail is preserved.

### Consensus signal

```
BULLISH  → all three horizons predict UP
BEARISH  → all three horizons predict DOWN
MIXED    → disagreement across horizons
```

---

## Portfolio covered

**eToro equities:**

| Ticker | Name |
|--------|------|
| LLY | Eli Lilly & Co |
| NVDA | NVIDIA Corporation |
| ALV.DE | Allianz SE |
| BTC-USD | Bitcoin |
| BABA | Alibaba Group ADR |

**Accumulation ETFs (long-term):**

| Ticker | Name |
|--------|------|
| EXUS.L | MSCI World ex USA ETF |
| ICGA.DE | MSCI China ETF |
| SGLN.L | Physical Gold ETC |
| EMIM.AS | iShares Core MSCI EM IMI ETF |
| MEUD.PA | Core Stoxx Europe 600 |
| SJPA.MI | iShares Core MSCI Japan IMI ETF |

### ML watchlist (macro context universe)

The watchlist expands the training universe beyond the personal portfolio. Models learn cross-asset correlations and market regime signals from this broader dataset.

| Group | Tickers | Purpose |
|-------|---------|---------|
| US Tech | AAPL MSFT GOOGL AMZN META TSLA NVDA | Core tech sentiment |
| Semiconductors | AMD AVGO ASML TSM | Chip sector comparison for NVDA |
| Swiss Blue Chip | NESN.SW NOVN.SW ROG.SW | Defensive European signal |
| Pharma / Health | NVO LLY JNJ PFE AZN MRK ABBV UNH IBB | Sector context for LLY |
| German equities | ALV.DE SIE.DE BMW.DE BAS.DE | European macro proxy |
| Portfolio ETFs | EXUS.L ICGA.DE SGLN.L EMIM.AS MEUD.PA SJPA.MI | Direct portfolio coverage |
| Global index ETFs | VWCE.DE IWDA.AS CSPX.L | Broad market regime |
| Crypto | BTC-USD ETH-USD | Crypto market regime |
| Emerging markets | BABA TSM BHP RIO VALE | EM macro signal |
| Traditional commodities | GLD SLV XOM CVX | Geopolitical / inflation proxy |
| New-economy commodities | URA LIT DBA | AI datacenter energy, EV supply chain, food inflation |
| Defensives | XLP XLU | Crisis hedge — rise during risk-off rotations |

---

## Notebook structure

| Block | Description |
|-------|-------------|
| 1 | Install dependencies (first run only) |
| 2 | Imports + global seed |
| 3 | Portfolio configuration |
| 4 | Paths + CSV + folders |
| 5 | Price download + FX rates + market context (VIX, SPY) |
| 6 | Feature engineering |
| 7 | Independent ML ensembles D+1 / D+2 / D+3 |
| 7B | Feature importances → `model_metadata.csv` |
| 7C | Monthly SGD recalibration |
| 8 | Validate past forecasts + adaptive weight update + save new forecasts |
| 9 | Portfolio P&L analysis (fees, breakeven, exit targets) |
| 10 | Exit signals |
| 11 | 1 / 3 / 5 / 10-year projection |
| 12 | DCA simulation |
| 12B | Auto-cleanup of charts older than 30 business days |
| 13 | Chart generation |
| 14 | Final summary + HTML email |

---

## Repository structure

```
├── PrevisaoCarteira.ipynb           ← main notebook
├── AnaliseV5/
│   ├── predictions_log.csv          ← full forecast + validation history
│   ├── ensemble_weights.json        ← current weights per model per horizon
│   ├── model_metadata.csv           ← daily feature importances (RF + GB)
│   └── AnaliseGraficos/
│       └── TICKER_YYYYMMDD.png     ← one chart per asset per day
├── config/
│   ├── my_portfolio.json            ← personal portfolio tickers
│   └── watchlist.json               ← extended ML training universe
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← daily automation schedule
├── README.md                        ← this file (English)
└── README_pt.md                     ← Portuguese version
```

---

## Automation (GitHub Actions)

```
Weekdays 15:35 Barcelona (13:35 UTC)
  │
  ├─ Clone repository
  ├─ Install Python dependencies
  ├─ Execute full notebook (~8 min)
  │   ├─ Download prices + FX + VIX + SPY
  │   ├─ Validate previous forecasts
  │   ├─ Retrain models with updated history
  │   ├─ Update ensemble weights
  │   ├─ Save new D+1 / D+2 / D+3 forecasts
  │   ├─ Generate charts
  │   └─ Write HTML email report
  └─ Auto-commit with timestamp → push to repository
```

Failures trigger an automatic email notification from GitHub.

**Why 15:35?** European and crypto markets have closed. US markets are open but yesterday's data is final. This maximises data coverage without lookahead bias.

---

## Tech stack

```
Python 3.11
├── yfinance          — market data (prices, FX, VIX, SPY)
├── scikit-learn      — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy    — data processing and feature computation
└── matplotlib        — chart generation

GitHub Actions        — free daily automation
```

---

## Accuracy context

- A random directional forecast has 50% accuracy by definition.
- This system targets 55–65% directional accuracy on personal portfolio tickers.
- Accuracy below 52% over 30+ validations signals model degradation.
- No single accuracy figure justifies financial decisions on its own — this is a personal analytical tool, not financial advice.

---

## About

Built by **Vicky Costa** — Data Analyst | Data Science student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
