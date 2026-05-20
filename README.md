# Carteira Inteligente — ML Portfolio Forecasting System

> A self-learning investment portfolio analysis and forecasting system built from scratch.
> Runs automatically every weekday at 17:45 (Barcelona / CEST) via GitHub Actions.

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

### Why independent ensembles per horizon — not a single model

Most tutorials train one model and extrapolate its output forward in time. That is a mistake: the patterns that predict tomorrow's move are structurally different from those that predict a move three days out. Tomorrow's price is dominated by short-term momentum and overnight sentiment. A 3-day horizon is more influenced by trend persistence and macro regime. Training a single model and reusing it across horizons conflates these different dynamics. Three separate ensembles are trained:

```
D+1 Ensemble ──► RF_d1 · GB_d1 · SGD_d1  ──► weights_d1
D+2 Ensemble ──► RF_d2 · GB_d2 · SGD_d2  ──► weights_d2
D+3 Ensemble ──► RF_d3 · GB_d3 · SGD_d3  ──► weights_d3
```

Each ensemble learns the temporal signature of its own horizon independently.

### Why three different models in each ensemble

Using a single model per horizon would give a single point of failure. Different algorithms make different types of errors on the same data, so combining them reduces variance without increasing bias. Each model in the ensemble was chosen for a specific reason:

| Model | Configuration | Why it was chosen |
|-------|--------------|-------------------|
| **Random Forest** | 300 trees, max depth 6 | Robust generaliser. Bootstrapped trees prevent overfitting on noisy market data. Works well even when some features are irrelevant — a common situation in financial time series where indicator usefulness changes with regime. Acts as the ensemble's stability anchor. |
| **Gradient Boosting** | 200 estimators, learning rate 0.05 | Captures patterns the Random Forest misses by sequentially correcting its own residual errors. Especially good at detecting short-term momentum signals and subtle interactions between indicators. The low learning rate (0.05) slows convergence intentionally — it prevents the model from memorising noise. |
| **SGD Classifier** | log_loss | A linear model deliberately included to act as a counterweight. When both non-linear models agree on something that is actually noise, the SGD Classifier — which cannot model non-linear interactions — acts as a dissenting vote and pulls the ensemble toward more conservative estimates. Its simplicity is a feature, not a limitation. |

The SGD model undergoes **full monthly recalibration**: scaler refit + retrain from scratch. This is necessary because the SGD Classifier uses standardised features. If the distribution of prices and indicators shifts gradually (e.g., after a major market repricing), the old scaler no longer represents the current data, and the model's linear coefficients become anchored to an obsolete baseline. Monthly recalibration keeps the linear anchor aligned with current market conditions without requiring daily recalibration overhead.

### Feature engineering — what the model sees

Each model receives a set of technical indicators computed from historical price data, plus external market context:

| Feature | Description | Why it matters |
|---------|-------------|----------------|
| `sma_20`, `sma_50` | 20-day and 50-day simple moving averages | Short vs medium-term trend alignment. Crossovers between them are a classical regime-change signal. |
| `rsi_14` | Relative Strength Index (14-day) | Detects whether the asset is overextended in either direction. RSI > 70 (overbought) and RSI < 30 (oversold) are historically mean-reverting conditions. |
| `macd`, `macd_signal` | MACD line and signal line | Captures momentum and trend reversals through crossovers. Useful for detecting when a trend is gaining or losing steam. |
| `bb_upper`, `bb_lower`, `bb_width` | Bollinger Bands (20-day, 2σ) | Encodes both volatility regime and price extremity. When price hits the upper/lower band, the model can factor in the probability of reversion. Band width signals whether the asset is in a quiet or explosive period. |
| `atr_14` | Average True Range (14-day) | The expected daily move in absolute terms. Helps the model distinguish between a +1% move that is within normal range and one that is exceptional. |
| `ret_1d`, `ret_5d` | 1-day and 5-day returns | Direct momentum features. Recent returns are among the most predictive short-horizon features. |
| `spy_ret_1d` | S&P 500 return (T-1) | Global market context. NVDA on a day after the S&P dropped 2% behaves differently from NVDA on a neutral day. This feature allows the model to condition its forecast on the broad market state. |
| `vix_level` | CBOE VIX closing level (T-1) | The market's implied volatility — the "fear gauge". A VIX of 30 means a fundamentally different environment from a VIX of 14. Without this, the model cannot distinguish bull-market and crisis-regime behaviour. |
| `vix_change` | VIX daily change (T-1) | Captures fear *acceleration*, not just fear level. A VIX that is rising sharply often leads different outcomes than the same absolute VIX level that has been stable for weeks. |

**Why T-1 for external context features:** when the pipeline runs at 17:45 Barcelona time, European markets have just closed but US markets are still open. Yesterday's US close (T-1) is the most recent complete and final data point for SPY and VIX. Using the in-progress T-0 values would constitute lookahead bias — the model would be trained on information that did not yet exist at forecast time.

### Adaptive ensemble weights with temporal decay

After each validation cycle, model weights are updated using accuracy over a rolling window with exponential decay:

```
weight(model) ∝ accuracy(model) · Σ decay^(days_ago)
```

More recent correct predictions carry more weight than older ones. If a model starts underperforming systematically, the ensemble lowers its vote share automatically — without any manual intervention.

**Why exponential decay rather than a flat rolling window:** a flat window gives equal importance to a correct prediction from 28 days ago and one from yesterday. Markets shift. A model that was the best predictor in February may have simply found a pattern in a trending market that no longer exists after a regime change. The exponential decay function ensures recent performance is weighted disproportionately more, making the system reactive to regime changes rather than stuck in its own past.

### Business day target dates — why this matters

Forecasts use `pd.offsets.BDay(N)` to compute target dates, not `pd.Timedelta(days=N)`. The difference: if today is Friday, D+1 using Timedelta resolves to Saturday — a non-trading day with no price. D+1 using BDay resolves to Monday, the next actual trading session. Using calendar days would cause the validator to look for prices that do not exist, producing silent NaN validations and corrupting the accuracy tracking.

### Validation and audit trail

Every forecast written to `output/predictions_log.csv` includes:

| Column | Description |
|--------|-------------|
| `pred_date` | Date the forecast was made |
| `target_date` | Date the forecast refers to (BDay-adjusted) |
| `horizon` | 1, 2, or 3 |
| `direction` | `up` or `down` |
| `confidence` | Ensemble weighted probability |
| `actual_price` | Filled on validation day (initially `NaN`) |
| `correct` | `True` / `False` (filled on validation day) |

Nothing is deleted or overwritten. The full audit trail is preserved indefinitely.

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

The watchlist expands the training universe beyond the personal portfolio. Models learn cross-asset correlations and market regime signals from this broader dataset, producing more contextually-aware forecasts for the portfolio's own assets.

| Group | Tickers | Why it is here |
|-------|---------|----------------|
| US Big Tech | AAPL MSFT GOOGL AMZN META TSLA NVDA | General tech sector sentiment |
| Semiconductors | AMD AVGO ASML TSM | Sector benchmark for NVDA |
| Swiss Blue Chip | NESN.SW NOVN.SW ROG.SW | Defensive European signal |
| Pharma / Health | NVO LLY JNJ PFE AZN MRK ABBV UNH IBB XBI | Sector context for LLY |
| German equities | ALV.DE SIE.DE BMW.DE BAS.DE | European macro proxy |
| Portfolio ETFs | EXUS.L ICGA.DE SGLN.L EMIM.AS MEUD.PA SJPA.MI | Direct portfolio coverage |
| Global index ETFs | VWCE.DE IWDA.AS CSPX.L | Broad market regime |
| Crypto | BTC-USD ETH-USD | Crypto market regime |
| EM tech / resources | BABA TSM BHP RIO VALE | Emerging market macro signal |
| Traditional commodities | GLD SLV XOM CVX COPX | Geopolitical / inflation proxy |
| New-economy commodities | URA LIT DBA | AI datacenter energy (uranium), EV supply chain (lithium), food inflation |
| Defensives | XLP XLU | Crisis hedge — rise during risk-off rotations |
| Bonds | TLT AGG HYG TIP EMB LQD SHY | Interest rate regime and credit conditions |
| India | INDA INFY WIT | Fastest-growing large EM, tech outsourcing signal |
| Brazil | EWZ ITUB | Commodity-linked EM, BRL risk proxy |
| REITs | VNQ VNQI O PLD | Rate sensitivity indicator |
| Volatility | UVXY VXX | Real-time fear / hedging demand |
| US Sectors | XLF XLK XLE XLV XLI XLY | Sector rotation detection |
| China | MCHI FXI | Direct coverage for BABA and ICGA.DE context |
| Japan | EWJ | Coverage for SJPA.MI context |
| Europe broad | VGK | European market breadth |
| Latin America | ILF | Regional EM diversification signal |
| Thematic | ICLN CIBR BOTZ ITA PHO | Clean energy, cybersecurity, robotics, defence, water |
| Regional banks | KRE | US small/mid bank stress indicator |
| Dividend / quality | VYM NOBL | Quality factor signal |
| Broad market | QQQ IWM RSP | Growth vs value vs equal-weight rotation |
| Frontier markets | FM ASEA | Peripheral EM signal |

---

## Repository structure

```
├── main.py                          ← pipeline orchestrator
├── config/
│   ├── my_portfolio.json            ← personal portfolio (tickers, units, entry prices)
│   └── watchlist.json               ← extended ML training universe
├── data/
│   ├── downloader.py                ← market data download via yfinance
│   └── storage.py                   ← CSV read/write helpers
├── features/
│   └── engineering.py               ← technical indicators + ML feature matrix
├── models/
│   ├── ensemble.py                  ← RF + GB + SGD training + adaptive weight updates
│   └── validator.py                 ← forecast validation against realised prices
├── portfolio/
│   ├── pnl.py                       ← P&L, fees, breakeven, exit targets
│   ├── exit_signals.py              ← exit signal logic
│   ├── projections.py               ← 1 / 3 / 5 / 10-year projections
│   └── dca.py                       ← DCA simulation
├── reports/
│   ├── charts.py                    ← chart generation (matplotlib)
│   └── email_report.py              ← HTML email builder
├── output/
│   ├── predictions_log.csv          ← full forecast + validation history
│   ├── ensemble_weights.json        ← current weights per model per horizon
│   ├── model_metadata.csv           ← daily feature importances (RF + GB)
│   ├── resumo_diario.html           ← latest HTML email (committed daily)
│   ├── ultima_recalibracao.json     ← SGD recalibration timestamp
│   ├── models/                      ← serialised model files (.joblib)
│   └── charts/                      ← one chart per asset per day (auto-cleaned after 30 days)
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← daily automation schedule
├── requirements.txt
├── README.md                        ← this file (English)
└── README_pt.md                     ← Portuguese version
```

---

## Automation (GitHub Actions)

```
Mon–Fri 17:45 Barcelona (15:45 UTC, accounting for ~2h GitHub scheduler delay)
  │
  ├─ Job 1: check if pipeline already ran today
  │   └─ reads predictions_log.csv — if today's date exists, skip (~10s)
  │
  └─ Job 2: execute pipeline (only if not yet run today)
      ├─ Clone repository
      ├─ Install Python 3.11 + dependencies
      ├─ Run main.py (~8 minutes)
      │   ├─ Download prices + FX + VIX + SPY
      │   ├─ Compute features
      │   ├─ Validate previous forecasts
      │   ├─ Retrain models with updated history
      │   ├─ Update ensemble weights
      │   ├─ Save new D+1 / D+2 / D+3 forecasts
      │   ├─ Generate charts
      │   └─ Build HTML email report
      ├─ Commit updated output files → push
      └─ Send HTML email via Gmail SMTP
```

**Why three cron entries:** GitHub Actions' scheduler is subject to queue delays of up to 2–3 hours under high load. Three separate cron triggers (30 minutes apart) are registered, but the anti-duplication check in Job 1 ensures the pipeline only executes once per day even if multiple crons fire. This guarantees delivery without requiring a paid Actions plan with priority scheduling.

**Why 17:45 Barcelona:** Frankfurt, Paris, London, Milan and Amsterdam all close at 17:30 CEST. Running at 17:45 captures the actual day-close prices for all European ETFs in the portfolio (EMIM.AS, MEUD.PA, SJPA.MI, ICGA.DE, EXUS.L, SGLN.L). US equities (LLY, NVDA, BABA) are still trading at this time — yfinance returns the most recent intraday price, not the day's close.

Failures trigger an automatic email notification from GitHub.

---

## Tech stack

```
Python 3.11
├── yfinance          — market data (prices, FX rates, VIX, SPY)
├── scikit-learn      — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy    — data processing and feature computation
├── joblib            — model serialisation
└── matplotlib        — chart generation

GitHub Actions        — free daily automation
Gmail SMTP            — HTML email delivery
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
