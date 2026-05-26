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
- D+1 / D+2 / D+3 directional forecasts per asset with previous-day Var%
- Long-term ETF projections (1 / 3 / 5 / 10 years)
- Model accuracy tracking, stratified by portfolio vs watchlist

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

### Calendar-aware target dates — why this matters

Forecasts use `pandas-market-calendars` with per-exchange mappings to compute target dates. `pd.offsets.BDay(N)` handles weekends but ignores market holidays. For example, if today is the Friday before a US bank holiday, `BDay(1)` returns Monday, but `pandas-market-calendars` for NYSE returns Tuesday — the actual next trading session. European exchanges (LSE, XETR, XAMS, XPAR, XMIL, SIX) have different holiday calendars from NYSE, and holiday-unaware date computation would silently produce target dates with no price data, corrupting the validation audit trail.

Each ticker is mapped to its exchange:

| Exchange | Calendar | Tickers |
|----------|----------|---------|
| NYSE (default) | `NYSE` | LLY, NVDA, BABA, BTC-USD, watchlist US |
| London | `LSE` | EXUS.L, SGLN.L, CSPX.L |
| Frankfurt / Xetra | `XETR` | ALV.DE, SIE.DE, BMW.DE, BAS.DE, DHER.DE, VWCE.DE, ICGA.DE |
| Amsterdam | `XAMS` | EMIM.AS, IWDA.AS |
| Paris | `XPAR` | MEUD.PA |
| Milan | `XMIL` | SJPA.MI |
| Swiss Exchange | `SIX` | NESN.SW, NOVN.SW, ROG.SW |

### Accuracy stratification — portfolio vs watchlist

The watchlist contains ~90 tickers used as macro context. These are not held assets — they are training signals. Accuracy figures for watchlist tickers are structurally lower (less data history, no entry-price anchoring) and should never be mixed with portfolio accuracy. The system tracks and reports accuracy separately:

- **Portfolio accuracy** — the number that matters for day-to-day decisions
- **Watchlist accuracy** — internal signal quality, not reported in the email

This distinction was introduced after observing a 28% blended accuracy figure that made the system appear to be performing at chance. The correct portfolio-only figure was 33% — still low due to limited validation history (< 30 samples per ticker at that point), but structurally different.

### Validation and audit trail

Every forecast written to `output/predictions_log.csv` includes:

| Column | Description |
|--------|-------------|
| `ticker` | Asset identifier |
| `pred_date` | Date the forecast was made |
| `target_date` | Date the forecast refers to (calendar-adjusted per exchange) |
| `horizon` | 1, 2, or 3 |
| `direction` | `up` or `down` |
| `ref_price` | Closing price on the day the forecast was made — the true reference for the direction check |
| `pred_price` | ATR-estimated target price (`close ± ATR × 0.5 × √horizon`) — informational only |
| `confidence` | Ensemble weighted probability |
| `actual_price` | Filled on validation day (initially `NaN`) |
| `actual_change_pct` | `(actual_price / ref_price − 1) × 100` — filled on validation |
| `correct` | `True` if actual ≥ ref_price (UP) or actual ≤ ref_price (DOWN); filled on validation day |
| `atr_at_prediction` | ATR14 at the time the forecast was made |
| `predicted_price` | Reserved for future use |
| `model_rf` | Individual Random Forest vote (`up`/`down`) |
| `model_gb` | Individual Gradient Boosting vote |
| `model_sgd` | Individual SGD Classifier vote |

Nothing is deleted or overwritten. The full audit trail is preserved indefinitely. New columns are added via a backwards-compatible migration block in `data/storage.py` — existing rows are backfilled where possible (e.g., `actual_change_pct` is derived from already-stored `actual_price` and `pred_price`).

### Consensus signal

```
BULLISH  → all three horizons predict UP
BEARISH  → all three horizons predict DOWN
MIXED    → disagreement across horizons
```

---

## Email report

The daily HTML email is designed to be read on mobile. It contains four sections:

### 1 — ML forecasts table

| Column | Content |
|--------|---------|
| Ativo | Ticker + asset name |
| Preço | Current closing price (EUR where applicable via FX) |
| Var% | Previous-day close-to-close change |
| D+1 | Directional forecast + confidence |
| D+2 | Directional forecast + confidence |
| D+3 | Directional forecast + confidence |
| Consenso | BULLISH / BEARISH / MISTO |

Each asset row is prefixed with ✅ or ❌ reflecting whether yesterday's D+1 forecast was correct.

**Legend:** ✅ = yesterday's D+1 prediction was correct · ❌ = yesterday's D+1 prediction was wrong

### 2 — ETF accumulation table

Long-term projections for accumulation ETFs at 1 / 3 / 5 / 10 years under pessimistic / base / optimistic growth scenarios. Values in EUR.

### 3 — Accuracy panel

Cumulative directional accuracy for **portfolio tickers only**, over the last 30 business days. Displayed with a chart-by-chart breakdown and a portfolio-level headline figure.

### 4 — Charts

One chart per portfolio asset, showing the last 120 trading days of price history with:
- Price + SMA20 + SMA50 + Bollinger Bands
- Entry price line (opening position)
- D+1 / D+2 / D+3 prediction arrows (calendar-aware dates)
- RSI (14-day)
- MACD + histogram
- Cumulative D+1 accuracy curve (once ≥ 3 validations are available)

Validation markers on charts show **D+1 only**: a ● for a correct prediction and an × for a wrong one. D+2 and D+3 are trained separately and shown in the email table, but are not plotted on the chart to avoid stacking multiple markers on the same target date.

---

## Public repository

Charts are published to a separate public repository ([smart-wallet-ml](https://github.com/srxkatsumi/smart-wallet-ml)) with a **10-day sliding window delay**. The public repo contains:

- One chart per portfolio asset per trading day, for the window D-19 to D-10
- An auto-generated README with the date of last update

No prices, positions, entry prices, or portfolio holdings are disclosed in the public repo. Chart filenames contain ticker symbols — that is intentional.

The sync runs as step 8 of the GitHub Actions workflow. It clones the public repo, copies the relevant charts, calls `scripts/gen_public_readme.py` to generate the README with the correct dates, and pushes. The 10-day delay prevents real-time signal copying while still allowing the charts to serve as a technical portfolio showcase.

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
| DHER.DE | Delivery Hero SE |

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
│   ├── settings.py                  ← all constants, paths, hyperparameters, TICKER_CALENDAR
│   ├── my_portfolio.json            ← personal portfolio (tickers, units, entry prices)
│   ├── portfolio.json               ← portfolio config with asset names
│   └── watchlist.json               ← extended ML training universe
├── data/
│   ├── downloader.py                ← market data download via yfinance
│   ├── storage.py                   ← CSV read/write helpers + backwards-compat migrations
│   └── calendars.py                 ← calendar-aware target date computation per exchange
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
├── scripts/
│   └── gen_public_readme.py         ← generates README for the public repo (called by CI)
├── output/
│   ├── predictions_log.csv          ← full forecast + validation history (never deleted)
│   ├── ensemble_weights.json        ← current weights per model per horizon
│   ├── model_metadata.csv           ← daily feature importances (RF + GB)
│   ├── resumo_diario.html           ← latest HTML email (committed daily)
│   ├── ultima_recalibracao.json     ← SGD recalibration timestamp
│   ├── models/                      ← serialised model files (.joblib)
│   └── charts/                      ← one chart per asset per day (auto-cleaned after 30 days)
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← daily automation (9 steps)
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
      ├─ 1. Checkout repository
      ├─ 2. Install Python 3.11
      ├─ 3. Install dependencies (pip install -r requirements.txt)
      ├─ 4. Run main.py (~8 minutes)
      │   ├─ Download prices + FX + VIX + SPY
      │   ├─ Compute features
      │   ├─ Validate previous forecasts
      │   ├─ Monthly SGD recalibration (if due)
      │   ├─ Retrain all models with updated history
      │   ├─ Update ensemble weights
      │   ├─ Save new D+1 / D+2 / D+3 forecasts
      │   ├─ Generate charts
      │   └─ Build HTML email report
      ├─ 5. Commit output files → push (predictions_log, weights, charts, html)
      ├─ 6. Prepare email subject date
      ├─ 7. Send HTML email via Gmail SMTP
      ├─ 8. Sync public repo (10-day delayed charts + auto-generated README)
      └─ 9. On failure: send failure notification email
```

**Why three cron entries:** GitHub Actions' scheduler is subject to queue delays of up to 2–3 hours under high load. Three separate cron triggers (30 minutes apart) are registered, but the anti-duplication check in Job 1 ensures the pipeline only executes once per day even if multiple crons fire.

**Why 17:45 Barcelona:** Frankfurt, Paris, London, Milan and Amsterdam all close at 17:30 CEST. Running at 17:45 captures the actual day-close prices for all European ETFs in the portfolio. US equities (LLY, NVDA, BABA) are still trading at this time — yfinance returns the most recent intraday price, not the day's close.

---

## Tech stack

```
Python 3.11
├── yfinance                 — market data (prices, FX rates, VIX, SPY)
├── scikit-learn             — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy           — data processing and feature computation
├── joblib                   — model serialisation
├── matplotlib               — chart generation
└── pandas-market-calendars  — per-exchange holiday calendars for target date computation

GitHub Actions               — free daily automation
Gmail SMTP                   — HTML email delivery
```

---

## Unit tests

8 automated tests run in GitHub Actions **before** `main.py`. If any test fails, the pipeline halts — the models do not train on potentially corrupt data.

```
pytest tests/ -v
```

| File | Tests | What it validates |
|------|-------|-------------------|
| `tests/test_features.py` | 2 | RSI14 always in [0, 100] on random and monotone price data |
| `tests/test_ensemble.py` | 2 | Ensemble and per-model probabilities always in [0, 1]; direction consistent with probability |
| `tests/test_pnl.py` | 4 | Breakeven > purchase price when fees > 0; breakeven == purchase price when fees = 0; USD→EUR conversion correct |

All tests use synthetic data — no network calls, no files on disk.

---

## Accuracy context

- A random directional forecast has 50% accuracy by definition.
- This system targets 55–65% directional accuracy on **portfolio tickers only**.
- Accuracy below 52% over 30+ portfolio validations signals model degradation.
- Accuracy figures are meaningless before ~30 validations per ticker — the system needs time to build a statistically meaningful sample.
- No single accuracy figure justifies financial decisions on its own — this is a personal analytical tool, not financial advice.

---

## Changelog

### Migration: Jupyter Notebook → modular Python
The original system was a single Jupyter notebook (AnaliseV5). It was migrated to a modular Python package to enable automated GitHub Actions execution, proper dependency management, and maintainability.

### Implemented improvements
- ✅ **Modular Python package** — `main.py` + `data/` + `features/` + `models/` + `portfolio/` + `reports/`
- ✅ **7-column email ML table** — Ativo · Preço · **Var%** · D+1 · D+2 · D+3 · Consenso
- ✅ **Var% column** — previous-day close-to-close change per asset in the email
- ✅ **✅/❌ legend in email** — clarifies what the icons represent (D+1 accuracy from the day before)
- ✅ **Accuracy stratification** — portfolio accuracy separated from watchlist; reported independently
- ✅ **predictions_log.csv new columns** — `actual_change_pct`, `atr_at_prediction`, `predicted_price`, `model_rf`, `model_gb`, `model_sgd`; backwards-compatible migration in `storage.py`
- ✅ **ATR at prediction stored** — `atr_at_prediction` captures the market volatility context at the time each forecast was made
- ✅ **Calendar-aware target dates** — `pandas-market-calendars` with per-exchange mapping replaces `pd.offsets.BDay` (which ignores market holidays)
- ✅ **Chart markers: D+1 only** — validation markers on charts show D+1 predictions only; D+2/D+3 stacking on the same target date removed
- ✅ **Future arrows: BDay-corrected** — prediction arrows on charts point to correct trading days (Friday → Monday, not Saturday)
- ✅ **Public repository** — `smart-wallet-ml` with 10-day delayed charts, synced daily by GitHub Actions step 8
- ✅ **Mobile-optimised email layout** — horizontal scroll tables with negative-margin trick for full-width display on ~412px viewports (Samsung Galaxy S26+)
- ✅ **`ref_price` column** — stores the actual closing price at prediction time; `actual_change_pct` and the correctness check now use this as the reference instead of the ATR-estimated `pred_price`
- ✅ **Correctness check fixed** — `correct = actual ≥ ref_price` (UP) / `actual ≤ ref_price` (DOWN); pure direction check, no longer requires the stock to reach the ATR target
- ✅ **Execution order fixed** — validate past predictions and update ensemble weights *before* training, so models always train with today's accurate weights rather than yesterday's stale ones
- ✅ **`save_model_metadata()` no longer retrains models** — `feature_importances_` are read from the models already trained in `train_all()`, eliminating a duplicate full training pass
- ✅ **Batched downloads with sleep** — yfinance requests split into groups of 20 with a 2-second pause between batches; eliminates silent NaN failures from rate limiting on large watchlists
- ✅ **SGLN.L price in EUR in email** — GBX (pence) tickers are now converted to EUR before display; price is consistent with all other assets in the ML table
- ✅ **Unit tests** — 8 pytest tests across 3 modules: RSI bounds (features), ensemble probability bounds, P&L fee logic; tests run in GitHub Actions before `main.py` and halt the pipeline on failure
- ✅ **Feature importance drift alert** — daily email panel with Spearman rank correlation (ρ) between today's feature ranking and the reference period; flags drift when ρ < 0.70 or the top feature changes
- ✅ **Retry on git push** — up to 3 attempts with `git pull --rebase` + 15s pause between each; eliminates data loss from transient push failures
- ✅ **GitHub artifact on failed push** — `predictions_log.csv` and `ensemble_weights.json` uploaded as a workflow artifact (retained 7 days) if all push attempts fail
- ✅ **Forward fill for NaN in VIX/SPY** — detects NaN in the last 3 days, applies ffill, and shows an amber warning band in the email when activated
- ✅ **Stock split detection** — price variation >40% from `ref_price` marks the validation as `NaN` instead of `False`; threshold configurable via `SPLIT_DETECTION_THRESHOLD`
- ✅ **✅/❌ icon fix** — `_acertou_ontem()` now filters by `target_date` instead of `pred_date`; shows whether the prediction *targeting* yesterday was correct, not the one *made* yesterday — fixes missing icons on Mondays and for tickers whose markets close after the pipeline runs
- ✅ **Feature drift panel legend** — added description of ρ, the reference period, and arrow meaning (↑ ↓ →) to the drift section in the email
- ✅ **Dynamic badge in public repo** — `last sync` badge powered by `shields.io/github/last-commit`; updates automatically on every page view without any JSON file or extra configuration

---

## Roadmap

> **Principle:** Stability → Observability → Publication → Model → Advanced features. Never the other way around. A better model on an unstable pipeline produces better results you can't trust.

### Week 1 — Pipeline stability

Before any model improvement, the pipeline must be fail-safe. Low effort, high impact.

| # | Item | Description |
|---|------|-------------|
| 1 | ✅ Retry on git push | 5 lines in the YAML. Eliminates data loss from a transient push failure. |
| 2 | ✅ GitHub artifact on failed push | Upload `predictions_log.csv` as a workflow artifact if the push fails — manual recovery safety net. |
| 3 | ✅ Forward fill for NaN in VIX/SPY | Use T-1 value if T-0 returns NaN; add a warning in the email when this happens. |
| 4 | ✅ Stock split detection | Price variation >40% in one day marks open validations as `NaN` instead of `False` — avoids penalising models for a corporate action. |

### Week 2 — Observability

The pipeline runs but doesn't tell you when it's degrading. That changes here.

| # | Item | Description |
|---|------|-------------|
| 5 | ✅ Feature importance drift alert | `model_metadata.csv` already stores daily importances — reads it and adds a drift panel to the email with Spearman rank correlation. |
| 6 | ⬜ Telegram as email fallback | ~20 lines; activates when Gmail fails — ensures the daily report is always delivered. *(deferred to end)* |
| 7 | ✅ Dynamic badge in public repo | `shields.io/github/last-commit` badge — updates automatically on every page view. |

### Week 3 — Public repo completion

Prepare everything for publication.

| # | Item | Description |
|---|------|-------------|
| 8 | ⬜ `predictions_log_public.csv` | Anonymised version of the log (no tickers, no prices) — proves real accuracy to anyone who opens the public repo. |
| 9 | ⬜ "Reliability" section in README | Group unit tests + fallbacks + retry logic in a single section. |
| 10 | ⬜ Semantic git tags | Tag each version milestone (`v1.0.0`, `v1.1.0`, …) to anchor the changelog in git history. |

### Weeks 4–5 — Model quality

Only here. Not before. The pipeline must be stable before touching the model.

| # | Item | Description |
|---|------|-------------|
| 11 | ⬜ Walk-Forward Validation | Replace single train/test split with rolling walk-forward for honest out-of-sample accuracy. |
| 12 | ⬜ Market regime as explicit feature | VIX-based regime label (low / medium / high volatility) as model input — context-specific pattern learning. |

### Week 6 — Reporting

| # | Item | Description |
|---|------|-------------|
| 13 | ⬜ Correlation matrix in email | Portfolio asset correlation heatmap — real concentration risk in stress scenarios. |
| 14 | ⬜ Fix SGLN.L projection | Three scenarios (pessimistic / base / optimistic) instead of a single historical growth rate. |

### Weeks 7–8 — Publication

| # | Item | Description |
|---|------|-------------|
| 15 | ⬜ Final READMEs (EN + PT) | Full revision of both READMEs before public launch. |
| 16 | ⬜ Public repo launch | Publish `smart-wallet-ml` as a complete, documented project. |
| 17 | ⬜ LinkedIn article | Tell the story of the project — from notebook to automated ML pipeline. |

### After publication — no fixed deadline

| # | Item | Description |
|---|------|-------------|
| 18 | ⬜ Fundamental event features | Earnings dates, FOMC weeks, options expiry. Requires a reliable external API; European coverage is limited. |
| 19 | ⬜ D+1 price regressor | Only with 1 year of clean accumulated data. |

---

## About

Built by **Vicky Costa** — Data Analyst | Data Science student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
