# Smart Wallet ML: an honest test of short-term market forecasting

An automated pipeline that runs every weekday after the close, forecasts the 1 to 3 day direction of a real ETF portfolio and a 616-asset watchlist, logs every forecast **before** the outcome, checks it against the official close, and asks one question with statistical discipline: **is any of this better than chance?**

The answer so far is the most useful part of the project: **short-term direction shows no demonstrated edge, while the size of the move (volatility) is predictable.** Both results are measured, documented and reproducible below.

[![Last Updated](https://img.shields.io/github/last-commit/srxkatsumi/smart-wallet-ml?label=last%20updated&color=brightgreen)](https://github.com/srxkatsumi/smart-wallet-ml/commits/main)
[![GitHub Actions](https://img.shields.io/badge/Automated-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart-wallet-ml/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-GitHub%20Pages-1a1740)](https://srxkatsumi.github.io/smart-wallet-ml/)

> ⚠️ **Study project. Nothing here is investment advice, and the forecasts must not be used as the basis for real investment decisions.**

---

## Results first

| Question | Answer | Evidence |
|---|---|---|
| Does the model predict 1 to 3 day direction better than a naive rule? | **No.** "Always ▲" beats it by 2 to 7 percentage points on every horizon | 2-year walk-forward on the portfolio, 11,472 forecasts (`scripts/estudo_vantagem.py`) |
| Does a higher stated confidence mean more hits? | **No.** Spearman correlation between confidence and hit is between -0.01 and +0.01 | Same study; no slice (asset, horizon, VIX regime, model agreement) survives Benjamini-Hochberg across 57 slices |
| Is the stated confidence useful as a probability? | **No.** Brier skill score -0.16 to -0.19 vs always saying 50% | Live log; confidence is now calibrated and shrunk to 50% (λ = 0) |
| Do adaptive ensemble weights help? | **No.** Same vote as a simple mean on ~54k forecasts per horizon | `scripts/ablation_ensemble_weights.py`; production now uses the simple mean |
| Is the size of the move predictable? | **Yes.** EWMA volatility beats the asset's historical variance (QLIKE 5% to 10% lower, both halves of the period) | `scripts/estudo_volatilidade.py`; GARCH and HAR did not beat EWMA |
| Can we predict a "big move" tomorrow (> 1 ATR)? | **Maybe.** Logistic model: AUC 0.61, BSS +1.4% vs base rate, well calibrated, but p = 0.146 after multiple-testing correction | Same study; now running as a live challenger |

A well-measured negative result is a result. The project keeps the direction forecasts running (they are cheap and they keep the evaluation honest), shows them calibrated to what the data supports, and is shifting its emphasis toward risk and volatility, where there is evidence.

---

## How it works

```
GitHub Actions (weekdays, after the close)
  │
  ├─ Download: 616 tickers + VIX, SPY, BTC, GLD (yfinance); a missing last session
  │             is rebuilt from hourly candles (European and Asian exchanges lag)
  ├─ Validate: every past forecast whose target date arrived is checked against
  │             the official close of its reference bar
  ├─ Features: 33 per asset (trend, momentum, volatility, volume, cross-asset, calendar)
  ├─ Calibrate: confidence vs real hits (shrinkage until 20 aligned days, then isotonic)
  ├─ Train + forecast: production ensemble (RF + GB + SGD) per asset and horizon
  ├─ Research: 13 model families + live challengers, report-only
  ├─ Risk: per-asset expected move (EWMA), drawdown, VaR; market context; events
  └─ Publish: emails, GitHub Pages site, append-only logs committed to the repo
```

---

## Methodology

**Target.** For horizon N (1, 2, 3 sessions), the label is up or down only when the move exceeds `0.3 × ATR14 × √N`; smaller moves are excluded from **training** to reduce label noise.

**Validation rule.** A forecast is correct when the official close on the target date is ≥ the reference close (for ▲) or ≤ it (for ▼). Two known caveats, both on the methodology to-do list:
- an unchanged price currently counts as correct for either direction;
- evaluation counts every day, including the small moves the model never trained on. Accuracy on moves above the training threshold is computed (`acc_mov_relevante`) and will be reported side by side.

**No look-ahead.**
- Horizons are counted from each asset's last real bar (`ref_date`) on its own exchange calendar (NYSE, Xetra, Paris, Madrid, Tokyo, Hong Kong...).
- External context uses the previous day's close.
- Walk-forward retrains only on data strictly before each test day, with a gap equal to the horizon.

**Significance.**
- The trading day is the unit of analysis: hundreds of assets on the same day are not independent observations.
- Accuracy is tested with a t-test on daily accuracy, reported only after 100 forecasts across 20 days.
- Model comparisons use Diebold-Mariano on the daily mean loss differential, with h−1 lags and the Harvey-Leybourne-Newbold small-sample correction.
- Families and slices are corrected with Benjamini-Hochberg.
- Every accuracy is shown with a 95% CI and the "always ▲" baseline, not against a fixed 50%.

**Jinshi, the validation gate.** No signal is promoted to production unless it passes all four:
1. it beats the baseline with BH-corrected significance;
2. the effect exceeds the cost of trading (0.15%);
3. it wins in both chronological halves;
4. it has a stated economic rationale.

---

## Models

**Production (what the daily email shows).** Per asset and per horizon, three classifiers, combined by simple mean:

| Model | Setup | Role |
|---|---|---|
| Random Forest | 100 trees, depth 5, isotonic calibration | Robust to noisy, partly irrelevant features |
| Gradient Boosting | 100 estimators, depth 3, learning rate 0.05, isotonic calibration | Sequential correction of residual errors |
| SGD Classifier | log loss, L2, incremental updates, monthly full recalibration | Linear counterweight to the two tree models |

The combined probability is then calibrated against real hits before it is shown.

**Research families (report-only).** Classical boosting and SVM (XGBoost, LightGBM, CatBoost, SVM), hidden states (Markov chain, HMM), time series (ARIMA, SARIMA, ETS, Holt-Winters, Prophet), recurrent networks (LSTM, GRU), attention (Transformer, simplified TFT and N-BEATS), Bayesian (Gaussian process, MC-dropout BNN), generative (VAE, GAN discriminator), reinforcement learning (DQN, PPO), contrarian sanity checks, efficient architectures (TCN, DLinear, NLinear, PatchTST), foundation models (Chronos, TimesFM- and Moirai-inspired), conformal prediction and drift detection (ADWIN, Page-Hinkley). 13 families in total; their consensus is shown in the email but never changes production.

**Live challengers (logged daily, judged by Jinshi after 20 aligned days).**
- **Extended ensemble:** production plus the 13 families, with weights shrunk toward equal until performance is sustained.
- **Regime:** the production trio plus VIX/VIX3M, the 10-year yield change and the dollar.
- **Volatility:** a logistic model for "big move tomorrow" and EWMA volatility for the next 1, 3 and 5 sessions.
- **Regression experiment:** a Random Forest regressor with a 90% conformal interval, compared with a paired Diebold-Mariano test against an ATR heuristic. It has its own email; see `models/regressor_experimento.py`.

**MaoMao.** A small spiking neural network (leaky integrate-and-fire) that turns daily market sensors into a mood and learns which fears were justified three days later. It is shown on the site and has not yet been through Jinshi.

---

## Features (33 per asset)

| Group | Features |
|---|---|
| Trend and momentum | distance to SMA20/SMA50, SMA cross, RSI14, MACD (line, signal, histogram), returns over 1 day, 5 days, 1, 3, 6 and 12 months |
| Volatility and volume | Bollinger width and position, ATR14, 10-day volatility, volume ratio, OBV trend (clipped), VWAP distance |
| Extremes | distance to the 52-week high and low |
| Cross-asset context (previous day) | SPY return, VIX level, change and regime, BTC and gold returns, 20-day correlation with SPY |
| Calendar | day of week, month, options-expiry week |
| Asset class | stock, equity ETF, commodity ETF, crypto |

The calendar features are included as candidates, not as established facts. Documented calendar anomalies (day-of-week, January, "sell in May") tend to weaken after publication (McLean & Pontiff, *Journal of Finance*, 2016), and nothing in this project's results suggests otherwise.

---

## What it produces

**Emails.**
| Email | When | Contents |
|---|---|---|
| Daily | Every run | System health, upcoming events, risk, calibrated forecasts, extended ensemble, accuracy with CI, drift, correlation, the 13 families |
| Weekly | Last session of the week (Saturday morning) | Families with CI and corrected p-values, calibration (Brier), diversification, challenger scoreboards |
| Experiments | Every run | Regression challenger vs champion |
| Alert | Only on anomaly days | VIX regime shift or a ≥ 15% daily VIX move, or a portfolio asset moving ≥ 2.5x its own ATR |

**Site** ([srxkatsumi.github.io/smart-wallet-ml](https://srxkatsumi.github.io/smart-wallet-ml/), English and Portuguese):
- **Home and Favorites:** charts and forecasts for the portfolio.
- **Watchlist:** 616 assets; click any of them for the same charts.
- **Models, Experiments and Alerts.**

**Logs.** `output/predictions_log.csv` is append-only and never rewritten. Every row stores the code version, calibration rule, a hash of the input bar and the run time, so any number in an email can be traced back.

---

## Universe

- **Portfolio (public):** 6 accumulating ETFs: ICGA.DE, PPFB.DE, EXUS.MI, LYP6.DE, IS3N.DE, EUNN.DE. Holdings that must stay private are read from a GitHub secret (`PORTFOLIO_PRIVADO`) and only ever appear in the emails, never in the repository or on the site.
- **Watchlist:** 616 tickers across US, European, Spanish, Japanese, Hong Kong and US-listed Chinese stocks, UCITS and US ETFs, bonds, commodities and crypto (`config/watchlist.json`). All prices are shown in EUR, converted with each day's exchange rate.

---

## Automation and reliability

- **Schedule.** The workflow is scheduled at 20:30 UTC on weekdays (21:30 in Barcelona in winter, 22:30 in summer), with two fallback triggers. GitHub often starts it later; a full run takes about 75 minutes.
- **Before the run.** 271 tests in 41 files run first and stop the pipeline on failure. The 8 tests for the classical models run in a separate process because XGBoost and PyTorch conflict in the same process on macOS.
- **During the run.**
  - Every step that can fail without blocking is recorded and shown in the email's health block.
  - Assets with a stale last bar are flagged.
  - An anomaly guard ignores stale data.
- **After the run.** Validation checks the output, the push is retried with recovery, and an emergency artifact is uploaded if the push fails.

**Known limitation: the repository is the database.** Logs, state and generated site data are committed every day. It is free and fully auditable, but it caused most of the operational incidents (push races, rebase failures; see [README_Errors_EN.md](README_Errors_EN.md)). Moving the data to a small hosted Postgres with an API is on the roadmap.

---

## Repository structure

```
main.py                 daily pipeline
config/                 portfolio, watchlist (616), settings, macro event calendar
data/                   download, exchange calendars, storage and log migrations
features/               the 33 features
models/                 production ensemble, calibration, validator, research families
research/               research runner and live challengers (extended, regime, volatility)
evaluation/             significance, Diebold-Mariano, anomaly, risk, events, health, traceability
jinshi/                 validation gate and backtests
maomao/                 spiking neural network "pet"
reports/                daily, weekly, experiments and alert emails
docs/                   GitHub Pages site (EN + PT) and its export scripts
scripts/                studies and ablations (direction edge, volatility, regime features, weights)
tests/                  41 test files
output/                 append-only logs and state (committed by the bot)
```

---

## Tech stack

Python 3.11 · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch · hmmlearn · Prophet · Chronos (Amazon) · SHAP · LIME · MLflow · yfinance · pandas-market-calendars · Matplotlib · pytest · GitHub Actions · GitHub Pages

---

## Limitations

- Direction forecasts have no demonstrated edge; treat them as an experiment, not a signal.
- The models do not read news. Earnings and central-bank decisions are flagged in the email, but are not inputs.
- The aligned evaluation history started on 2026-11-07. Several live tests need about 20 trading days before they report anything.
- A free data source (yfinance) means occasional gaps; they are filled from hourly candles or flagged, never silently invented.

---

## More

- [CHANGELOG.md](CHANGELOG.md): what changed and when.
- [ROADMAP.md](ROADMAP.md): what is planned.
- [README_Errors_EN.md](README_Errors_EN.md): every incident, with root cause and fix.
- [README_pt.md](README_pt.md): this README in Portuguese.
- [README_educativo.md](README_educativo.md): a long-form explanation in Portuguese.

---

## About

Built by **Vicky Costa**, Data Analyst.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycostasanches/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
