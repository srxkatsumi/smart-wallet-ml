# Roadmap

## Next (November and December 2026)

| Item | Why | Status |
|---|---|---|
| Ties no longer count as correct; accuracy on moves above the training threshold shown side by side | Evaluation should measure what the model was trained for | Planned |
| Watchlist data health ("612 of 616 with fresh data", list of stalled tickers) | 616 assets can go stale silently | Planned |
| Risk block and events calendar on the site | Today they only exist in the email | Planned |
| Isotonic calibration, Jinshi verdict for the extended ensemble and the regime challenger | Needs 20 aligned trading days (around 2026-12-04) | Waiting for data |
| Jinshi verdict for the volatility challengers | Around 2026-12-16 | Waiting for data |
| Monthly email (Jinshi verdicts, Diebold-Mariano, changes) | Long-horizon reading | Planned |
| Move logs and state out of git into a small hosted Postgres, with a read-only API | The repository-as-database caused most operational incidents | Planned (free tier) |
| "Ask the Portfolio": an LLM agent that answers questions over the prediction log in SQL, evaluated against the answers the pipeline already computes | Natural-language access to the audit trail, with a built-in test set | Planned (free LLM tier or local model) |


> **Principle:** Stability → Observability → Publication → Model → Advanced features. Never the other way around. A better model on an unstable pipeline produces better results you can't trust.

## Week 1: Pipeline stability

Before any model improvement, the pipeline must be fail-safe. Low effort, high impact.

| # | Item | Description |
|---|------|-------------|
| 1 | ✅ Retry on git push | 5 lines in the YAML. Eliminates data loss from a transient push failure. |
| 2 | ✅ GitHub artifact on failed push | Upload `predictions_log.csv` as a workflow artifact if the push fails — manual recovery safety net. |
| 3 | ✅ Forward fill for NaN in VIX/SPY | Use T-1 value if T-0 returns NaN; add a warning in the email when this happens. |
| 4 | ✅ Stock split detection | Price variation >40% in one day marks open validations as `NaN` instead of `False` — avoids penalising models for a corporate action. |

## Week 2: Observability

The pipeline runs but doesn't tell you when it's degrading. That changes here.

| # | Item | Description |
|---|------|-------------|
| 5 | ✅ Feature importance drift alert | `model_metadata.csv` already stores daily importances — reads it and adds a drift panel to the email with Spearman rank correlation. |
| 6 | ✅ Telegram as email fallback | `continue-on-error` on the Gmail step + Telegram bot fallback; activates when Gmail fails — ensures the daily report is always delivered. |
| 7 | ✅ Dynamic badge in public repo | `shields.io/github/last-commit` badge — updates automatically on every page view. |

## Week 3: Public repo completion

Prepare everything for publication.

| # | Item | Description |
|---|------|-------------|
| 8 | ✅ `predictions_log_public.csv` | Anonymised version of the log (no tickers, no prices) — proves real accuracy to anyone who opens the public repo. |
| 9 | ✅ "Reliability" section in README | Group unit tests + fallbacks + retry logic in a single section. |
| 10 | ✅ Semantic git tags | Tag each version milestone (`v1.0.0`, `v1.1.0`, …) to anchor the changelog in git history. |

## Weeks 4–5: Model quality

Only here. Not before. The pipeline must be stable before touching the model.

| # | Item | Description |
|---|------|-------------|
| 11 | ✅ Walk-Forward Validation | `TimeSeriesSplit(n_splits=5, gap=1)` — gap=1 prevents same-day lookahead between training and validation folds. |
| 12 | ✅ Market regime as explicit feature | `vix_regime` label (0 / 1 / 2) based on VIX thresholds (< 15 / 15–25 / ≥ 25); added to FEATURE_COLS; SGD models reinitialise automatically on feature count change. |

## Week 6: Reporting

| # | Item | Description |
|---|------|-------------|
| 13 | ✅ Correlation matrix in email | Portfolio asset return correlation heatmap (120-day window) embedded as base64 PNG; graceful fallback if matplotlib unavailable. |
| 14 | ✅ ETF projection scenarios | 10-year column shows three fixed scenarios: Pess = 3% / Base = 8% / Otim = 15%; eliminates the misleading single historical rate. |

## Weeks 7–8: Publication

| # | Item | Description |
|---|------|-------------|
| 15 | ✅ Final READMEs (EN + PT) | Full revision of both READMEs including educational deep-dive (`README_educativo.md`), all new features, and updated roadmap. |
| 16 | ✅ Public repo launch | `smart-wallet-ml` published with `predictions_log_public.csv`, 10-day delayed charts, and auto-generated README. |
| 17 | ⬜ LinkedIn article | Tell the story of the project — from notebook to automated ML pipeline. *(external content, not code — no fixed deadline)* |

## After publication

| # | Item | Description |
|---|------|-------------|
| 18 | ⬜ Fundamental event features | Earnings dates, FOMC weeks, options expiry. Requires a reliable external API; European coverage is limited. |
| 19 | ⬜ D+1 price regressor | Only with 1 year of clean accumulated data. |

## Research framework: 38 models across 13 families

All 25 original models are implemented and tested. Transversal layers (phases 9-15) are complete. Phases 16-20 are fully implemented.

| Phase | Family | Models | Status |
|-------|--------|--------|--------|
| 0 | Classic baseline | RF, GB, SGD | ✅ |
| 1 | Hidden state | Markov, HMM | ✅ |
| 2 | Advanced classic | XGBoost, LightGBM, CatBoost, SVM | ✅ |
| 3 | Time series | ARIMA, SARIMA, ETS, Holt-Winters, Prophet | ✅ |
| 4 | Recurrent neural | LSTM, GRU | ✅ |
| 5 | Attention neural | Transformer, TFT, N-BEATS | ✅ |
| 6 | Bayesian | Gaussian Process, BNN (MC Dropout) | ✅ |
| 7 | Generative | VAE, GAN | ✅ |
| 8 | Reinforcement | DQN, PPO | ✅ |
| 9 | Evaluation | Diebold-Mariano, McNemar, Ljung-Box, domain metrics | ✅ |
| 10 | Explainability | SHAP, Attention weights, LIME | ✅ |
| 11 | Meta-learning | Stacking, Optuna | ✅ |
| 12 | Tracking | MLflow (daily + research runner) | ✅ |
| 13 | Information theory | Shannon entropy, Mutual information, Permutation entropy, Transfer entropy | ✅ |
| 14 | Transfer learning | MMD, CORAL, Fine-tuning, Cross-domain evaluation | ✅ |
| 15 | Email report | Per-lot eToro table, ETF section, monthly buy recommendation | ✅ |
| 16 | Contrarian / Sanity tests | CB (Contrarian Baseline), EWI (Error-Weighted Inverter), PEL (Predictive Error Learning) | ✅ |
| 17 | Efficient architectures (post-2022) | TCN, DLinear, NLinear, PatchTST | ✅ |
| 18 | Foundation Models (2023-2024) | Chronos (Amazon), TimesFM (Google), Moirai (Salesforce) | ✅ |
| 19 | Calibrated uncertainty | Conformal Prediction (own implementation; MAPIE dropped in Aug 2026, see ERR-009) | ✅ |
| 20 | Drift detection | ADWIN, Page-Hinkley | ✅ |


## November 2026: statistical honesty and volatility

| # | Item | Status |
|---|------|--------|
| 1 | Calibrated confidence in the email (shrinkage until 20 days, then isotonic) | ✅ / ⏳ isotonic around 2026-12-04 |
| 2 | CIs, "always ▲" baseline, p-values and HLN in every reading | ✅ |
| 3 | System health, events and risk in the daily email | ✅ |
| 4 | Weekly email | ✅ |
| 5 | Prediction traceability | ✅ |
| 6 | Extended ensemble and regime challenger through Jinshi | ⏳ around 2026-12-04 |
| 7 | Volatility challengers through Jinshi | ⏳ around 2026-12-16 |
| 8 | Monthly email (Jinshi verdict, Diebold-Mariano, changes) | ⬜ |
| 9 | Gold consistency (PPFB.DE × SGLN.L) and euro-area inflation in the calendar | ⬜ |
