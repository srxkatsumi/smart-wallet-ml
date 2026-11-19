# Changelog

Detailed incident write-ups live in [README_Errors_EN.md](README_Errors_EN.md).


## Incidents & fixes

### 2026-11-19: calibrated confidence on the site and context without the day's data

- **Site:** Home, Favorites and Watchlist showed raw confidence while the email showed the calibrated one (the same forecast read 50% in one and 64% in the other). The site now shows the calibrated value with the raw one below.
- **Context (VIX, SPY, BTC, GLD):** when Yahoo returned the day's row with no price, the code repeated the previous day (SPY and GLD on 2026-11-18), feeding every asset's features. The empty row is now dropped and the session is built from hourly candles, as already done for assets; if that also fails, the email warns.
- **Validator:** the `yfinance` import was lost in the 2026-11-06 audit, so pending predictions for tickers that left the watchlist were never checked.

### 2026-11-12 to 2026-11-18: email review, honest calibration and volatility challengers

- **Portfolio:** SGLN.L left the portfolio (same ETC as PPFB.DE) and stays in the watchlist. Private holdings now live in a GitHub secret and only appear in the emails.
- **Site:** the deploy triggered by the daily run published the commit from before the new data, leaving the site a day behind (ERR-016). The accuracy card showed "-" while significance had fewer than 20 days.
- **External email review (ERR-018):** counters with no universe or window, "25 models consensus" (it is 13), inverted consensus count, drift legend missing its second criterion and the families section labelled as weekly. All fixed and labelled.
- **Confidence:** a 2-year study showed directional confidence has no relation to hits in any slice (asset, horizon, VIX regime, model agreement), and "always ▲" beats the model by 2 to 7 points. Confidence is now calibrated by shrinkage until there are 20 days for the isotonic curve; today λ = 0 and the email shows 50 with the raw value beside it.
- **Ensemble weights:** adaptive weights gave exactly the same vote as a simple mean across ~54k predictions per horizon. Production now uses the simple mean.
- **Volatility:** move size is predictable where direction is not. EWMA beat the historical variance (QLIKE 5% to 10% lower); GARCH and HAR did not beat EWMA; the logistic model for "big move tomorrow" reached AUC 0.61 and BSS +1.4% vs the base rate without surviving multiple-testing correction. The expected move is now in the email; the logistic model and EWMA run as live challengers.
- **New in the email:** system health, upcoming events, risk, CIs, "always ▲" baseline, HLN in Diebold-Mariano, version footer and the weekly email. Barcelona time zone with automatic DST (ERR-017).
- **Traceability:** every prediction stores `versao_codigo`, `versao_calibracao`, `hash_dados` and `executado_em`.
- **Watchlist:** clicking an asset opens the same charts as Favorites (price with SMA and Bollinger, RSI, MACD, accuracy and forecasts). 88 tickers added (616 in total): 11 UCITS accumulating ETFs on Xetra, 29 European stocks (Germany, France, Netherlands, Italy, Switzerland), 8 Spanish IBEX stocks, 7 UK stocks via US ADRs, 11 Japanese stocks in Tokyo, 7 Chinese stocks in Hong Kong, 8 US-listed Chinese stocks, fintech/BNPL names (KLAR, XYZ, TOST, SEZL) and 3 gold-miner ETFs. New tickers inherit their exchange calendar from the suffix (.MC, .PA, .AS, .DE, .T, .HK...), the page converts yen and Hong Kong dollars to EUR, and the currency/dividend cache fetches missing tickers immediately.

### 2026-08-17: Silent gaps found during a project-wide weak-point review

Not a single incident — a deliberate audit turned up three things that were quietly wrong without ever throwing an error:

1. **Test suite disconnected from CI.** 21 test files existed, `pytest` was in `requirements.txt`, but the workflow never invoked it. `tests/test_email_phase15.py` had been broken (`ImportError`) for weeks — it tested a PnL/lots email section removed in an earlier refactor — and nothing noticed. Fix: deleted the stale test, added `pytest.ini` (`--continue-on-collection-errors`, so one broken file can never again silently void the whole suite), wired `pytest tests/` in as the first step of the job. Commit `5d4078378`.
2. **SGD "incremental learning" was inert since 2026-06-26.** `output/models/*.pkl` is git-ignored (a deliberate fix after stale post-migration models silently broke `partial_fit` and produced zero predictions that day — see the guard added at the time, `_load_sgd`'s `n_features_in_` check), but no cache step ever restored the folder either — so every run silently fell back to a full `.fit()` from scratch, every day, never `partial_fit`. Fix: added an `actions/cache@v4` step for `output/models/` (keyed like the existing research-model cache); the pre-existing feature-count guard means a future feature change forces a safe re-init instead of repeating the June incident. Commit `5d4078378`.
3. **`PPFB.DE` missing from `ASSET_CLASSES`/`TICKER_CALENDAR`.** Present in the portfolio, absent from both hand-maintained dicts — fell back to `asset_class=0`/`NYSE` silently instead of its real values (iShares Physical Gold ETC, Xetra, commodity ETF). Fix: added the correct entries plus `tests/test_config_coverage.py`, which now fails loudly in CI if a future portfolio ticker is missing from either dict. Commit `5d4078378`.

Also fixed the same day: the regression-experiment's Diebold-Mariano test used the default `h=1` for every horizon, understating the long-run variance for D+2/D+3 (overlapping forecast windows are serially correlated — the original 1995 paper recommends `h = forecast horizon`). Now calls `diebold_mariano(e_challenger, e_champion, h=day)`. See "Regression experiment" above.

### 2026-08-10 to 2026-08-11: Fallback retries raced each other; one trading day's cycle silently never ran

**Symptom:** no email arrived for the 2026-08-11 close, even though the workflow showed "success" runs that evening.

**Root cause 1 — race between fallback attempts:** with the research pipeline + regression experiment, a full run now takes ~1h20-2h — longer than the gap between the three fallback cron triggers. On 2026-08-10, a second attempt started its own full ~2h27m run because the "already ran today" check only grepped the *committed* CSV, which the first (still-running) attempt hadn't pushed yet. The loser's push was rejected, and the recovery path (`git pull --rebase`) itself failed because `output/models/*.pkl` is never committed, leaving the working tree permanently "dirty" for rebase purposes.

**Fix:** the anti-duplication check now also queries the GitHub Actions API for any run already in progress or succeeded today (catches the race in seconds, not ~2h later at push time); the push retry yields to a same-day run that already published instead of trying to merge two independent full pipeline runs. Commit `3ef79b576`.

**Root cause 2 — `hoje` computed mid-run:** `main.py` computed `hoje = pd.Timestamp.now().normalize()` after downloading market data for the whole watchlist, not at the top of `main()`. A run starting at 23:38 UTC that took until 01:31 UTC the next day logged its predictions with `pred_date` = the *later* date — technically correct market-data-wise (NYSE had already closed), but it meant the anti-duplication check for the real next day saw "already done" and skipped all three of that day's attempts, so that trading day's actual close was never processed.

**Fix:** moved `hoje = pd.Timestamp.now().normalize()` to the first line of `main()`. Commit `3ece5070d`. A secondary defensive fix — the "✅ Validar dados gerados" step recomputing `today` via shell `date` *after* `main.py` finished could itself land on the wrong side of the same midnight boundary — was patched to accept either today's or yesterday's `pred_date`. Commit `3ea9e8f47`.

### 2026-08-05: `mapie` incompatible with the pinned scikit-learn — regression challenger failed for 100% of tickers

**Symptom:** the new regression-experiment email shipped with "0 ativos" — the challenger model failed to train for every single ticker.

**Root cause:** `mapie<1` (resolved to 0.9.x) depends on scikit-learn internals (`EnsembleRegressor.__sklearn_tags__`) that don't exist in the pinned `scikit-learn==1.8.0` — a version incompatibility invisible locally because a throwaway test venv had resolved an older, compatible scikit-learn without that pin.

**Fix:** dropped the `mapie` dependency; split-conformal prediction intervals are now computed by hand (fit on the first ~80% of the training window, calibrate the interval on the most recent ~20% — same manual/version-independent philosophy already used by `models/conformal.py` for the production classifiers, for exactly this reason). Commit `a4a602db5`.

### 2026-07-30: `KeyError: 'etoro'` after removing the eToro holdings from the portfolio

**Symptom:** three consecutive scheduled runs failed immediately with `KeyError: 'etoro'`.

**Root cause:** `config/portfolio.json`'s `etoro` key was removed (holdings sold, no longer tracked), but `data/storage.py:load_my_tickers()` and `main.py` still accessed `portfolio_cfg["etoro"]` directly instead of `.get("etoro", [])`.

**Fix:** both call sites now default to an empty list when the key is absent; the research pipeline (previously scoped only to eToro tickers) now falls back to the ETF portfolio when `etoro` is empty. Commit `e32e748c`.

### 2026-06-05 to 2026-06-09: Pipeline silent failure (4 missed days)

**Symptom:** No automated runs and no daily email from 2026-06-05 through 2026-06-08 (4 trading days).

**Root cause 1: Invalid YAML in workflow file (primary):**
Commit `18d3902` added a data-validation step to `.github/workflows/executar_diario.yml`. The validation used a multi-line `python -c "..."` block whose Python code started at column 0, while the YAML `run: |` block required all content to be indented at least 10 spaces (matching the first line of the block). GitHub Actions' YAML parser rejected the entire workflow file with `Invalid workflow file: error on line 99`. Because the file was syntactically invalid, **no jobs ran at all**: not the pipeline, not the email, not even the "already ran today" check. The scheduler continued firing but every attempt immediately errored out before executing a single step. After 3 consecutive failures, GitHub paused the scheduled runs entirely.

**Fix:** Collapsed the multi-line Python into a one-liner: `python -c "import pandas as pd; log=pd.read_csv(...); print(len(...))"`, valid YAML on a single line. Commit `82e3f3f`.

**Root cause 2: Unpinned `scikit-learn` installing version 2.x:**
With the YAML fixed (run #65), the pipeline ran and failed immediately on all 95 tickers with `No module named '_loss'`. scikit-learn 2.x restructured internal Cython extensions (including `_loss`), which `GradientBoostingClassifier` and other estimators rely on at training time. Because `requirements.txt` had `scikit-learn` with no version pin, GitHub Actions installed the latest available version (2.x), which was incompatible with the project. The local environment had scikit-learn 1.8.0, which worked correctly.

**Fix:** Pinned `scikit-learn>=1.5,<2` in `requirements.txt`. Commit `49669e6`.

**Root cause 3: All dependencies unpinned:**
The broader lesson was that all 26 packages in `requirements.txt` were unpinned. Any package releasing a breaking update would cause the same class of failure without any change to project code.

**Fix:** Pinned all 26 dependencies to the exact versions confirmed working in the local environment. Commit `d6f7be1`.

---

## Migration: Jupyter Notebook → modular Python
The original system was a single Jupyter notebook (AnaliseV5). It was migrated to a modular Python package to enable automated GitHub Actions execution, proper dependency management, and maintainability.

## Implemented improvements
- ✅ **Contrarian family in Monday research runner** — CB (Contrarian Baseline), EWI (Error-Weighted Inverter) and PEL (Predictive Error Learning) added as the 9th family in the weekly model comparison. If CB ranks first, an amber warning appears in the email flagging potential systematic errors in the main ensemble.
- ✅ **Modular Python package** — `main.py` + `data/` + `features/` + `models/` + `reports/`
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
- ✅ **Schedule moved to after US market close** — cron retargeted to 22:00 UTC (midnight Barcelona CEST); NYSE/NASDAQ close at 20:00 UTC so all tickers have their day-close price available at validation time — fixes missing ✅/❌ icons for US equities and crypto
- ✅ **`predictions_log_public.csv`** — anonymised version of the audit log (no tickers, no prices) published daily to the public repo; contains `asset_type` (portfolio/watchlist), direction, confidence, outcome, and individual model votes — allows anyone to verify real accuracy
- ✅ **Reliability section** — all protection mechanisms grouped in one place: unit tests (pre-run), VIX/SPY forward fill and stock split detection (runtime), git push retry and emergency artifact (post-run)
- ✅ **Semantic git tags** — `v1.0.0` (Week 1: stability), `v1.1.0` (Week 2: observability), `v1.2.0` (Week 3: public repo); each tag anchors a milestone in git history with a descriptive message
- ✅ **Walk-Forward Validation gap** — `TimeSeriesSplit(n_splits=5, gap=1)`; `gap=1` inserts one day between each training fold and the validation fold, preventing same-day lookahead and producing more honest cross-validated accuracy estimates
- ✅ **Market regime feature (`vix_regime`)** — discrete VIX-based label (0 = low, 1 = medium, 2 = high) added to `FEATURE_COLS`; SGD models detect a feature count mismatch on load and reinitialise automatically, so adding new features never silently corrupts existing models
- ✅ **Correlation matrix in email** — portfolio asset return heatmap (120-day daily returns) embedded directly as a base64 PNG in the HTML email; no file attachments; graceful no-op if matplotlib is unavailable
- ✅ **ETF long-term projection scenarios** — 10-year column now shows three fixed growth scenarios (Pess = 3% · Base = 8% · Otim = 15%) instead of a single historical rate; eliminates misleading projections for tickers with limited or unrepresentative price history (e.g., SGLN.L in GBX)
- ✅ **Educational README** — `README_educativo.md` (PT-BR): in-depth documentation of every feature with formulas and original authors, adaptive weight numerical example, bug history, and full pipeline explanation
- ✅ **`predictions_log_public.csv` in public repo** — anonymised audit log published daily alongside the 10-day delayed charts; contains `asset_type` (portfolio/watchlist), direction, confidence, outcome, and individual model votes — verifiable accuracy for anyone
