# Error Log — Carteira Inteligente

History of issues found, root causes, and applied solutions.

---

## ERR-001 — 3 days without predictions (Jun 2, 3, 4 2026)

**Symptom**
`predictions_log.csv` received no new entries for June 2, 3, and 4 2026. Automatic runs executed (commits exist), HTML was generated, but it showed "0 ativos" and no new predictions were saved.

**Root cause**
Commit `050700b` (Jun 2 2026) modified `requirements.txt` (pinned `xgboost>=2.0,<3.0`). This change **invalidated the GitHub Actions pip cache**, forcing a clean install of all packages. `yfinance`, without a version pin (bare `yfinance`), installed a new version that silently failed to download all tickers — `raw_data = {}` → `featured_data = {}` → `resultados_ml = {}` → 0 predictions. The pipeline completed without error; the code had no guard for this case.

**Fix applied**
- Commit `7cc1ada`: pinned `yfinance>=1.3.0` in `requirements.txt` (confirmed working version)
- Validation step in workflow (added in `18d3902`) already catches this: fails if `COUNT < 10` predictions → sends alert email

**Lost data**
Jun 2, 3, 4 2026 — predictions are unrecoverable (market already closed, historical prices do not equal real-time predictions).

---

## ERR-002 — Friday (Jun 5) email not received

**Symptom**
The automatic run on June 5 2026 executed, but the report email was never received.

**Root cause**
Race condition between a manual push (local run) and the scheduled workflow:
1. Local pipeline generated 285 intraday predictions (market still open at ~18:32 UTC)
2. Manual push sent `predictions_log.csv` to GitHub
3. Scheduled workflow started concurrently, processed, and attempted push → conflict on `predictions_log.csv`
4. Automatic rebase failed on all 3 retries → workflow ended in error → success email never sent

Additionally, the intraday predictions were invalid (prices captured while market was open).

**Fix applied**
- Pre-commit hook (`.claude/hooks/pre_commit_check.sh`): blocks committing `predictions_log.csv` with today's data before 21:00 UTC (NYSE still open)
- Commit `18d3902`: added `permissions: contents: write` to workflow (missing permission caused push failures in some contexts)
- Commit `3f933f9`: removed the intraday predictions from June 5

---

## ERR-003 — Fake EMIM.AS row in predictions_log.csv

**Symptom**
Row with `ticker=EMIM.AS`, `pred_date=2026-06-05`, `target_date=2026-05-15` in `predictions_log.csv` — target_date in the past, which is impossible for a real prediction.

**Root cause**
Row created while testing the pre-commit hook (CHECK 1 — open market validation). The test injected a fake row to simulate the "today's data already exists" scenario. The row was not fully cleaned up after testing.

**Fix applied**
Row removed. It was already clean in commit `3f933f9` (which removed all 2026-06-05 entries).

---

## ERR-004 — 3 failing tests in test_research_runner.py

**Symptom**
`python -m pytest tests/test_research_runner.py` failed with `TypeError: _build_consensus() missing 1 required positional argument: 'research_weights'`.

**Root cause**
Commit `050700b` added a new required parameter `research_weights` to `_build_consensus` in `research/runner.py`, but the tests were not updated to pass this argument.

**Affected tests**
- `test_build_consensus_all_up`
- `test_build_consensus_all_down`
- `test_build_consensus_multiple_tickers`

**Fix applied**
Commit `18d3902`: all 3 tests fixed by passing `{}` as the third argument: `_build_consensus(rows, ["NVDA"], {})`.

---

## ERR-006 — research/runner.py with invalid syntax lines

**Symptom**
`research/runner.py` had lines `SYNTAX ERROR HERE` and `SYNTAX ERROR` at the end of the file.

**Root cause**
Accidental test lines inserted while testing the pre-commit hook (CHECK 2 — test verification on commit).

**Fix applied**
Commit `7cc1ada`: lines removed.

---

## ERR-007 — Email not sent on 2026-06-09 (Monday)

**Symptom**
The pipeline ran normally on 2026-06-09 and generated 285 predictions, but the daily email never arrived.

**Root cause**
Race condition between an early manual run and the scheduled cron:
1. Manual run (`workflow_dispatch`) at **06:47 UTC** generated predictions and committed the HTML — no email (the `github.event_name == 'schedule'` condition wasn't met)
2. The **17:30 UTC** cron started the `verificar` job, found `predictions_log.csv` with today's data → `ja_executou=true`
3. The `executar-notebook` job was skipped entirely (`if: ja_executou == 'false'`) — including the email step
4. Result: the pipeline ran in the morning with no email; the afternoon cron ran nothing

**Fix applied**
Commit `ceeeeb2`: the `verificar` job now distinguishes between a cron trigger and a manual trigger:
- **Schedule (cron)**: ignores `ja_executou` → always executes → email always sent
- **workflow_dispatch**: previous behaviour (skip if already ran today)

The pipeline is idempotent — running twice on the same day never duplicates predictions.

---

## ERR-008 — `KeyError: 'etoro'` after removing eToro holdings from the portfolio (2026-07-30)

**Symptom**
Three consecutive scheduled runs failed immediately with `KeyError: 'etoro'`.

**Root cause**
The `etoro` key in `config/portfolio.json` was removed (holdings sold, no longer tracked), but `data/storage.py:load_my_tickers()` and `main.py` still accessed `portfolio_cfg["etoro"]` directly instead of `.get("etoro", [])`.

**Fix applied**
Both call sites now default to an empty list when the key is absent; the research pipeline (previously scoped only to eToro tickers) now falls back to the ETF portfolio when `etoro` is empty. Commit `e32e748c`.

---

## ERR-009 — `mapie` incompatible with pinned scikit-learn — regression challenger failed for 100% of tickers (2026-08-05)

**Symptom**
The new regression-experiment email shipped with "0 ativos" — the challenger model failed to train for every single ticker.

**Root cause**
`mapie<1` (resolved to 0.9.x) depends on scikit-learn internals (`EnsembleRegressor.__sklearn_tags__`) that don't exist in the pinned `scikit-learn==1.8.0` — a version incompatibility invisible locally because a throwaway test venv had resolved an older, compatible scikit-learn without that pin.

**Fix applied**
Dropped the `mapie` dependency; split-conformal prediction intervals are now computed by hand (fit on the first ~80% of the training window, calibrate the interval on the most recent ~20% — same manual/version-independent philosophy already used by `models/conformal.py` for the production classifiers, for exactly this reason). Commit `a4a602db5`.

---

## ERR-010 — Fallback retries raced each other; one trading day's cycle silently never ran (2026-08-10 to 2026-08-11)

**Symptom**
No email arrived for the 2026-08-11 close, even though the workflow showed "success" runs that evening.

**Root cause 1 — race between fallback attempts**
With the research pipeline + regression experiment, a full run now takes ~1h20-2h — longer than the gap between the three fallback cron triggers. On 2026-08-10, a second attempt started its own full ~2h27m run because the "already ran today" check only grepped the *committed* CSV, which the first (still-running) attempt hadn't pushed yet. The loser's push was rejected, and the recovery path (`git pull --rebase`) itself failed because `output/models/*.pkl` is never committed, leaving the working tree permanently "dirty" for rebase purposes.

**Root cause 2 — `hoje` computed mid-run**
`main.py` computed `hoje = pd.Timestamp.now().normalize()` after downloading market data for the whole watchlist, not at the top of `main()`. A run starting at 23:38 UTC that took until 01:31 UTC the next day logged its predictions with `pred_date` = the *later* date — technically correct market-data-wise (NYSE had already closed), but it meant the anti-duplication check for the real next day saw "already done" and skipped all three of that day's attempts, so that trading day's actual close was never processed.

**Fix applied**
- The anti-duplication check now also queries the GitHub Actions API for any run already in progress or succeeded today (catches the race in seconds, not ~2h later at push time); the push retry yields to a same-day run that already published instead of trying to merge two independent full pipeline runs. Commit `3ef79b576`.
- Moved `hoje = pd.Timestamp.now().normalize()` to the first line of `main()`. Commit `3ece5070d`.
- Secondary defensive fix: the "✅ Validar dados gerados" step recomputing `today` via shell `date` *after* `main.py` finished could itself land on the wrong side of the same midnight boundary — patched to accept either today's or yesterday's `pred_date`. Commit `3ea9e8f47`.

---

## ERR-011 — Silent gaps found during a project-wide weak-point review (2026-08-17)

Not a single incident — a deliberate audit turned up three things that were quietly wrong without ever throwing an error:

1. **Test suite disconnected from CI.** 21 test files existed, `pytest` was in `requirements.txt`, but the workflow never invoked it. `tests/test_email_phase15.py` had been broken (`ImportError`) for weeks — it tested a PnL/lots email section removed in an earlier refactor — and nothing noticed. Fix: deleted the stale test, added `pytest.ini` (`--continue-on-collection-errors`, so one broken file can never again silently void the whole suite), wired `pytest tests/` in as the first step of the job. Commit `5d4078378`.
2. **SGD "incremental learning" was inert since 2026-06-26.** `output/models/*.pkl` is git-ignored (a deliberate fix after stale post-migration models silently broke `partial_fit` and produced zero predictions that day — see the guard added at the time, `_load_sgd`'s `n_features_in_` check), but no cache step ever restored the folder either — so every run silently fell back to a full `.fit()` from scratch, every day, never `partial_fit`. Fix: added an `actions/cache@v4` step for `output/models/` (keyed like the existing research-model cache); the pre-existing feature-count guard means a future feature change forces a safe re-init instead of repeating the June incident. Commit `5d4078378`.
3. **`PPFB.DE` missing from `ASSET_CLASSES`/`TICKER_CALENDAR`.** Present in the portfolio, absent from both hand-maintained dicts — fell back to `asset_class=0`/`NYSE` silently instead of its real values (iShares Physical Gold ETC, Xetra, commodity ETF). Fix: added the correct entries plus `tests/test_config_coverage.py`, which now fails loudly in CI if a future portfolio ticker is missing from either dict. Commit `5d4078378`.

Also fixed the same day: the regression-experiment's Diebold-Mariano test used the default `h=1` for every horizon, understating the long-run variance for D+2/D+3 (overlapping forecast windows are serially correlated — the original 1995 paper recommends `h = forecast horizon`). Now calls `diebold_mariano(e_challenger, e_champion, h=day)`.

---

## Production validations (current state)

| Validation | Where | What it catches |
|------------|-------|----------------|
| `verificar` job | Workflow | Runs manually: skips if already run today. Crons: also checks the Actions API for an in-progress/succeeded run before starting |
| `✅ Validar dados gerados` | Workflow | Fails if `COUNT < 10` predictions (today's or yesterday's date), HTML or weights missing |
| `pytest tests/` | Workflow | Runs before `main.py`; halts the pipeline on any test failure |
| `actions/cache` (research + SGD models) | Workflow | Persists trained models between runs; feature-count guard forces safe re-init on mismatch |
| `tests/test_config_coverage.py` | CI test | Fails if a portfolio ticker is missing from `ASSET_CLASSES`/`TICKER_CALENDAR` |
| Pre-commit hook CHECK 1 | Local | Blocks committing predictions_log.csv with intraday data |
| Pre-commit hook CHECK 2 | Local | Blocks commit if related tests are failing |
| `permissions: contents: write` | Workflows | Ensures push never fails due to missing permission |

---

*Last updated: 2026-08-17*
