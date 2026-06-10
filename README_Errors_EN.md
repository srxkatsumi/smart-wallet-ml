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

## ERR-005 — Climate workflow silently failing on git push

**Symptom**
The `daily_update.yml` workflow in the `climate-adaptive-ml` project ran the pipeline but failed to push updated predictions.

**Root cause**
Missing `permissions: contents: write` in the `run-pipeline` job. Without this permission, the `GITHUB_TOKEN` is not authorized to write to the repository, and `git push` fails.

**Fix applied**
Added to `.github/workflows/daily_update.yml`:
```yaml
jobs:
  run-pipeline:
    permissions:
      contents: write
```
Committed manually by the user (policy: climate commits are always manual).

---

## ERR-006 — research/runner.py with invalid syntax lines

**Symptom**
`research/runner.py` had lines `SYNTAX ERROR HERE` and `SYNTAX ERROR` at the end of the file.

**Root cause**
Accidental test lines inserted while testing the pre-commit hook (CHECK 2 — test verification on commit).

**Fix applied**
Commit `7cc1ada`: lines removed.

---

## Production validations (current state)

| Validation | Where | What it catches |
|------------|-------|----------------|
| `verificar` job | Workflow | Prevents pipeline from running twice on the same day |
| `✅ Validar dados gerados` | Workflow | Fails if `COUNT < 10` predictions, HTML or weights missing |
| Pre-commit hook CHECK 1 | Local | Blocks committing predictions_log.csv with intraday data |
| Pre-commit hook CHECK 2 | Local | Blocks commit if related tests are failing |
| `permissions: contents: write` | Workflows | Ensures push never fails due to missing permission |

---

*Last updated: 2026-06-07*
