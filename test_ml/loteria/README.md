# Mega Sena ML Experiment

> **Hypothesis:** Can machine learning predict lottery numbers better than random chance?
> **Expected answer:** No. And we have the data to prove it.

[![Automated](https://img.shields.io/badge/Updated-Daily%20Mon–Fri-4c8f6b?logo=github-actions)](../../.github/workflows/executar_diario.yml)

| | |
|---|---|
| **Started** | 28 May 2026 |
| **Horizon** | 5 years — material for PhD research |
| **Author** | Vicky Costa |

---

## What is this?

This sub-project applies the same adaptive ensemble ML pipeline used for stock forecasting to an unambiguously random process: the Brazilian Mega Sena lottery. The pipeline runs every weekday (Mon–Fri) with two distinct roles:

- **Daily (Mon–Fri):** downloads results and processes 300 historical draws at a time — the model learns from the past. No future predictions are made.
- **Monday only:** validates last week's draw outcomes, then uses everything learned to generate 5 sequences for each of the 3 upcoming draws (Mon/Thu/Sat).

This separation is intentional: the model accumulates knowledge from the full historical record before making any forward-looking prediction. Over ≈10 weekdays from 28 May 2026, the backfill builds a longitudinal dataset from concurso 1 (1996) to the present.

The purpose is not to win the lottery. The purpose is to produce a quantitative, reproducible demonstration that machine learning cannot extract predictive signal from a process with no signal to extract.

This is a controlled negative experiment. In science, a well-designed experiment that disproves a hypothesis is as valuable as one that confirms it.

---

## The lottery

**Mega Sena** (Caixa Econômica Federal, Brazil):
- Pool: 60 balls numbered 1–60
- Drawn per game: 6 balls
- Draws per week: 3 — **Monday**, **Thursday**, **Saturday**
- Data source: [loterias.caixa.gov.br](https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx)

---

## Random baseline — the theoretical ceiling

Before running a single model, we know what to expect from a completely random strategy:

| Metric | Formula | Value |
|--------|---------|-------|
| P(any number drawn) | 6 / 60 | 10.0% |
| Expected matches per sequence | 6 × (6/60) | **0.60** |
| P(0 matches) | C(54,6) / C(60,6) | ≈ 47.4% |
| P(1 match) | C(6,1)·C(54,5) / C(60,6) | ≈ 37.7% |
| P(2 matches) | C(6,2)·C(54,4) / C(60,6) | ≈ 13.2% |
| P(3 matches — Terno) | C(6,3)·C(54,3) / C(60,6) | ≈ 2.44% |
| P(4 matches — Quadra) | C(6,4)·C(54,2) / C(60,6) | ≈ 0.18% |
| P(5 matches — Quina) | C(6,5)·C(54,1) / C(60,6) | ≈ 0.0064% |
| P(6 matches — Sena) | C(6,6) / C(60,6) | ≈ 0.000179% |

**If our ML model averages significantly more than 0.60 matches per sequence over many draws, that would be extraordinary and demand investigation.** The null hypothesis is that it won't.

---

## ML architecture

The same three-model ensemble used for stock forecasting is applied here. Each model learns from the entire draw history, treating the problem as: *"for number X, given the history of past draws, what is the probability that X is drawn today?"*

### Models

| Model | Configuration | Role in ensemble |
|-------|--------------|-----------------|
| **Random Forest** | 100 trees, max depth 4, class_weight="balanced" | Captures non-linear frequency patterns. Bootstrapped trees prevent extreme overfitting on noisy frequency data. |
| **Gradient Boosting** | 100 estimators, lr 0.05, max depth 3 | Iteratively corrects residuals — better at identifying subtle interactions between frequency features. Low lr = resistant to overfitting noise. |
| **SGD Classifier** | log_loss, L2, class_weight="balanced" | Linear model. Acts as a regulariser — if it disagrees with both non-linear models, the ensemble moves toward a more conservative probability. |

All three models output a probability for each of the 60 numbers. The ensemble combines them via adaptive weighted voting.

### Adaptive weights

After each validated draw, model weights are updated using exponential decay:

```
weight(model) ∝ hit_rate(model) × Σ decay^(draws_ago)
```

More recent correct predictions count more. If a model starts performing systematically worse, its vote share decreases automatically.

**Note:** Because lottery draws are independent and identically distributed (i.i.d.) by design, we expect the weights to fluctuate randomly around 1/3 each, never settling on a consistent winner. This is itself a diagnostic test: persistent weight divergence would suggest non-randomness in the data.

### Prediction strategy

For each upcoming draw (Mon/Thu/Sat), the system generates **5 sequences**:

| Sequence | Method |
|----------|--------|
| 1 | Top-6 numbers by ensemble probability (deterministic) |
| 2–5 | Weighted random sampling using ensemble probabilities as weights |

This gives a deterministic "best guess" plus 4 diverse alternatives, covering a wider portion of the probability space than repeating the same sequence.

### Features (per number, per draw)

| Feature | Description | Why it's "supposed" to matter |
|---------|-------------|-------------------------------|
| `freq_5d` | Times drawn in last 5 games | "Hot" numbers — often cited by lottery enthusiasts |
| `freq_10d` | Times drawn in last 10 games | Medium-term frequency |
| `freq_20d` | Times drawn in last 20 games | Baseline frequency |
| `freq_50d` | Times drawn in last 50 games | Long-term frequency |
| `draws_since_last` | Draws elapsed since last appearance | "Cold" numbers — "due" theory |
| `freq_trend` | freq_5d / freq_20d | Acceleration of appearances |
| `deviation` | freq_20d − 0.10 | Departure from expected 10% rate |
| `decade` | Numeric group (1–10, 11–20, etc.) | Decade distribution patterns |
| `is_even` | Binary | Even/odd distribution tendencies |
| `is_prime` | Binary | Prime number theory (purely anecdotal) |
| `prev_sum` | Sum of previous draw's numbers | "Balance" theory |
| `prev_mean` | Mean of previous draw | Distribution center |
| `prev_spread` | max − min of previous draw | Spread theory |
| `day_mon/thu/sat` | Draw day one-hot | Whether certain numbers are more common on certain days |

**None of these features should predict a truly random draw.** The model will find apparent patterns — it always does, even in pure noise. The question is whether those patterns generalize to unseen draws.

---

## Live results

### Last 7 draws — Concurso {{LAST_CONCURSO}} ({{LAST_DATE}})

<!-- LAST_WEEK_START -->
_Loading..._
<!-- LAST_WEEK_END -->

### Next predictions

<!-- NEXT_PREDS_START -->
_Loading..._
<!-- NEXT_PREDS_END -->

### Accumulated statistics

<!-- STATS_START -->
_Loading..._
<!-- STATS_END -->

---

## Why this matters (the PhD angle)

This experiment began on **28 May 2026** as a long-term personal research project. The goal over the next **5 years** is to accumulate enough rigorous, timestamped data to use as empirical foundation for doctoral research in Machine Learning and predictive modelling.

The central research question:

> *Can supervised machine learning identify exploitable structure in a process that is provably random by design?*

The Mega Sena draw mechanism uses certified physical randomness (numbered balls, audited by government regulators). There is no hidden variable, no market microstructure, no human psychology. If ML cannot beat random chance here, it provides a clean baseline for evaluating ML performance in domains that are claimed to be random but may not be (financial markets, weather patterns, biological sequences).

The experimental design is rigorous:
- Pre-registered hypothesis (impossible to predict > baseline)
- Objective outcome metric (number of matches vs theoretical expectation)
- Reproducible: all data, code, and predictions are public and timestamped
- Longitudinal: results accumulate weekly without manual intervention — no cherry-picking

**5-year roadmap toward PhD:**

| Year | Milestone |
|------|-----------|
| 2026 | Baseline established — first 100 draws, walk-forward backfill |
| 2027 | Statistical significance test — t-test against μ = 0.60 with N ≥ 150 |
| 2028 | Feature ablation study — which features are genuine noise vs artefact |
| 2029 | Comparative study — Mega Sena vs stock market: can the same model architecture distinguish random from non-random domains? |
| 2030 | PhD dissertation material — full longitudinal dataset, reproducible pipeline, peer-reviewed methodology |

After 1–2 years of data, this becomes a statistically meaningful test. With N = 150 draws and 5 sequences each (750 sequence-evaluations), a one-sample t-test against the null hypothesis of μ = 0.60 matches would have power > 80% to detect an effect size of Δ = 0.15 matches — a 25% improvement over random, which would already be extraordinary.

---

## Repository structure

```
test_ml/loteria/
├── main.py               ← weekly runner (download → validate → train → predict → update README)
├── config.py             ← constants, paths, hyperparameters
├── data/
│   ├── downloader.py     ← fetches results from loterias.caixa.gov.br
│   └── storage.py        ← predictions log + weights I/O
├── features/
│   └── engineering.py    ← per-number feature matrix from draw history
├── models/
│   └── ensemble.py       ← RF + GB + SGD + adaptive weight update
├── reports/
│   └── summary.py        ← generates Markdown sections for this README
└── output/               ← generated files (gitignored except CSVs)
    ├── mega_sena_results.csv    ← cached official results
    ├── predictions_log.csv      ← all predictions + validation outcomes
    └── ensemble_weights.json    ← current model weights
```

---

## How to run locally

```bash
cd test_ml/loteria
pip install pandas numpy scikit-learn requests beautifulsoup4 lxml openpyxl
python main.py          # normal run (processes next 300 historical draws + predicts upcoming)
python main.py --force  # force re-download of results
```

If the automatic download fails, download the results manually:
1. Go to [loterias.caixa.gov.br/Paginas/Mega-Sena.aspx](https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx)
2. Click **"Resultados da Mega-Sena por ordem crescente"**
3. Save the file as `output/mega_sena_manual.html`
4. Run `python main.py`

---

*This project does not encourage gambling. It is a data science experiment with a predetermined conclusion. Use the lottery numbers at your own risk — which is to say, do not.*

Built by **Vicky Costa** — Data Analyst · Data Science student · future PhD candidate
Started **28 May 2026** · 5-year longitudinal experiment
