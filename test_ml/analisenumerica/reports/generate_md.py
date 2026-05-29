"""Generate output/previsoes.md — the primary output of the experiment."""
import pandas as pd
from pathlib import Path
from config import OUTPUT_MD, BALLS_PER_DRAW, RANDOM_EXPECTED_MATCHES, PRIZE_TIERS
from data.storage import load_baselines


def _dezenas(row) -> str:
    return " ".join(f"{int(row[f'n{i}']):02d}" for i in range(1, 7))


def _acertos_str(row) -> str:
    """Return the raw match count, or ⏳ if pending."""
    if row.get("validated") == True:
        return str(int(row["matches"])) if pd.notna(row.get("matches")) else "—"
    return "⏳"


def _acuracia_str(row) -> str:
    if row.get("validated") == True and pd.notna(row.get("matches")):
        return f"{int(row['matches']) / BALLS_PER_DRAW * 100:.1f}%"
    return "⏳"


def _premio_str(row) -> str:
    if row.get("validated") != True:
        return "⏳"
    m = int(row["matches"]) if pd.notna(row.get("matches")) else 0
    return PRIZE_TIERS.get(m, "—")


def _day_pt(day: str) -> str:
    return {"Monday": "Seg", "Thursday": "Qui", "Saturday": "Sab"}.get(day, day[:3])


def _stats_block(validated: pd.DataFrame) -> str:
    if validated.empty:
        return "_Nenhuma validação ainda._\n"

    n_draws   = validated["target_concurso"].nunique()
    n_seqs    = len(validated)
    avg       = validated["matches"].mean()
    best      = int(validated["matches"].max())
    vs_rand   = avg - RANDOM_EXPECTED_MATCHES

    prizes = {k: 0 for k in PRIZE_TIERS}
    for m in validated["matches"].dropna():
        m = int(m)
        if m in prizes:
            prizes[m] += 1

    lines = [
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Sorteios analisados | {n_draws} |",
        f"| Sequências avaliadas | {n_seqs} |",
        f"| Média de acertos/sequência | **{avg:.3f}** |",
        f"| Baseline aleatório (teórico) | {RANDOM_EXPECTED_MATCHES:.3f} |",
        f"| vs Baseline | **{vs_rand:+.3f}** |",
        f"| Melhor resultado | {best} acertos |",
    ]
    for n_match, name in sorted(PRIZE_TIERS.items(), reverse=True):
        count = prizes.get(n_match, 0)
        if count > 0:
            lines.append(f"| {name} ({n_match} acertos) | {count}× |")

    return "\n".join(lines) + "\n"


def _baseline_comparison(validated: pd.DataFrame) -> str:
    baselines_df = load_baselines()
    if validated.empty:
        return "_Sem dados suficientes para comparação._\n"

    ml_avg = validated["matches"].mean()
    lines = [
        "| Estratégia | Média acertos/seq | vs Teórico (0.60) |",
        "|-----------|------------------|-------------------|",
        f"| 🤖 ML Ensemble | **{ml_avg:.3f}** | **{ml_avg - RANDOM_EXPECTED_MATCHES:+.3f}** |",
    ]

    for strategy, label, emoji in [
        ("hot",    "Hot (+ frequentes)",     "🔥"),
        ("cold",   "Cold (- frequentes)",    "❄️"),
        ("random", "Aleatório (Monte Carlo)", "🎲"),
    ]:
        b_val = baselines_df[
            (baselines_df.get("strategy") == strategy) &
            (baselines_df.get("validated") == True)
        ] if not baselines_df.empty else pd.DataFrame()
        if not b_val.empty:
            avg = pd.to_numeric(b_val["matches"], errors="coerce").mean()
            lines.append(f"| {emoji} {label} | {avg:.3f} | {avg - RANDOM_EXPECTED_MATCHES:+.3f} |")

    lines.append(f"| 📊 Baseline teórico | 0.600 | +0.000 |")
    return "\n".join(lines) + "\n"


def _table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"| {str(row['prediction_date'])[:10]} "
            f"| {str(row['target_date'])[:10]} "
            f"| {_day_pt(str(row['draw_day']))} "
            f"| {int(row['target_concurso'])} "
            f"| {int(row['seq_num'])} "
            f"| `{_dezenas(row)}` "
            f"| {_acertos_str(row)} "
            f"| {_acuracia_str(row)} "
            f"| {_premio_str(row)} |"
        )
    return "\n".join(rows)


def save_md(pred_df: pd.DataFrame, results: pd.DataFrame):
    today      = pd.Timestamp.today().strftime("%Y-%m-%d")
    last_c     = int(results["concurso"].iloc[-1])
    last_date  = pd.Timestamp(results["data"].iloc[-1]).strftime("%d/%m/%Y")

    validated = pred_df[pred_df["validated"] == True].copy()
    validated["matches"] = pd.to_numeric(validated["matches"], errors="coerce")

    pending   = pred_df[pred_df["validated"] != True].copy()
    pending   = pending.sort_values(["target_date", "seq_num"])

    header = (
        "| Data Previsão | Data Sorteio | Dia | Concurso | Seq "
        "| Dezenas | Acertos | Acurácia | Prêmio |"
    )
    sep = (
        "|--------------|-------------|-----|----------|-----"
        "|---------|---------|---------|--------|"
    )

    # Pending block
    if pending.empty:
        pending_block = "_Sem previsões pendentes._"
    else:
        pending_block = header + "\n" + sep + "\n" + _table_rows(pending)

    n_validated = len(validated)
    n_concursos = validated["target_concurso"].nunique() if not validated.empty else 0

    md = f"""# Mega Sena — Previsões ML

> Gerado automaticamente · {today} · Último sorteio registado: Concurso {last_c} ({last_date})

---

## Estatísticas acumuladas

{_stats_block(validated)}

---

## Comparação com baselines

{_baseline_comparison(validated)}

---

## ⏳ Próximos sorteios (pendentes)

{pending_block}

---

## 📋 Histórico completo

O histórico completo de previsões ({n_validated} sequências · {n_concursos} sorteios validados) está disponível em formato CSV:

→ **[predictions_log.csv](predictions_log.csv)**

Colunas: `prediction_date` · `target_concurso` · `target_date` · `draw_day` · `seq_num` · `n1`–`n6` · `matches` · `prize` · `acuracia` · `validated` · `actual_n1`–`actual_n6`
"""

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"[generate_md] Guardado: {OUTPUT_MD}  "
          f"({len(validated)} validadas · {len(pending)} pendentes)")
