"""
Rastreabilidade: cada previsão guarda qual código, qual calibração e quais dados
a geraram, para que qualquer número do email possa ser reproduzido e explicado
depois (por exemplo: "este ▲ 50 saiu da calibração por encolhimento λ=0,00 de
18/11, com o código abc1234, sobre a barra de 17/11").

Colunas novas no predictions_log.csv (linhas antigas ficam vazias):
  versao_codigo      commit curto do repositório (GITHUB_SHA no Actions)
  versao_calibracao  regra usada naquele horizonte, ex. "enc λ=0.00 @2026-11-18"
  hash_dados         hash curto da última barra (OHLCV + data) que o modelo viu
  executado_em       horário UTC da execução
"""
import hashlib
import os
import subprocess

import pandas as pd


def versao_codigo() -> str:
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=True)
        sujo = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+local" if sujo else "")
    except Exception:
        return "desconhecida"


def versao_calibracao(curves: dict | None, h: int) -> str:
    curves = curves or {}
    c = curves.get(f"d{h}")
    quando = curves.get("updated", "?")
    if not c:
        return "sem calibração (bruta)"
    if "lambda" in c:
        return f"enc λ={c['lambda']:.2f} @{quando}"
    return f"iso n={c.get('n', '?')} @{quando}"


def hash_dados(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return ""
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    ultima = df[cols].iloc[-1]
    texto = str(pd.Timestamp(df.index[-1]).date()) + "|" + "|".join(
        f"{float(v):.6f}" if pd.notna(v) else "nan" for v in ultima.values)
    return hashlib.sha1(texto.encode()).hexdigest()[:10]


def contexto_execucao(curves: dict | None) -> dict:
    return {
        "versao_codigo": versao_codigo(),
        "executado_em": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "calibracao": {h: versao_calibracao(curves, h) for h in [1, 2, 3]},
    }
