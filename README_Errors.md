# Registo de Erros — Carteira Inteligente

Histórico de problemas encontrados, causas raiz e soluções aplicadas.

---

## ERR-001 — 3 dias sem previsões (Jun 2, 3, 4 de 2026)

**Sintoma**
`predictions_log.csv` não recebeu novas entradas para os dias 2, 3 e 4 de junho de 2026. As execuções automáticas correram (commits existem), o HTML foi gerado, mas mostrava "0 ativos" e nenhuma previsão nova foi guardada.

**Causa raiz**
O commit `050700b` (2 Jun 2026) alterou `requirements.txt` (fixou `xgboost>=2.0,<3.0`). Esta alteração **invalidou o cache pip** do GitHub Actions, forçando uma instalação limpa de todos os pacotes. O `yfinance`, sem versão fixada (`yfinance` a seco), instalou uma versão nova que falhou silenciosamente ao descarregar todos os tickers — `raw_data = {}` → `featured_data = {}` → `resultados_ml = {}` → 0 previsões. O pipeline completou sem erro, o código não tinha nenhum guard para este caso.

**Solução aplicada**
- Commit `7cc1ada`: pin `yfinance>=1.3.0` em `requirements.txt` (versão confirmada a funcionar)
- O passo de validação no workflow (adicionado em `18d3902`) já captura este cenário: falha se `COUNT < 10` previsões → envia email de alerta

**Dados perdidos**
Jun 2, 3, 4 de 2026 — previsões irrecuperáveis (mercado já fechado, preços históricos não equivalem a previsões em tempo real).

---

## ERR-002 — Email de sexta-feira (Jun 5) não recebido

**Sintoma**
A execução automática de 5 de junho de 2026 correu, mas o email com o relatório não chegou.

**Causa raiz**
Race condition entre uma execução manual (push local) e o workflow agendado:
1. Pipeline local gerou 285 previsões intraday (mercado ainda aberto às ~18:32 UTC)
2. Push manual enviou `predictions_log.csv` para o GitHub
3. Workflow agendado arrancou em simultâneo, processou, e tentou push → conflito em `predictions_log.csv`
4. Rebase automático falhou nas 3 tentativas → workflow terminou em erro → email de sucesso nunca enviado

Adicionalmente, as previsões intraday eram inválidas (preços com mercado aberto).

**Solução aplicada**
- Pre-commit hook (`.claude/hooks/pre_commit_check.sh`): bloqueia commit de `predictions_log.csv` com dados de hoje antes das 21h00 UTC (NYSE ainda aberto)
- Commit `18d3902`: adicionou `permissions: contents: write` ao workflow (ausência causava falhas de push em alguns contextos)
- Commit `3f933f9`: removeu as previsões intraday do dia 5 de junho

---

## ERR-003 — Linha falsa EMIM.AS no predictions_log.csv

**Sintoma**
Linha com `ticker=EMIM.AS`, `pred_date=2026-06-05`, `target_date=2026-05-15` no ficheiro `predictions_log.csv` — target_date no passado, o que é impossível numa previsão real.

**Causa raiz**
Linha criada durante teste do pre-commit hook (CHECK 1 — validação de mercado aberto). O teste injetou uma linha falsa para simular o cenário de "dados de hoje já existem". A linha não foi completamente removida após o teste.

**Solução aplicada**
Linha removida. Já estava limpa no commit `3f933f9` (que removeu todas as entradas de 2026-06-05).

---

## ERR-004 — 3 testes falharam em test_research_runner.py

**Sintoma**
`python -m pytest tests/test_research_runner.py` falhava com `TypeError: _build_consensus() missing 1 required positional argument: 'research_weights'`.

**Causa raiz**
O commit `050700b` adicionou um novo parâmetro obrigatório `research_weights` à função `_build_consensus` em `research/runner.py`, mas os testes não foram atualizados para passar esse argumento.

**Testes afetados**
- `test_build_consensus_all_up`
- `test_build_consensus_all_down`
- `test_build_consensus_multiple_tickers`

**Solução aplicada**
Commit `18d3902`: os 3 testes foram corrigidos passando `{}` como terceiro argumento: `_build_consensus(rows, ["NVDA"], {})`.

---

## ERR-006 — research/runner.py com linhas de sintaxe inválida

**Sintoma**
`research/runner.py` tinha as linhas `SYNTAX ERROR HERE` e `SYNTAX ERROR` no final do ficheiro.

**Causa raiz**
Linhas de teste acidentais inseridas durante testes do pre-commit hook (CHECK 2 — verificação de testes ao commitar).

**Solução aplicada**
Commit `7cc1ada`: linhas removidas.

---

## ERR-007 — Email não enviado em 2026-06-09 (Segunda-feira)

**Sintoma**
O pipeline correu normalmente em 2026-06-09 e gerou 285 previsões, mas o email diário nunca chegou.

**Causa raiz**
Race condition entre run manual matinal e o cron agendado:
1. Run manual (workflow_dispatch) às **06:47 UTC** gerou previsões e commitou o HTML — sem email (condição `github.event_name == 'schedule'` não satisfeita)
2. Cron das **17:30 UTC** arrancou o job `verificar`, encontrou `predictions_log.csv` com dados de hoje → `ja_executou=true`
3. Job `executar-notebook` foi saltado inteiramente (`if: ja_executou == 'false'`) — incluindo o passo de email
4. Resultado: pipeline correu de manhã sem email; cron da tarde não correu nada

**Solução aplicada**
Commit `ceeeeb2`: job `verificar` agora distingue entre trigger de cron e trigger manual:
- **Schedule (cron)**: ignora `ja_executou` → sempre executa → email sempre enviado
- **workflow_dispatch**: comportamento anterior (salta se já correu hoje)

O pipeline é idempotente — correr duas vezes no mesmo dia não duplica previsões.

---

## Validações em produção (estado atual)

| Validação | Onde | O que deteta |
|-----------|------|--------------|
| `verificar` job | Workflow | Runs manuais: salta se já executou. Crons: sempre executa |
| `✅ Validar dados gerados` | Workflow | Falha se `COUNT < 10` previsões, HTML ou weights em falta |
| Pre-commit hook CHECK 1 | Local | Bloqueia commit de predictions_log.csv com dados intraday |
| Pre-commit hook CHECK 2 | Local | Bloqueia commit se testes relacionados falharem |
| `permissions: contents: write` | Workflows | Garante que o push nunca falha por falta de permissão |

---

*Última actualização: 2026-06-10*
