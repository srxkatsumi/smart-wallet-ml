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

## ERR-008 — `KeyError: 'etoro'` após remover as posições do eToro da carteira (2026-07-30)

**Sintoma**
Três execuções agendadas consecutivas falharam imediatamente com `KeyError: 'etoro'`.

**Causa raiz**
A chave `etoro` de `config/portfolio.json` foi removida (posições vendidas, deixaram de ser acompanhadas), mas `data/storage.py:load_my_tickers()` e `main.py` continuavam a acessar `portfolio_cfg["etoro"]` diretamente em vez de `.get("etoro", [])`.

**Solução aplicada**
Ambos os pontos de acesso agora usam uma lista vazia por default quando a chave está ausente; o pipeline de research (antes restrito só aos tickers do eToro) agora recai na carteira de ETFs quando `etoro` está vazio. Commit `e32e748c`.

---

## ERR-009 — `mapie` incompatível com o scikit-learn fixado — desafiante de regressão falhou pra 100% dos tickers (2026-08-05)

**Sintoma**
O novo email do experimento de regressão saiu com "0 ativos" — o modelo desafiante falhou o treino pra todos os tickers.

**Causa raiz**
`mapie<1` (resolveu pra 0.9.x) depende de internals do scikit-learn (`EnsembleRegressor.__sklearn_tags__`) que não existem no `scikit-learn==1.8.0` fixado — uma incompatibilidade de versão invisível localmente porque um venv de teste descartável tinha resolvido um scikit-learn mais antigo e compatível, sem esse pin.

**Solução aplicada**
Removida a dependência do `mapie`; os intervalos de predição por conformal prediction agora são calculados manualmente (treina no primeiro ~80% da janela, calibra o intervalo no ~20% mais recente — mesma filosofia manual/independente de versão já usada por `models/conformal.py` pros classificadores de produção, exatamente por este motivo). Commit `a4a602db5`.

---

## ERR-010 — Tentativas de fallback entraram em corrida; o ciclo de um dia de negociação nunca correu (2026-08-10 a 2026-08-11)

**Sintoma**
Nenhum email chegou pro fecho de 2026-08-11, mesmo o workflow mostrando runs de "sucesso" naquela noite.

**Causa raiz 1 — corrida entre tentativas de fallback**
Com o pipeline de research + experimento de regressão, um run completo já leva ~1h20-2h — mais que o intervalo entre as três tentativas de fallback do cron. Em 2026-08-10, uma segunda tentativa começou seu próprio run completo de ~2h27m porque a checagem "já rodou hoje" só olhava o CSV *commitado*, que a primeira tentativa (ainda a correr) ainda não tinha publicado. O push da tentativa perdedora foi rejeitado, e a recuperação (`git pull --rebase`) falhou também, porque `output/models/*.pkl` nunca é commitado, deixando a árvore de trabalho permanentemente "suja" pro rebase.

**Causa raiz 2 — `hoje` calculado a meio da execução**
`main.py` calculava `hoje = pd.Timestamp.now().normalize()` depois de baixar os dados de mercado de toda a watchlist, não no início de `main()`. Um run que começou às 23h38 UTC e levou até 01h31 UTC do dia seguinte gravou as suas previsões com `pred_date` = a data *posterior* — correto do ponto de vista dos dados de mercado (a NYSE já tinha fechado), mas isso significava que a checagem anti-duplicação do dia seguinte real via "já feito" e saltava as três tentativas daquele dia — o fecho real daquele dia de negociação nunca foi processado.

**Solução aplicada**
- A checagem anti-duplicação agora também consulta a API do GitHub Actions por qualquer run já em curso ou já bem-sucedido hoje (apanha a corrida em segundos, não ~2h depois no push); a retentativa de push cede a um run do mesmo dia que já publicou em vez de tentar mesclar dois pipelines completos independentes. Commit `3ef79b576`.
- Movido `hoje = pd.Timestamp.now().normalize()` pra primeira linha de `main()`. Commit `3ece5070d`.
- Correção defensiva secundária: o step "✅ Validar dados gerados" recalculava `today` via `date` do shell *depois* do `main.py` terminar, podendo cair do lado errado da mesma virada de meia-noite — corrigido pra aceitar o `pred_date` de hoje ou de ontem. Commit `3ea9e8f47`.

---

## ERR-011 — Lacunas silenciosas encontradas numa revisão de pontos fracos do projeto (2026-08-17)

Não foi um incidente único — uma auditoria deliberada encontrou três coisas erradas em silêncio, sem nunca lançar erro:

1. **Suite de testes desligada do CI.** Existiam 21 arquivos de teste, `pytest` estava no `requirements.txt`, mas o workflow nunca o invocava. `tests/test_email_phase15.py` estava quebrado (`ImportError`) há semanas — testava uma seção de PnL/lotes do email removida num refactor anterior — e ninguém percebeu. Solução: removido o teste obsoleto, adicionado `pytest.ini` (`--continue-on-collection-errors`, pra um arquivo quebrado nunca mais silenciar a suite inteira), e ligado `pytest tests/` como primeiro passo do job. Commit `5d4078378`.
2. **"Aprendizado incremental" do SGD estava inerte desde 2026-06-26.** `output/models/*.pkl` está no `.gitignore` (uma correção deliberada depois de modelos desatualizados quebrarem o `partial_fit` em silêncio e produzirem zero previsões naquele dia — ver a guarda adicionada na altura, o cheque de `n_features_in_` em `_load_sgd`), mas nenhum step de cache restaurava a pasta — então todo run caía silenciosamente num `.fit()` do zero, todos os dias, nunca `partial_fit`. Solução: adicionado um step `actions/cache@v4` pra `output/models/` (mesma chave do cache de research já existente); a guarda de contagem de features já existente protege contra repetir o incidente de junho. Commit `5d4078378`.
3. **`PPFB.DE` ausente de `ASSET_CLASSES`/`TICKER_CALENDAR`.** Presente na carteira, ausente dos dois dicionários mantidos à mão — caía nos defaults silenciosos (`asset_class=0`/NYSE) em vez dos valores reais (iShares Physical Gold ETC, Xetra, ETC de commodity). Solução: adicionadas as entradas corretas mais `tests/test_config_coverage.py`, que agora falha alto no CI se um futuro ticker da carteira ficar de fora de algum dos dois dicionários. Commit `5d4078378`.

Também corrigido no mesmo dia: o teste de Diebold-Mariano do experimento de regressão usava o default `h=1` pros três horizontes, subestimando a variância de longo prazo pro D+2/D+3 (janelas de previsão sobrepostas são serialmente correlacionadas — o paper original de 1995 recomenda `h` = horizonte de previsão). Agora chama `diebold_mariano(e_challenger, e_champion, h=day)`.

---

## Validações em produção (estado atual)

| Validação | Onde | O que deteta |
|-----------|------|--------------|
| `verificar` job | Workflow | Runs manuais: salta se já executou. Crons: também consulta a API do Actions por um run em curso/bem-sucedido antes de arrancar |
| `✅ Validar dados gerados` | Workflow | Falha se `COUNT < 10` previsões (data de hoje ou de ontem), HTML ou weights em falta |
| `pytest tests/` | Workflow | Corre antes do `main.py`; pára o pipeline em qualquer falha de teste |
| `actions/cache` (modelos research + SGD) | Workflow | Persiste modelos treinados entre runs; guarda de contagem de features força reinit seguro se houver mismatch |
| `tests/test_config_coverage.py` | Teste CI | Falha se um ticker da carteira ficar ausente de `ASSET_CLASSES`/`TICKER_CALENDAR` |
| Pre-commit hook CHECK 1 | Local | Bloqueia commit de predictions_log.csv com dados intraday |
| Pre-commit hook CHECK 2 | Local | Bloqueia commit se testes relacionados falharem |
| `permissions: contents: write` | Workflows | Garante que o push nunca falha por falta de permissão |

---

*Última actualização: 2026-08-17*
