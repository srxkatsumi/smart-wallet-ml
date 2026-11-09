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

## ERR-012: Pesos adaptativos do ensemble colapsavam para um único dia de mercado (2026-08-20)

**Sintoma**
Uma revisão externa apontou que o mecanismo de pesos adaptativos (`models/validator.py:update_ensemble_weights`) parecia reagir rápido demais ao ruído de curto prazo. A matemática do decay em si estava certa (o dia mais recente pesa só ~10% de uma janela de 30, não ~90% como a revisão assumiu), mas rodar um ablation direto (pesos adaptativos vs. voto majoritário simples de RF/GB/SGD sobre as 61.823 previsões validadas reais) mostrou o mecanismo a performar **pior** que pesos fixos, de forma muito significativa: −1,6 a −2,0pp de acurácia nos três horizontes (McNemar, p<0,0001).

**Causa raiz**
`validadas[validadas["horizon"] == day_n].tail(30)` pega as últimas 30 **linhas**, não os últimos 30 dias. O watchlist tem mais de 500 tickers validados no mesmo `target_date`. Confirmado diretamente nos dados: `tail(30)` para D+1 devolvia 30 linhas, todas do mesmo dia. A janela que o código e o README descreviam como ~30 dias de histórico era, na prática, ~30 tickers de um único dia de mercado, e o decay exponencial ordenava esses tickers (ordem arbitrária, não cronológica) como se fossem passos temporais sequenciais. `MIN_VALIDATIONS_WEIGHT=5` piorava o efeito: os pesos já estavam a oscilar com histórico de um único dia desde os primeiros dias de operação. O mesmo padrão existia em `research/runner.py:_update_research_weights`, com efeito menor (a carteira de pesquisa tem só 12 tickers, então a janela colapsava para ~5 dias em vez de 1).

**Solução aplicada**
Reescrita a janela para os últimos 30 `target_date`/`week_date` **distintos**, não as últimas 30 linhas. Dentro da janela, a acurácia é calculada primeiro por dia, e só depois o decay é aplicado entre dias, para um dia com mais tickers não pesar mais que um dia com menos. Aplicado em `models/validator.py` e `research/runner.py`. Testes de regressão adicionados (`tests/test_validator.py`, `tests/test_research_runner.py`) que reproduzem o cenário de um dia com muitos tickers e travam o comportamento correto. `scripts/ablation_ensemble_weights.py` guarda o ablation usado para medir o problema e confirmar a correção: os pesos corrigidos recuperam as −1,6 a −2,0pp que o bug estava a custar, e ficam estatisticamente indistinguíveis de pesos fixos (0 divergências de McNemar) no histórico atual, ainda curto demais para separar RF/GB/SGD de forma confiável.

---

## ERR-013: Confiança do ensemble sem informação em produção — calibração isotônica pós-hoc (2026-11-05)

**Sintoma**
Não foi um incidente — uma revisão deliberada da qualidade do modelo mediu a acurácia real por faixa de confiança reportada e achou acurácia praticamente plana entre o decil mais baixo (<55%) e o mais alto (>80%) de confiança, nas ~48 mil previsões já validadas de todo o watchlist.

**Causa raiz**
`models/ensemble.py` já calibra RF e GB individualmente com `CalibratedClassifierCV(method="isotonic")`, mas essa calibração é feita sobre as *dobras de treino* de cada modelo isoladamente. O SGD nunca passa por calibração (usa `predict_proba` bruto do `log_loss`). A confiança final reportada é a média ponderada dos três (`confidence = max(prob_ens, 1 - prob_ens)`), e média de probabilidades já calibradas individualmente não preserva calibração do conjunto — e o componente SGD não calibrado degrada ainda mais o resultado combinado. Medido diretamente: Brier score de 0,2855–0,2921 nos três horizontes, **pior** do que simplesmente prever 0,5 constante (Brier=0,25) — a confiança reportada hoje é menos honesta que não dizer nada.

**Solução aplicada**
Novo módulo `models/calibration.py`: ajusta uma curva isotônica por horizonte (confiança bruta → acurácia empírica), agregada por todos os tickers do watchlist (por ticker o histórico é pequeno demais — 66 a 150 validações — pra um fit confiável; agregado há ~48 mil). Pontos agrupados em bins de frequência igual antes do fit (reliability diagram padrão) — ajustar isotonic direto em floats quase todos únicos deixa uma única observação extrema fixar o valor da curva na borda. Guarda mínimo de 200 validações por horizonte (`MIN_N_CALIBRATION`) antes de ativar; sem isso, cai no valor bruto sem alterar. Reajusta a cada 30 dias (`CONFIDENCE_CALIB_DAYS`, mesma cadência do `monthly_recalibration` do SGD) e reaplica a curva nova a todo o histórico já validado, não só às previsões novas do dia.

Gravado em coluna nova `confidence_calibrated`, lado a lado com `confidence` (nunca sobrescrita — o valor bruto é o que alimenta o próximo refit). Resultado no histórico real: Brier cai de 0,2855–0,2921 para ~0,25 nos três horizontes, e a curva calibrada fica dentro de ±3pp de 50% em toda a faixa — confirmação numérica de que a confiança não carrega sinal explorável hoje, em vez de inventar uma calibração que os dados não sustentam. Testes de regressão em `tests/test_calibration.py`, incluindo um caso sintético com confiança genuinamente informativa (recupera curva crescente) e um com confiança pura ruído (mantém-se plana) — trava os dois comportamentos.

---

## ERR-014: Auditoria completa: horizontes desalinhados, famílias de pesquisa sem previsão real, medições enviesadas (2026-11-06)

**Sintoma**
Não foi um incidente. Uma auditoria do código achou o preço do SGLN.L no site em €0,84 (real: ~€70), e 4 das 13 famílias de pesquisa com a mesma previsão de alta todos os dias desde setembro.

**Causa raiz e correção**
- **Horizontes:** o alvo era calculado a partir do relógio da execução, que começa entre 00h48 e 02h59 UTC, e o feed às vezes vem com um dia a menos. O "D+1" cobria 2 a 3 pregões, enquanto o modelo treina para 1. Agora o alvo conta a partir do último fecho de cada ticker, gravado em `ref_date`. Tickers sem barra recente ficam sem previsão e fora do alerta.
- **Famílias de pesquisa:** `predict()` recebia 1 linha, então as famílias sequenciais devolviam a taxa-base. As de séries temporais devolviam o primeiro valor ajustado do treino. A linha usada era a última com rótulo, de 1 a 4 pregões atrás. Os três horizontes eram validados com o mesmo preço, e o retreino exigia uma data de segunda-feira, que quase nunca ocorre. Agora `predict()` recebe as 60 linhas mais recentes, as sequências incluem a barra do dia, cada horizonte usa o próprio fecho e o retreino é por idade do modelo. Famílias que falham ficam sem voto em vez de votar em alta.
- **Modelos:** o EWI invertia sempre e o PEL aplicava os lags na ordem errada. A recompensa do DQN era mal definida, o discriminador do GAN não estimava P(alta) e o Conformal inflava a probabilidade (média de 0,80).
- **Feature `obv_trend`:** a razão explodia quando a média do OBV passava perto de zero (3.625 vezes a escala normal no IS3N.DE), derrubando o VAE com NaN e distorcendo modelos lineares e neurais. Agora é limitada a ±5.
- **Blend 50/50:** nunca chegou ao log, porque atualizava os horizontes, mas não o `preds_dict`. Foi removido: o consenso de pesquisa é só relatório.
- **Medição:** WFV e backtest usavam rótulos futuros em D+2/D+3. O gap de CV era curto para esses horizontes, e a acurácia do SGD era medida nos dados de treino. Os p-valores tratavam ~530 tickers do mesmo dia como independentes: agora o dia é a unidade. O Brier da calibração agora é medido fora da amostra, e o SHAP usa dados escalados.
- **Moeda:** `data/downloader.py` já convertia o SGLN.L para EUR, e os exports do site convertiam de novo. O câmbio de hoje também era aplicado à série inteira, misturando FX no `correct`: agora a conversão usa o câmbio de cada dia.
- **Persistência:** a chave semanal do cache fazia só a primeira execução da semana salvar modelos. Estado do MaoMao e calibração não eram commitados.

**Efeito no histórico**
O log foi mantido como auditoria. Métricas, pesos e a calibração usam só linhas com `ref_date` (`data/storage.py:aligned_rows`), então a acurácia recomeça do zero e só é reportada com 100 previsões em 20 dias. As conclusões anteriores do ERR-013 e sobre as 13 famílias precisam ser medidas de novo com os dados alinhados.

**Ainda aberto**
`confidence_calibrated` continua sem aparecer no site ou no email: a decisão fica para depois de ter 20 dias alinhados.

---

## ERR-015: Previsões dos ETFs europeus saíam para um pregão já encerrado (2026-11-09)

**Sintoma**
Na primeira execução com os horizontes alinhados (ERR-014), as previsões dos ~20 tickers europeus, incluindo os 7 ETFs da carteira, partiram da barra de 05/11 com alvo em 06/11, um dia que já tinha fechado quando a previsão saiu.

**Causa raiz**
Na execução da madrugada, a série diária do yfinance ainda não tem o pregão europeu do dia anterior. Antes do ERR-014 o atraso ficava escondido no horizonte de 2 a 3 pregões.

**Correção**
Quando falta o último pregão encerrado no calendário do ticker, `data/downloader.py` monta essa barra a partir dos candles de 1 hora, que já a têm. O fechamento montado difere do oficial em até ~0,3%, porque o último candle não inclui o leilão. Por isso a validação compara com o fechamento oficial da barra de referência assim que ele aparece na série (`official_close`). Medido nos 7 ETFs para 06/11: diferença de -0,26% a +0,06%.

Também saíram da watchlist 15 tickers sem nenhuma previsão nos últimos 10 dias (sem dados no yfinance ou com treino sempre falhando, como o SHV).

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

*Última actualização: 2026-11-05*
