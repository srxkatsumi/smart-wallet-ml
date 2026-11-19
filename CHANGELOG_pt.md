# Changelog

Os incidentes em detalhe estão em [README_Errors.md](README_Errors.md).


## Incidentes e correções recentes

Histórico mais antigo e detalhado em `README_Errors.md`. Resumo dos incidentes desta fase (experimento de regressão + auditoria de pontos fracos):

### 2026-11-19: site com a confiança calibrada e contexto sem o dado do dia

- **Site:** Início, Favoritos e Watchlist mostravam a confiança bruta, enquanto o email mostrava a calibrada (a mesma previsão aparecia como 50% num e 64% no outro). Agora o site mostra a calibrada, com a bruta embaixo.
- **Contexto (VIX, SPY, BTC, GLD):** quando o Yahoo devolvia a linha do dia sem preço, o código repetia a véspera (SPY e GLD em 18/11), e isso entrava nas features de todos os ativos. Agora a linha vazia sai e o pregão é montado pelos candles de 1 hora, como já acontecia com os ativos; se nem assim houver dado, o email avisa.
- **Validador:** o import do `yfinance` tinha se perdido na auditoria de 06/11, e as previsões pendentes de tickers que saíram da watchlist não eram conferidas.

### 2026-11-12 a 2026-11-18: revisão do email, calibração honesta e desafiantes de volatilidade

- **Carteira:** o SGLN.L saiu da carteira por ser o mesmo ETC do PPFB.DE e ficou na watchlist. Posições privadas passaram a viver num secret do GitHub e só aparecem nos emails.
- **Site:** o deploy disparado pela execução diária publicava o commit anterior aos dados novos, deixando o site um dia atrasado (ERR-016). O card de acurácia mostrava "-" enquanto a significância não tinha 20 dias.
- **Revisão externa do email (ERR-018):** contadores sem universo nem janela, "Consenso dos 25 modelos" (são 13), contagem do consenso invertida, legenda do drift sem o segundo critério e a seção das famílias rotulada como semanal. Tudo corrigido e rotulado.
- **Confiança:** um estudo de 2 anos mostrou que a confiança direcional não tem relação com o acerto em nenhum recorte (ativo, horizonte, regime do VIX, acordo entre modelos), e que "sempre ▲" acerta 2 a 7 pontos mais. A confiança passou a ser calibrada por encolhimento até haver 20 dias para a curva isotônica; hoje λ = 0, e o email mostra 50 com a bruta ao lado.
- **Pesos do ensemble:** os pesos adaptativos davam exatamente o mesmo voto que a média simples em ~54 mil previsões por horizonte. A produção passou a usar a média simples.
- **Volatilidade:** o tamanho do movimento é previsível onde a direção não é. A EWMA bateu a variância histórica (QLIKE 5% a 10% menor); GARCH e HAR não bateram a EWMA; a logística para "movimento grande amanhã" teve AUC 0,61 e BSS +1,4% contra a taxa de base, sem passar na correção múltipla. A oscilação esperada entrou no email; a logística e a EWMA correm como desafiantes ao vivo.
- **Novo no email:** saúde do sistema, eventos das próximas sessões, risco, IC, referência "sempre ▲", HLN no Diebold-Mariano, rodapé com versões e o email semanal. Fuso de Barcelona com horário de verão automático (ERR-017).
- **Rastreabilidade:** cada previsão grava `versao_codigo`, `versao_calibracao`, `hash_dados` e `executado_em`.
- **Watchlist:** clicar num ativo abre os mesmos gráficos dos Favoritos (preço com SMA e Bollinger, RSI, MACD, acurácia e previsões). Entraram 88 tickers (616 no total): 11 ETFs UCITS de acumulação na Xetra, 29 ações europeias (Alemanha, França, Holanda, Itália, Suíça), 8 espanholas do IBEX, 7 britânicas pelos ADRs americanos, 11 japonesas em Tóquio, 7 chinesas em Hong Kong, 8 chinesas listadas nos EUA, fintechs de pagamento parcelado (KLAR, XYZ, TOST, SEZL) e 3 ETFs de mineradoras de ouro. Tickers novos herdam o calendário da bolsa pelo sufixo (.MC, .PA, .AS, .DE, .T, .HK...), a página converte iene e dólar de Hong Kong para euro, e o cache de moeda e dividendos busca na hora os que faltam.

### 2026-08-17: Lacunas silenciosas encontradas numa revisão de pontos fracos do projeto

Não foi um incidente único — uma auditoria deliberada encontrou três coisas erradas em silêncio, sem nunca lançar erro:

1. **Suite de testes desligada do CI.** Existiam 21 arquivos de teste, `pytest` estava no `requirements.txt`, mas o workflow nunca o invocava. `tests/test_email_phase15.py` estava quebrado (`ImportError`) há semanas — testava uma seção de PnL/lotes do email removida num refactor anterior — e ninguém percebeu. Correção: removido o teste obsoleto, adicionado `pytest.ini` (`--continue-on-collection-errors`, pra um arquivo quebrado nunca mais silenciar a suite inteira), e ligado `pytest tests/` como primeiro passo do job. Commit `5d4078378`.
2. **"Aprendizado incremental" do SGD estava inerte desde 2026-06-26.** `output/models/*.pkl` está no `.gitignore` (uma correção deliberada depois de modelos desatualizados quebrarem o `partial_fit` em silêncio e produzirem zero previsões naquele dia — ver a guarda adicionada na altura, o cheque de `n_features_in_` em `_load_sgd`), mas nenhum passo de cache restaurava a pasta — então todo run caía silenciosamente num `.fit()` do zero, todos os dias, nunca `partial_fit`. Correção: adicionado um step `actions/cache@v4` pra `output/models/` (mesma chave do cache de research já existente); a guarda de contagem de features já existente protege contra repetir o incidente de junho. Commit `5d4078378`.
3. **`PPFB.DE` ausente de `ASSET_CLASSES`/`TICKER_CALENDAR`.** Presente na carteira, ausente dos dois dicionários mantidos à mão — caía nos defaults silenciosos (`asset_class=0`/NYSE) em vez dos valores reais (iShares Physical Gold ETC, Xetra, ETC de commodity). Correção: adicionadas as entradas corretas mais `tests/test_config_coverage.py`, que agora falha alto no CI se um futuro ticker da carteira ficar de fora de algum dos dois dicionários. Commit `5d4078378`.

Também corrigido no mesmo dia: o teste de Diebold-Mariano do experimento de regressão usava o default `h=1` pros três horizontes, subestimando a variância de longo prazo pro D+2/D+3 (janelas de previsão sobrepostas são serialmente correlacionadas — o paper original de 1995 recomenda `h` = horizonte de previsão). Agora chama `diebold_mariano(e_challenger, e_champion, h=day)`. Ver "Experimento de regressão" acima.

### 2026-08-10 a 2026-08-11: tentativas de fallback entraram em corrida; o ciclo de um dia de negociação nunca correu

**Sintoma:** nenhum email chegou pro fecho de 2026-08-11, mesmo o workflow mostrando runs de "sucesso" naquela noite.

**Causa raiz 1 — corrida entre tentativas de fallback:** com o pipeline de research + experimento de regressão, um run completo já leva ~1h20-2h — mais que o intervalo entre as três tentativas de fallback do cron. Em 2026-08-10, uma segunda tentativa começou seu próprio run completo de ~2h27m porque a checagem "já rodou hoje" só olhava o CSV *commitado*, que a primeira tentativa (ainda a correr) ainda não tinha publicado. O push da tentativa perdedora foi rejeitado, e a recuperação (`git pull --rebase`) falhou também, porque `output/models/*.pkl` nunca é commitado, deixando a árvore de trabalho permanentemente "suja" pro rebase.

**Correção:** a checagem anti-duplicação agora também consulta a API do GitHub Actions por qualquer run já em curso ou já bem-sucedido hoje (apanha a corrida em segundos, não ~2h depois no push); a retentativa de push cede a um run do mesmo dia que já publicou em vez de tentar mesclar dois pipelines completos independentes. Commit `3ef79b576`.

**Causa raiz 2 — `hoje` calculado a meio da execução:** `main.py` calculava `hoje = pd.Timestamp.now().normalize()` depois de baixar os dados de mercado de toda a watchlist, não no início de `main()`. Um run que começou às 23h38 UTC e levou até 01h31 UTC do dia seguinte gravou as suas previsões com `pred_date` = a data *posterior* — correto do ponto de vista dos dados de mercado (a NYSE já tinha fechado), mas isso significava que a checagem anti-duplicação do dia seguinte real via "já feito" e saltava as três tentativas daquele dia — o fecho real daquele dia de negociação nunca foi processado.

**Correção:** movido `hoje = pd.Timestamp.now().normalize()` pra primeira linha de `main()`. Commit `3ece5070d`. Uma correção defensiva secundária — o step "✅ Validar dados gerados" recalculava `today` via `date` do shell *depois* do `main.py` terminar, podendo cair do lado errado da mesma virada de meia-noite — foi corrigida pra aceitar o `pred_date` de hoje ou de ontem. Commit `3ea9e8f47`.

### 2026-08-05: `mapie` incompatível com o scikit-learn fixado — desafiante de regressão falhou para 100% dos tickers

**Sintoma:** o novo email do experimento de regressão saiu com "0 ativos" — o modelo desafiante falhou o treino para todos os tickers.

**Causa raiz:** `mapie<1` (resolveu pra 0.9.x) depende de internals do scikit-learn (`EnsembleRegressor.__sklearn_tags__`) que não existem no `scikit-learn==1.8.0` fixado — uma incompatibilidade de versão invisível localmente porque um venv de teste descartável tinha resolvido um scikit-learn mais antigo e compatível, sem esse pin.

**Correção:** removida a dependência do `mapie`; os intervalos de predição por conformal prediction agora são calculados manualmente (treina no primeiro ~80% da janela, calibra o intervalo no ~20% mais recente — mesma filosofia manual/independente de versão já usada por `models/conformal.py` pros classificadores de produção, exatamente por este motivo). Commit `a4a602db5`.

### 2026-07-30: `KeyError: 'etoro'` depois de remover as posições do eToro da carteira

**Sintoma:** três execuções agendadas consecutivas falharam imediatamente com `KeyError: 'etoro'`.

**Causa raiz:** a chave `etoro` de `config/portfolio.json` foi removida (posições vendidas, deixaram de ser acompanhadas), mas `data/storage.py:load_my_tickers()` e `main.py` continuavam a acessar `portfolio_cfg["etoro"]` diretamente em vez de `.get("etoro", [])`.

**Correção:** ambos os pontos de acesso agora usam uma lista vazia por default quando a chave está ausente; o pipeline de research (antes restrito só aos tickers do eToro) agora recai na carteira de ETFs quando `etoro` está vazio. Commit `e32e748c`.

---

## Migração: Jupyter Notebook → Python modular
O sistema original era um único Jupyter notebook (AnaliseV5). Foi migrado para um package Python modular para permitir execução automática via GitHub Actions, gestão de dependências e manutenibilidade.

## Melhorias implementadas
- ✅ **Família Contrarian no research runner de segunda-feira** — CB (Contrarian Baseline), EWI (Error-Weighted Inverter) e PEL (Predictive Error Learning) adicionados como 9ª família na comparação semanal de modelos. Se o CB ficar em primeiro lugar, um aviso âmbar aparece no email sinalizando possíveis erros sistemáticos no ensemble principal.
- ✅ **Package Python modular** — `main.py` + `data/` + `features/` + `models/` + `reports/`
- ✅ **Tabela ML de 7 colunas no email** — Ativo · Preço · **Var%** · D+1 · D+2 · D+3 · Consenso
- ✅ **Coluna Var%** — variação fecho-a-fecho do dia anterior por ativo no email
- ✅ **Legenda ✅/❌ no email** — clarifica o significado dos ícones (acurácia D+1 do dia anterior)
- ✅ **Estratificação de acurácia** — acurácia da carteira separada da watchlist; reportadas de forma independente
- ✅ **Novas colunas no predictions_log.csv** — `actual_change_pct`, `atr_at_prediction`, `predicted_price`, `model_rf`, `model_gb`, `model_sgd`; migração retrocompatível em `storage.py`
- ✅ **ATR no momento da previsão guardado** — `atr_at_prediction` captura o contexto de volatilidade de mercado no momento em que cada previsão foi feita
- ✅ **Datas-alvo por calendário de bolsa** — `pandas-market-calendars` com mapeamento por bolsa substitui `pd.offsets.BDay` (que ignorava feriados de mercado)
- ✅ **Marcadores nos gráficos: apenas D+1** — os marcadores de validação nos gráficos mostram só previsões D+1; sobreposição D+2/D+3 na mesma data-alvo eliminada
- ✅ **Setas de previsão: datas corrigidas** — as setas apontam para os dias de negociação corretos (sexta → segunda, não sábado)
- ✅ **Repositório público** — `smart-wallet-ml` com gráficos com atraso de 10 dias, sincronizado diariamente pelo GitHub Actions (step 8)
- ✅ **Email otimizado para mobile** — tabelas com scroll horizontal e truque de margem negativa para largura total em viewports de ~412px (Samsung Galaxy S26+)
- ✅ **Coluna `ref_price`** — guarda o preço de fecho real no dia da previsão; `actual_change_pct` e a verificação de acerto passam a usar esta referência em vez do `pred_price` estimado pelo ATR
- ✅ **Verificação de acerto corrigida** — `correct = actual ≥ ref_price` (UP) / `actual ≤ ref_price` (DOWN); verificação de direcção pura, sem exigir que o ativo atinja o alvo ATR
- ✅ **Ordem de execução corrigida** — validar previsões anteriores e actualizar pesos do ensemble *antes* do treino, para que os modelos treinem sempre com os pesos actualizados de hoje e não com os de ontem
- ✅ **`save_model_metadata()` sem retreino** — `feature_importances_` lidas dos modelos já treinados em `train_all()`, eliminando uma passagem de treino duplicada
- ✅ **Downloads em batches com sleep** — pedidos ao yfinance divididos em grupos de 20 com pausa de 2 segundos entre grupos; elimina falhas silenciosas de NaN por rate limiting em watchlists grandes
- ✅ **Preço SGLN.L em EUR no email** — tickers em GBX (pence) são agora convertidos para EUR antes de serem mostrados; preço consistente com todos os outros ativos na tabela ML
- ✅ **Testes unitários** — 8 testes pytest em 3 módulos: limites do RSI (features), limites de probabilidade do ensemble, lógica de fees no P&L; os testes correm no GitHub Actions antes do `main.py` e param o pipeline em caso de falha
- ✅ **Feature importance drift como alerta** — painel diário no email com correlação de Spearman (ρ) entre o ranking de features de hoje e o período de referência; sinaliza drift quando ρ < 0,70 ou a feature principal muda
- ✅ **Retry no git push** — até 3 tentativas com `git pull --rebase` + pausa de 15s entre cada uma; elimina perda de dados por falha transitória no push
- ✅ **Artefacto GitHub em push falhado** — `predictions_log.csv` e `ensemble_weights.json` guardados como artefacto do workflow (retido 7 dias) se todas as tentativas falharem
- ✅ **Forward fill para NaN em VIX/SPY** — detecta NaN nos últimos 3 dias, aplica ffill e mostra banda âmbar no email quando activado
- ✅ **Detecção de stock split** — variação >40% face ao `ref_price` marca a validação como `NaN` em vez de `False`; threshold configurável via `SPLIT_DETECTION_THRESHOLD`
- ✅ **Correcção dos ícones ✅/❌** — `_acertou_ontem()` passa a filtrar por `target_date` em vez de `pred_date`; mostra se a previsão que *apontava* para ontem acertou, não a que foi *feita* ontem — corrige ícones em falta às segundas-feiras e para tickers cujo mercado fecha depois do pipeline correr
- ✅ **Legenda no painel drift** — adicionada descrição de ρ, do período de referência e do significado das setas (↑ ↓ →) na secção de drift do email
- ✅ **Badge dinâmico no repo público** — badge `last sync` via `shields.io/github/last-commit`; actualiza automaticamente em cada visualização, sem ficheiro JSON nem configuração extra
- ✅ **Horário movido para após o fecho dos mercados EUA** — cron reajustado para 22h00 UTC (meia-noite Barcelona CEST); NYSE/NASDAQ fecham às 20h00 UTC, por isso todos os tickers têm o preço de fecho disponível no momento da validação — corrige os ícones ✅/❌ em falta para ações americanas e crypto
- ✅ **`predictions_log_public.csv`** — versão anonimizada do log de auditoria (sem tickers, sem preços) publicada diariamente no repo público; contém `asset_type` (portfolio/watchlist), direção, confiança, resultado e votos individuais dos modelos — permite a qualquer pessoa verificar a acurácia real
- ✅ **Seção Confiabilidade** — todos os mecanismos de proteção reunidos num único lugar: testes unitários (antes da execução), forward fill VIX/SPY e detecção de split (durante a execução), retry do push e artifact de emergência (após a execução)
- ✅ **Tags semânticos no git** — `v1.0.0` (Semana 1: estabilidade), `v1.1.0` (Semana 2: observabilidade), `v1.2.0` (Semana 3: repo público); cada tag ancora um marco no histórico git com mensagem descritiva
- ✅ **Gap no Walk-Forward Validation** — `TimeSeriesSplit(n_splits=5, gap=1)`; `gap=1` insere um dia entre cada fold de treino e o de validação, evitando lookahead no mesmo dia e produzindo estimativas de acurácia mais honestas
- ✅ **Feature de regime de mercado (`vix_regime`)** — label discreta baseada no VIX (0 = baixo, 1 = médio, 2 = alto) adicionada a `FEATURE_COLS`; os modelos SGD detetam automaticamente a mudança no número de features ao carregar e reinicializam — nunca corrompem modelos existentes silenciosamente
- ✅ **Matriz de correlação no email** — heatmap de retornos diários da carteira (janela 120 dias) embebido diretamente como PNG base64 no email HTML; sem anexos; fallback silencioso se matplotlib não estiver disponível
- ✅ **Cenários de projeção ETF para longo prazo** — a coluna 10 anos mostra três cenários fixos (Pess = 3% · Base = 8% · Otim = 15%) em vez de uma única taxa histórica; elimina projeções enganadoras para tickers com historial curto ou não representativo (e.g., SGLN.L em GBX)
- ✅ **README educativo** — `README_educativo.md` (PT-BR): documentação detalhada de todas as features com fórmulas e autores originais, exemplo numérico de pesos adaptativos, historial de bugs, e explicação completa do pipeline
- ✅ **`predictions_log_public.csv` no repo público** — log de auditoria anonimizado publicado diariamente a par dos gráficos com atraso de 10 dias; contém `asset_type` (portfolio/watchlist), direção, confiança, resultado e votos individuais dos modelos — acurácia verificável por qualquer pessoa
