# /roadmap — Guardião do roadmap, READMEs e valores técnicos

Quando esta skill for invocada, execute todos os passos abaixo em ordem. Nunca pule um passo. Leia todos os arquivos diretamente do disco antes de qualquer análise; nunca use versões em memória.

---

## EQUIPA DE SKILLS — PROTOCOLO DE GOVERNANÇA

Este projeto é gerido por uma equipa de skills especializadas. O `/roadmap` é a fonte de verdade. Nenhuma skill age sem consultar o roadmap primeiro.

**Skills disponíveis:**
- `/roadmap` — Guardião. Audita, aprova, reporta estado. Nunca implementa código.
- `/model-add` — Implementa um modelo específico num projeto. Requer aprovação do roadmap.
- `/eval` — Corre testes estatísticos e métricas de avaliação. Nunca modifica modelos.
- `/xai` — Gera relatórios de explicabilidade (SHAP, attention, LIME). Nunca treina modelos.
- `/experiment` — Gere MLflow e DVC. Nunca altera experimentos passados.
- `/sync-check` — Audita o que falta por projeto. Apenas lê, nunca implementa.

**Protocolo obrigatório para todas as skills:**
1. Ler o roadmap e verificar se a tarefa está na lista aprovada
2. Se não está na lista: parar e pedir aprovação ao utilizador
3. Se está aprovada: executar
4. Após concluir: correr testes e verificar que nada quebrou
5. Reportar resultado — só depois o utilizador decide se faz commit

**Regra de ouro — NADA vai para git sem passar nos testes:**
- Carteira: `pytest tests/` deve passar a 100%
- Mega Sena: `pytest test_ml/analisenumerica/tests/` deve passar a 100%
- Qualquer novo modelo deve ter o seu próprio ficheiro de teste antes de ser considerado implementado
- Se os testes falharem: corrigir antes de reportar conclusão. Nunca reportar "feito" com testes a falhar.

---

## REGISTO MESTRE DE MODELOS

Estes são os 25 modelos aprovados organizados em 7 famílias. Este registo é a fonte de verdade para `/sync-check` e `/model-add`.

| # | Família | Modelos | Projectos alvo |
|---|---------|---------|----------------|
| 1 | Clássico | RF, GB, SGD, XGBoost, LightGBM, CatBoost, SVM | Carteira, Mega Sena |
| 2 | Séries temporais | ARIMA, SARIMA, ETS, Holt-Winters, Prophet | Carteira, Mega Sena, E-commerce |
| 3 | Estado oculto | Markov, HMM | Carteira, Mega Sena |
| 4 | Redes neurais | LSTM, GRU, Transformer, TFT, N-BEATS | Carteira, Mega Sena |
| 5 | Bayesiano | Gaussian Process, BNN | Carteira, Mega Sena |
| 6 | Generativo | VAE, GAN | Carteira, Mega Sena |
| 7 | Reinforcement | Q-Learning, PPO | Carteira |

**Estado de implementação por projecto:**

| Modelo | Carteira | Mega Sena | E-commerce |
|--------|----------|-----------|------------|
| RF | ✅ | ✅ | ⬜ |
| GB | ✅ | ✅ | ⬜ |
| SGD | ✅ | ✅ | ⬜ |
| XGBoost | ⬜ | ⬜ | ⬜ |
| LightGBM | ⬜ | ⬜ | ⬜ |
| CatBoost | ⬜ | ⬜ | ⬜ |
| SVM | ⬜ | ⬜ | ⬜ |
| ARIMA | ⬜ | ⬜ | ⬜ |
| SARIMA | ⬜ | ⬜ | ⬜ |
| ETS | ⬜ | ⬜ | ⬜ |
| Holt-Winters | ⬜ | ⬜ | ⬜ |
| Prophet | ⬜ | ⬜ | ⬜ |
| Markov | ⬜ | ⬜ | ⬜ |
| HMM | ⬜ | ⬜ | ⬜ |
| LSTM | ⬜ | ⬜ | ⬜ |
| GRU | ⬜ | ⬜ | ⬜ |
| Transformer | ⬜ | ⬜ | ⬜ |
| TFT | ⬜ | ⬜ | ⬜ |
| N-BEATS | ⬜ | ⬜ | ⬜ |
| Gaussian Process | ⬜ | ⬜ | ⬜ |
| BNN | ⬜ | ⬜ | ⬜ |
| VAE | ⬜ | ⬜ | ⬜ |
| GAN | ⬜ | ⬜ | ⬜ |
| Q-Learning | ⬜ | — | ⬜ |
| PPO | ⬜ | — | ⬜ |

**Camadas transversais (além dos modelos):**

| Camada | Componente | Estado |
|--------|-----------|--------|
| Avaliação | Diebold-Mariano test | ⬜ |
| Avaliação | McNemar test | ⬜ |
| Avaliação | Ljung-Box test | ⬜ |
| Avaliação | Métricas por domínio | ⬜ |
| Explicabilidade | SHAP values | ⬜ |
| Explicabilidade | Attention weights | ⬜ |
| Explicabilidade | LIME | ⬜ |
| Meta-learning | Stacking com meta-learner | ⬜ |
| Meta-learning | Optuna (hyperopt) | ⬜ |
| Rastreamento | MLflow | ⬜ |
| Rastreamento | DVC | ⬜ |
| Teoria | Entropia de Shannon | ⬜ |
| Teoria | Informação Mútua | ⬜ |
| Transferência | Transfer Learning entre domínios | ⬜ |

---

## ROADMAP DE IMPLEMENTAÇÃO — 14 FASES

Este é o plano de execução oficial. Todas as skills devem seguir esta ordem. Nenhuma fase deve ser iniciada sem aprovação explícita do utilizador.

**Princípio de ordenação:** dependências mínimas primeiro. Cada fase reutiliza o que a anterior instalou.

| Fase | O que implementar | Família | Dependências novas | Projectos | Estado |
|------|------------------|---------|-------------------|-----------|--------|
| **0** | RF, GB, SGD | Clássico | nenhuma | Carteira, Mega Sena | ✅ |
| **1** | Markov, HMM | Estado oculto | nenhuma | Carteira, Mega Sena | ⬜ |
| **2** | XGBoost, LightGBM, CatBoost, SVM | Clássico avançado | `xgboost`, `lightgbm`, `catboost` | Carteira, Mega Sena | ⬜ |
| **3** | ARIMA, SARIMA, ETS, Holt-Winters, Prophet | Séries temporais | `statsmodels`, `prophet` | Carteira, Mega Sena | ⬜ |
| **4** | LSTM, GRU | Neural básico | `torch` | Carteira, Mega Sena | ⬜ |
| **5** | Transformer, TFT, N-BEATS | Neural avançado | `torch` (já tem) | Carteira, Mega Sena | ⬜ |
| **6** | Gaussian Process, BNN | Bayesiano | `torch` (já tem) | Carteira, Mega Sena | ⬜ |
| **7** | VAE, GAN | Generativo | `torch` (já tem) | Carteira, Mega Sena | ⬜ |
| **8** | Q-Learning, PPO | Reinforcement | `gymnasium`, `stable-baselines3` | Carteira | ⬜ |
| **9** | Diebold-Mariano, McNemar, Ljung-Box, métricas por domínio | Avaliação | `scipy` | Todos | ⬜ |
| **10** | SHAP, Attention weights, LIME | Explicabilidade | `shap`, `lime` | Todos | ⬜ |
| **11** | Stacking com meta-learner, Optuna | Meta-learning | `optuna` | Todos | ⬜ |
| **12** | MLflow, DVC | Rastreamento | `mlflow`, `dvc` | Todos | ⬜ |
| **13** | Entropia de Shannon, Informação Mútua | Teoria da informação | `scipy` (já tem) | Todos | ⬜ |
| **14** | Transfer Learning entre domínios | Transferência | nenhuma nova | Todos | ⬜ |
| **15** | Reestruturação do email: secção acções + ETFs + recomendação mensal | Relatório | nenhuma nova | Carteira | ⬜ |

**Contagem total:**
- Modelos: 25 (3 já implementados + 22 a implementar em fases 1-8)
- Famílias: 7
- Testes estatísticos: 3 (Fase 9)
- Ferramentas XAI: 3 (Fase 10)
- Meta-learning: 2 componentes (Fase 11)
- Rastreamento: 2 ferramentas (Fase 12)
- Teoria: 2 componentes (Fase 13)
- Transfer Learning: 1 (Fase 14)

**Domínios alvo:** Loteria (Mega Sena) → Acções (Carteira) → E-commerce (futuro)

**Regra para os agentes:** antes de iniciar qualquer fase, verificar se a fase anterior está marcada como ✅. Nunca saltar fases sem aprovação explícita do utilizador.

---

### ESPECIFICAÇÃO DETALHADA — FASE 15: Email

#### Secção 1: Acções (eToro)

Colunas obrigatórias na tabela HTML do email:

| Coluna | Origem | Cálculo |
|--------|--------|---------|
| Ativo | portfolio/transactions.csv ou equivalente | — |
| Data compra | registo de compra | — |
| Preço compra | registo de compra | valor pago por unidade na data |
| Fecho ontem | yfinance | preço de fecho do dia anterior |
| D+1 | predictions_log.csv | previsão com seta ▲/▼ |
| D+2 | predictions_log.csv | previsão com seta ▲/▼ |
| D+3 | predictions_log.csv | previsão com seta ▲/▼ |
| Alvo 15% (€) | calculado | preço_compra × 1,15 |
| % feito | calculado | (fecho_ontem - preço_compra) / preço_compra × 100 |
| % pendente | calculado | (alvo_15 - fecho_ontem) / preço_compra × 100 |

Regras:
- Um lote de compra = uma linha. NVDA comprado em datas diferentes = linhas separadas.
- Se % feito >= 15%: destacar célula em verde (alvo atingido).
- Se % pendente < 0: o ativo já ultrapassou o alvo — destacar em dourado.

#### Secção 2: ETFs (longo prazo)

Colunas obrigatórias:

| Coluna | Origem | Cálculo |
|--------|--------|---------|
| ETF | portfolio/transactions.csv | — |
| Ticker | portfolio/transactions.csv | — |
| Data compra | registo de compra | — |
| Preço compra | registo de compra | valor pago por unidade |
| Fecho ontem | yfinance | — |
| D+1 | predictions_log.csv | ▲/▼ + preço estimado |
| D+2 | predictions_log.csv | ▲/▼ + preço estimado |
| D+3 | predictions_log.csv | ▲/▼ + preço estimado |
| Δ vs compra (%) | calculado | (fecho_ontem - preço_compra) / preço_compra × 100 |

Regras:
- Label da secção: "ETFs — Visão de Longo Prazo"
- Separada visualmente da secção de acções.
- Sem coluna de Alvo 15% (ETFs são longo prazo, sem alvo de saída fixo).

#### Secção 3: Recomendação mensal de compra ETF

Condição de activação: apenas se hoje for a 1.ª semana útil do mês (dias 1 a 7, dias úteis apenas).

Conteúdo:
- Para cada ETF da carteira (SGLN.L, EXUS.L, ICGA.DE):
  - Analisar previsões D+1 a D+5 da semana
  - Identificar o dia com menor preço previsto
  - Apresentar: "Melhor dia para comprar [ETF]: [dia da semana] ([data]) — preço estimado [€]"
- Label da secção: "Recomendação de Compra — Esta Semana"
- Aparece uma vez por mês, no topo do email, antes das tabelas de previsões.

---

## CONTEXTO DO PROJETO

Este repositório contém dois projetos independentes que correm no mesmo pipeline GitHub Actions. Um terceiro projecto (e-commerce) está planeado para 2026.

**Carteira Inteligente (projeto principal):**
Pipeline MLOps de previsão de carteira de investimentos. Escrito em Python 3.11, corre todos os dias úteis às 22h00 (Barcelona) via GitHub Actions. Ensemble de RF + GB + SGD com pesos adaptativos e decaimento temporal para prever a direção de preços em D+1, D+2 e D+3.

**Mega Sena ML Experiment (subprojeto em `test_ml/analisenumerica/`):**
Experimento científico negativo controlado. Aplica o mesmo ensemble adaptativo a um processo declaradamente aleatório (Mega Sena). Iniciado em 28/05/2026. Horizonte de 5 anos para uso como base de doutoramento. Corre diariamente (backfill de 300 sorteios por execução). Previsões apenas às segundas-feiras.

**E-commerce (planeado):**
Terceiro domínio para previsão de sazonalidades e tendências de preço. Domínio com padrões reais, contraste com Mega Sena (aleatório) e Carteira (ruidoso).

---

## REGRAS GLOBAIS PARA TODOS OS READMES

Estas regras aplicam-se a qualquer atualização de README, sem exceções:

- Nunca usar travessões em texto corrido nem em listas. Usar vírgulas, ponto e vírgula ou frases separadas. Nunca o símbolo — no meio de uma frase.
- Todos os READMEs escritos em primeira pessoa, pois o projeto foi construído por uma única pessoa.
- README.md em inglês técnico.
- README_pt.md e README_educativo.md em português brasileiro. Usar "arquivo" (não "ficheiro"), "atualizar" (não "actualizar"), "portfólio" mantém-se, termos técnicos ML mantêm-se em inglês.
- Horário de execução sempre como 22h00 (Barcelona) em todos os arquivos.
- Valores dos hiperparâmetros nos READMEs devem refletir exatamente o que está em `config/settings.py` (stock) ou `test_ml/analisenumerica/config.py` (Mega Sena). Nunca publicar valores aproximados ou históricos.

---

## PASSO 0 — Auditoria técnica: código versus README

Este é o passo mais crítico. Leia os arquivos de código indicados e compare com o que os READMEs afirmam. Registre cada divergência encontrada.

### 0-A: Hiperparâmetros do pipeline de ações

Leia `config/settings.py` e extraia os valores reais de:

| Parâmetro | Valor no código | O que o README.md afirma | Divergência? |
|-----------|----------------|--------------------------|--------------|

Valores a verificar (extrair do código, não assumir):
- `N_ESTIMATORS_RF` e `MAX_DEPTH_RF` (o README diz "300 trees, max depth 6" mas o código pode ter valores diferentes)
- `N_ESTIMATORS_GB`, `MAX_DEPTH_GB`, `LEARNING_RATE_GB` (o README diz "200 estimators" mas verificar)
- `N_SPLITS_CV` e `CV_GAP` (README diz "TimeSeriesSplit(n_splits=5, gap=1)")
- `SPLIT_DETECTION_THRESHOLD` (README diz ">40%")
- `RECALIBRATION_DAYS`
- `DOWNLOAD_BATCH_SIZE` e `DOWNLOAD_BATCH_SLEEP` (README diz "batches of 20 with 2s pause")

### 0-B: Features do pipeline de ações

Leia `features/engineering.py` e extraia a lista completa de `FEATURE_COLS`. Compare com a tabela de features no README.md:

- Contar quantas features existem no código.
- Verificar se `vol_ratio`, `obv_trend` e `asset_class` estão documentados na tabela do README (foram adicionados recentemente e provavelmente ainda faltam).
- Verificar se a definição de `target_d1/d2/d3` no README reflete o threshold ATR atual. O código usa `ATR14 * 0.3 * sqrt(horizon)` como threshold com dias neutros como NaN. O README ainda mostra a versão binária simples?
- Verificar se o README menciona `CalibratedClassifierCV` com `TimeSeriesSplit(n_splits=3)` isotonic.

### 0-C: Hiperparâmetros da Mega Sena

Leia `test_ml/analisenumerica/config.py` e extraia:

| Parâmetro | Valor no código | Valor no README do subprojeto | Divergência? |
|-----------|----------------|-------------------------------|--------------|

Valores a verificar:
- `N_BALLS` (deve ser 60), `BALLS_PER_DRAW` (deve ser 6)
- `N_ESTIMATORS_RF` e `MAX_DEPTH_RF` (README diz "100 árvores, profundidade máx. 4")
- `N_ESTIMATORS_GB`, `MAX_DEPTH_GB`, `LEARNING_RATE_GB` (README diz "100 estimadores, lr 0.05, profundidade máx. 3")
- `N_SPLITS_CV`
- `DAILY_BATCH_SIZE` (deve ser 300), `RETRAIN_INTERVAL` (deve ser 50)
- `RANDOM_EXPECTED_MATCHES` (deve ser 0.6 = 6 * 6/60)
- `PRIZE_TIERS` (deve ter: 6=Sena, 5=Quina, 4=Quadra, 3=Terno, 2=Duque)

### 0-D: Features da Mega Sena

Leia `test_ml/analisenumerica/features/engineering.py` e extraia a lista de `FEATURE_COLS`. Compare com a tabela de features no README:

- Contar quantas features existem no código.
- Verificar se `pair_freq_prev` está documentada (foi adicionada recentemente).
- Verificar se os 4 janelas de frequência (`freq_5d`, `freq_10d`, `freq_20d`, `freq_50d`) batem com `FREQ_WINDOWS = [5, 10, 20, 50]`.

### 0-E: Contagem do roadmap

Leia `README.md` e `README_pt.md` do projeto principal e conte:

- Quantos itens ✅ em cada arquivo.
- Quantos itens ⬜ em cada arquivo.
- Os dois totais devem ser iguais. Se não forem, há divergência.

### 0-F: Datas e horários

- O horário de execução está como 22h00 em ambos os READMEs? (Não "~22h30", não "20h30 UTC sozinho", não outra variação.)
- A data de início da Mega Sena está como 28 de maio de 2026 em ambos os READMEs do subprojeto?
- O horizonte de 5 anos está mencionado?

### 0-H: Completude de modelos por projecto

Ler o REGISTO MESTRE DE MODELOS acima e verificar em cada projecto quais ficheiros de modelo existem:

- `models/` na Carteira: listar ficheiros .py presentes
- `test_ml/analisenumerica/models/` na Mega Sena: listar ficheiros .py presentes

Para cada modelo do registo, verificar se existe um ficheiro correspondente. Produzir tabela:

| Família | Modelo | Carteira | Mega Sena | E-commerce |
|---------|--------|----------|-----------|------------|

Marcar ✅ se o ficheiro existe e tem implementação, ⬜ se falta, — se não aplicável ao projecto.

---

### 0-G: Audit de travessões

Varrer os seguintes arquivos em busca do símbolo — (travessão) em texto corrido:
- `README.md`
- `README_pt.md`
- `test_ml/analisenumerica/README.md`
- `test_ml/analisenumerica/README_pt.md`

Listar cada ocorrência com o número de linha e o trecho de texto. Indicar se é em código ou tabela (aceitável) ou em texto corrido (não permitido).

---

## PASSO 1 — Estado atual dos roadmaps

### Roadmap do pipeline de ações (README.md)

Listar todos os itens numerados com seu estado atual lido diretamente do README:

| # | Semana | Item | Estado |
|---|--------|------|--------|
| 1 | Semana 1 | Retry no git push | ? |
| 2 | Semana 1 | Artefato GitHub em push falhado | ? |
| 3 | Semana 1 | Forward fill para NaN em VIX/SPY | ? |
| 4 | Semana 1 | Detecção de stock split | ? |
| 5 | Semana 2 | Feature importance drift alert | ? |
| 6 | Semana 2 | Telegram como fallback de email | ? |
| 7 | Semana 2 | Badge dinâmico no repo público | ? |
| 8 | Semana 3 | predictions_log_public.csv | ? |
| 9 | Semana 3 | Seção "Confiabilidade" no README | ? |
| 10 | Semana 3 | Git tags semânticos | ? |
| 11 | Semanas 4-5 | Walk-Forward Validation | ? |
| 12 | Semanas 4-5 | Regime de mercado como feature | ? |
| 13 | Semana 6 | Matriz de correlação no email | ? |
| 14 | Semana 6 | Cenários de projeção ETF | ? |
| 15 | Semanas 7-8 | READMEs finais (EN + PT) | ? |
| 16 | Semanas 7-8 | Lançamento do repo público | ? |
| 17 | Semanas 7-8 | Artigo LinkedIn | ? |
| 18 | Pós-publicação | Features de eventos fundamentalistas | ? |
| 19 | Pós-publicação | Regressor de preço D+1 | ? |

Substituir os ? pelo que os READMEs mostram (✅ ou ⬜).

Registrar também os itens que estão implementados no código mas ainda não documentados nos READMEs como itens do roadmap:
- Calibração com CalibratedClassifierCV (TimeSeriesSplit n_splits=3, isotonic)
- Volume features: vol_ratio e obv_trend
- Threshold ATR nos targets (não mais binário simples; dias neutros filtrados do treino)
- asset_class como feature (0=ação, 1=ETF, 2=cripto, 3=commodity ETF)
- Precisão separada UP/DOWN no email com novos ícones ✅/📉/❌

### Roadmap da Mega Sena (5 anos)

Verificar no README do subprojeto o estado dos 5 marcos anuais:

| Ano | Marco | Estado |
|-----|-------|--------|
| 2026 | Baseline estabelecido (primeiros 100 sorteios, backfill walk-forward) | Em andamento |
| 2027 | Teste de significância estatística (t-test contra mu=0,60 com N>=150) | Pendente |
| 2028 | Estudo de ablação de features | Pendente |
| 2029 | Estudo comparativo (Mega Sena vs mercado financeiro) | Pendente |
| 2030 | Material de tese de doutoramento | Pendente |

Verificar também se os seguintes itens de implementação estão documentados nos READMEs do subprojeto:
- Backfill diário (300 sorteios por execução) com previsão apenas às segundas-feiras
- Feature `pair_freq_prev` (co-ocorrência com sorteio anterior)
- Baselines comparativos: hot (top-6 mais frequentes), cold (bottom-6 menos frequentes), random (Monte Carlo)
- Seção de comparação de baselines no relatório gerado

---

## PASSO 2 — Coerência entre os READMEs

### Pipeline de ações: README.md vs README_pt.md

Verificar ponto a ponto:

1. Número de itens ✅ no roadmap é igual nos dois?
2. Número de entradas no changelog é igual nos dois?
3. Os valores de hiperparâmetros nas tabelas dos dois são idênticos?
4. O horário de execução está como 22h00 nos dois?
5. A tabela de features tem as mesmas linhas nos dois?
6. As colunas do predictions_log.csv listadas nos dois são as mesmas?
7. A contagem de testes unitários (8 testes, 3 módulos) está nos dois?
8. A seção de acurácia (Portfolio vs Watchlist) está nos dois com o mesmo conteúdo?

Se algum ponto divergir, listar claramente a divergência antes de continuar.

### Mega Sena: README.md vs README_pt.md

Verificar:

1. A data de início (28 May 2026 / 28 de maio de 2026) está correta e igual nos dois?
2. O horizonte de 5 anos está nos dois?
3. A tabela de probabilidades teóricas tem os mesmos valores nos dois?
4. A lista de features tem o mesmo número de linhas nos dois?
5. A estratégia de previsão (5 sequências por sorteio) está descrita nos dois?
6. O roadmap de 5 anos tem os mesmos marcos nos dois?
7. As instruções de execução local estão sincronizadas?

---

## PASSO 3 — Próximos passos para o pipeline de ações

Com base no que foi encontrado nos passos anteriores, apresentar:

1. **Divergências técnicas que exigem atualização imediata dos READMEs** (valores de hiperparâmetros errados, features não documentadas, definição de target desatualizada).

2. **O próximo item de roadmap a implementar**, com nome e impacto esperado.

3. **Os 2 itens seguintes na fila** para contexto de sequência.

Usar sempre o princípio: Estabilidade > Observabilidade > Publicação > Modelo > Features avançadas. Nunca propor um item de modelo se houver itens de estabilidade ou observabilidade pendentes.

---

## PASSO 4 — Checklist de README para a próxima implementação

Antes de qualquer implementação, gerar um checklist específico:

```
README UPDATE CHECKLIST — [nome do próximo item]

README.md:
  [ ] Adicionar entrada no Changelog com descrição detalhada em inglês
  [ ] Marcar item como ✅ no Roadmap
  [ ] Atualizar qualquer seção técnica afetada (Automation, Features, Mermaid, etc.)
  [ ] Verificar se a estrutura do repositório precisa de atualização
  [ ] Confirmar que não há travessões introduzidos
  [ ] Confirmar que está em primeira pessoa
  [ ] Confirmar que os valores de hiperparâmetros batem com config/settings.py

README_pt.md:
  [ ] Adicionar entrada no Changelog com tradução completa em português brasileiro
  [ ] Marcar item como ✅ no Roadmap
  [ ] Atualizar qualquer seção técnica afetada (mesmas seções do EN)
  [ ] Verificar coerência de terminologia com entradas anteriores
  [ ] Confirmar vocabulário PT-BR ("arquivo", "atualizar", "seção")
  [ ] Confirmar que não há travessões introduzidos

README_educativo.md (se existir):
  [ ] Verificar se o item implementado afeta alguma seção explicativa
  [ ] Atualizar exemplos concretos se necessário
  [ ] Verificar seção de limitações honestas se relevante

Mega Sena (se a mudança afetar o subprojeto):
  [ ] test_ml/analisenumerica/README.md atualizado em inglês
  [ ] test_ml/analisenumerica/README_pt.md atualizado em PT-BR
  [ ] Valores de hiperparâmetros batem com test_ml/analisenumerica/config.py

VALIDAÇÃO FINAL:
  [ ] Ambos os READMEs têm o mesmo número de itens ✅ no roadmap
  [ ] Ambos os changelogs têm o mesmo número de entradas
  [ ] Nenhuma seção técnica ficou com informação desatualizada
  [ ] Horário 22h00 correto em todos os arquivos
  [ ] Zero travessões em texto corrido
  [ ] Hiperparâmetros corretos e verificados contra o código
```

---

## PASSO 5 — Quando pedido para atualizar ou criar READMEs

Se o usuário pedir para atualizar os READMEs ou criar o README_educativo.md, seguir as instruções abaixo.

### README.md (Inglês técnico) — ATUALIZAR, NÃO SUBSTITUIR

Ler o README.md atual e aplicar apenas as seguintes alterações, preservando todo o conteúdo existente:

1. Atualizar o horário de execução para 22h00 (Barcelona) em todas as ocorrências.

2. Remover todos os travessões em texto corrido. Substituir por vírgulas, ponto e vírgula ou frases separadas conforme o contexto.

3. Corrigir a tabela de modelos com os valores reais do código:

   ```
   | Model | Configuration | Role in ensemble |
   |-------|--------------|-----------------|
   | Random Forest | 100 trees, max depth 5, class_weight balanced | Stability anchor. Bootstrapped trees resist overfitting on noisy market data. Works well even when some features are irrelevant. |
   | Gradient Boosting | 100 estimators, lr 0.05, max depth 3 | Iteratively corrects residuals. Better at detecting short-term momentum and subtle feature interactions. |
   | SGD Classifier | log_loss, L2, monthly full recalibration | Linear counterweight. When both non-linear models agree on noise, the SGD pulls the ensemble toward more conservative estimates. |
   ```

4. Corrigir a definição de targets para refletir o threshold ATR atual:

   ```
   target_d1 = 1 if move > ATR14 x 0.3 x sqrt(1), 0 if move < -(ATR14 x 0.3 x sqrt(1)), NaN if neutral
   target_d2 = 1 if move > ATR14 x 0.3 x sqrt(2), 0 if move < -(ATR14 x 0.3 x sqrt(2)), NaN if neutral
   target_d3 = 1 if move > ATR14 x 0.3 x sqrt(3), 0 if move < -(ATR14 x 0.3 x sqrt(3)), NaN if neutral
   ```
   Neutral days are excluded from training. The model only learns from days with a clear directional signal.

5. Atualizar a tabela de features para incluir vol_ratio, obv_trend e asset_class.

6. Adicionar menção à calibração CalibratedClassifierCV com TimeSeriesSplit(n_splits=3) isotonic.

7. Adicionar os seguintes diagramas Mermaid imediatamente após a seção Overview:

   **BLOCO A: Daily Pipeline**
   ```mermaid
   flowchart TD
       A["GitHub Actions\n3 crons Mon-Fri\nanti-duplication check"] --> B["Duplicate check\nreads predictions_log.csv\nskip if today exists"]
       B -->|not yet run| C["Price download\nbatches of 20 tickers\n2s sleep between batches"]
       C --> D["Forward fill\nVIX/SPY NaN detection\namber warning in email"]
       D --> E["Feature engineering\nSMA20/50 - RSI14 - MACD\nBollinger - ATR14 - ret_1d/5d\nvol_ratio - obv_trend\nspy_ret_1d - vix_level - vix_regime - asset_class"]
       E --> F["Past forecast validation\nref_price vs actual_price\nstock split detection 40pct"]
       F --> G["Ensemble weight update\nexponential temporal decay\nrolling 30-day window"]
       G --> H["Train 3 independent ensembles\nD+1 - D+2 - D+3\nCalibratedClassifierCV isotonic"]
       H --> I["Save new forecasts\nD+1 / D+2 / D+3\npredictions_log.csv"]
       I --> J["Chart generation\n12 charts per day\ncleanup after 30 days"]
       J --> K["HTML email build\nML table - P&L - accuracy UP/DOWN\ndrift alert - charts"]
       K --> L["Git commit with retry\n3 attempts - 15s pause\ngit pull --rebase"]
       L -->|push fails| M["Safety artifact\npredictions_log.csv\nretained 7 days"]
       L -->|push ok| N["Gmail SMTP send\nHTML email\nfailure email on error"]
       N --> O["Public repo sync\n10-day delay window\nanonymised CSV + charts"]
   ```

   **BLOCO B: ML Architecture**
   ```mermaid
   flowchart TD
       F["Input features 20 total\nSMA20/50 - RSI14 - MACD - Bollinger\nATR14 - ret_1d/5d - vol_ratio - obv_trend\nspy_ret_1d - vix_level/change/regime - asset_class"] --> E1["D+1 Ensemble"]
       F --> E2["D+2 Ensemble"]
       F --> E3["D+3 Ensemble"]
       E1 --> RF1["Random Forest\n100 trees - max_depth 5\nCalibratedClassifierCV isotonic"]
       E1 --> GB1["Gradient Boosting\n100 est. - lr 0.05\nCalibratedClassifierCV isotonic"]
       E1 --> SG1["SGD Classifier\nlog_loss - L2 - monthly recal."]
       RF1 & GB1 & SG1 --> W1["Adaptive weights\nexponential decay\nrolling 30-day accuracy"]
       W1 --> P1["Forecast D+1\nUP/DOWN + confidence"]
       E2 --> W2["Adaptive weights D+2"]
       E3 --> W3["Adaptive weights D+3"]
       W2 --> P2["Forecast D+2"]
       W3 --> P3["Forecast D+3"]
       P1 & P2 & P3 --> V["Validation\nactual_price vs ref_price\ncorrect = True/False/NaN"]
       V --> WU["Weight update\nmore recent = more weight\nweight proportional to accuracy x sum(decay^t)"]
   ```

8. Manter o changelog e roadmap existentes sem alterações de conteúdo. Apenas remover travessões se existirem.

---

### README_pt.md (Português brasileiro técnico) — ATUALIZAR, NÃO SUBSTITUIR

Ler o README_pt.md atual e aplicar apenas as seguintes alterações, preservando todo o conteúdo existente:

1. Atualizar o horário de execução para 22h00 (Barcelona) em todas as ocorrências.

2. Remover todos os travessões em texto corrido.

3. Corrigir a tabela de modelos com os valores reais (100 árvores, max depth 5, 100 estimadores).

4. Corrigir a definição de targets para refletir o threshold ATR.

5. Atualizar a tabela de features para incluir vol_ratio, obv_trend e asset_class.

6. Adicionar menção à calibração.

7. Adicionar os mesmos diagramas Mermaid com labels em português brasileiro.

8. Confirmar vocabulário PT-BR: "arquivo" (não "ficheiro"), "atualizar" (não "actualizar"), "seção" (não "secção").

9. Manter changelog e roadmap existentes sem alterações de conteúdo.

---

### README_educativo.md — CRIAR NOVO ARQUIVO

Escrito em português brasileiro, em primeira pessoa, sem travessões.

**Estrutura obrigatória:**

**1. Introdução pessoal**
Por que construí este projeto. O que fazia antes (análise manual) e o que o sistema faz agora.

**2. Como o sistema funciona no dia a dia**
Explicação simples do pipeline diário em linguagem acessível. Incluir o diagrama Mermaid da pipeline com labels em português brasileiro simples.

**3. O que é Machine Learning neste contexto**
Classificação binária com exemplos reais dos ativos (NVDA, BTC, ALV.DE). Explicar o threshold ATR: o modelo só aprende em dias com sinal claro; dias neutros são ignorados no treino.

**4. Os três modelos do ensemble**
Para cada modelo: o que faz, por que está no ensemble, o que seria diferente sem ele. Incluir o diagrama Mermaid da arquitetura ML. Explicar a calibração: por que CalibratedClassifierCV evita probabilidades infladas (ex: DHER.DE previsto com 90% antes da calibração).

**5. As features: o que o modelo vê**

Para cada feature: o que mede, como é calculada, quem criou e quando, por que foi incluída.

Autores a citar:
- ATR: J. Welles Wilder Jr., 1978
- RSI: J. Welles Wilder Jr., 1978
- MACD: Gerald Appel, 1979
- Bollinger Bands: John Bollinger, anos 1980
- SMA: uso histórico sem autor único
- vol_ratio e obv_trend: derivados do On Balance Volume (Joseph Granville, 1963)

**6. Pesos adaptativos**
Decaimento exponencial em linguagem simples.
Fórmula: `peso(modelo) proporcional a acuracia(modelo) x soma decaimento^(dias_atras)`
Exemplo numérico concreto: RF acertou 6 de 10, GB acertou 4 de 10; como ficam os pesos.

**7. Como as previsões são validadas**
ref_price, actual_price e o cálculo de correct.
Bug do pred_price que causou 23% de acurácia e como o ref_price corrigiu.
Fórmulas:
- `correct = actual >= ref_price` (UP)
- `correct = actual <= ref_price` (DOWN)

**8. As métricas do email explicadas**

ACURÁCIA GERAL: `correct_count / total_validated x 100`, janela 30 dias úteis, apenas carteira.

VAR%: `(preco_hoje - preco_ontem) / preco_ontem x 100`

CONFIDENCE: `soma (peso_modelo x prob_modelo)` para cada modelo.

CONSENSO: BULLISH (D+1 + D+2 + D+3 todos UP); BEARISH (todos DOWN); MISTO (discordância).

ICONES POR ATIVO: ✅ = previu subida e subiu; 📉 = previu queda e caiu; ❌ = errou.

ACURÁCIA SEPARADA UP/DOWN: precisão quando prevê subida vs quando prevê queda. Um modelo que prevê sempre UP tem 50% de acurácia global mas zero valor real.

RO DE SPEARMAN (feature drift): fórmula `rho = 1 - (6 x soma d2) / (n x (n2 - 1))`. Charles Spearman, 1904. rho próximo de 1 = modelo estável; rho < 0,70 = regime de mercado pode ter mudado.

ATR ESTIMADO: `close +/- ATR x 0,5 x raiz(horizonte)`. J. Welles Wilder Jr., 1978.

**9. Por que calendários por bolsa são importantes**
Exemplo: NYSE fecha no Thanksgiving, LSE não. Sem pandas-market-calendars o modelo procuraria preços em dias sem negociação.

**10. O audit trail: por que nunca apago nada**
predictions_log.csv imutável. Por que é o ativo mais valioso do projeto. Como permitiu detectar o bug do ref_price retroativamente.

**11. Limitações honestas**
O modelo não lê notícias, não sabe de earnings, não distingue queda de 0,1% de queda de 8%. A acurácia ainda está em fase de validação.

**12. Roadmap em linguagem simples**
O roadmap técnico traduzido para o que cada item significa na prática.

---

## PASSO 6 — Resumo executivo do analista

Terminar sempre com este bloco. Preencher todos os campos com os valores reais encontrados nos passos anteriores:

```
ESTADO DO PROJETO — [data de hoje]

PIPELINE DE ACOES
Semana atual:         [Semana X — nome da fase]
Itens concluídos:     [X de 19]
Próximo item:         [#N — nome]
READMEs:              [Sincronizados / Divergências encontradas: listar]
Travessões:           [Nenhum / Encontrados em: arquivo linha X]
Horário:              [22h00 correto / Desatualizado em: arquivo]
Hiperparâmetros:      [Corretos / Divergência: README diz X, código tem Y]
Features documentadas:[X de Y / Faltam: listar]
Targets:              [ATR threshold documentado / Ainda binário no README]

MEGA SENA ML EXPERIMENT
Data de início:       [28/05/2026 correto / Divergência]
Ano do roadmap:       [2026 — Baseline em andamento]
Backfill estimado:    [X de 3011 sorteios processados]
Features:             [pair_freq_prev documentado / Falta]
Baselines:            [hot/cold/random documentados / Faltam]
READMEs:              [Sincronizados / Divergências: listar]
Travessões:           [Nenhum / Encontrados em: arquivo linha X]

MODELOS — CARTEIRA
Implementados:      [listar]
Pendentes:          [X de 25]

MODELOS — MEGA SENA
Implementados:      [listar]
Pendentes:          [X de 25]

CAMADAS TRANSVERSAIS
Avaliação:          [X de 4 componentes]
Explicabilidade:    [X de 3 componentes]
Meta-learning:      [X de 2 componentes]
Rastreamento:       [MLflow / DVC: estado]
Teoria:             [X de 2 componentes]
Transferência:      [estado]

ACAO IMEDIATA RECOMENDADA:
[uma frase clara e direta sobre o que fazer agora]
```
