# Roadmap

## Próximos passos (novembro e dezembro de 2026)

| Item | Por quê | Estado |
|---|---|---|
| Empate deixa de contar como acerto; acurácia nos movimentos acima do limiar de treino lado a lado | A avaliação deve medir aquilo para que o modelo foi treinado | Planejado |
| Saúde da watchlist ("612 de 616 com dados", lista dos parados) | 616 ativos podem ficar sem dados em silêncio | Planejado |
| Bloco de risco e calendário de eventos no site | Hoje só existem no email | Planejado |
| Calibração isotônica e veredito do Jinshi para o ensemble ampliado e o desafiante de regime | Precisa de 20 pregões alinhados (por volta de 4/12/2026) | Aguardando dados |
| Veredito do Jinshi para os desafiantes de volatilidade | Por volta de 16/12/2026 | Aguardando dados |
| Email mensal (vereditos do Jinshi, Diebold-Mariano, mudanças) | Leitura de horizonte longo | Planejado |
| Tirar logs e estado do git para um Postgres pequeno, com uma API só de leitura | O repositório como banco de dados causou a maior parte dos incidentes | Planejado (plano gratuito) |
| "Pergunte à Carteira": um agente com LLM que responde perguntas sobre o log em SQL, avaliado contra as respostas que o pipeline já calcula | Acesso em linguagem natural à trilha de auditoria, com conjunto de testes embutido | Planejado (LLM gratuito ou local) |


> **Princípio:** Estabilidade → Observabilidade → Publicação → Modelo → Features avançadas. Nunca o contrário. Um modelo melhor num pipeline instável produz resultados melhores em que não podes confiar.

## Semana 1 — Estabilidade do pipeline

Antes de qualquer melhoria de modelo, o pipeline tem de ser à prova de falha.

| # | Item | Descrição |
|---|------|-----------|
| 1 | ✅ Retry no git push | 5 linhas no YAML. Elimina perda de dados por falha transitória no push. |
| 2 | ✅ Artefacto GitHub em push falhado | Upload do `predictions_log.csv` como artefacto do workflow se o push falhar — safety net para recuperação manual. |
| 3 | ✅ Forward fill para NaN em VIX/SPY | Usar o valor T-1 se T-0 devolver NaN; adicionar aviso no email quando acontece. |
| 4 | ✅ Detecção de stock split | Variação >40% num dia marca validações em aberto como `NaN` em vez de `False` — evita penalizar modelos por acção corporativa. |

## Semana 2 — Observabilidade

O pipeline corre mas não te diz quando está a degradar. Isso muda aqui.

| # | Item | Descrição |
|---|------|-----------|
| 5 | ✅ Feature importance drift como alerta | O `model_metadata.csv` já guarda as importances diárias — lê o ficheiro e adiciona um painel de drift no email com correlação de Spearman. |
| 6 | ⬜ Telegram como fallback de email | ~20 linhas; activa quando o Gmail falha — garante que o relatório diário é sempre entregue. *(adiado para o final)* |
| 7 | ✅ Badge dinâmico no repo público | Badge `shields.io/github/last-commit` — actualiza automaticamente em cada visualização. |

## Semana 3 — Repo público completo

Preparar tudo para a publicação.

| # | Item | Descrição |
|---|------|-----------|
| 8 | ✅ `predictions_log_public.csv` | Versão anonimizada do log (sem tickers, sem preços) — prova de acurácia real para qualquer pessoa que abra o repo público. |
| 9 | ✅ Secção "Confiabilidade" no README | Agrupar testes unitários + fallbacks + retry numa secção única. |
| 10 | ✅ Git tags semânticos | Tag em cada marco de versão (`v1.0.0`, `v1.1.0`, …) para ancorar o changelog no histórico git. |

## Semanas 4–5 — Qualidade do modelo

Só aqui. Não antes. O pipeline tem de estar estável antes de mexer no modelo.

| # | Item | Descrição |
|---|------|-----------|
| 11 | ✅ Walk-Forward Validation | `TimeSeriesSplit(n_splits=5, gap=1)` — `gap=1` impede lookahead entre o fold de treino e o de validação, produzindo estimativas de acurácia mais honestas. |
| 12 | ✅ Regime de mercado como feature | Label `vix_regime` (0 / 1 / 2) com base em thresholds de VIX (< 15 / 15–25 / ≥ 25); adicionada a `FEATURE_COLS`; os modelos SGD reinicializam automaticamente ao detectar mudança no número de features. |

## Semana 6 — Relatório

| # | Item | Descrição |
|---|------|-----------|
| 13 | ✅ Matriz de correlação no email | Heatmap de correlação de retornos diários da carteira (janela de 120 dias) embebido como PNG base64 no email HTML; fallback silencioso se matplotlib não estiver disponível. |
| 14 | ✅ Cenários de projeção ETF | A coluna 10 anos mostra três cenários fixos: Pess = 3% · Base = 8% · Otim = 15%; elimina a taxa histórica enganadora para tickers com historial curto ou não representativo. |

## Semanas 7–8 — Publicação

| # | Item | Descrição |
|---|------|-----------|
| 15 | ✅ READMEs finais (EN + PT) | Revisão completa de ambos os READMEs, incluindo `README_educativo.md` (PT-BR) com todas as features novas e roadmap atualizado. |
| 16 | ✅ Lançamento do repo público | `smart-wallet-ml` publicado com `predictions_log_public.csv`, gráficos com atraso de 10 dias e README gerado automaticamente. |
| 17 | ⬜ Artigo LinkedIn | Contar a história do projecto — do notebook ao pipeline ML automatizado. *(conteúdo externo, não código — sem prazo fixo)* |

## Depois da publicação — sem prazo fixo

| # | Item | Descrição |
|---|------|-----------|
| 18 | ⬜ Features de eventos fundamentalistas | Datas de earnings, semanas FOMC, expiração de opções. Requer API externa fiável; cobertura europeia é limitada. |
| 19 | ⬜ Regressor de preço D+1 | Só com 1 ano de dados limpos acumulados. |

## Framework de investigação — 38 modelos em 13 famílias

Os 25 modelos originais estão implementados e testados. As camadas transversais (fases 9-15) estão completas. As fases 16-20 estão completamente implementadas.

| Fase | Família | Modelos | Estado |
|------|---------|---------|--------|
| 0 | Clássico base | RF, GB, SGD | ✅ |
| 1 | Estado oculto | Markov, HMM | ✅ |
| 2 | Clássico avançado | XGBoost, LightGBM, CatBoost, SVM | ✅ |
| 3 | Séries temporais | ARIMA, SARIMA, ETS, Holt-Winters, Prophet | ✅ |
| 4 | Neural recorrente | LSTM, GRU | ✅ |
| 5 | Neural com atenção | Transformer, TFT, N-BEATS | ✅ |
| 6 | Bayesiano | Gaussian Process, BNN (MC Dropout) | ✅ |
| 7 | Generativo | VAE, GAN | ✅ |
| 8 | Reinforcement | DQN, PPO | ✅ |
| 9 | Avaliação | Diebold-Mariano, McNemar, Ljung-Box, métricas por domínio | ✅ |
| 10 | Explicabilidade | SHAP, Attention weights, LIME | ✅ |
| 11 | Meta-learning | Stacking, Optuna | ✅ |
| 12 | Rastreamento | MLflow (pipeline diário + research runner) | ✅ |
| 13 | Teoria da informação | Entropia de Shannon, MI, Permutation entropy, Transfer entropy | ✅ |
| 14 | Transfer Learning | MMD, CORAL, Fine-tuning, Cross-domain evaluation | ✅ |
| 15 | Email report | Tabela eToro por lote, secção ETF, recomendação mensal | ✅ |
| 16 | Contrarian / Testes de sanidade | CB (Contrarian Baseline), EWI, PEL | ✅ |
| 17 | Arquitecturas eficientes (pós-2022) | TCN, DLinear, NLinear, PatchTST | ✅ |
| 18 | Foundation Models (2023-2024) | Chronos (Amazon), TimesFM (Google), Moirai (Salesforce) | ✅ |
| 19 | Incerteza calibrada | Conformal Prediction (implementação própria; o MAPIE saiu em agosto de 2026, ver ERR-009) | ✅ |
| 20 | Detecção de drift | ADWIN, Page-Hinkley | ✅ |


## Novembro de 2026: honestidade estatística e volatilidade

| # | Item | Estado |
|---|------|--------|
| 1 | Confiança calibrada no email (encolhimento até 20 dias, depois isotônica) | ✅ / ⏳ isotônica por volta de 4/12 |
| 2 | IC, referência "sempre ▲", p-valores e HLN em todas as leituras | ✅ |
| 3 | Saúde do sistema, eventos e risco no email diário | ✅ |
| 4 | Email semanal | ✅ |
| 5 | Rastreabilidade das previsões | ✅ |
| 6 | Ensemble ampliado e desafiante de regime no Jinshi | ⏳ por volta de 4/12 |
| 7 | Desafiantes de volatilidade no Jinshi | ⏳ por volta de 16/12 |
| 8 | Email mensal (veredito do Jinshi, Diebold-Mariano, mudanças) | ⬜ |
| 9 | Consistência do ouro (PPFB.DE × SGLN.L) e inflação da zona do euro no calendário | ⬜ |
