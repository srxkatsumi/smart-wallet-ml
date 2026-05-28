# /roadmap — Guardião do roadmap e da documentação do projeto Carteira Inteligente

Quando esta skill for invocada, execute **todos** os passos abaixo em ordem.

---

## CONTEXTO DO PROJETO

Tens acesso completo ao repositório GitHub srxkatsumi/smart_wallet.
Começa sempre por ler a versão mais recente de todos os ficheiros relevantes antes de qualquer alteração. Nunca uses versões em memória.

O projeto é o **Carteira Inteligente**, um sistema de MLOps de previsão de carteira de investimentos construído do zero. Escrito em Python 3.11, corre via GitHub Actions todos os dias úteis às 22h00 (Barcelona), e usa um ensemble de RF + GB + SGD com pesos adaptativos e decaimento temporal para prever a direção de preços em D+1/D+2/D+3.

---

## REGRAS GLOBAIS PARA TODOS OS READMES

Estas regras aplicam-se a qualquer atualização de README, sem exceções:

- Nunca usar travessões em texto corrido nem em listas. Usar bullets (-) ou numeração para listas. Em texto corrido usar vírgulas, ponto e vírgula ou frases separadas.
- Todos os READMEs devem estar escritos em primeira pessoa, pois o projeto foi construído por uma única pessoa.
- README.md em inglês técnico.
- README_pt.md e README_educativo.md em português brasileiro, não europeu. ("arquivo" não "ficheiro", "atualizar" não "actualizar", "portfólio" mantém-se, termos técnicos ML mantêm-se em inglês).
- Horário de execução sempre como 22h00 (Barcelona) em todos os ficheiros.

---

## PASSO 1 — Ler o estado atual do roadmap

Leia `README.md` e identifique:
- Quantos itens estão ✅ concluídos
- Quantos itens estão ⬜ pendentes e quais são
- Em que semana do roadmap o projeto se encontra agora

---

## PASSO 2 — Verificar coerência entre os READMEs

Leia `README_pt.md` e verifique:
- Todos os itens ✅ no `README.md` estão também ✅ no `README_pt.md`?
- Todos os itens do changelog do `README.md` existem no `README_pt.md` com tradução completa?
- As secções de automação, timing e explicações técnicas estão sincronizadas entre os dois ficheiros?
- O horário aparece como 22h00 em ambos?
- Existem travessões em algum dos dois?

Se houver divergências, listá-las claramente antes de continuar.

---

## PASSO 3 — Próximos passos com prazo estimado

Com base nos itens ⬜ pendentes, apresentar:

1. **O próximo item a implementar** com nome, descrição e impacto esperado.
2. **Os 3 itens seguintes** na fila para contexto de sequência.
3. **Alertas de prazo** se algum item da semana atual ainda está ⬜ quando já devia estar concluído.

Usar sempre o princípio do roadmap: **Estabilidade → Observabilidade → Publicação → Modelo → Features avançadas**. Nunca propor um item de modelo se houver itens de estabilidade ou observabilidade pendentes.

---

## PASSO 4 — Checklist de README para a próxima implementação

Antes de qualquer implementação, gerar um checklist específico para o próximo item. Os READMEs devem ser atualizados no mesmo commit que o código.

```
README UPDATE CHECKLIST — [nome do próximo item]

README.md:
  [ ] Adicionar entrada no Changelog com descrição detalhada em inglês
  [ ] Marcar item como ✅ no Roadmap
  [ ] Atualizar qualquer secção técnica afetada (Automation, Features, Mermaid, etc.)
  [ ] Verificar se a estrutura de repositório precisa de atualização
  [ ] Confirmar que não há travessões introduzidos
  [ ] Confirmar que está em primeira pessoa

README_pt.md:
  [ ] Adicionar entrada no Changelog com tradução completa em português brasileiro
  [ ] Marcar item como ✅ no Roadmap
  [ ] Atualizar qualquer secção técnica afetada (mesmas secções do EN)
  [ ] Verificar coerência de terminologia com entradas anteriores
  [ ] Confirmar vocabulário PT-BR ("arquivo", "atualizar", etc.)
  [ ] Confirmar que não há travessões introduzidos

README_educativo.md (se existir):
  [ ] Verificar se o item implementado afeta alguma secção explicativa
  [ ] Atualizar exemplos concretos se necessário
  [ ] Verificar limitações honestas (secção 11) se relevante

VALIDAÇÃO FINAL:
  [ ] Ambos os READMEs têm o mesmo número de itens ✅ no roadmap
  [ ] Ambos os changelogs têm o mesmo número de entradas
  [ ] Nenhuma secção técnica ficou com informação desatualizada
  [ ] Horário 22h00 correto em todos os ficheiros
```

---

## PASSO 5 — Quando for pedido para atualizar ou criar READMEs

Se o utilizador pedir para atualizar os READMEs ou criar o README_educativo.md, seguir as instruções abaixo.

### README.md (Inglês técnico) — ATUALIZAR, NÃO SUBSTITUIR

Ler o README.md atual e aplicar apenas as seguintes alterações, preservando todo o conteúdo existente:

1. Atualizar o horário de execução para 22h00 (Barcelona) em todas as ocorrências.

2. Remover todos os travessões em texto corrido e em listas. Substituir por vírgulas, ponto e vírgula ou bullets conforme o contexto.

3. Adicionar um diagrama Mermaid completo com dois blocos imediatamente após a secção Overview:

   **BLOCO A: Daily Pipeline**
   ```mermaid
   flowchart TD
       A["⏰ GitHub Actions\n3 crons Mon–Fri\nanti-duplication check"] --> B["🔍 Duplicate check\nreads predictions_log.csv\nskip if today exists"]
       B -->|not yet run| C["📥 Price download\nbatches of 20 tickers\n2s sleep between batches"]
       C --> D["🔄 Forward fill\nVIX/SPY NaN detection\namber warning in email"]
       D --> E["⚙️ Feature engineering\nSMA20/50 · RSI14 · MACD\nBollinger · ATR14 · ret_1d/5d\nspy_ret_1d · vix_level · vix_change"]
       E --> F["✅ Past forecast validation\nref_price vs actual_price\nstock split detection >40%"]
       F --> G["⚖️ Ensemble weight update\nexponential temporal decay\nrolling 30-day window"]
       G --> H["🤖 Train 3 independent ensembles\nD+1 · D+2 · D+3"]
       H --> I["💾 Save new forecasts\nD+1 / D+2 / D+3\npredictions_log.csv"]
       I --> J["📊 Chart generation\n12 charts per day\ncleanup after 30 days"]
       J --> K["📧 HTML email build\nML table · P&L · accuracy\ndrift alert · charts"]
       K --> L["🔁 Git commit with retry\n3 attempts · 15s pause\ngit pull --rebase"]
       L -->|push fails| M["🛟 Safety artifact\npredictions_log.csv\nretained 7 days"]
       L -->|push ok| N["📤 Gmail SMTP send\nHTML email\nfailure email on error"]
       N --> O["🌐 Public repo sync\n10-day delay window\nanonymised CSV + charts"]
       O --> P["📉 Drift alert check\nSpearman ρ RF features\nflag if ρ < 0.70"]
   ```

   **BLOCO B: ML Architecture**
   ```mermaid
   flowchart TD
       F["Input features\nSMA · RSI · MACD · Bollinger\nATR · returns · VIX · SPY"] --> E1["D+1 Ensemble"]
       F --> E2["D+2 Ensemble"]
       F --> E3["D+3 Ensemble"]
       E1 --> RF1["🌲 Random Forest\n300 trees · max_depth 6"]
       E1 --> GB1["📈 Gradient Boosting\n200 est. · lr 0.05"]
       E1 --> SG1["📐 SGD Classifier\nlog_loss · monthly recal."]
       RF1 & GB1 & SG1 --> W1["⚖️ Adaptive weights\nexponential decay\nrolling 30-day accuracy"]
       W1 --> P1["Forecast D+1\nUP/DOWN + confidence"]
       E2 --> W2["⚖️ Adaptive weights D+2"]
       E3 --> W3["⚖️ Adaptive weights D+3"]
       W2 --> P2["Forecast D+2"]
       W3 --> P3["Forecast D+3"]
       P1 & P2 & P3 --> V["✅ Validation\nactual_price vs ref_price\ncorrect = True/False/NaN"]
       V --> WU["🔄 Weight update\nmore recent = more weight\nweight ∝ accuracy × Σdecay^t"]
   ```

4. Manter o changelog e roadmap existentes sem alterações de conteúdo. Apenas remover travessões se existirem.

---

### README_pt.md (Português brasileiro técnico) — ATUALIZAR, NÃO SUBSTITUIR

Ler o README_pt.md atual e aplicar apenas as seguintes alterações, preservando todo o conteúdo existente:

1. Atualizar o horário de execução para 22h00 (Barcelona) em todas as ocorrências.

2. Remover todos os travessões em texto corrido e em listas.

3. Adicionar os mesmos diagramas Mermaid com labels em português brasileiro.

4. Confirmar que o texto está em português brasileiro, não europeu. Substituir: "ficheiro" por "arquivo", "actualizar" por "atualizar", "portfólio" mantém-se, "perceção" por "percepção" (BR), etc.

5. Manter todos os termos técnicos em inglês: ensemble, drift, Random Forest, Gradient Boosting, pipeline, backtest, etc.

6. Manter changelog e roadmap existentes sem alterações de conteúdo.

---

### README_educativo.md — CRIAR NOVO ARQUIVO

Este README é educativo e explica o projeto a alguém com conhecimentos básicos de programação mas sem background de ML ou finanças. Escrito em português brasileiro, em primeira pessoa, sem travessões.

**Estrutura obrigatória:**

**1. Introdução pessoal**
Por que construí este projeto. O que fazia antes (analisar manualmente) e o que o sistema faz agora.

**2. Como o sistema funciona no dia a dia**
Explicação simples do pipeline diário em linguagem acessível. Incluir o diagrama Mermaid da pipeline com labels em português brasileiro simples, sem jargão.

**3. O que é Machine Learning neste contexto**
Explicar classificação binária de forma simples com exemplos reais dos ativos (NVDA, BTC, ALV.DE).

**4. Os três modelos do ensemble e por que escolhi cada um**
Para cada modelo explicar em linguagem simples: o que faz, por que está no ensemble, o que seria diferente sem ele. Incluir o diagrama Mermaid da arquitetura ML.

**5. As features: o que o modelo vê**
Para cada feature explicar: o que mede em linguagem simples, como é calculada (fórmula simplificada), quem criou este indicador e quando, por que a incluí no modelo com exemplo concreto.
- ATR: J. Welles Wilder Jr., 1978
- RSI: J. Welles Wilder Jr., 1978
- MACD: Gerald Appel, 1979
- Bollinger Bands: John Bollinger, anos 1980
- SMA: uso histórico sem autor único definido

**6. Como os pesos adaptativos funcionam**
Explicar o decaimento exponencial em linguagem simples.
Fórmula: `weight(model) ∝ accuracy(model) × Σ decay^(days_ago)`
Explicar cada termo da fórmula. Dar um exemplo numérico concreto (ex: RF acertou 6 de 10, GB acertou 4 de 10, como ficam os pesos).

**7. Como as previsões são validadas**
Explicar ref_price, actual_price e o cálculo de correct.
Explicar por que o bug do pred_price causou 23% de acurácia e como o ref_price o corrigiu.
Fórmulas:
- `correct = actual >= ref_price` (UP)
- `correct = actual <= ref_price` (DOWN)

**8. As métricas do email explicadas**

ACURÁCIA GERAL
- Fórmula: `correct_count / total_validated × 100`
- Parâmetro: janela de 30 dias úteis, apenas portfólio
- Como se lê: acima de 52% indica que o modelo tem edge real

VAR%
- Fórmula: `(preço_hoje - preço_ontem) / preço_ontem × 100`
- Como se lê: variação de fechamento para fechamento

CONFIDENCE (ex: ▼ 70%)
- Fórmula: `Σ (peso_modelo × prob_modelo)` para cada modelo
- Como se lê: 70% significa que o ensemble tem 70% de convicção na direção prevista

CONSENSO (BULLISH / BEARISH / MISTO)
- BULLISH se D+1 + D+2 + D+3 todos UP
- BEARISH se D+1 + D+2 + D+3 todos DOWN
- MISTO se há discordância entre horizontes

✅/❌ POR ATIVO
- O que significa: se a previsão D+1 que apontava para ontem estava correta
- Como é calculado: compara target_date de ontem com actual_price disponível

ρ DE SPEARMAN (feature drift)
- Explicar correlação de Spearman em linguagem simples
- Fórmula: `ρ = 1 - (6 × Σd²) / (n × (n²-1))`
  onde d = diferença de rank entre hoje e referência e n = número de features
- Criado por Charles Spearman, psicólogo britânico, 1904
- Como se lê: ρ próximo de 1 = modelo estável, ρ < 0.70 = regime de mercado pode ter mudado

ATR ESTIMADO (pred_price)
- Fórmula: `close ± ATR × 0.5 × √horizon`
- Criado por J. Welles Wilder Jr., 1978
- Como se lê: estimativa informacional do preço alvo, não é uma previsão de preço

**9. Por que calendários por exchange são importantes**
Explicar o problema dos feriados com exemplo concreto (NYSE fecha no Thanksgiving, LSE não). Sem pandas-market-calendars o modelo procuraria preços em dias sem negociação e corromperia o audit trail.

**10. O audit trail: por que nunca apago nada**
Explicar a filosofia do predictions_log.csv imutável. Por que é o ativo mais valioso do projeto. Como permitiu detectar o bug do ref_price retroativamente.

**11. Limitações honestas**
O que o modelo não sabe fazer:
- Não lê notícias nem eventos fundamentalistas
- Não sabe que há earnings amanhã
- Não distingue uma queda de 0.1% de uma de 8%
- A acurácia ainda está em fase de validação e pode estar abaixo de 50%
Ser honesto sobre o que o sistema ainda não consegue.

**12. Roadmap em linguagem simples**
O roadmap técnico traduzido para o que cada item significa na prática para um utilizador comum.

---

## PASSO 6 — Resumo executivo

Terminar sempre com este bloco:

```
ESTADO DO PROJETO — [data de hoje]

Semana atual:       Semana X — [nome da fase]
Itens concluídos:   X de Y
Próximo item:       #N — [nome]
READMEs:            ✅ Sincronizados / ⚠️ Divergências encontradas
Travessões:         ✅ Nenhum / ⚠️ Encontrados em [ficheiro]
Horário:            ✅ 22h00 correto / ⚠️ Desatualizado em [ficheiro]

PRÓXIMO PASSO: [uma frase clara com o que fazer a seguir]
```
