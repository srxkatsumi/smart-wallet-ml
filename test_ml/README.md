# test_ml — Laboratório de Machine Learning

Quatro áreas de investigação independentes que partilham infraestrutura de modelos e protocolo de validação.

## Projetos

### 🎱 [loteria/](loteria/) — Mega Sena ML Experiment

Aplicação do mesmo ensemble adaptativo (RF + GB + SGD) da Carteira Inteligente a um processo declaradamente aleatório: a loteria Mega Sena brasileira.

| | |
|---|---|
| **Objetivo** | Provar experimentalmente que ML não consegue superar o acaso num processo i.i.d. |
| **Dados** | 3000+ sorteios históricos (1996 → presente), 60 bolas, 6 por sorteio |
| **Execução** | Automática, todos os dias úteis via GitHub Actions |
| **Horizonte** | 5 anos, material para investigação de doutoramento |
| **Resultado atual** | Ensemble: 0.70 matches/sorteio vs baseline aleatório: 0.60 (+16%) |

O projeto documenta formalmente que, mesmo com 9 famílias de modelos (RF/GB/SGD, Markov/HMM, LSTM/GRU, Transformer, Bayesiano, etc.), nenhum supera consistentemente o baseline aleatório em dados genuinamente i.i.d.

---

### 📈 [carteira/](carteira/) — Experimentos ML para a Carteira Inteligente

Área reservada para prototipagem e validação de algoritmos destinados ao pipeline diário da Carteira Inteligente.

| | |
|---|---|
| **Objetivo** | Validar algoritmos antes de promovê-los ao pipeline de produção |
| **Dados** | Preços diários de 95 ativos (ações, ETFs, cripto), horizonte D+1/D+2/D+3 |
| **Estado atual** | Área preparada; experimentos ativos ficam em `novo_algoritmo/` |

---

### 🔬 [novo_algoritmo/](novo_algoritmo/) — Inventário do Especialista

Onde o Especialista guarda as análises, backtests e diários de investigação de algoritmos novos. Cada ciclo de investigação segue 5 fases: Problemas, Hipóteses, MVP, Validação, Iteração.

| | |
|---|---|
| **Ciclo 1** | ContextGate concluído e validado (+6.1 pp com voto unânime) |
| **Ciclo 2** | Agendado para 2026-07-31 |

---

### 🌦️ [clima/](clima/) — Previsão Meteorológica (planeado)

Área reservada para um futuro experimento de previsão de clima. Vazia por enquanto.

---

## O que os projetos têm em comum

Todos usam:
- **Mesmo ensemble base:** RF + GB + SGD com pesos adaptativos
- **Mesmo protocolo de validação:** walk-forward sem lookahead
- **Mesmas famílias de modelos:** clássico, neural, bayesiano, séries temporais, contrarian
- **Mesma métrica de qualidade:** acurácia normalizada vs baseline ingênuo

A diferença fundamental é o **sinal no dado**:
- Loteria: sem sinal (i.i.d.), ML não melhora, prova negativa documentada
- Carteira: sinal existe, ML melhora, acurácia atual 53.8% vs 50% aleatório

---

## Estrutura

```
test_ml/
├── README.md                  ← este arquivo
├── loteria/                   ← Mega Sena ML Experiment
│   ├── main.py                ← runner semanal automático
│   ├── models/                ← 9 famílias de modelos
│   ├── features/              ← engenharia de features por bola/sorteio
│   ├── data/                  ← download e armazenamento
│   ├── output/                ← resultados, previsões, pesos
│   ├── README.md              ← documentação em inglês
│   └── README_pt.md           ← documentação em português
├── carteira/                  ← experimentos para o pipeline da Carteira
│   └── (vazio, área reservada)
├── novo_algoritmo/            ← análises e diários do Especialista
│   ├── NewAlgoritmo.md        ← diário de investigação (5 fases)
│   ├── context_gate.py        ← implementação do ContextGate
│   └── backtest_gate.py       ← validação experimental
└── clima/                     ← previsão meteorológica (planeado, vazio)
```
