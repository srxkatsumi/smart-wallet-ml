# test_ml — Laboratório de Machine Learning

Dois projetos de ML independentes que partilham a mesma infraestrutura de modelos e pipeline de validação.

---

## Projetos

### 🎱 [loteria/](loteria/) — Mega Sena ML Experiment

Aplicação do mesmo ensemble adaptativo (RF + GB + SGD) da Carteira Inteligente a um processo declaradamente aleatório: a loteria Mega Sena brasileira.

| | |
|---|---|
| **Objetivo** | Provar experimentalmente que ML não consegue superar o acaso num processo i.i.d. |
| **Dados** | 3000+ sorteios históricos (1996 → presente), 60 bolas, 6 por sorteio |
| **Execução** | Automática — todos os dias úteis via GitHub Actions |
| **Horizonte** | 5 anos — material para investigação de doutoramento |
| **Resultado atual** | Ensemble: 0.70 matches/sorteio vs baseline aleatório: 0.60 (+16%) |

O projeto documenta formalmente que, mesmo com 9 famílias de modelos (RF/GB/SGD, Markov/HMM, LSTM/GRU, Transformer, Bayesiano, etc.), nenhum supera consistentemente o baseline aleatório em dados genuinamente i.i.d.

---

### 📈 [carteira/](carteira/) — Experimentos ML para a Carteira Inteligente

Investigação e prototipagem de algoritmos novos para melhorar o pipeline da Carteira Inteligente (previsão diária de ativos financeiros).

| | |
|---|---|
| **Objetivo** | Inventar e validar algoritmos que melhorem a acurácia do ensemble da Carteira |
| **Dados** | Preços diários de 95 ativos (ações, ETFs, cripto) — horizonte D+1/D+2/D+3 |
| **Metodologia** | Ciclo de 5 fases: Problemas → Hipóteses → MVP → Validação → Iteração |
| **Estado atual** | Ciclo 1 concluído — ContextGate validado; Ciclo 2 agendado para 2026-07-31 |

#### Subprojetos

| Pasta | O que é |
|---|---|
| [`carteira/novo_algoritmo/`](carteira/novo_algoritmo/) | ContextGate — porteiro contextual que aprende quais modelos favorecer conforme o contexto do mercado |

---

## O que têm em comum

Ambos os projetos usam:
- **Mesmo ensemble base:** RF + GB + SGD com pesos adaptativos
- **Mesmo protocolo de validação:** walk-forward sem lookahead
- **Mesmas famílias de modelos:** clássico, neural, bayesiano, séries temporais, contrarian
- **Mesma métrica de qualidade:** acurácia normalizada vs baseline ingênuo

A diferença fundamental é o **sinal no dado**:
- Loteria → sem sinal (i.i.d.) → ML não melhora → prova negativa documentada
- Carteira → sinal existe → ML melhora → acurácia atual 53.8% vs 50% aleatório

---

## Estrutura

```
test_ml/
├── README.md              ← este ficheiro
├── loteria/               ← Mega Sena ML Experiment
│   ├── main.py            ← runner semanal automático
│   ├── models/            ← 9 famílias de modelos
│   ├── features/          ← engenharia de features por bola/sorteio
│   ├── data/              ← download + armazenamento
│   ├── output/            ← resultados, previsões, pesos
│   ├── README.md          ← documentação do projeto em inglês
│   └── README_pt.md       ← documentação em português
└── carteira/
    └── novo_algoritmo/    ← ContextGate + backtests
        ├── NewAlgoritmo.md    ← diário de investigação (5 fases)
        ├── context_gate.py    ← implementação do ContextGate
        └── backtest_gate.py   ← validação experimental
```
