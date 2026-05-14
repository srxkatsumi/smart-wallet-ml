# 📈 Carteira Inteligente v5

> Projeto pessoal de análise e previsão de carteira de investimentos, construído do zero por uma analista de dados que investe o seu próprio dinheiro e quis perceber melhor o que está a acontecer com ele.

---

## O que é isto?

Tenho uma carteira de investimentos dividida entre ações (eToro) e ETFs de acumulação de longo prazo. Durante um tempo geria tudo manualmente — abria apps, via números a vermelho e a verde, e tomava decisões no feeling.

Decidi mudar isso.

Este projeto é um notebook Python que executa automaticamente todos os dias úteis às 15h30 (Barcelona) e faz três coisas:

1. **Analisa a carteira** — Ganho/Perda real em euros, breakeven com fees incluídos, alvo de saída
2. **Prevê a direção dos próximos 3 dias** — Usando Machine Learning (Random Forest, Gradient Boosting, Logistic Regression em ensemble) com indicadores técnicos + contexto de mercado (VIX, SPY)
3. **Aprende com os erros** — Cada previsão é validada quando a data chega. O modelo que acertar mais passa a ter mais peso nas decisões seguintes

Não é um oráculo. É um sistema que fica menos burro a cada dia que passa.

---

## Carteira coberta

**eToro (ações):**
- Eli Lilly (LLY)
- NVIDIA (NVDA)
- Allianz (ALV.DE)
- Bitcoin (BTC-USD)
- Alibaba ADR (BABA)

**ETFs de acumulação (longo prazo):**
- MSCI World ex USA (EXUS.L)
- MSCI China (ICGA.DE)
- Physical Gold (SGLN.L)
- MSCI EM IMI (EMIM.AS)
- Stoxx Europe 600 (MEUD.PA)
- MSCI Japan IMI (SJPA.MI)

---

## Stack técnica

```
Python 3.11
├── yfinance          — dados de mercado em tempo real
├── scikit-learn      — Random Forest, Gradient Boosting, Logistic Regression
├── pandas / numpy    — processamento de dados
└── matplotlib        — visualizações

GitHub Actions        — execução automática diária (gratuito)
```

---

## Funcionalidades

### Modelos ML independentes por horizonte
Ao contrário da maioria dos exemplos que encontras online, este sistema treina **3 ensembles separados** — um para D+1, outro para D+2, outro para D+3. Cada um aprende o seu próprio padrão. Não extrapolam uns dos outros.

### Contexto de mercado
Além dos indicadores técnicos do próprio ativo, cada modelo recebe o **VIX e o retorno do SPY do dia anterior** como features. A ideia é que a NVDA num dia de pânico global se comporta diferente da NVDA num dia neutro.

### Pesos adaptativos com decay temporal
O ensemble atualiza os pesos de cada modelo com base no histórico real de acertos. Previsões mais recentes pesam mais do que previsões antigas (decay exponencial). Se o Random Forest começar a errar sistematicamente, o sistema aprende a confiar menos nele.

### Validação automática
Cada previsão feita fica guardada no CSV com `actual_price = NaN`. Quando a data chega, o sistema preenche o preço real e calcula se acertou. Tudo auditável, nada apagado.

---

## Estrutura do repositório

```
├── PrevisaoCarteira_v5.ipynb        ← notebook principal (14 blocos)
├── AnaliseV5/
│   ├── predictions_log.csv          ← histórico completo de previsões
│   ├── ensemble_weights.json        ← pesos atuais de cada modelo
│   └── AnaliseGraficos/
│       └── TICKER_v5_YYYYMMDD.png  ← gráfico por ativo por dia
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← agendamento automático
└── README.md
```

---

## Como corre automaticamente

```
Seg-Sex 15h30 Barcelona
  │
  ├─ GitHub Actions clona o repositório
  ├─ Instala dependências
  ├─ Executa o notebook completo (~8 minutos)
  ├─ Valida previsões anteriores
  ├─ Guarda novas previsões (D+1, D+2, D+3 para cada ativo)
  ├─ Atualiza gráficos
  └─ Commit automático com timestamp → push para o repositório
```

Se falhar, o GitHub envia email de notificação.

---

## Output do Bloco 14 (exemplo)

```
Ativo        Preço       D+1          D+2          D+3      Consenso
SJPA.MI      67.76    UP(60%)      UP(61%)      UP(66%)   📈 BULLISH
ALV.DE      380.80    UP(56%)      UP(65%)      UP(66%)   📈 BULLISH
NVDA        231.60  DOWN(55%)    DOWN(55%)    DOWN(60%)   📉 BEARISH
BTC-USD   79822.72    UP(59%)      UP(55%)      UP(63%)   📈 BULLISH
```

---

## Aviso importante

> As previsões são probabilísticas. 55–60% de acurácia direcional já está acima do aleatório mas não é suficiente para tomar decisões financeiras isoladamente. Este projeto é uma ferramenta de análise pessoal, não aconselhamento financeiro.

---

## Sobre

Construído por **Vicky Costa** — Analista de Dados | Estudante de Ciência de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)