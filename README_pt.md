# Carteira Inteligente — Sistema de Previsão de Carteira com Machine Learning

> Um sistema de análise e previsão de carteira de investimentos que aprende sozinho, construído do zero.
> Executa automaticamente todos os dias úteis às 15h35 (Barcelona / CET+1) via GitHub Actions.

[![GitHub Actions](https://img.shields.io/badge/Automatizado-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart_wallet/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

---

## O que é isto?

Tenho uma carteira de investimentos dividida entre ações (eToro) e ETFs de acumulação de longo prazo. Durante um tempo geria tudo manualmente — abria apps, via números a vermelho e a verde, e tomava decisões no feeling.

Decidi mudar isso.

Este projeto é um notebook Python que executa automaticamente todos os dias úteis às 15h35 (Barcelona) e faz três coisas:

1. **Analisa a carteira** — Ganho/Perda real em euros, breakeven com fees incluídos, alvo de saída
2. **Prevê a direção dos próximos 3 dias** — Usando Machine Learning (Random Forest, Gradient Boosting, SGD em ensemble) com indicadores técnicos + contexto de mercado (VIX, SPY)
3. **Aprende com os erros** — Cada previsão é validada quando a data chega. O modelo que acertar mais passa a ter mais peso nas decisões seguintes

Não é um oráculo. É um sistema que fica menos burro a cada dia que passa.

---

## Como o Machine Learning funciona aqui

### O problema que resolvemos

A pergunta que o modelo tenta responder é simples: **"o preço vai subir ou cair nos próximos N dias?"**

Isso é o que em ML chamamos de classificação binária:
- `1` = sobe (UP)
- `0` = cai (DOWN)

E fazemos isso para 3 horizontes de tempo independentes: amanhã (D+1), depois de amanhã (D+2) e em 3 dias úteis (D+3).

### Por que 3 modelos separados e não um só?

A maioria dos exemplos online treina um único modelo e depois "extrapola" os resultados para D+2 e D+3. Isso é um erro. O padrão que faz uma ação subir amanhã é diferente do padrão que faz ela subir em 3 dias. Por isso treinamos ensembles totalmente independentes:

```
Ensemble D+1 ──► Random Forest + Gradient Boosting + SGD ──► pesos_d1
Ensemble D+2 ──► Random Forest + Gradient Boosting + SGD ──► pesos_d2
Ensemble D+3 ──► Random Forest + Gradient Boosting + SGD ──► pesos_d3
```

### Os 3 algoritmos usados e por que cada um

| Algoritmo | Para que serve aqui |
|-----------|---------------------|
| **Random Forest** (300 árvores, profundidade 6) | É o "generalista robusto". Resiste bem ao overfitting e captura padrões não-lineares. Funciona como ancora de estabilidade do ensemble. |
| **Gradient Boosting** (200 estimadores, taxa 0.05) | É o "especialista em detalhes". Aprende os padrões que o Random Forest não consegue capturar, especialmente sinais de momentum de curto prazo. |
| **SGD Classifier** (função de perda logarítmica) | É a "âncora linear". Um modelo simples que impede que o ensemble fique louco com ruído de curto prazo. Recalibrado do zero todo mês. |

### As features (o que o modelo "vê" de cada ativo)

Além do preço histórico, o modelo recebe uma série de indicadores técnicos calculados automaticamente:

| Feature | O que representa |
|---------|------------------|
| `sma_20`, `sma_50` | Médias móveis de 20 e 50 dias — tendência de curto e médio prazo |
| `rsi_14` | RSI de 14 dias — diz se o ativo está sobrecomprado (>70) ou sobrevendido (<30) |
| `macd`, `macd_signal` | Indicador de momentum — captura cruzamentos de tendência |
| `bb_upper`, `bb_lower`, `bb_width` | Bandas de Bollinger — volatilidade e posição do preço em relação à banda |
| `atr_14` | Average True Range — magnitude esperada do movimento diário |
| `ret_1d`, `ret_5d` | Retorno dos últimos 1 e 5 dias |
| `spy_ret_1d` | Retorno do S&P 500 no dia anterior — contexto global do mercado |
| `vix_level` | Nível do VIX no dia anterior — termômetro do medo do mercado |
| `vix_change` | Variação do VIX no dia anterior — aceleração do medo |

**Por que o VIX e o SPY?** A NVDA num dia de pânico global se comporta de forma muito diferente da NVDA num dia neutro. Ao incluir essas features, o modelo consegue condicionar a sua previsão ao estado emocional geral do mercado.

**Prevenção de data leakage:** todas as features de contexto usam valores do dia anterior (T-1). Quando os mercados europeus abrem às 15h35, só existe o fechamento de NY do dia anterior. Usar dados de T-0 seria trapaça — o modelo teria informações que não existiam no momento da previsão.

### Pesos adaptativos com decaimento temporal

Depois de cada ciclo de validação, os pesos de cada modelo no ensemble são atualizados com base no histórico real de acertos, com decaimento exponencial:

```
peso(modelo) ∝ acurácia(modelo) · Σ decaimento^(dias_atrás)
```

Acertos mais recentes pesam mais do que acertos antigos. Se um modelo começar a errar sistematicamente, o ensemble automaticamente passa a confiar menos nele — sem intervenção manual.

### Validação e auditoria completa

Cada previsão fica guardada no CSV com todos os detalhes. Quando a data-alvo chega, o sistema preenche o preço real e registra se acertou ou errou. Nada é apagado. O histórico completo fica preservado e auditável.

### Sinal de consenso

```
BULLISH  → os 3 horizontes preveem SUBIDA
BEARISH  → os 3 horizontes preveem QUEDA
MISTO    → há discordância entre os horizontes
```

---

## Carteira coberta

**eToro (ações):**

| Ticker | Ativo |
|--------|-------|
| LLY | Eli Lilly & Co |
| NVDA | NVIDIA Corporation |
| ALV.DE | Allianz SE |
| BTC-USD | Bitcoin |
| BABA | Alibaba Group ADR |

**ETFs de acumulação (longo prazo):**

| Ticker | Ativo |
|--------|-------|
| EXUS.L | MSCI World ex USA ETF |
| ICGA.DE | MSCI China ETF |
| SGLN.L | Physical Gold ETC |
| EMIM.AS | iShares Core MSCI EM IMI ETF |
| MEUD.PA | Core Stoxx Europe 600 |
| SJPA.MI | iShares Core MSCI Japan IMI ETF |

### Watchlist ML (universo de contexto macroeconômico)

A watchlist expande o universo de treinamento além da carteira pessoal. Os modelos aprendem correlações entre ativos e sinais de regime de mercado a partir desse dataset mais amplo.

| Grupo | Tickers | Por que está aqui |
|-------|---------|-------------------|
| Big Tech EUA | AAPL MSFT GOOGL AMZN META TSLA NVDA | Sentimento geral do setor de tecnologia |
| Semicondutores | AMD AVGO ASML TSM | Benchmark setorial para comparar com a NVDA |
| Blue Chips Suíças | NESN.SW NOVN.SW ROG.SW | Sinal europeu defensivo |
| Farmacêuticas / Saúde | NVO LLY JNJ PFE AZN MRK ABBV UNH IBB | Contexto setorial para a LLY |
| Ações Alemãs | ALV.DE SIE.DE BMW.DE BAS.DE | Proxy do macro europeu |
| ETFs da carteira | EXUS.L ICGA.DE SGLN.L EMIM.AS MEUD.PA SJPA.MI | Cobertura direta da carteira pessoal |
| ETFs de índice global | VWCE.DE IWDA.AS CSPX.L | Regime amplo do mercado global |
| Cripto | BTC-USD ETH-USD | Regime do mercado cripto |
| Mercados emergentes | BABA TSM BHP RIO VALE | Sinal macro de emergentes |
| Commodities tradicionais | GLD SLV XOM CVX | Proxy geopolítico e de inflação |
| Novas commodities | URA LIT DBA | Urânio (energia para datacenters de IA), Lítio (cadeia da EV), Agricultura (inflação real) |
| Setores defensivos | XLP XLU | Hedge de crise — sobem quando há rotação para risk-off |

---

## Estrutura do notebook

| Bloco | Descrição |
|-------|-----------|
| 1 | Instalar dependências (só 1ª vez) |
| 2 | Imports + seed global |
| 3 | Configuração da carteira |
| 4 | Caminhos + CSV + pastas |
| 5 | Download de preços + câmbio + contexto (VIX, SPY) |
| 6 | Feature engineering |
| 7 | Ensembles ML independentes D+1 / D+2 / D+3 |
| 7B | Feature importances → `model_metadata.csv` |
| 7C | Recalibração mensal do SGD |
| 8 | Validar previsões antigas + atualizar pesos + guardar novas previsões |
| 9 | Análise da carteira: G/P, breakeven, alvos de saída |
| 10 | Sinais de saída |
| 11 | Projeção 1 / 3 / 5 / 10 anos |
| 12 | Simulação DCA |
| 12B | Limpeza automática de gráficos com mais de 30 dias úteis |
| 13 | Geração de gráficos |
| 14 | Resumo final + email HTML |

---

## Estrutura do repositório

```
├── PrevisaoCarteira.ipynb           ← notebook principal
├── AnaliseV5/
│   ├── predictions_log.csv          ← histórico completo de previsões e validações
│   ├── ensemble_weights.json        ← pesos atuais por modelo e por horizonte
│   ├── model_metadata.csv           ← feature importances diárias (RF + GB)
│   └── AnaliseGraficos/
│       └── TICKER_YYYYMMDD.png     ← um gráfico por ativo por dia
├── config/
│   ├── my_portfolio.json            ← tickers da carteira pessoal
│   └── watchlist.json               ← universo estendido para contexto ML
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← agendamento automático diário
├── README.md                        ← versão em inglês
└── README_pt.md                     ← este arquivo (português)
```

---

## Como funciona a automação (GitHub Actions)

```
Seg–Sex 15h35 Barcelona (13h35 UTC)
  │
  ├─ GitHub clona o repositório
  ├─ Instala as dependências Python
  ├─ Executa o notebook completo (~8 minutos)
  │   ├─ Baixa preços + câmbio + VIX + SPY
  │   ├─ Valida previsões anteriores
  │   ├─ Treina os modelos com o histórico atualizado
  │   ├─ Atualiza os pesos do ensemble
  │   ├─ Guarda as novas previsões D+1 / D+2 / D+3
  │   ├─ Gera os gráficos
  │   └─ Escreve o relatório HTML do email
  └─ Commit automático com timestamp → push para o repositório
```

Se falhar, o GitHub envia um email de notificação automaticamente.

**Por que 15h35?** Os mercados europeus e o cripto já fecharam. Os EUA estão abertos mas o dado do dia anterior está finalizado. Isso maximiza a cobertura de dados sem introduzir lookahead bias (usar informação do futuro no treino).

---

## Stack técnica

```
Python 3.11
├── yfinance          — dados de mercado em tempo real (preços, câmbio, VIX, SPY)
├── scikit-learn      — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy    — processamento de dados e cálculo de features
└── matplotlib        — geração de gráficos

GitHub Actions        — automação diária gratuita
```

---

## Contexto sobre acurácia

- Uma previsão direcional aleatória tem 50% de acurácia por definição.
- Este sistema tem como alvo 55–65% de acurácia direcional nos ativos da carteira pessoal.
- Acurácia abaixo de 52% ao longo de 30+ validações é sinal de degradação do modelo.
- Nenhum número de acurácia, por si só, justifica decisões financeiras — este é um projeto de análise pessoal, não aconselhamento financeiro.

---

## Sobre

Construído por **Vicky Costa** — Analista de Dados | Estudante de Ciência de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
