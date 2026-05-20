# Carteira Inteligente — Sistema de Previsão de Carteira com Machine Learning

> Um sistema de análise e previsão de carteira de investimentos que aprende sozinho, construído do zero.
> Executa automaticamente todos os dias úteis às 17h45 (Barcelona / CEST) via GitHub Actions.

[![GitHub Actions](https://img.shields.io/badge/Automatizado-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart_wallet/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

---

## O que é isto?

Tenho uma carteira de investimentos dividida entre ações (eToro) e ETFs de acumulação de longo prazo. Durante um tempo geria tudo manualmente — abria apps, via números a vermelho e a verde, e tomava decisões no feeling.

Decidi mudar isso.

Este projeto é um pipeline Python que executa automaticamente todos os dias úteis às 17h45 (Barcelona) e faz três coisas:

1. **Analisa a carteira** — Ganho/Perda real em euros, breakeven com fees incluídos, alvo de saída
2. **Prevê a direção dos próximos 3 dias** — Usando Machine Learning (Random Forest, Gradient Boosting, SGD em ensemble) com indicadores técnicos + contexto de mercado (VIX, SPY)
3. **Aprende com os erros** — Cada previsão é validada quando a data chega. O modelo que acertar mais passa a ter mais peso nas decisões seguintes

Não é um oráculo. É um sistema que fica menos burro a cada dia que passa.

---

## Como o Machine Learning funciona aqui

### O problema que resolvemos

A pergunta que o modelo tenta responder é simples: **"o preço vai subir ou cair nos próximos N dias úteis?"**

Isso é o que em ML chamamos de classificação binária:
- `1` = sobe (UP)
- `0` = cai (DOWN)

E fazemos isso para 3 horizontes de tempo independentes: amanhã (D+1), depois de amanhã (D+2) e em 3 dias úteis (D+3).

### Por que 3 ensembles separados e não um só modelo?

A maioria dos exemplos online treina um único modelo e depois "extrapola" os resultados para D+2 e D+3. Isso é um erro. O padrão que faz uma ação subir amanhã é estruturalmente diferente do padrão que faz ela subir em 3 dias. O movimento de amanhã é dominado por momentum de curto prazo e sentimento overnight. Um horizonte de 3 dias é mais influenciado pela persistência da tendência e pelo regime macro. Misturar esses horizontes num único modelo conflaciona dinâmicas diferentes. Por isso treinamos ensembles totalmente independentes:

```
Ensemble D+1 ──► RF_d1 · GB_d1 · SGD_d1  ──► pesos_d1
Ensemble D+2 ──► RF_d2 · GB_d2 · SGD_d2  ──► pesos_d2
Ensemble D+3 ──► RF_d3 · GB_d3 · SGD_d3  ──► pesos_d3
```

Cada ensemble aprende a assinatura temporal do seu horizonte de forma independente.

### Por que 3 algoritmos diferentes em cada ensemble?

Usar um único modelo por horizonte seria um único ponto de falha. Algoritmos diferentes cometem tipos diferentes de erros sobre os mesmos dados — combiná-los reduz a variância sem aumentar o viés. Cada modelo foi escolhido por uma razão específica:

| Algoritmo | Configuração | Por que foi escolhido |
|-----------|-------------|----------------------|
| **Random Forest** | 300 árvores, profundidade máx. 6 | Generalizador robusto. As árvores com bootstrap resistem bem ao overfitting em dados ruidosos de mercado. Funciona bem mesmo quando alguns indicadores são irrelevantes — situação comum em séries temporais financeiras onde a utilidade de cada feature muda com o regime. Funciona como âncora de estabilidade do ensemble. |
| **Gradient Boosting** | 200 estimadores, taxa 0.05 | Captura padrões que o Random Forest não consegue capturar, corrigindo iterativamente os seus próprios erros residuais. Especialmente bom a detetar sinais de momentum de curto prazo e interações subtis entre indicadores. A taxa de aprendizagem baixa (0.05) atrasa a convergência intencionalmente — impede que o modelo memorize ruído. |
| **SGD Classifier** | log_loss | Um modelo linear incluído deliberadamente como contrapeso. Quando os dois modelos não-lineares concordam com algo que é na verdade ruído, o SGD — que não consegue modelar interações não-lineares — age como voto discordante e puxa o ensemble para estimativas mais conservadoras. A sua simplicidade é uma feature, não uma limitação. |

O modelo SGD passa por **recalibração mensal completa**: refit do scaler + retreino do zero. Isso é necessário porque o SGD usa features normalizadas. Se a distribuição de preços e indicadores mudar gradualmente (por exemplo, após uma grande repricing do mercado), o scaler antigo já não representa os dados atuais, e os coeficientes lineares do modelo ficam ancorados a uma baseline obsoleta. A recalibração mensal mantém a âncora linear alinhada com as condições atuais do mercado, sem o overhead de recalibrar diariamente.

### As features — o que o modelo "vê" de cada ativo

Além do preço histórico, o modelo recebe uma série de indicadores técnicos calculados automaticamente, mais contexto externo de mercado:

| Feature | O que representa | Por que importa |
|---------|-----------------|-----------------|
| `sma_20`, `sma_50` | Médias móveis de 20 e 50 dias | Alinhamento de tendência de curto vs médio prazo. Os cruzamentos entre elas são um sinal clássico de mudança de regime. |
| `rsi_14` | RSI de 14 dias | Deteta se o ativo está sobreextendido em qualquer direção. RSI > 70 (sobrecomprado) e RSI < 30 (sobrevendido) são historicamente condições de mean-reversion. |
| `macd`, `macd_signal` | Linha MACD e linha de sinal | Captura momentum e reversões de tendência através de cruzamentos. Útil para detetar quando uma tendência está a ganhar ou a perder força. |
| `bb_upper`, `bb_lower`, `bb_width` | Bandas de Bollinger (20 dias, 2σ) | Codifica tanto o regime de volatilidade como a extremidade do preço. Quando o preço atinge a banda superior/inferior, o modelo pode fatorar a probabilidade de reversão. A largura da banda sinaliza se o ativo está num período calmo ou explosivo. |
| `atr_14` | Average True Range de 14 dias | O movimento diário esperado em termos absolutos. Ajuda o modelo a distinguir entre um movimento de +1% que está dentro do intervalo normal e um que é excecional. |
| `ret_1d`, `ret_5d` | Retorno dos últimos 1 e 5 dias | Features de momentum direto. Os retornos recentes estão entre as features mais preditivas para horizontes curtos. |
| `spy_ret_1d` | Retorno do S&P 500 (T-1) | Contexto global do mercado. A NVDA num dia após o S&P cair 2% comporta-se de forma diferente da NVDA num dia neutro. Esta feature permite que o modelo condicione a sua previsão ao estado do mercado amplo. |
| `vix_level` | Nível de fecho do VIX (T-1) | A volatilidade implícita do mercado — o "termômetro do medo". Um VIX de 30 significa um ambiente fundamentalmente diferente de um VIX de 14. Sem isso, o modelo não consegue distinguir comportamentos de bull market e de crise. |
| `vix_change` | Variação diária do VIX (T-1) | Captura a *aceleração* do medo, não só o seu nível. Um VIX a subir rapidamente frequentemente leva a resultados diferentes do mesmo nível absoluto de VIX que se manteve estável durante semanas. |

**Por que T-1 para as features de contexto externo:** quando o pipeline corre às 17h45 (Barcelona), os mercados europeus acabaram de fechar mas os EUA ainda estão abertos. O fecho de NY do dia anterior (T-1) é o dado mais recente, completo e final disponível para o SPY e o VIX. Usar os valores em progresso de T-0 constituiria data leakage — o modelo estaria a ser treinado com informação que ainda não existia no momento da previsão.

### Pesos adaptativos com decaimento temporal

Depois de cada ciclo de validação, os pesos de cada modelo no ensemble são atualizados com base no histórico real de acertos, com decaimento exponencial:

```
peso(modelo) ∝ acurácia(modelo) · Σ decaimento^(dias_atrás)
```

Acertos mais recentes pesam mais do que acertos antigos. Se um modelo começar a errar sistematicamente, o ensemble automaticamente passa a confiar menos nele — sem intervenção manual.

**Por que decaimento exponencial e não uma janela fixa?** Uma janela fixa dá igual importância a um acerto de 28 dias atrás e a um de ontem. Os mercados mudam. Um modelo que era o melhor preditor em fevereiro pode simplesmente ter encontrado um padrão num mercado em tendência que já não existe depois de uma mudança de regime. O decaimento exponencial garante que a performance recente tenha peso desproporcionalmente maior, tornando o sistema reativo a mudanças de regime em vez de ficar preso no seu próprio passado.

### Datas-alvo em dias úteis — por que isso importa

As previsões usam `pd.offsets.BDay(N)` para calcular as datas-alvo, e não `pd.Timedelta(days=N)`. A diferença: se hoje é sexta-feira, D+1 usando Timedelta resolve para sábado — um dia sem negociação, sem preço. D+1 usando BDay resolve para segunda-feira, a próxima sessão real. Usar dias de calendário faria o validador procurar preços que não existem, produzindo validações NaN silenciosas e corrompendo o tracking de acurácia.

### Sinal de consenso

```
BULLISH  → os 3 horizontes preveem SUBIDA
BEARISH  → os 3 horizontes preveem QUEDA
MISTO    → há discordância entre os horizontes
```

### Validação e auditoria completa

Cada previsão fica guardada no CSV com todos os detalhes. Quando a data-alvo chega, o sistema preenche o preço real e registra se acertou ou errou. Nada é apagado. O histórico completo fica preservado e auditável.

| Coluna | Descrição |
|--------|-----------|
| `pred_date` | Data em que a previsão foi feita |
| `target_date` | Data a que a previsão se refere (ajustada para dias úteis) |
| `horizon` | 1, 2 ou 3 |
| `direction` | `up` ou `down` |
| `confidence` | Probabilidade ponderada do ensemble |
| `actual_price` | Preenchido no dia da validação (inicialmente `NaN`) |
| `correct` | `True` / `False` (preenchido no dia da validação) |

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

A watchlist expande o universo de treinamento além da carteira pessoal. Os modelos aprendem correlações entre ativos e sinais de regime de mercado a partir desse dataset mais amplo, produzindo previsões mais contextualmente conscientes para os ativos da carteira.

| Grupo | Tickers | Por que está aqui |
|-------|---------|-------------------|
| Big Tech EUA | AAPL MSFT GOOGL AMZN META TSLA NVDA | Sentimento geral do setor de tecnologia |
| Semicondutores | AMD AVGO ASML TSM | Benchmark setorial para comparar com a NVDA |
| Blue Chips Suíças | NESN.SW NOVN.SW ROG.SW | Sinal europeu defensivo |
| Farmacêuticas / Saúde | NVO LLY JNJ PFE AZN MRK ABBV UNH IBB XBI | Contexto setorial para a LLY |
| Ações Alemãs | ALV.DE SIE.DE BMW.DE BAS.DE | Proxy do macro europeu |
| ETFs da carteira | EXUS.L ICGA.DE SGLN.L EMIM.AS MEUD.PA SJPA.MI | Cobertura direta da carteira pessoal |
| ETFs de índice global | VWCE.DE IWDA.AS CSPX.L | Regime amplo do mercado global |
| Cripto | BTC-USD ETH-USD | Regime do mercado cripto |
| EM tech / recursos | BABA TSM BHP RIO VALE | Sinal macro de emergentes |
| Commodities tradicionais | GLD SLV XOM CVX COPX | Proxy geopolítico e de inflação |
| Novas commodities | URA LIT DBA | Urânio (energia para datacenters de IA), Lítio (cadeia da EV), Agricultura (inflação real) |
| Setores defensivos | XLP XLU | Hedge de crise — sobem quando há rotação para risk-off |
| Obrigações | TLT AGG HYG TIP EMB LQD SHY | Regime de taxas de juro e condições de crédito |
| Índia | INDA INFY WIT | Maior EM em crescimento, sinal de outsourcing tecnológico |
| Brasil | EWZ ITUB | EM ligado a commodities, proxy de risco BRL |
| REITs | VNQ VNQI O PLD | Indicador de sensibilidade às taxas de juro |
| Volatilidade | UVXY VXX | Procura de hedging e medo em tempo real |
| Setores EUA | XLF XLK XLE XLV XLI XLY | Deteção de rotação setorial |
| China | MCHI FXI | Cobertura direta para contexto BABA e ICGA.DE |
| Japão | EWJ | Contexto para SJPA.MI |
| Europa ampla | VGK | Largura do mercado europeu |
| América Latina | ILF | Sinal de diversificação EM regional |
| Temáticos | ICLN CIBR BOTZ ITA PHO | Energia limpa, cibersegurança, robótica, defesa, água |
| Bancos regionais EUA | KRE | Indicador de stress bancário de pequeno e médio porte |
| Dividendos / qualidade | VYM NOBL | Sinal do fator qualidade |
| Mercado amplo | QQQ IWM RSP | Rotação crescimento vs valor vs equal-weight |
| Mercados de fronteira | FM ASEA | Sinal EM periférico |

---

## Estrutura do repositório

```
├── main.py                          ← orquestrador do pipeline
├── config/
│   ├── my_portfolio.json            ← carteira pessoal (tickers, unidades, preços de entrada)
│   └── watchlist.json               ← universo estendido de treino ML
├── data/
│   ├── downloader.py                ← download de dados de mercado via yfinance
│   └── storage.py                   ← leitura/escrita de CSVs
├── features/
│   └── engineering.py               ← indicadores técnicos + matriz de features ML
├── models/
│   ├── ensemble.py                  ← treino RF + GB + SGD + atualização de pesos adaptativos
│   └── validator.py                 ← validação de previsões contra preços realizados
├── portfolio/
│   ├── pnl.py                       ← G/P, fees, breakeven, alvos de saída
│   ├── exit_signals.py              ← lógica de sinais de saída
│   ├── projections.py               ← projeções 1 / 3 / 5 / 10 anos
│   └── dca.py                       ← simulação DCA
├── reports/
│   ├── charts.py                    ← geração de gráficos (matplotlib)
│   └── email_report.py              ← construção do email HTML
├── output/
│   ├── predictions_log.csv          ← histórico completo de previsões e validações
│   ├── ensemble_weights.json        ← pesos atuais por modelo e por horizonte
│   ├── model_metadata.csv           ← feature importances diárias (RF + GB)
│   ├── resumo_diario.html           ← email HTML mais recente (commitado diariamente)
│   ├── ultima_recalibracao.json     ← timestamp da última recalibração do SGD
│   ├── models/                      ← modelos serializados (.joblib)
│   └── charts/                      ← um gráfico por ativo por dia (limpeza automática após 30 dias)
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← agendamento automático diário
├── requirements.txt
├── README.md                        ← versão em inglês
└── README_pt.md                     ← este arquivo (português)
```

---

## Como funciona a automação (GitHub Actions)

```
Seg–Sex 17h45 Barcelona (15h45 UTC, compensando ~2h de delay típico do GitHub)
  │
  ├─ Job 1: verificar se já executou hoje
  │   └─ lê predictions_log.csv — se a data de hoje já existe, sai em ~10s
  │
  └─ Job 2: executar pipeline (só se ainda não correu hoje)
      ├─ Clonar repositório
      ├─ Instalar Python 3.11 + dependências
      ├─ Executar main.py (~8 minutos)
      │   ├─ Baixar preços + câmbio + VIX + SPY
      │   ├─ Calcular features
      │   ├─ Validar previsões anteriores
      │   ├─ Treinar modelos com histórico atualizado
      │   ├─ Atualizar pesos do ensemble
      │   ├─ Guardar novas previsões D+1 / D+2 / D+3
      │   ├─ Gerar gráficos
      │   └─ Construir email HTML
      ├─ Commit dos ficheiros de output → push
      └─ Enviar email HTML via Gmail SMTP
```

**Por que três entradas de cron:** o agendador do GitHub Actions está sujeito a delays de fila de 2 a 3 horas sob alta carga. Três gatilhos de cron separados (com 30 minutos de diferença) são registados, mas a verificação anti-duplicação no Job 1 garante que o pipeline só executa uma vez por dia, mesmo que múltiplos crons disparem. Isso garante a entrega sem necessitar de um plano Actions pago com agendamento prioritário.

**Por que 17h45 Barcelona:** Frankfurt, Paris, Londres, Milão e Amsterdam fecham às 17h30 CEST. Ao correr às 17h45, o pipeline apanha o fecho real do dia para todos os ETFs europeus da carteira (EMIM.AS, MEUD.PA, SJPA.MI, ICGA.DE, EXUS.L, SGLN.L). As ações americanas (LLY, NVDA, BABA) ainda estão abertas a essa hora — o yfinance devolve o preço intraday mais recente, não o fecho do dia.

Se falhar, o GitHub envia um email de notificação automaticamente.

---

## Stack técnica

```
Python 3.11
├── yfinance          — dados de mercado em tempo real (preços, câmbio, VIX, SPY)
├── scikit-learn      — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy    — processamento de dados e cálculo de features
├── joblib            — serialização de modelos
└── matplotlib        — geração de gráficos

GitHub Actions        — automação diária gratuita
Gmail SMTP            — entrega de email HTML
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
