# Carteira Inteligente — Sistema de Previsão de Carteira com Machine Learning

> Um sistema de análise e previsão de carteira de investimentos que aprende sozinho, construído do zero.
> Executa automaticamente todos os dias úteis às ~22h30 (Barcelona / CEST) — após o fecho de todos os mercados — via GitHub Actions.

[![GitHub Actions](https://img.shields.io/badge/Automatizado-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart_wallet/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

---

## O que é isto?

Tenho uma carteira de investimentos dividida entre ações (eToro) e ETFs de acumulação de longo prazo. Durante um tempo geria tudo manualmente — abria apps, via números a vermelho e a verde, e tomava decisões no feeling.

Decidi mudar isso.

Este projeto é um pipeline Python que executa automaticamente todos os dias úteis às ~22h30 (Barcelona) e faz três coisas:

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

A maioria dos exemplos online treina um único modelo e depois "extrapola" os resultados para D+2 e D+3. Isso é um erro. O padrão que faz uma ação subir amanhã é estruturalmente diferente do padrão que faz ela subir em 3 dias. O movimento de amanhã é dominado por momentum de curto prazo e sentimento overnight. Um horizonte de 3 dias é mais influenciado pela persistência da tendência e pelo regime macro. Por isso treinamos ensembles totalmente independentes:

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
| **Random Forest** | 300 árvores, profundidade máx. 6 | Generalizador robusto. As árvores com bootstrap resistem bem ao overfitting em dados ruidosos de mercado. Funciona bem mesmo quando alguns indicadores são irrelevantes — situação comum em séries temporais financeiras. Funciona como âncora de estabilidade do ensemble. |
| **Gradient Boosting** | 200 estimadores, taxa 0.05 | Captura padrões que o Random Forest não consegue capturar, corrigindo iterativamente os seus próprios erros residuais. Especialmente bom a detetar sinais de momentum de curto prazo e interações subtis entre indicadores. A taxa de aprendizagem baixa (0.05) impede que o modelo memorize ruído. |
| **SGD Classifier** | log_loss | Um modelo linear incluído deliberadamente como contrapeso. Quando os dois modelos não-lineares concordam com algo que é na verdade ruído, o SGD age como voto discordante e puxa o ensemble para estimativas mais conservadoras. A sua simplicidade é uma feature, não uma limitação. |

O modelo SGD passa por **recalibração mensal completa**: refit do scaler + retreino do zero. Isso é necessário porque o SGD usa features normalizadas. Se a distribuição de preços e indicadores mudar gradualmente, o scaler antigo já não representa os dados atuais e os coeficientes lineares ficam ancorados a uma baseline obsoleta.

### As features — o que o modelo "vê" de cada ativo

| Feature | O que representa | Por que importa |
|---------|-----------------|-----------------|
| `sma_20`, `sma_50` | Médias móveis de 20 e 50 dias | Alinhamento de tendência de curto vs médio prazo. |
| `rsi_14` | RSI de 14 dias | Deteta se o ativo está sobreextendido. RSI > 70 (sobrecomprado) e RSI < 30 (sobrevendido) são historicamente condições de mean-reversion. |
| `macd`, `macd_signal` | Linha MACD e linha de sinal | Captura momentum e reversões de tendência através de cruzamentos. |
| `bb_upper`, `bb_lower`, `bb_width` | Bandas de Bollinger (20 dias, 2σ) | Codifica o regime de volatilidade e a extremidade do preço. |
| `atr_14` | Average True Range de 14 dias | O movimento diário esperado em termos absolutos. |
| `ret_1d`, `ret_5d` | Retorno dos últimos 1 e 5 dias | Features de momentum direto. |
| `spy_ret_1d` | Retorno do S&P 500 (T-1) | Contexto global do mercado. |
| `vix_level` | Nível de fecho do VIX (T-1) | O "termómetro do medo" do mercado. |
| `vix_change` | Variação diária do VIX (T-1) | Captura a *aceleração* do medo, não só o seu nível. |
| `vix_regime` | Label de regime do VIX: 0 = baixo (VIX < 15), 1 = médio (15 ≤ VIX < 25), 2 = alto (VIX ≥ 25) | Sinal discreto de regime de mercado — permite ao modelo aprender padrões diferentes por contexto de volatilidade. Sem ela, um mercado calmo e uma crise parecem iguais para o vetor de features. |

**Por que T-1 para as features de contexto externo:** os valores T-1 são usados para garantir consistência com o processo de treino. Durante o treino, cada linha usa contexto T-1 (os dados SPY/VIX disponíveis *antes* da sessão que está a ser prevista). Usar T-0 em produção introduziria um desfasamento treino/inferência — o modelo receberia uma estrutura temporal para a qual nunca foi treinado.

### Pesos adaptativos com decaimento temporal

Depois de cada ciclo de validação, os pesos de cada modelo são atualizados com decaimento exponencial:

```
peso(modelo) ∝ acurácia(modelo) · Σ decaimento^(dias_atrás)
```

Acertos mais recentes pesam mais do que acertos antigos. Se um modelo começar a errar sistematicamente, o ensemble automaticamente passa a confiar menos nele — sem intervenção manual.

### Datas-alvo por calendário de mercado — por que isso importa

As previsões usam `pandas-market-calendars` com mapeamento por bolsa para calcular as datas-alvo. `pd.offsets.BDay(N)` trata fins de semana mas ignora feriados de mercado. Por exemplo, se hoje é a sexta-feira antes de um feriado bancário dos EUA, `BDay(1)` devolve segunda, mas `pandas-market-calendars` para a NYSE devolve terça — a sessão real seguinte. As bolsas europeias (LSE, XETR, XAMS, XPAR, XMIL, SIX) têm calendários de feriados diferentes da NYSE. Uma data-alvo inválida produziria validações NaN silenciosas que corrompem o histórico de acurácia.

Mapeamento de cada ticker para a sua bolsa:

| Bolsa | Calendário | Tickers |
|-------|-----------|---------|
| NYSE (padrão) | `NYSE` | LLY, NVDA, BABA, BTC-USD, watchlist EUA |
| Londres | `LSE` | EXUS.L, SGLN.L, CSPX.L |
| Frankfurt / Xetra | `XETR` | ALV.DE, SIE.DE, BMW.DE, BAS.DE, DHER.DE, VWCE.DE, ICGA.DE |
| Amsterdão | `XAMS` | EMIM.AS, IWDA.AS |
| Paris | `XPAR` | MEUD.PA |
| Milão | `XMIL` | SJPA.MI |
| Bolsa Suíça | `SIX` | NESN.SW, NOVN.SW, ROG.SW |

### Estratificação de acurácia — carteira vs watchlist

A watchlist contém ~90 tickers usados como contexto macroeconómico. Não são ativos detidos — são sinais de treino. A acurácia dos tickers da watchlist é estruturalmente mais baixa e nunca deve ser misturada com a acurácia da carteira. O sistema rastreia e reporta separadamente:

- **Acurácia da carteira** — o número que importa para as decisões do dia a dia
- **Acurácia da watchlist** — qualidade interna do sinal, não reportada no email

Esta distinção foi introduzida depois de observar uma acurácia misturada de 28% que fazia o sistema parecer aleatório. A figura correta da carteira era 33% — ainda baixa por falta de histórico de validações suficiente (< 30 amostras por ticker nesse momento), mas estruturalmente diferente.

### Validação e auditoria completa

Cada previsão fica guardada no `output/predictions_log.csv` com todos os detalhes:

| Coluna | Descrição |
|--------|-----------|
| `ticker` | Identificador do ativo |
| `pred_date` | Data em que a previsão foi feita |
| `target_date` | Data a que a previsão se refere (ajustada por calendário de bolsa) |
| `horizon` | 1, 2 ou 3 |
| `direction` | `up` ou `down` |
| `ref_price` | Preço de fecho no dia em que a previsão foi feita — referência real para verificar a direcção |
| `pred_price` | Preço-alvo estimado pelo ATR (`fecho ± ATR × 0,5 × √horizonte`) — informativo |
| `confidence` | Probabilidade ponderada do ensemble |
| `actual_price` | Preenchido no dia da validação (inicialmente `NaN`) |
| `actual_change_pct` | `(actual_price / ref_price − 1) × 100` — preenchido na validação |
| `correct` | `True` se actual ≥ ref_price (UP) ou actual ≤ ref_price (DOWN); preenchido na validação |
| `atr_at_prediction` | ATR14 no momento em que a previsão foi feita |
| `predicted_price` | Reservado para uso futuro |
| `model_rf` | Voto individual do Random Forest (`up`/`down`) |
| `model_gb` | Voto individual do Gradient Boosting |
| `model_sgd` | Voto individual do SGD Classifier |

Nada é apagado. O histórico completo fica preservado indefinidamente. Novas colunas são adicionadas via migração retrocompatível em `data/storage.py` — as linhas existentes são preenchidas retroativamente onde possível.

### Sinal de consenso

```
BULLISH  → os 3 horizontes preveem SUBIDA
BEARISH  → os 3 horizontes preveem QUEDA
MISTO    → há discordância entre os horizontes
```

---

## Email diário

O email HTML é desenhado para ser lido em telemóvel. Tem quatro secções:

### 1 — Tabela ML de previsões

| Coluna | Conteúdo |
|--------|---------|
| Ativo | Ticker + nome do ativo |
| Preço | Preço de fecho atual (em EUR onde aplicável via câmbio) |
| Var% | Variação fecho-a-fecho do dia anterior |
| D+1 | Previsão direcional + confiança |
| D+2 | Previsão direcional + confiança |
| D+3 | Previsão direcional + confiança |
| Consenso | BULLISH / BEARISH / MISTO |

Cada linha é prefixada com ✅ ou ❌ — se a previsão D+1 de ontem foi correta ou errada.

**Legenda:** ✅ acertou a previsão D+1 do dia anterior · ❌ errou a previsão D+1 do dia anterior

### 2 — Tabela ETFs de acumulação

Projeções de longo prazo para os ETFs de acumulação a 1 / 3 / 5 / 10 anos nos cenários pessimista / base / optimista. Valores em EUR.

### 3 — Painel de acurácia

Acurácia direcional cumulativa **apenas para os tickers da carteira**, nos últimos 30 dias úteis. Com gráfico por ativo e figura global de carteira.

### 4 — Gráficos

Um gráfico por ativo da carteira com os últimos 120 dias úteis:
- Preço + SMA20 + SMA50 + Bandas de Bollinger
- Linha de preço de abertura de posição
- Setas de previsão D+1 / D+2 / D+3 (datas por calendário de bolsa)
- RSI (14 dias)
- MACD + histograma
- Curva de acurácia cumulativa D+1 (depois de ≥ 3 validações disponíveis)

Os marcadores de validação nos gráficos mostram **apenas D+1**: ● para previsão correta e × para errada. D+2 e D+3 são treinados separadamente e aparecem na tabela do email, mas não são plotados no gráfico para evitar sobreposição de marcadores na mesma data-alvo.

---

## Repositório público

Os gráficos são publicados num repositório público separado ([smart-wallet-ml](https://github.com/srxkatsumi/smart-wallet-ml)) com um **atraso de 10 dias em janela deslizante**. O repo público contém:

- Um gráfico por ativo da carteira por dia de negociação, para a janela D-19 a D-10
- Um README gerado automaticamente com a data da última atualização

Não são divulgados preços de entrada, posições, volumes ou outros dados da carteira. Os nomes dos ficheiros de gráfico contêm os tickers — isso é intencional.

A sincronização corre como step 8 do workflow GitHub Actions. Clona o repo público, copia os gráficos relevantes, gera o README com as datas corretas via `scripts/gen_public_readme.py` e faz push. O atraso de 10 dias impede cópia de sinais em tempo real.

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
| DHER.DE | Delivery Hero SE |

**ETFs de acumulação (longo prazo):**

| Ticker | Ativo |
|--------|-------|
| EXUS.L | MSCI World ex USA ETF |
| ICGA.DE | MSCI China ETF |
| SGLN.L | Physical Gold ETC |
| EMIM.AS | iShares Core MSCI EM IMI ETF |
| MEUD.PA | Core Stoxx Europe 600 |
| SJPA.MI | iShares Core MSCI Japan IMI ETF |

### Watchlist ML (universo de contexto macroeconómico)

A watchlist expande o universo de treinamento além da carteira pessoal. Os modelos aprendem correlações entre ativos e sinais de regime de mercado a partir desse dataset mais amplo.

| Grupo | Tickers | Por que está aqui |
|-------|---------|-------------------|
| Big Tech EUA | AAPL MSFT GOOGL AMZN META TSLA NVDA | Sentimento geral do setor tecnológico |
| Semicondutores | AMD AVGO ASML TSM | Benchmark setorial para a NVDA |
| Blue Chips Suíças | NESN.SW NOVN.SW ROG.SW | Sinal europeu defensivo |
| Farmacêuticas / Saúde | NVO LLY JNJ PFE AZN MRK ABBV UNH IBB XBI | Contexto setorial para a LLY |
| Ações Alemãs | ALV.DE SIE.DE BMW.DE BAS.DE | Proxy do macro europeu |
| ETFs da carteira | EXUS.L ICGA.DE SGLN.L EMIM.AS MEUD.PA SJPA.MI | Cobertura direta da carteira |
| ETFs de índice global | VWCE.DE IWDA.AS CSPX.L | Regime amplo do mercado |
| Cripto | BTC-USD ETH-USD | Regime do mercado cripto |
| EM tech / recursos | BABA TSM BHP RIO VALE | Sinal macro de emergentes |
| Commodities tradicionais | GLD SLV XOM CVX COPX | Proxy geopolítico e de inflação |
| Novas commodities | URA LIT DBA | Urânio (datacenters de IA), Lítio (EV), Agricultura (inflação) |
| Setores defensivos | XLP XLU | Hedge de crise — sobem em rotações risk-off |
| Obrigações | TLT AGG HYG TIP EMB LQD SHY | Regime de taxas de juro e crédito |
| Índia | INDA INFY WIT | Maior EM em crescimento, sinal de outsourcing tecnológico |
| Brasil | EWZ ITUB | EM ligado a commodities, proxy de risco BRL |
| REITs | VNQ VNQI O PLD | Indicador de sensibilidade às taxas |
| Volatilidade | UVXY VXX | Procura de hedging e medo em tempo real |
| Setores EUA | XLF XLK XLE XLV XLI XLY | Deteção de rotação setorial |
| China | MCHI FXI | Cobertura para BABA e ICGA.DE |
| Japão | EWJ | Contexto para SJPA.MI |
| Europa ampla | VGK | Largura do mercado europeu |
| América Latina | ILF | Sinal EM regional |
| Temáticos | ICLN CIBR BOTZ ITA PHO | Energia limpa, cibersegurança, robótica, defesa, água |
| Bancos regionais EUA | KRE | Indicador de stress bancário |
| Dividendos / qualidade | VYM NOBL | Sinal do fator qualidade |
| Mercado amplo | QQQ IWM RSP | Rotação crescimento vs valor vs equal-weight |
| Mercados de fronteira | FM ASEA | Sinal EM periférico |

---

## Estrutura do repositório

```
├── main.py                          ← orquestrador do pipeline
├── config/
│   ├── settings.py                  ← constantes, caminhos, hiperparâmetros, TICKER_CALENDAR
│   ├── my_portfolio.json            ← carteira pessoal (tickers, unidades, preços de entrada)
│   ├── portfolio.json               ← config da carteira com nomes dos ativos
│   └── watchlist.json               ← universo estendido de treino ML
├── data/
│   ├── downloader.py                ← download de dados de mercado via yfinance
│   ├── storage.py                   ← leitura/escrita de CSVs + migrações retrocompatíveis
│   └── calendars.py                 ← cálculo de datas-alvo por calendário de bolsa
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
├── scripts/
│   └── gen_public_readme.py         ← gera README do repo público (chamado pelo CI)
├── output/
│   ├── predictions_log.csv          ← histórico completo de previsões e validações (nunca apagado)
│   ├── ensemble_weights.json        ← pesos atuais por modelo e por horizonte
│   ├── model_metadata.csv           ← feature importances diárias (RF + GB)
│   ├── resumo_diario.html           ← email HTML mais recente (commitado diariamente)
│   ├── ultima_recalibracao.json     ← timestamp da última recalibração do SGD
│   ├── models/                      ← modelos serializados (.joblib)
│   └── charts/                      ← um gráfico por ativo por dia (limpeza automática 30 dias)
├── .github/
│   └── workflows/
│       └── executar_diario.yml      ← automação diária (9 steps)
├── requirements.txt
├── README.md                        ← versão em inglês
└── README_pt.md                     ← este ficheiro (português)
```

---

## Como funciona a automação (GitHub Actions)

```
Seg–Sex ~22h30 Barcelona CEST (20h30 UTC — após o fecho de todos os mercados)
  │
  ├─ Job 1: verificar se já executou hoje
  │   └─ lê predictions_log.csv — se a data de hoje já existe, sai em ~10s
  │
  └─ Job 2: executar pipeline (só se ainda não correu hoje)
      ├─ 1. Checkout do repositório
      ├─ 2. Instalar Python 3.11
      ├─ 3. Instalar dependências (pip install -r requirements.txt)
      ├─ 4. Correr testes unitários (pytest tests/ -v) — pára o pipeline se falhar
      ├─ 5. Executar main.py (~8 minutos)
      │   ├─ Baixar preços + câmbio + VIX + SPY
      │   ├─ Calcular features
      │   ├─ Validar previsões anteriores
      │   ├─ Atualizar pesos do ensemble
      │   ├─ Recalibração mensal do SGD (se necessário)
      │   ├─ Treinar modelos com os pesos actualizados
      │   ├─ Guardar novas previsões D+1 / D+2 / D+3
      │   ├─ Gerar gráficos
      │   └─ Construir email HTML
      ├─ 6. Commit dos ficheiros de output → push (até 3 tentativas + git pull --rebase)
      ├─ 7. Enviar email HTML via Gmail SMTP
      ├─ 8. Sincronizar repo público (gráficos com atraso 10 dias + README gerado)
      └─ 9. Em caso de falha: artefacto de emergência + email de notificação de erro
```

**Por que três entradas de cron:** o agendador do GitHub Actions pode atrasar 2–3 horas. Três crons separados (com 30 min de diferença) garantem a execução. A verificação anti-duplicação no Job 1 garante que o pipeline só executa uma vez por dia.

**Por que após o fecho dos EUA:** a NYSE e o NASDAQ fecham às 20h00 UTC (16h00 ET). Ao correr às ~20h30 UTC, o pipeline tem acesso ao preço real de fecho do dia para todos os ativos da carteira — incluindo ações americanas (LLY, NVDA, BABA) e crypto (BTC-USD). Isto permite validar as previsões no próprio dia em que os mercados fecham, e os ícones ✅/❌ de acurácia no email aparecem sempre preenchidos quando o relatório chega.

---

## Stack técnica

```
Python 3.11
├── yfinance                 — dados de mercado (preços, câmbio, VIX, SPY)
├── scikit-learn             — RandomForestClassifier, GradientBoostingClassifier, SGDClassifier
├── pandas / numpy           — processamento de dados e cálculo de features
├── joblib                   — serialização de modelos
├── matplotlib               — geração de gráficos
└── pandas-market-calendars  — calendários de feriados por bolsa para cálculo de datas-alvo

GitHub Actions               — automação diária gratuita
Gmail SMTP                   — entrega de email HTML
```

---

## Confiabilidade

O pipeline tem múltiplas camadas independentes de proteção, desde antes da primeira linha de código rodar até depois do último arquivo ser enviado ao repositório.

### Antes da execução: testes unitários

8 testes automáticos correm no GitHub Actions **antes** do `main.py`. Se algum falhar, o pipeline para imediatamente — os modelos não treinam com dados potencialmente corrompidos.

```
pytest tests/ -v
```

| Arquivo | Testes | O que valida |
|---------|--------|--------------|
| `tests/test_features.py` | 2 | RSI14 sempre em [0, 100] com dados aleatórios e monotônicos |
| `tests/test_ensemble.py` | 2 | Probabilidades do ensemble e por modelo sempre em [0, 1]; direção coerente com a probabilidade |
| `tests/test_pnl.py` | 4 | Breakeven > preço de compra quando fees > 0; breakeven == preço de compra quando fees = 0; conversão USD→EUR correta |

Todos os testes usam dados sintéticos — sem chamadas de rede, sem arquivos em disco.

### Durante a execução: validação dos dados de mercado

| Proteção | Gatilho | Comportamento |
|----------|---------|---------------|
| Forward fill do VIX/SPY | VIX ou SPY retorna NaN nos últimos 3 dias | Usa o valor de T-1; exibe faixa âmbar de aviso no email |
| Detecção de stock split | `abs(actual / ref_price - 1) > 40%` | Marca a validação como `NaN` em vez de `False`; não penaliza o modelo por evento corporativo |

### Após a execução: rede de segurança do push

O git push usa um loop de tentativas: até 3 tentativas, com `git pull --rebase` e pausa de 15 segundos entre cada uma. Se todas as 3 falharem, `predictions_log.csv` e `ensemble_weights.json` são salvos como artifact do GitHub Actions (retidos por 7 dias), permitindo recuperação manual sem perda de dados.

```
for i in 1 2 3; do
    git push && break
    git pull --rebase && sleep 15
done
```

---

## Contexto sobre acurácia

- Uma previsão direcional aleatória tem 50% de acurácia por definição.
- Este sistema tem como alvo 55–65% de acurácia direcional **apenas nos tickers da carteira**.
- Acurácia abaixo de 52% ao longo de 30+ validações da carteira é sinal de degradação.
- A acurácia não tem significado estatístico antes de ~30 validações por ticker — o sistema precisa de tempo para construir uma amostra representativa.
- Nenhum número de acurácia, por si só, justifica decisões financeiras — este é um projeto de análise pessoal, não aconselhamento financeiro.

---

## Changelog

### Migração: Jupyter Notebook → Python modular
O sistema original era um único Jupyter notebook (AnaliseV5). Foi migrado para um package Python modular para permitir execução automática via GitHub Actions, gestão de dependências e manutenibilidade.

### Melhorias implementadas
- ✅ **Package Python modular** — `main.py` + `data/` + `features/` + `models/` + `portfolio/` + `reports/`
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
- ✅ **Horário movido para após o fecho dos mercados EUA** — cron reajustado para ~20h30 UTC (22h30 Barcelona CEST); NYSE/NASDAQ fecham às 20h00 UTC, por isso todos os tickers têm o preço de fecho disponível no momento da validação — corrige os ícones ✅/❌ em falta para ações americanas e crypto
- ✅ **`predictions_log_public.csv`** — versão anonimizada do log de auditoria (sem tickers, sem preços) publicada diariamente no repo público; contém `asset_type` (portfolio/watchlist), direção, confiança, resultado e votos individuais dos modelos — permite a qualquer pessoa verificar a acurácia real
- ✅ **Seção Confiabilidade** — todos os mecanismos de proteção reunidos num único lugar: testes unitários (antes da execução), forward fill VIX/SPY e detecção de split (durante a execução), retry do push e artifact de emergência (após a execução)
- ✅ **Tags semânticos no git** — `v1.0.0` (Semana 1: estabilidade), `v1.1.0` (Semana 2: observabilidade), `v1.2.0` (Semana 3: repo público); cada tag ancora um marco no histórico git com mensagem descritiva
- ✅ **Gap no Walk-Forward Validation** — `TimeSeriesSplit(n_splits=5, gap=1)`; `gap=1` insere um dia entre cada fold de treino e o de validação, evitando lookahead no mesmo dia e produzindo estimativas de acurácia mais honestas
- ✅ **Feature de regime de mercado (`vix_regime`)** — label discreta baseada no VIX (0 = baixo, 1 = médio, 2 = alto) adicionada a `FEATURE_COLS`; os modelos SGD detetam automaticamente a mudança no número de features ao carregar e reinicializam — nunca corrompem modelos existentes silenciosamente
- ✅ **Matriz de correlação no email** — heatmap de retornos diários da carteira (janela 120 dias) embebido diretamente como PNG base64 no email HTML; sem anexos; fallback silencioso se matplotlib não estiver disponível
- ✅ **Cenários de projeção ETF para longo prazo** — a coluna 10 anos mostra três cenários fixos (Pess = 3% · Base = 8% · Otim = 15%) em vez de uma única taxa histórica; elimina projeções enganadoras para tickers com historial curto ou não representativo (e.g., SGLN.L em GBX)
- ✅ **README educativo** — `README_educativo.md` (PT-BR): documentação detalhada de todas as features com fórmulas e autores originais, exemplo numérico de pesos adaptativos, historial de bugs, e explicação completa do pipeline
- ✅ **`predictions_log_public.csv` no repo público** — log de auditoria anonimizado publicado diariamente a par dos gráficos com atraso de 10 dias; contém `asset_type` (portfolio/watchlist), direção, confiança, resultado e votos individuais dos modelos — acurácia verificável por qualquer pessoa

---

## Roadmap

> **Princípio:** Estabilidade → Observabilidade → Publicação → Modelo → Features avançadas. Nunca o contrário. Um modelo melhor num pipeline instável produz resultados melhores em que não podes confiar.

### Semana 1 — Estabilidade do pipeline

Antes de qualquer melhoria de modelo, o pipeline tem de ser à prova de falha.

| # | Item | Descrição |
|---|------|-----------|
| 1 | ✅ Retry no git push | 5 linhas no YAML. Elimina perda de dados por falha transitória no push. |
| 2 | ✅ Artefacto GitHub em push falhado | Upload do `predictions_log.csv` como artefacto do workflow se o push falhar — safety net para recuperação manual. |
| 3 | ✅ Forward fill para NaN em VIX/SPY | Usar o valor T-1 se T-0 devolver NaN; adicionar aviso no email quando acontece. |
| 4 | ✅ Detecção de stock split | Variação >40% num dia marca validações em aberto como `NaN` em vez de `False` — evita penalizar modelos por acção corporativa. |

### Semana 2 — Observabilidade

O pipeline corre mas não te diz quando está a degradar. Isso muda aqui.

| # | Item | Descrição |
|---|------|-----------|
| 5 | ✅ Feature importance drift como alerta | O `model_metadata.csv` já guarda as importances diárias — lê o ficheiro e adiciona um painel de drift no email com correlação de Spearman. |
| 6 | ⬜ Telegram como fallback de email | ~20 linhas; activa quando o Gmail falha — garante que o relatório diário é sempre entregue. *(adiado para o final)* |
| 7 | ✅ Badge dinâmico no repo público | Badge `shields.io/github/last-commit` — actualiza automaticamente em cada visualização. |

### Semana 3 — Repo público completo

Preparar tudo para a publicação.

| # | Item | Descrição |
|---|------|-----------|
| 8 | ✅ `predictions_log_public.csv` | Versão anonimizada do log (sem tickers, sem preços) — prova de acurácia real para qualquer pessoa que abra o repo público. |
| 9 | ✅ Secção "Confiabilidade" no README | Agrupar testes unitários + fallbacks + retry numa secção única. |
| 10 | ✅ Git tags semânticos | Tag em cada marco de versão (`v1.0.0`, `v1.1.0`, …) para ancorar o changelog no histórico git. |

### Semanas 4–5 — Qualidade do modelo

Só aqui. Não antes. O pipeline tem de estar estável antes de mexer no modelo.

| # | Item | Descrição |
|---|------|-----------|
| 11 | ✅ Walk-Forward Validation | `TimeSeriesSplit(n_splits=5, gap=1)` — `gap=1` impede lookahead entre o fold de treino e o de validação, produzindo estimativas de acurácia mais honestas. |
| 12 | ✅ Regime de mercado como feature | Label `vix_regime` (0 / 1 / 2) com base em thresholds de VIX (< 15 / 15–25 / ≥ 25); adicionada a `FEATURE_COLS`; os modelos SGD reinicializam automaticamente ao detectar mudança no número de features. |

### Semana 6 — Relatório

| # | Item | Descrição |
|---|------|-----------|
| 13 | ✅ Matriz de correlação no email | Heatmap de correlação de retornos diários da carteira (janela de 120 dias) embebido como PNG base64 no email HTML; fallback silencioso se matplotlib não estiver disponível. |
| 14 | ✅ Cenários de projeção ETF | A coluna 10 anos mostra três cenários fixos: Pess = 3% · Base = 8% · Otim = 15%; elimina a taxa histórica enganadora para tickers com historial curto ou não representativo. |

### Semanas 7–8 — Publicação

| # | Item | Descrição |
|---|------|-----------|
| 15 | ✅ READMEs finais (EN + PT) | Revisão completa de ambos os READMEs, incluindo `README_educativo.md` (PT-BR) com todas as features novas e roadmap atualizado. |
| 16 | ✅ Lançamento do repo público | `smart-wallet-ml` publicado com `predictions_log_public.csv`, gráficos com atraso de 10 dias e README gerado automaticamente. |
| 17 | ⬜ Artigo LinkedIn | Contar a história do projecto — do notebook ao pipeline ML automatizado. *(conteúdo externo, não código — sem prazo fixo)* |

### Depois da publicação — sem prazo fixo

| # | Item | Descrição |
|---|------|-----------|
| 18 | ⬜ Features de eventos fundamentalistas | Datas de earnings, semanas FOMC, expiração de opções. Requer API externa fiável; cobertura europeia é limitada. |
| 19 | ⬜ Regressor de preço D+1 | Só com 1 ano de dados limpos acumulados. |

### Framework de investigação — 38 modelos em 12 famílias (base do doutoramento)

Este projeto é o domínio principal de um framework de investigação ML multi-domínio desenvolvido no âmbito de um programa de doutoramento. As mesmas famílias de modelos são aplicadas em três domínios de previsibilidade crescente: loteria (aleatório por definição), mercado financeiro (ruidoso) e e-commerce (padrões sazonais). A hipótese central é que a performance dos modelos escala com a previsibilidade do domínio.

Os 25 modelos originais estão implementados e testados. As camadas transversais (fases 9-15) estão completas. As fases 16-20, que cobrem a nova geração de modelos (arquitecturas eficientes pós-2022, foundation models, incerteza calibrada e detecção de drift), estão aprovadas e pendentes de implementação.

| Fase | Família | Modelos | Carteira | Mega Sena |
|------|---------|---------|----------|-----------|
| 0 | Clássico base | RF, GB, SGD | ✅ | ✅ |
| 1 | Estado oculto | Markov, HMM | ✅ | ✅ |
| 2 | Clássico avançado | XGBoost, LightGBM, CatBoost, SVM | ✅ | ✅ |
| 3 | Séries temporais | ARIMA, SARIMA, ETS, Holt-Winters, Prophet | ✅ | ✅ |
| 4 | Neural recorrente | LSTM, GRU | ✅ | ✅ |
| 5 | Neural com atenção | Transformer, TFT, N-BEATS | ✅ | ✅ |
| 6 | Bayesiano | Gaussian Process, BNN (MC Dropout) | ✅ | ✅ |
| 7 | Generativo | VAE, GAN | ✅ | ✅ |
| 8 | Reinforcement | DQN, PPO | ✅ | — |
| 9 | Avaliação | Diebold-Mariano, McNemar, Ljung-Box, métricas por domínio | ✅ | ✅ |
| 10 | Explicabilidade | SHAP, Attention weights, LIME | ✅ | ✅ |
| 11 | Meta-learning | Stacking, Optuna | ✅ | ✅ |
| 12 | Rastreamento | MLflow (pipeline diário + research runner), DVC | ✅ | ✅ |
| 13 | Teoria da informação | Entropia de Shannon, MI, Permutation entropy, Transfer entropy | ✅ | ✅ |
| 14 | Transfer Learning | MMD, CORAL, Fine-tuning, Cross-domain evaluation | ✅ | ✅ |
| 15 | Email report | Tabela eToro por lote, secção ETF, recomendação mensal | ✅ | — |
| 16 | Contrarian / Testes de sanidade | CB (Contrarian Baseline), EWI, PEL | ⏳ | — |
| 17 | Arquitecturas eficientes (pós-2022) | TCN, DLinear, NLinear, PatchTST | ⏳ | ⏳ |
| 18 | Foundation Models (2023-2024) | Chronos (Amazon), TimesFM (Google), Moirai (Salesforce) | ⏳ | ⏳ |
| 19 | Incerteza calibrada | Conformal Prediction (MAPIE) | ⏳ | ⏳ |
| 20 | Detecção de drift | ADWIN, Page-Hinkley | ⏳ | — |

---

## Sobre

Construído por **Vicky Costa** — Analista de Dados | Estudante de Ciência de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycosta/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
