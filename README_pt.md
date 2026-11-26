# Carteira Inteligente: um teste honesto de previsão de mercado de curto prazo

Um pipeline automático que roda todos os dias úteis depois do fechamento, prevê a direção de 1 a 3 dias de uma carteira real de ETFs e de uma watchlist de 615 ativos, registra cada previsão **antes** do resultado, confere com o fechamento oficial e faz uma pergunta com disciplina estatística: **alguma coisa aqui é melhor que o acaso?**

A resposta até agora é a parte mais útil do projeto: **a direção de curto prazo não tem vantagem demonstrada, mas o tamanho do movimento (volatilidade) é previsível.** Os dois resultados estão medidos, documentados e reproduzíveis abaixo.

[![Última atualização](https://img.shields.io/github/last-commit/srxkatsumi/smart-wallet-ml?label=atualizado&color=brightgreen)](https://github.com/srxkatsumi/smart-wallet-ml/commits/main)
[![GitHub Actions](https://img.shields.io/badge/Automatizado-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/srxkatsumi/smart-wallet-ml/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-GitHub%20Pages-1a1740)](https://srxkatsumi.github.io/smart-wallet-ml/pt/)

> ⚠️ **Projeto de estudo. Nada aqui é recomendação de investimento, e as previsões não devem servir de base para decisões reais.**

---

## Primeiro, os resultados

| Pergunta | Resposta | Evidência |
|---|---|---|
| O modelo prevê a direção de 1 a 3 dias melhor que uma regra ingênua? | **Não.** "Sempre ▲" ganha de 2 a 7 pontos percentuais em todos os horizontes | Walk-forward de 2 anos na carteira, 11.472 previsões (`scripts/estudo_vantagem.py`) |
| Confiança maior quer dizer mais acertos? | **Não.** Correlação de Spearman entre confiança e acerto de -0,01 a +0,01 | Mesmo estudo; nenhum recorte (ativo, horizonte, regime do VIX, acordo entre modelos) sobrevive ao Benjamini-Hochberg entre 57 recortes |
| A confiança serve como probabilidade? | **Não.** Brier skill score de -0,16 a -0,19 contra dizer sempre 50% | Log ao vivo; a confiança agora é calibrada e encolhida para 50% (λ = 0) |
| Os pesos adaptativos do ensemble ajudam? | **Não.** Mesmo voto que a média simples em ~54 mil previsões por horizonte | `scripts/ablation_ensemble_weights.py`; a produção usa a média simples |
| O tamanho do movimento é previsível? | **Sim.** A volatilidade da EWMA bate a variância histórica do ativo (QLIKE 5% a 10% menor, nas duas metades do período) | `scripts/estudo_volatilidade.py`; GARCH e HAR não bateram a EWMA |
| Dá para prever um "movimento grande" amanhã (mais de 1 ATR)? | **Talvez.** Logística: AUC 0,61, BSS +1,4% contra a taxa de base, bem calibrada, mas p = 0,146 depois da correção para múltiplos testes | Mesmo estudo; agora roda como desafiante ao vivo |

Um resultado negativo bem medido é um resultado. O projeto mantém as previsões de direção rodando (são baratas e mantêm a avaliação honesta), mostra-as calibradas ao que os dados sustentam e está mudando a ênfase para risco e volatilidade, onde há evidência.

---

## Como funciona

```
GitHub Actions (dias úteis, depois do fechamento)
  │
  ├─ Download: 615 tickers + VIX, SPY, BTC, GLD (yfinance); o último pregão que
  │             faltar é montado pelos candles de 1 hora (bolsas europeias e asiáticas atrasam)
  ├─ Validação: cada previsão cuja data-alvo chegou é conferida contra o
  │             fechamento oficial da barra de referência
  ├─ Features: 33 por ativo (tendência, momento, volatilidade, volume, contexto, calendário)
  ├─ Calibração: confiança contra acertos reais (encolhimento até 20 dias alinhados, depois isotônica)
  ├─ Treino e previsão: ensemble de produção (RF + GB + SGD) por ativo e horizonte
  ├─ Pesquisa: 13 famílias de modelos + desafiantes ao vivo, só relatório
  ├─ Risco: oscilação esperada por ativo (EWMA), queda desde o pico, VaR; contexto; eventos
  └─ Publicação: emails, site no GitHub Pages, logs só de acréscimo commitados no repositório
```

---

## Metodologia

**Alvo.** Para o horizonte N (1, 2, 3 pregões), o rótulo é alta ou baixa só quando o movimento passa de `0,3 × ATR14 × √N`; movimentos menores ficam fora do **treino** para reduzir o ruído dos rótulos.

**Regra de acerto.** A previsão acerta quando o fechamento oficial na data-alvo é ≥ ao de referência (para ▲) ou ≤ (para ▼). Duas ressalvas conhecidas, ambas na lista de metodologia:
- um preço que não mudou conta hoje como acerto para as duas direções;
- a avaliação conta todos os dias, inclusive os movimentos pequenos que o modelo nunca viu no treino. A acurácia nos movimentos acima do limiar de treino já é calculada (`acc_mov_relevante`) e vai aparecer lado a lado.

**Sem olhar o futuro.**
- Os horizontes contam a partir da última barra real de cada ativo (`ref_date`), no calendário da própria bolsa (NYSE, Xetra, Paris, Madri, Tóquio, Hong Kong...).
- O contexto externo usa o fechamento do dia anterior.
- O walk-forward retreina só com dados anteriores a cada dia de teste, com intervalo igual ao horizonte.

**Significância.**
- A unidade de análise é o dia de mercado: centenas de ativos no mesmo dia não são observações independentes.
- A acurácia é testada com um teste t sobre a acurácia diária, e só é reportada depois de 100 previsões em 20 dias.
- Comparações entre modelos usam Diebold-Mariano sobre a média diária da diferença de perda, com h−1 lags e a correção de amostra pequena de Harvey-Leybourne-Newbold.
- Famílias e recortes são corrigidos por Benjamini-Hochberg.
- Toda acurácia aparece com IC 95% e a referência "sempre ▲", e não contra 50% fixo.

**Jinshi, o portão de validação.** Nenhum sinal vai para a produção sem passar nos quatro critérios:
1. bater a referência com significância corrigida por BH;
2. ter um efeito maior que o custo de operar (0,15%);
3. vencer nas duas metades do período;
4. ter uma razão econômica declarada.

---

## Modelos

**Produção (o que o email diário mostra).** Por ativo e por horizonte, três classificadores combinados pela média simples:

| Modelo | Configuração | Papel |
|---|---|---|
| Random Forest | 100 árvores, profundidade 5, calibração isotônica | Robusto a features ruidosas ou irrelevantes |
| Gradient Boosting | 100 estimadores, profundidade 3, taxa 0,05, calibração isotônica | Corrige os erros residuais em sequência |
| SGD Classifier | log loss, L2, atualização incremental, recalibração completa mensal | Contrapeso linear aos dois modelos de árvore |

A probabilidade combinada é calibrada contra os acertos reais antes de ser mostrada.

**Famílias de pesquisa (só relatório).** Boosting clássico e SVM (XGBoost, LightGBM, CatBoost, SVM), estados ocultos (cadeia de Markov, HMM), séries temporais (ARIMA, SARIMA, ETS, Holt-Winters, Prophet), redes recorrentes (LSTM, GRU), atenção (Transformer, TFT e N-BEATS simplificados), bayesianos (processo gaussiano, BNN com MC dropout), generativos (VAE, discriminador de GAN), aprendizado por reforço (DQN, PPO), testes de sanidade contrarian, arquiteturas eficientes (TCN, DLinear, NLinear, PatchTST), foundation models (Chronos, inspirados no TimesFM e no Moirai), predição conformal e detecção de drift (ADWIN, Page-Hinkley). São 13 famílias; o consenso delas aparece no email, mas nunca muda a produção.

**Desafiantes ao vivo (registrados todo dia, julgados pelo Jinshi com 20 dias alinhados).**
- **Ensemble ampliado:** a produção mais as 13 famílias, com pesos encolhidos para iguais até haver desempenho sustentado.
- **Regime:** o trio da produção mais VIX/VIX3M, a variação do juro de 10 anos e o dólar.
- **Volatilidade:** uma logística para "movimento grande amanhã" e a volatilidade da EWMA para os próximos 1, 3 e 5 pregões.
- **Experimento de regressão:** um Random Forest de regressão com intervalo conformal de 90%, comparado por Diebold-Mariano pareado contra uma heurística de ATR. Tem email próprio; ver `models/regressor_experimento.py`.

**MaoMao.** Uma pequena rede neural pulsada (neurônios LIF) que transforma sensores diários do mercado num humor e aprende quais medos se confirmaram três dias depois. Aparece no site e ainda não passou pelo Jinshi.

---

## Features (33 por ativo)

| Grupo | Features |
|---|---|
| Tendência e momento | distância à SMA20/SMA50, cruzamento das médias, RSI14, MACD (linha, sinal, histograma), retornos de 1 dia, 5 dias, 1, 3, 6 e 12 meses |
| Volatilidade e volume | largura e posição nas Bandas de Bollinger, ATR14, volatilidade de 10 dias, razão de volume, tendência do OBV (limitada), distância ao VWAP |
| Extremos | distância à máxima e à mínima de 52 semanas |
| Contexto (dia anterior) | retorno do SPY, nível, variação e regime do VIX, retornos do BTC e do ouro, correlação de 20 dias com o SPY |
| Calendário | dia da semana, mês, semana de vencimento de opções |
| Classe do ativo | ação, ETF de ações, ETF de commodity, cripto |

As features de calendário entram como candidatas, não como fatos estabelecidos. As anomalias de calendário documentadas (dia da semana, janeiro, "sell in May") tendem a enfraquecer depois de publicadas (McLean & Pontiff, *Journal of Finance*, 2016), e nada nos resultados deste projeto sugere o contrário.

---

## O que o sistema produz

**Emails.**
| Email | Quando | Conteúdo |
|---|---|---|
| Diário | Toda execução | Saúde do sistema, eventos, risco, previsões calibradas, ensemble ampliado, acurácia com IC, drift, correlação, as 13 famílias |
| Semanal | Último pregão da semana (sábado de manhã) | Famílias com IC e p corrigido, calibração (Brier), diversificação, placares dos desafiantes |
| Experimentos | Toda execução | O desafiante de regressão contra o campeão |
| Alerta | Só em dias de anomalia | VIX entrando no regime alto ou variando 15% ou mais num dia, ou um ativo da carteira movendo-se 2,5 vezes ou mais o próprio ATR |

**Site** ([srxkatsumi.github.io/smart-wallet-ml](https://srxkatsumi.github.io/smart-wallet-ml/pt/), em português e inglês):
- **Início e Favoritos:** gráficos e previsões da carteira.
- **Watchlist:** 615 ativos; clicar em qualquer um abre os mesmos gráficos.
- **Modelos, Experimentos e Alertas.**

**Logs.** O `output/predictions_log.csv` só recebe linhas novas e nunca é reescrito. Cada linha guarda a versão do código, a regra de calibração, um hash da barra de entrada e o horário da execução, para que qualquer número do email possa ser rastreado.

---

## Universo

- **Carteira (pública):** 6 ETFs de acumulação: ICGA.DE, PPFB.DE, EXUS.MI, LYP6.DE, IS3N.DE, EUNN.DE. Posições que precisam ficar privadas vêm de um secret do GitHub (`PORTFOLIO_PRIVADO`) e só aparecem nos emails, nunca no repositório nem no site.
- **Watchlist:** 615 tickers entre ações americanas, europeias, espanholas, japonesas, de Hong Kong e chinesas listadas nos EUA, ETFs UCITS e americanos, títulos, commodities e cripto (`config/watchlist.json`). Todos os preços aparecem em euro, convertidos pelo câmbio de cada dia.

---

## Automação e confiabilidade

- **Horário.** O workflow roda na madrugada depois de cada pregão, às 03:07 UTC de terça a sábado, para os emails chegarem por volta das 5h30 em Barcelona no inverno (6h30 no verão). Uma execução completa leva cerca de 75 minutos. Três disparos de reserva mais tarde, às 03:37, 04:13 e 06:17 UTC, cobrem agendamentos descartados. Todos evitam os minutos :00 e :30, onde o GitHub mais atrasa ou descarta agendamentos. Uma execução é pulada se outra já rodou no mesmo dia.
- **Antes da execução.** 271 testes em 41 arquivos rodam primeiro e param o pipeline se falharem. Os 8 testes dos modelos clássicos rodam em processo separado, porque o XGBoost e o PyTorch entram em conflito no mesmo processo no macOS.
- **Durante a execução.**
  - Toda etapa que pode falhar sem bloquear fica registrada e aparece no bloco de saúde do email.
  - Ativos com a última barra velha são sinalizados.
  - O alerta ignora dados velhos.
- **Depois da execução.** A validação confere a saída, o push tem nova tentativa com recuperação, e um artefato de emergência é salvo se o push falhar.

**Limitação conhecida: o repositório é o banco de dados.** Logs, estado e os dados do site são commitados todos os dias. É gratuito e totalmente auditável, mas causou a maior parte dos incidentes operacionais (corridas entre pushes, rebase travado; ver [README_Errors.md](README_Errors.md)). Levar os dados para um Postgres pequeno com uma API está no roadmap.

---

## Estrutura do repositório

```
main.py                 pipeline diário
config/                 carteira, watchlist (615), configurações, calendário macro
data/                   download, calendários das bolsas, armazenamento e migrações do log
features/               as 33 features
models/                 ensemble de produção, calibração, validador, famílias de pesquisa
research/               runner de pesquisa e desafiantes ao vivo (ampliado, regime, volatilidade)
evaluation/             significância, Diebold-Mariano, anomalia, risco, eventos, saúde, rastreabilidade
jinshi/                 portão de validação e backtests
maomao/                 o "pet", uma rede neural pulsada
reports/                emails diário, semanal, de experimentos e de alerta
docs/                   site no GitHub Pages (EN + PT) e os scripts de exportação
scripts/                estudos e ablações (vantagem da direção, volatilidade, regime, pesos)
tests/                  41 arquivos de teste
output/                 logs só de acréscimo e estado (commitados pelo bot)
```

---

## Stack técnica

Python 3.11 · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch · hmmlearn · Prophet · Chronos (Amazon) · SHAP · LIME · MLflow · yfinance · pandas-market-calendars · Matplotlib · pytest · GitHub Actions · GitHub Pages

---

## Limitações

- As previsões de direção não têm vantagem demonstrada; trate-as como experimento, não como sinal.
- Os modelos não leem notícias. Resultados trimestrais e decisões de bancos centrais são avisados no email, mas não entram nos modelos.
- O histórico de avaliação alinhado começou em 07/11/2026. Vários testes ao vivo precisam de cerca de 20 pregões antes de dizer alguma coisa.
- A fonte de dados gratuita (yfinance) tem falhas ocasionais; elas são completadas pelos candles de 1 hora ou sinalizadas, nunca inventadas em silêncio.

---

## Mais

- [CHANGELOG_pt.md](CHANGELOG_pt.md): o que mudou e quando.
- [ROADMAP_pt.md](ROADMAP_pt.md): o que está planejado.
- [README_Errors.md](README_Errors.md): todos os incidentes, com causa e correção.
- [README.md](README.md): este README em inglês.
- [README_educativo.md](README_educativo.md): a explicação longa e didática.

---

## Sobre

Construído por **Vicky Costa**, Analista de Dados.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vickycosta-blue)](https://www.linkedin.com/in/vickycostasanches/)
[![Blog](https://img.shields.io/badge/Blog-vickycosta.com-purple)](https://www.vickycosta.com)
