# MaoMao

O pet do site (`docs/index.html`). Ao contrário do protótipo inicial (um `if/else` lendo números já
calculados), o MaoMao é uma rede neural pulsada de verdade: ele lê o mercado todos os dias, forma um humor
a partir disso, e **aprende**, com o tempo, quais sinais realmente valem a pena temer ou confiar — a partir
do que aconteceu depois, não da previsão de direção do ensemble.

**Dois humores independentes, duas redes separadas** (mesma arquitetura, sensores diferentes):

| | Carteira | Mercado |
|---|---|---|
| Vê | Só os 7 ETFs que você tem investidos de verdade | O watchlist inteiro (500+ tickers) |
| Estado | `output/maomao_state.json` | `output/maomao_market_state.json` |
| Orquestração | `maomao.daily.run_daily_step` | `maomao.daily.run_market_step` |
| `ANOMALY` vem de | `anomalia.json["alerta"]` (VIX **ou** um ETF específico rompendo o próprio ATR) | só o VIX (`anomalia.json["market"]`) — uma queda isolada de 1 ETF da carteira não é "o mercado" |

São dois `Brain` (`maomao/brain.py`) independentes, não uma rede de 8 pools misturada — mais simples de
auditar, e cada um aprende e esquece no seu próprio ritmo.

## De onde veio a ideia

O MaoMao é inspirado no [Pestinha](https://github.com/mille-okws/pestinha) (`mille-okws`), um pet de
desktop cujo comportamento é decidido por uma rede neural pulsada de 470 neurônios LIF (integração e
disparo com vazamento), com canais sensoriais que habituam, pools de interneurônios (humor) com pesos
fixos, motores de decisão, e aprendizado real por uma regra de três fatores modulada por recompensa
(carinho do usuário).

**Isto é uma implementação original, escrita do zero para este projeto — nenhuma linha do código do
Pestinha foi copiada.** Essa decisão foi deliberada, não só por respeito à autoria: o Pestinha é licenciado
sob GPL-3.0-or-later, uma licença copyleft — copiar o código deles para dentro deste repositório forçaria
(no mínimo o módulo, possivelmente o projeto todo, dependendo de como fosse integrado) a ficar sob a mesma
licença. Como este é um projeto de carteira financeira, não queríamos que uma funcionalidade de diversão
arrastasse essa obrigação para o resto do repositório. O que se reaproveitou foi a **arquitetura** (ideias
e estrutura não são protegidas por direitos de autor, só a expressão/código em si) — adaptada do zero para
dados financeiros em vez de atividade de computador/carinho de usuário.

Licença do Pestinha, para quem quiser conferir: [GNU GPL-3.0-or-later](https://github.com/mille-okws/pestinha/blob/main/LICENSE).

## Diferença de cadência (importante)

O Pestinha simula continuamente, em passos de 1 milissegundo, porque reage em tempo real a mouse e
teclado — faz sentido, o estímulo dele muda a cada instante. **O mercado só fecha uma vez por dia.** Não há
nada de real para simular a cada milissegundo entre dois fechamentos; fazer isso seria só gastar CPU
simulando ruído.

Por isso o MaoMao chama a rede **uma vez por execução do pipeline** (`maomao/daily.py:run_daily_step`),
com o snapshot sensorial daquele dia — mas ainda roda a dinâmica LIF de verdade dentro dessa única chamada
(300 sub-passos, limiar, período refratário, ruído — ver `maomao/brain.py:Brain.step`), até a taxa de
disparo de cada pool se assentar. É a mesma técnica que `core.step(dt_ms)` do Pestinha usa, só que invocada
uma vez por dia em vez de a cada frame.

## Arquitetura

```text
output/predictions_log.csv ──► canais sensoriais (0..1) ──► pools de humor (LIF)
output/anomalia.json                                              │
                                                                   ▼
                                                     CALM · FEARFUL · EUPHORIC · CONFUSED
                                                                   │
                                                                   ▼
                                                   fila de "apostas" pendentes (3 dias)
                                                                   │
                                            retorno real dos 3 dias seguintes já aconteceu
                                                                   │
                                                                   ▼
                                          recompensa (+/-) ──► ajusta as sinapses aprendidas
```

### Canais sensoriais (`maomao/sensors.py`)

Calculados a partir dos artefactos que o pipeline principal já gera — não reimplementa nenhuma lógica de
deteção, só lê. `compute_sensors` (carteira) e `compute_market_sensors` (mercado) compartilham o mesmo
núcleo (`compute_sensors_core`), só trocando de onde vêm `RET_UP/DOWN`, `STREAK_*` e `DISAGREEMENT` —
carteira sobre os 7 ETFs, mercado sobre o watchlist inteiro:

| Canal | Carteira | Mercado |
|---|---|---|
| `RET_UP` / `RET_DOWN` | Retorno real médio dos 7 ETFs (`actual_price`/`ref_price`) | Retorno real médio do watchlist inteiro |
| `VOL_SPIKE` | Maior múltiplo de ATR do dia, entre os 7 ETFs (`anomalia.json`) | O mesmo número da carteira — aproximação, ver "Limitações do backfill" |
| `VIX_SPIKE` | Variação do VIX no dia | Igual — é sempre o mesmo VIX, não é específico de carteira nem mercado |
| `STREAK_UP` / `STREAK_DOWN` | Sequência de dias consecutivos de alta/baixa real, carteira | Idem, watchlist inteiro |
| `ANOMALY` | `anomalia.json["alerta"]` (VIX ou um ETF específico) | Só VIX (`anomalia.json["market"]`) |
| `DISAGREEMENT` | Fração dos 7 ETFs onde RF/GB/SGD discordam | Fração do watchlist inteiro |
| `CALM_STREAK` | Sequência de dias sem alerta de carteira | Sequência de dias sem alerta de mercado |

### Pools de humor (`maomao/brain.py`)

4 pools pequenos (8 neurônios LIF cada — bem menor que os 16-20 do Pestinha, de propósito: aqui há 1
amostra por dia, não uma a cada milissegundo). As ligações canal→pool (`BASE_WEIGHTS`) são **regra**,
escritas à mão, no mesmo espírito das tabelas de pesos do Pestinha — não são aprendidas.

| Pool | Reage a |
|---|---|
| `CALM` | Sequência de dias calmos, baixa volatilidade, sem alerta |
| `FEARFUL` | Pico de volatilidade/VIX, sequência de baixas, alerta ativo |
| `EUPHORIC` | Sequência de altas, baixa volatilidade (complacência) |
| `CONFUSED` | Os três modelos discordando entre si |

### Bug real encontrado e corrigido: `POOL_SIZE` estava a diluir o ganho, não só o ruído

`POOL_SIZE` deveria só controlar quantos neurônios ruidosos são promediados pra suavizar a taxa de saída —
o mesmo estímulo deveria dar o mesmo humor médio independente de `POOL_SIZE`. Até esta versão, não dava: o
drive de cada sub-passo era dividido por `POOL_SIZE` **e** a taxa de saída também dividia por `POOL_SIZE`
— dupla diluição. Medido (`scripts/tune_maomao_gain.py`): só trocar `POOL_SIZE` de 1 para 32, sem mudar
nenhum outro número, fazia `FEARFUL` cair de 0.87 para 0.003 pro mesmo estímulo exato.

Corrigido: o drive não divide mais por `POOL_SIZE` (cada neurônio do pool recebe o drive inteiro — coding
populacional, não diluição). Isso tornou a rede ~8x mais sensível com os `BASE_WEIGHTS` antigos, que nunca
tinham sido calibrados de verdade (só pareciam razoáveis porque o bug os diluía) — quase qualquer sinal
real passou a saturar no teto de disparo (~0.87, dado `REFRACTORY=2`), perdendo a gradação entre "medo
moderado" e "medo forte". Introduzido `WEIGHT_GAIN = 0.25` (aplicado a `BASE_WEIGHTS` e aos pesos
aprendidos, em `_effective_weight`), escolhido testando contra cenários sintéticos até recuperar gradação
sensata nos quatro pools — ver `scripts/tune_maomao_gain.py` pra reproduzir a calibração.

## O que é aprendido (e o que nunca é)

**O que é aprendido:** só as sinapses canal→`FEARFUL` e canal→`EUPHORIC` (`maomao/learning.py` +
`Brain.learn`). Quando um desses pools dispara forte num dia, isso entra numa fila de apostas pendentes.
Só quando os **3 dias reais seguintes já têm retorno validado** é que a aposta é julgada:

- `FEARFUL` é reforçado quando o medo "se confirma" — a queda continua nos dias seguintes.
- `FEARFUL` é punido quando é alarme falso — o mercado recupera.
- `EUPHORIC` é reforçado quando a alta continua; punido quando reverte.

Isso testa **clustering de retorno/volatilidade** — um fenômeno real e parcialmente documentado em
finanças (volatilidade tende a vir em grupos) — não a direção do próprio ensemble.

**O que o MaoMao nunca aprende, por desenho:** a direção prevista (`direction`/`confidence` do ensemble)
nunca entra como canal sensorial, nunca é usada como recompensa, e nenhum pool reage a "o modelo disse que
vai subir". Já medimos (`README_Errors.md`, ERR-013) que esse sinal é estatisticamente indistinguível de
aleatório em produção — ensinar o pet a confiar nele seria recriar o mesmo problema de falsa confiança
dentro de um sistema mais difícil de auditar.

**O que é regra, não aprendido:** as ligações canal→pool (`BASE_WEIGHTS`), os limiares de disparo da rede
LIF, e o limiar/janela da fila de apostas (`FIRE_THRESHOLD`, `HORIZON_DAYS` em `maomao/learning.py`).

## Replay histórico (`maomao/backfill.py`) — "dar 2 anos de experiência ao recém-nascido"

Sem isso, o MaoMao só acumularia 1 observação por dia a partir de quando este commit for publicado — meses
até ter sinapses minimamente confiáveis. `maomao/backfill.py` resolve isso revivendo o mercado real que já
aconteceu, em lote, pelos dois lados:

- **Carteira:** baixa o histórico completo (desde a data real de listagem) de cada um dos 7 ETFs via
  yfinance (`period="max"`) mais o VIX, e revive cada dia em ordem cronológica através do mesmo motor
  (sensores → rede → fila de apostas) que a produção usaria. Não é dado inventado — é o mercado real, só
  processado de uma vez em vez de esperar em tempo real. Reusa `evaluation.anomaly.detect_anomaly()` de
  verdade (esse módulo só depende de preço/ATR/VIX, nunca do ensemble — seguro pra qualquer recorte
  histórico causal). `DISAGREEMENT` fica sempre 0.0 (nunca existiu ensemble correndo nessas datas).
- **Mercado:** usa os ~6 meses que já temos documentados em `output/predictions_log.csv` (watchlist
  inteiro) em vez de baixar histórico novo de 500+ tickers — como pedido, aproveita o que já existe. Só o
  VIX é baixado (pequeno, rápido), pra reconstruir `anomaly["market"]` de cada dia real (o pipeline só
  guarda o `anomalia.json` de hoje, não um histórico).

**Resultado real do replay (gerado ao escrever este README, 2026-11-05, já com `WEIGHT_GAIN=0.25`):**

- **Carteira**, 17 anos replayed (2009-10-20 → 2026-11-05, 4371 dias de mercado): `FEARFUL` aprendeu pesos
  **negativos** nos canais de stress (`RET_DOWN -0.41`, `STREAK_DOWN -0.31`, `VIX_SPIKE -0.26`, `VOL_SPIKE
  -0.16`) e positivo em `RET_UP` (+0.17). Interpretação honesta: quando o medo disparou forte nos 7 ETFs da
  carteira, o retorno médio dos 3 dias seguintes tendeu a **recuperar**, não a continuar caindo — reversão
  à média de curto prazo depois de picos de volatilidade, um padrão real e conhecido em finanças (diferente
  de "prever direção", que já medimos ser ruído — ver ERR-013). `EUPHORIC` também aprendeu pesos pequenos e
  majoritariamente negativos (`STREAK_UP -0.10`, `CALM_STREAK -0.08`) — com o ganho calibrado, chegou a
  disparar algumas vezes nesses 17 anos, e o mesmo padrão de reversão de curto prazo apareceu, mais fraco.
- **Mercado**, ~55 dias úteis processados sobre o watchlist inteiro: com o ganho corrigido, `EUPHORIC`
  já cruzou o limiar de 50% algumas vezes e aprendeu pesos claramente positivos (`RET_UP +0.39`,
  `CALM_STREAK +0.39`, `STREAK_UP +0.33`, `DISAGREEMENT +0.28`) — ao contrário da carteira, aqui a euforia
  tendeu a **se confirmar** (ganhos continuaram) nos ~6 meses que temos. `FEARFUL` só aprendeu ajustes
  pequenos (não chegou a disparar fortes o bastante tantas vezes). Amostra ainda curta (~6 meses) — vale
  reler esse resultado à medida que mais dias reais se acumulam.

Rodar de novo: `python -m maomao.backfill` (a partir da raiz do repositório).

## Limitações do backfill (honestas, não escondidas)

- `VOL_SPIKE` no humor de mercado usa o mesmo múltiplo de ATR da carteira — não existe um cálculo de
  amplitude de volatilidade pro watchlist inteiro ainda. É uma aproximação documentada, não um cálculo de
  "o mercado" de verdade.
- O replay da carteira reconstrói ATR14/ret_1d com uma cópia da fórmula de
  `features/engineering.py:build_features` (não a função inteira, que exige contexto de SPY/BTC/ouro que
  este replay não usa) — `tests/test_maomao_backfill.py` trava que as duas fórmulas dão o mesmo número.
- SGLN.L é cotado em GBp (pence); o replay usa o preço bruto sem converter pra EUR. Isso não muda nada:
  multiplicar uma série toda por uma constante (a conversão de moeda) não muda retorno percentual nem a
  razão ATR/close — é exatamente como a produção trata isso também (ver `data/downloader.py`).

## O que falta (v1 → v2)

Cortes de âmbito deliberados, não esquecimentos:

- **Habituação por canal.** O Pestinha tem depressão sináptica (um estímulo repetido perde força,
  recupera com o tempo) em cada canal sensorial. A v1 não implementa isso — `ANOMALY`/`VOL_SPIKE` têm o
  mesmo peso no dia 1 e no dia 30 de uma crise prolongada.
- **Mais pools/canais.** O Pestinha tem 10 populações de interneurônios e 30 canais; o MaoMao começa com
  4 e 9. Dá pra crescer (ex: um pool de "fadiga" alimentado pela própria ANOMALY habituada, uma vez que a
  habituação exista).
- **Condicionamento de calendário.** O Pestinha antecipa padrões (`core/conditioning.py`, Rescorla-Wagner).
  Dava pra fazer o mesmo com `is_options_expiry`/`day_of_week`, que já existem nas features do pipeline
  principal, mas ainda não alimentam o MaoMao.
- **Interação do usuário.** Avaliado e descartado por agora (ver histórico da conversa) — o usuário não
  acessa o site todo dia, então um canal de recompensa dependente de clique/carinho ficaria sem sinal na
  maioria dos dias. Pode voltar a fazer sentido se o uso do site mudar.
- **Amplitude de mercado de verdade.** `VOL_SPIKE`/`ANOMALY` do humor de mercado ainda reaproveitam o
  cálculo da carteira (ver "Limitações do backfill"). Um cálculo de amplitude real (fração do watchlist
  rompendo o próprio ATR, não só os 7 ETFs) é o próximo passo mais honesto de melhoria.

## Persistência

Um JSON por humor — `output/maomao_state.json` (carteira) e `output/maomao_market_state.json` (mercado) —
cada um com pesos aprendidos, apostas pendentes, histórico curto de alertas e o log de humor dos últimos 30
dias. Mesmo papel que `memory/` no Pestinha, só que dois JSONs simples, porque o estado de cada um é muito
menor (9 canais × 2 pools aprendidos, não ~16 mil sinapses). Escrita atómica (ficheiro temporário e troca)
em `maomao/state.py:save_state`, path sempre explícito (nunca um global do módulo) — é assim que os dois
humores ficam independentes com o mesmo código.

Idempotente por data: o pipeline principal tem um padrão conhecido de reexecuções redundantes no mesmo dia
(`README_Errors.md`, ERR-010/ERR-013) — `maomao/daily.py` guarda a última data já processada (por humor) e
nunca reaplica aprendizado duas vezes para o mesmo dia.

## Onde aparece

O site (`docs/index.html` e `docs/pt/index.html`) lê `docs/assets/maomao_data.js` (gerado por
`docs/export_maomao.py` a partir dos dois estados) e desenha os dois humores lado a lado
(`docs/assets/pet.js`) — texto traduzido EN/PT, dados sempre os mesmos. Email e badge do README ficam como
possível próximo passo — ver histórico da conversa, não implementados ainda.
