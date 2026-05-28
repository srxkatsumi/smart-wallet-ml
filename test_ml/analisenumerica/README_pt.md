# Experimento ML — Mega Sena

> **Hipótese:** O Machine Learning consegue prever os números da Mega Sena melhor do que o acaso?
> **Resposta esperada:** Não. E temos os dados para provar.

[![Automatizado](https://img.shields.io/badge/Atualizado-Todo%20Domingo-4c8f6b?logo=github-actions)](../../.github/workflows/analise_numerica.yml)

---

## O que é isto?

Este sub-projeto aplica o mesmo pipeline de ensemble adaptativo usado para previsão de ações a um processo declaradamente aleatório: a Mega Sena. Toda semana faz o download dos resultados oficiais, retreina três classificadores, gera 5 sequências candidatas de 6 números e, após o sorteio real, verifica quantos números acertou.

O objetivo não é ganhar na loteria. O objetivo é produzir uma demonstração quantitativa e reproduzível de que o Machine Learning não consegue extrair sinal preditivo de um processo sem sinal a extrair.

Este é um **experimento negativo controlado**. Em ciência, um experimento bem desenhado que refuta uma hipótese tem o mesmo valor do que um que a confirma.

---

## A loteria

**Mega Sena** (Caixa Econômica Federal):
- Universo: 60 bolas numeradas de 1 a 60
- Sorteadas por jogo: 6 bolas
- Sorteios por semana: 3 — **Segunda**, **Quinta**, **Sábado**
- Fonte dos dados: [loterias.caixa.gov.br](https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx)

---

## Baseline aleatório — o teto teórico

Antes de rodar qualquer modelo, já sabemos o que esperar de uma estratégia completamente aleatória:

| Métrica | Fórmula | Valor |
|---------|---------|-------|
| P(qualquer número ser sorteado) | 6 / 60 | 10,0% |
| Acertos esperados por sequência | 6 × (6/60) | **0,60** |
| P(0 acertos) | C(54,6) / C(60,6) | ≈ 47,4% |
| P(1 acerto) | C(6,1)·C(54,5) / C(60,6) | ≈ 37,7% |
| P(2 acertos) | C(6,2)·C(54,4) / C(60,6) | ≈ 13,2% |
| P(3 acertos — Terno) | C(6,3)·C(54,3) / C(60,6) | ≈ 2,44% |
| P(4 acertos — Quadra) | C(6,4)·C(54,2) / C(60,6) | ≈ 0,18% |
| P(5 acertos — Quina) | C(6,5)·C(54,1) / C(60,6) | ≈ 0,0064% |
| P(6 acertos — Sena) | C(6,6) / C(60,6) | ≈ 0,000179% |

**Se o modelo ML obtiver consistentemente mais do que 0,60 acertos por sequência ao longo de muitos sorteios, isso seria extraordinário e exigiria investigação.** A hipótese nula é que isso não vai acontecer.

---

## Arquitetura ML

O mesmo ensemble de três modelos usado na previsão de ações é aplicado aqui. Cada modelo aprende a partir do histórico completo de sorteios, tratando o problema como: *"para o número X, dado o histórico de sorteios passados, qual é a probabilidade de X ser sorteado hoje?"*

### Os modelos

| Modelo | Configuração | Papel no ensemble |
|--------|-------------|-------------------|
| **Random Forest** | 100 árvores, profundidade máx. 4, class_weight="balanced" | Captura padrões não-lineares de frequência. As árvores com bootstrap resistem ao overfitting em dados ruidosos. |
| **Gradient Boosting** | 100 estimadores, lr 0.05, profundidade máx. 3 | Corrige os erros do modelo anterior iterativamente. Melhor a identificar interações subtis entre features de frequência. |
| **SGD Classifier** | log_loss, L2, class_weight="balanced" | Modelo linear. Age como regularizador — se discorda dos dois modelos não-lineares, puxa o ensemble para estimativas mais conservadoras. |

Os três modelos produzem uma probabilidade para cada um dos 60 números. O ensemble combina-as via votação ponderada adaptativa.

### Pesos adaptativos

Após cada sorteio validado, os pesos dos modelos são atualizados com decaimento exponencial:

```
peso(modelo) ∝ taxa_acertos(modelo) × Σ decaimento^(sorteios_atrás)
```

Acertos mais recentes pesam mais do que acertos antigos. Se um modelo piorar sistematicamente, a sua participação no voto diminui automaticamente.

**Nota importante:** Como os sorteios da Mega Sena são independentes e identicamente distribuídos (i.i.d.) por definição, esperamos que os pesos flutuem aleatoriamente em torno de 1/3 cada, sem nunca convergir para um vencedor consistente. Isso em si é um diagnóstico: divergência persistente dos pesos indicaria não-aleatoriedade nos dados.

### Estratégia de previsão

Para cada sorteio seguinte (Seg/Qui/Sab), o sistema gera **5 sequências**:

| Sequência | Método |
|-----------|--------|
| 1 | Top-6 números por probabilidade do ensemble (determinístico) |
| 2–5 | Amostragem aleatória ponderada pelas probabilidades do ensemble |

Isso dá um "melhor palpite" determinístico mais 4 alternativas diversas, cobrindo uma porção maior do espaço de probabilidade do que repetir sempre a mesma sequência.

### Features (por número, por sorteio)

| Feature | O que é | Por que "supostamente" importa |
|---------|---------|-------------------------------|
| `freq_5d` | Vezes sorteado nos últimos 5 jogos | Números "quentes" — muito citado por entusiastas |
| `freq_10d` | Vezes sorteado nos últimos 10 jogos | Frequência de médio prazo |
| `freq_20d` | Vezes sorteado nos últimos 20 jogos | Frequência de referência |
| `freq_50d` | Vezes sorteado nos últimos 50 jogos | Frequência de longo prazo |
| `draws_since_last` | Sorteios desde a última aparição | Números "frios" — teoria da "vez que vai sair" |
| `freq_trend` | freq_5d / freq_20d | Aceleração de aparições recentes |
| `deviation` | freq_20d − 0,10 | Desvio da frequência esperada de 10% |
| `decade` | Grupo numérico (1–10, 11–20, etc.) | Padrões de distribuição por dezena |
| `is_even` | Binário | Tendências de paridade par/ímpar |
| `is_prime` | Binário | Teoria dos números primos (puramente anedótico) |
| `prev_sum` | Soma das dezenas do sorteio anterior | Teoria do "equilíbrio" |
| `prev_mean` | Média do sorteio anterior | Centro de distribuição |
| `prev_spread` | máx − mín do sorteio anterior | Teoria da "amplitude" |
| `day_mon/thu/sat` | Dia da semana (one-hot) | Se certos números são mais comuns em certos dias |

**Nenhuma destas features deveria prever um sorteio verdadeiramente aleatório.** O modelo vai encontrar padrões aparentes — sempre encontra, mesmo em ruído puro. A questão é se esses padrões generalizam para sorteios futuros não vistos.

---

## Resultados em tempo real

### Últimos 7 sorteios — Concurso {{LAST_CONCURSO}} ({{LAST_DATE}})

<!-- LAST_WEEK_START -->
_A carregar..._
<!-- LAST_WEEK_END -->

### Próximas previsões

<!-- NEXT_PREDS_START -->
_A carregar..._
<!-- NEXT_PREDS_END -->

### Estatísticas acumuladas

<!-- STATS_START -->
_A carregar..._
<!-- STATS_END -->

---

## Por que isso importa (o ângulo do doutorado)

Este experimento é uma instância concreta de uma questão de investigação mais ampla:

> *O Machine Learning supervisionado consegue identificar estrutura explorável num processo que é provadamente aleatório por design?*

O mecanismo de sorteio da Mega Sena usa aleatoriedade física certificada (bolas numeradas, auditadas por reguladores governamentais). Não há variável oculta, microestrutura de mercado, ou psicologia humana. Se o ML não consegue superar o acaso aqui, fornece um baseline limpo para avaliar o desempenho de ML em domínios que são alegadamente aleatórios mas podem não ser (mercados financeiros, padrões climáticos, sequências biológicas).

O design experimental é rigoroso:
- Hipótese pré-registada (impossível prever acima do baseline)
- Métrica de resultado objetiva (acertos vs expectativa teórica)
- Reproduzível: todos os dados, código e previsões são públicos e com timestamp
- Longitudinal: os resultados acumulam semanalmente sem intervenção manual

Após 1–2 anos de dados, isto torna-se um teste estatisticamente significativo. Com N = 150 sorteios e 5 sequências cada (750 avaliações de sequência), um t-test de uma amostra contra a hipótese nula de μ = 0,60 acertos teria poder > 80% para detetar um effect size de Δ = 0,15 acertos — uma melhoria de 25% sobre o aleatório, que já seria extraordinária.

### Conexão com investigação em ML

O problema tem analogias diretas com outros domínios de investigação:

- **Hipótese dos mercados eficientes (EMH)**: Da mesma forma que a teoria financeira diz que não podes bater o mercado de forma consistente com informação pública, aqui testamos se não podes bater um gerador de números aleatório com informação histórica pública.

- **Overfitting em séries temporais**: O modelo vai provavelmente mostrar accuracy razoável no conjunto de treino (encontra padrões espúrios) mas falhar no conjunto de validação (generalização). Isso ilustra o problema de overfitting em dados com pouco sinal real.

- **Feature importance em dados ruidosos**: As features mais "importantes" segundo o Random Forest serão artefactos do conjunto de treino, não regularidades reais. Isso é um caso de estudo sobre como interpretar feature importance em domínios sem sinal.

---

## Estrutura do repositório

```
test_ml/analisenumerica/
├── main.py               ← executor semanal (download → valida → treina → prevê → atualiza README)
├── config.py             ← constantes, caminhos, hiperparâmetros
├── data/
│   ├── downloader.py     ← busca resultados em loterias.caixa.gov.br
│   └── storage.py        ← log de previsões + pesos (leitura/escrita)
├── features/
│   └── engineering.py    ← matriz de features por número a partir do histórico
├── models/
│   └── ensemble.py       ← RF + GB + SGD + atualização adaptativa de pesos
├── reports/
│   └── summary.py        ← gera as secções Markdown para este README
└── output/               ← arquivos gerados
    ├── mega_sena_results.csv    ← resultados oficiais em cache
    ├── predictions_log.csv      ← todas as previsões + resultados de validação
    └── ensemble_weights.json    ← pesos atuais dos modelos
```

---

## Como executar localmente

```bash
cd test_ml/analisenumerica
pip install pandas numpy scikit-learn requests beautifulsoup4 lxml
python main.py           # execução normal
python main.py --force   # forçar novo download dos resultados
```

Se o download automático falhar:
1. Acede a [loterias.caixa.gov.br/Paginas/Mega-Sena.aspx](https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx)
2. Clica em **"Resultados da Mega-Sena por ordem crescente"**
3. Guarda o arquivo como `output/mega_sena_manual.html`
4. Executa `python main.py`

---

*Este projeto não incentiva jogos de azar. É um experimento de ciência de dados com uma conclusão predeterminada. Use os números da loteria por sua própria conta e risco — ou melhor, não use.*

Construído por **Vicky Costa** — Data Analyst | Estudante de Ciência de Dados
