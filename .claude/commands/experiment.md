# /experiment — Rastreamento de Experimentos (MLflow + DVC)

Regista experimentos no MLflow e versiona dados com DVC. Nunca altera experimentos passados, nunca modifica modelos.

---

## PROTOCOLO OBRIGATÓRIO

### PASSO 1 — Verificar pré-condições

Ler `.claude/commands/roadmap.md` e verificar:
- Qual experimento o utilizador quer registar (modelo + projecto + métricas)
- Se o modelo está marcado como ✅ no REGISTO MESTRE

Se o utilizador não especificou o que registar, perguntar antes de continuar.

### PASSO 2 — Verificar estado do MLflow

Verificar se MLflow está instalado:
```
pip show mlflow
```

Se não estiver, listar que `mlflow` precisa ser adicionado a `requirements.txt` e pedir confirmação.

Verificar se existe um `mlruns/` ou `mlflow.db` no projecto. Se não existir, este será o primeiro registo — informar o utilizador.

### PASSO 3 — Estrutura de nomes de experimentos

Usar sempre esta convenção de nomes:

| Projecto | Nome do experimento MLflow |
|----------|---------------------------|
| Carteira | `carteira_<familia>_<modelo>` |
| Mega Sena | `megasena_<familia>_<modelo>` |
| E-commerce | `ecommerce_<familia>_<modelo>` |

Exemplos: `carteira_neural_lstm`, `megasena_classico_xgboost`, `ecommerce_timeseries_prophet`

### PASSO 4 — Registar no MLflow

Para cada experimento registar obrigatoriamente:

**Parâmetros (params):**
- Nome do modelo
- Hiperparâmetros usados
- Projecto e domínio
- Data de treino
- Número de amostras de treino

**Métricas (metrics):**
- Acurácia global
- Acurácia UP / DOWN (quando aplicável)
- p-value vs baseline (quando disponível de `/eval`)
- Número de previsões validadas

**Artefactos (artifacts):**
- Ficheiro do modelo serializado (`.pkl` ou `.pt`)
- Relatório SHAP se disponível
- `predictions_log.csv` snapshot

### PASSO 5 — DVC para dados

Se o utilizador pedir versionamento de dados:

Verificar se DVC está instalado:
```
pip show dvc
```

Ficheiros a versionar com DVC (nunca com git directamente):
- `data/prices_cache.csv` ou equivalente
- `test_ml/analisenumerica/data/resultados.csv`
- `output/predictions_log.csv`

Criar ou actualizar `.dvc` tracking files conforme necessário.

### PASSO 6 — Relatório

```
EXPERIMENTO REGISTADO — [data]

MLFLOW
Nome:               [carteira_neural_lstm]
Run ID:             [gerado pelo MLflow]
Parâmetros:         [lista dos principais]
Métricas chave:     [acurácia, p-value]
Artefactos:         [lista de ficheiros]

DVC (se aplicável)
Ficheiros versionados: [lista]

HISTÓRICO DE RUNS NESTE EXPERIMENTO:
[tabela com os últimos 5 runs: data, acurácia, modelo]

PRÓXIMO PASSO SUGERIDO:
[/eval para comparar runs / /sync-check para ver o que falta implementar]
```

---

## RESTRIÇÕES

- Nunca apagar ou modificar runs MLflow anteriores
- Nunca versionar com DVC ficheiros que já estão em `.gitignore` sem confirmar com o utilizador
- Nunca fazer commit
- Nunca alterar `mlflow.db` manualmente
