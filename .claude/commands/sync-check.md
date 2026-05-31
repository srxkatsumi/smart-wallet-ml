# /sync-check — Auditoria de Completude de Modelos

Verifica o que está implementado vs o que deveria estar em todos os projectos. Apenas lê e reporta, nunca implementa.

---

## PROTOCOLO OBRIGATÓRIO

### PASSO 1 — Ler o registo mestre

Ler `.claude/commands/roadmap.md` e extrair a tabela completa do REGISTO MESTRE DE MODELOS com os 25 modelos e os seus estados por projecto.

### PASSO 2 — Auditar ficheiros existentes

Verificar o que realmente existe no disco (não assumir — sempre ler):

**Carteira Inteligente:**
```
models/
```
Listar todos os ficheiros `.py` e identificar quais modelos estão implementados dentro de cada ficheiro.

**Mega Sena:**
```
test_ml/analisenumerica/models/
```
Idem.

**E-commerce (quando existir):**
Verificar se o directório do projecto existe. Se não existir, registar como "projecto não iniciado".

### PASSO 3 — Auditar camadas transversais

Para cada camada transversal, verificar se existe implementação:

| Camada | O que procurar |
|--------|---------------|
| SHAP | `pip show shap` + ficheiros em `output/xai/` |
| LIME | `pip show lime` + ficheiros em `output/xai/` |
| MLflow | `pip show mlflow` + directório `mlruns/` |
| DVC | `pip show dvc` + ficheiros `.dvc` no projecto |
| Testes estatísticos | Verificar se existe `eval/` ou equivalente no código |
| Entropia / Info Mútua | Procurar em `features/engineering.py` |

### PASSO 4 — Calcular gaps

Produzir tabela de gaps por projecto:

```
MODELOS PENDENTES — CARTEIRA
Família Clássico:       [lista do que falta]
Família Séries Temp.:   [lista do que falta]
Família Estado Oculto:  [lista do que falta]
Família Neural:         [lista do que falta]
Família Bayesiano:      [lista do que falta]
Família Generativo:     [lista do que falta]
Família RL:             [lista do que falta]

MODELOS PENDENTES — MEGA SENA
[idem]

CAMADAS TRANSVERSAIS PENDENTES
[lista do que falta]
```

### PASSO 5 — Relatório completo

```
SYNC-CHECK — [data]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARTEIRA INTELIGENTE
Modelos implementados:  [X de 25]
Modelos pendentes:      [25-X]

MEGA SENA
Modelos implementados:  [X de 25]
Modelos pendentes:      [25-X]

E-COMMERCE
Estado:                 [Não iniciado / X de 25]

CAMADAS TRANSVERSAIS
Avaliação estatística:  [X de 4]
Explicabilidade:        [X de 3]
Meta-learning:          [X de 2]
Rastreamento:           [MLflow: ✅/⬜ | DVC: ✅/⬜]
Teoria da informação:   [X de 2]
Transfer Learning:      [✅/⬜]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRÓXIMOS 3 MODELOS SUGERIDOS (por ordem de complexidade crescente):
1. [modelo] em [projecto] — família [X] — complexidade: baixa
2. [modelo] em [projecto] — família [X] — complexidade: média
3. [modelo] em [projecto] — família [X] — complexidade: alta

Para implementar qualquer um: /model-add [nome do modelo] [projecto]
```

---

## RESTRIÇÕES

- Apenas lê ficheiros, nunca escreve nem modifica
- Nunca fazer commit
- Se encontrar inconsistências entre o REGISTO MESTRE e o que existe no disco, reportar a discrepância mas não corrigir — sugerir ao utilizador que actualize o roadmap com `/roadmap`
- Sugerir sempre os próximos passos por ordem de complexidade crescente (Markov antes de GAN)
