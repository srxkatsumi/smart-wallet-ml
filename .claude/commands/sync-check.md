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
| MLflow | `pip show mlflow` + `mlflow.db` ou `mlruns/` |
| DVC | `pip show dvc` + ficheiros `.dvc` no projecto |
| Testes estatísticos | Verificar se existe `eval/` ou equivalente no código |
| Entropia / Info Mútua | Procurar em `features/engineering.py` |

### PASSO 3B — Auditoria de integração sistémica

**Esta é a verificação que distingue "código existe" de "código funciona no sistema real."**

Para cada camada transversal marcada como implementada, verificar os três níveis:

| Nível | Pergunta | Como verificar |
|-------|----------|---------------|
| Código | A função existe no módulo? | `grep -n "def "` no ficheiro |
| Pipeline | É chamada em `main.py`? | `grep -n "from evaluation\|import tracking\|log_run\|shap_tree\|dvc_track"` em `main.py` |
| CI/CD | O output é persistido no GitHub Actions? | Ler `.github/workflows/executar_diario.yml` e verificar se o ficheiro gerado aparece num `git add` |

Verificar também:
- **`.gitignore`**: ficheiros gerados pelas camadas (ex: `mlflow.db`, `output/xai/`) estão ignorados acidentalmente ou faltam ser ignorados?
- **Hiperparâmetros no MLflow**: os `params` registados incluem os valores reais de `config/settings.py`, ou apenas metadados genéricos (ticker, horizon)?
- **Código morto**: existem ficheiros em `models/` que não constam do roadmap e não são chamados em nenhum pipeline? Listar.

Reportar cada camada com três estados possíveis:
- ✅ **Integrado** — código existe, é chamado no pipeline, o output é persistido no CI
- ⚠️ **Parcial** — código existe e é chamado, mas o output não é persistido (ou vice-versa)
- ⬜ **Apenas código** — módulo existe mas não é chamado em `main.py` nem no CI

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

CAMADAS TRANSVERSAIS          Código    Pipeline  CI/git
Avaliação estatística:        [✅/⬜]   [✅/⬜]   [✅/⬜]
Explicabilidade (SHAP/LIME):  [✅/⬜]   [✅/⬜]   [✅/⬜]
Meta-learning:                [✅/⬜]   [✅/⬜]   [✅/⬜]
MLflow:                       [✅/⬜]   [✅/⬜]   [✅/⬜]
DVC:                          [✅/⬜]   [✅/⬜]   [✅/⬜]
Teoria da informação:         [✅/⬜]   [✅/⬜]   [✅/⬜]
Transfer Learning:            [✅/⬜]   [✅/⬜]   [✅/⬜]

INTEGRAÇÃO — PROBLEMAS DETECTADOS
.gitignore:         [OK / Falta: listar ficheiros]
MLflow params:      [Hiperparâmetros reais: ✅/⬜ | Só metadados: listar]
Código morto:       [Nenhum / Ficheiros sem pipeline: listar]
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
