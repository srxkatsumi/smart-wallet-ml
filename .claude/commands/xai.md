# /xai — Explicabilidade (XAI)

Gera relatórios de explicabilidade para modelos já treinados. Nunca treina modelos, nunca altera previsões.

---

## PROTOCOLO OBRIGATÓRIO

### PASSO 1 — Verificar pré-condições

Ler `.claude/commands/roadmap.md` e verificar:
- Qual modelo e projecto o utilizador quer explicar
- Se o modelo está marcado como ✅ no REGISTO MESTRE (se não, sugerir `/model-add` primeiro)

Se o utilizador não especificou, perguntar antes de continuar.

### PASSO 2 — Escolher ferramenta de explicabilidade

| Ferramenta | Para quê | Modelos compatíveis |
|------------|---------- |--------------------|
| SHAP TreeExplainer | Importância de features por previsão | RF, GB, XGBoost, LightGBM, CatBoost |
| SHAP DeepExplainer | Importância de features em redes neurais | LSTM, GRU, Transformer |
| Attention weights | Visualiza o que o Transformer "olhou" | Transformer, TFT |
| LIME | Explicações locais para qualquer modelo | Todos |
| Permutation importance | Importância global de features | Todos |

Seleccionar a ferramenta adequada ao modelo pedido. Se o utilizador pedir "SHAP" para um modelo neural, usar DeepExplainer e explicar a diferença.

### PASSO 3 — Verificar dependências

Verificar se `shap` e `lime` estão em `requirements.txt`. Se não estiverem, listar o que falta e pedir confirmação antes de adicionar.

### PASSO 4 — Gerar explicações

**SHAP — para modelos de árvore:**
- Global: gráfico de importância de features (bar plot com valores médios |SHAP|)
- Local: waterfall plot para as 3 previsões mais recentes
- Guardar em `output/xai/shap_<modelo>_<projecto>_<data>.png`

**Attention weights — para Transformer/TFT:**
- Mapa de atenção: quais posições temporais receberam mais atenção
- Guardar em `output/xai/attention_<modelo>_<projecto>_<data>.png`

**LIME:**
- Explicação local para a última previsão de cada activo/sequência
- Guardar em `output/xai/lime_<modelo>_<projecto>_<data>.html`

### PASSO 5 — Relatório

```
EXPLICABILIDADE — [modelo] — [projecto] — [data]

FERRAMENTA USADA:   [SHAP / Attention / LIME]
FICHEIROS GERADOS:  [lista de caminhos]

TOP 5 FEATURES MAIS IMPORTANTES (global):
1. [feature] — importância média: [valor]
2. [feature] — importância média: [valor]
3. [feature] — importância média: [valor]
4. [feature] — importância média: [valor]
5. [feature] — importância média: [valor]

INTERPRETAÇÃO:
[2-3 frases sobre o que as features mais importantes indicam sobre o comportamento do modelo]

NOTA ACADÉMICA:
[Para Mega Sena: lembrar que features importantes num processo aleatório são artefactos do treino, não padrões reais]

PRÓXIMO PASSO SUGERIDO:
[/experiment para registar / /eval para validar estatisticamente]
```

---

## RESTRIÇÕES

- Nunca modificar modelos ou previsões
- Nunca interpretar features importantes na Mega Sena como "padrões reais" — são artefactos de sobreajuste
- Nunca fazer commit
- Guardar sempre os ficheiros gerados antes de reportar
