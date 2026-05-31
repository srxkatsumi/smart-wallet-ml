# /eval — Framework de Avaliação Estatística

Corre testes de significância estatística e métricas de avaliação sobre modelos já treinados. Nunca modifica código de modelos.

---

## PROTOCOLO OBRIGATÓRIO

### PASSO 1 — Verificar pré-condições

Ler `.claude/commands/roadmap.md` e verificar:
- Qual projecto e modelo(s) o utilizador quer avaliar
- Se esses modelos estão marcados como ✅ no REGISTO MESTRE (se não, sugerir `/model-add` primeiro)

Se o utilizador não especificou projecto e modelo, perguntar antes de continuar.

### PASSO 2 — Identificar dados disponíveis

**Para Carteira:** verificar se `output/predictions_log.csv` existe e tem linhas validadas (`validated == True`)
**Para Mega Sena:** verificar se `test_ml/analisenumerica/output/predictions_log.csv` existe

Contar quantas previsões validadas existem. Se menos de 30, avisar que os testes podem não ter poder estatístico suficiente.

### PASSO 3 — Escolher os testes apropriados

O utilizador pode pedir testes específicos ou pedir avaliação completa. Testes disponíveis:

| Teste | Quando usar | O que responde |
|-------|-------------|----------------|
| Diebold-Mariano | Comparar dois modelos de previsão | Modelo A é estatisticamente melhor que B? |
| McNemar | Comparar dois classificadores | Os erros dos dois modelos são independentes? |
| Ljung-Box | Verificar autocorrelação nos resíduos | Os erros têm estrutura ou são ruído? |
| Baseline aleatório | Sempre | O modelo bate o acaso? |

**Métricas por domínio:**
- Carteira: Acurácia UP/DOWN, F1-score, AUC-ROC, Sharpe implícito
- Mega Sena: Accuracy@k (acertos por sequência), comparação vs baseline hot/cold/random
- E-commerce (futuro): MAE, MAPE, RMSE, cobertura de intervalos

### PASSO 4 — Executar avaliação

Ler os dados de previsões e calcular. Apresentar resultados em formato de tabela clara.

Para cada teste:
- Estatística do teste
- p-value
- Interpretação em linguagem directa: "O modelo X é significativamente melhor que o acaso (p=0.03)" ou "Não há evidência estatística de que X supera Y (p=0.42)"

### PASSO 5 — Relatório final

```
AVALIAÇÃO — [modelo] — [projecto] — [data]

DADOS
Previsões validadas:    [N]
Período:                [data início] a [data fim]
Horizonte:              [D+1 / D+2 / D+3 / sorteio]

MÉTRICAS
Acurácia global:        [X%]
Acurácia UP:            [X%]
Acurácia DOWN:          [X%]
vs baseline aleatório:  [+X% / -X%]

TESTES ESTATÍSTICOS
Diebold-Mariano:        [estatística] p=[valor] → [interpretação]
McNemar:                [estatística] p=[valor] → [interpretação]
Ljung-Box (resíduos):   [estatística] p=[valor] → [interpretação]

CONCLUSÃO
[Uma frase directa sobre o que os testes indicam]

PRÓXIMO PASSO SUGERIDO
[/xai para explicar os resultados / /experiment para registar / ou outra acção]
```

---

## RESTRIÇÕES

- Nunca modificar ficheiros de modelo ou de previsões
- Nunca apagar resultados de avaliações anteriores
- Nunca concluir que "o modelo prevê a loteria" — para Mega Sena, a conclusão é sempre enquadrada como experimento científico negativo controlado
- Nunca fazer commit
