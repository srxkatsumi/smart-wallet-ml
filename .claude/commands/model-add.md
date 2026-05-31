# /model-add — Implementador de Modelos

Implementa um modelo específico num projecto específico. Nunca age sem verificar aprovação no roadmap primeiro.

---

## PROTOCOLO OBRIGATÓRIO — executar sempre nesta ordem

### PASSO 1 — Identificar o que foi pedido

O utilizador deve especificar:
- **Modelo**: nome exacto conforme o REGISTO MESTRE DE MODELOS no `/roadmap`
- **Projecto**: `carteira`, `megasena` ou `ambos`

Se algum destes não foi especificado, perguntar antes de continuar. Não assumir.

### PASSO 2 — Verificar aprovação no roadmap

Ler `.claude/commands/roadmap.md` e localizar o modelo pedido na tabela REGISTO MESTRE DE MODELOS.

- Se o modelo **não está** no registo: parar aqui. Informar o utilizador e sugerir `/roadmap` para adicionar o modelo à lista antes de prosseguir.
- Se o modelo **está** no registo mas o estado já é ✅ no projecto alvo: informar que já está implementado. Perguntar se quer reimplementar ou verificar com `/sync-check`.
- Se o modelo **está** no registo e o estado é ⬜: continuar para o Passo 3.

### PASSO 3 — Verificar estrutura existente do projecto

Antes de escrever código, ler os ficheiros relevantes:

**Para Carteira:** ler `models/ensemble.py` e `config/settings.py`
**Para Mega Sena:** ler `test_ml/analisenumerica/models/ensemble.py` e `test_ml/analisenumerica/config.py`

Identificar:
- Interface actual dos modelos (como `train()` e `predict()` estão definidos)
- Dependências já instaladas em `requirements.txt`
- Se o novo modelo precisa de novas dependências

### PASSO 4 — Verificar dependências

Ler o `requirements.txt` do projecto alvo.

Se o modelo precisar de dependências novas (ex: `torch` para LSTM, `statsmodels` para ARIMA):
- Listar quais dependências serão adicionadas
- Pedir confirmação ao utilizador antes de alterar `requirements.txt`

### PASSO 5 — Implementar

Criar o ficheiro de modelo seguindo estas regras:

**Localização:**
- Carteira: `models/<nome_familia>.py` (ex: `models/timeseries.py`)
- Mega Sena: `test_ml/analisenumerica/models/<nome_familia>.py`

**Interface obrigatória — todo modelo deve implementar:**
```python
def train(X, y) -> model:
    """Treina o modelo. Retorna o objecto treinado."""

def predict(model, X) -> np.ndarray:
    """Retorna probabilidades entre 0 e 1 para cada amostra."""
```

**Regras de implementação:**
- Sem comentários que expliquem o óbvio
- Seed fixo em 42 para reprodutibilidade
- Compatível com a interface do `ensemble.py` existente
- Sem features extras além do necessário para o modelo funcionar

### PASSO 5-B — Criar teste obrigatório

Antes de declarar o modelo implementado, criar um ficheiro de teste:

**Localização:**
- Carteira: `tests/test_<nome_modelo>.py`
- Mega Sena: `test_ml/analisenumerica/tests/test_<nome_modelo>.py`

**O teste deve verificar obrigatoriamente:**
1. `train()` executa sem erros com dados sintéticos (mínimo 50 amostras)
2. `predict()` retorna um array com valores entre 0 e 1
3. O shape do output é compatível com o ensemble existente
4. O modelo não quebra com dados que contêm NaN (deve tratar ou lançar erro claro)

**Estrutura mínima do teste:**
```python
import numpy as np
import pytest
from models.<ficheiro> import train, predict

def _make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    y = rng.integers(0, 2, n)
    return X, y

def test_train_runs():
    X, y = _make_data()
    model = train(X, y)
    assert model is not None

def test_predict_bounds():
    X, y = _make_data()
    model = train(X, y)
    probs = predict(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
```

Após criar o teste, correr:
```
pytest tests/test_<nome_modelo>.py -v
```

Se falhar: corrigir o modelo ou o teste antes de avançar. Nunca passar ao Passo 6 com testes a falhar.

Depois correr a suite completa para garantir que nada quebrou:
```
pytest tests/ -v          # Carteira
pytest test_ml/analisenumerica/tests/ -v   # Mega Sena
```

### PASSO 6 — Reportar resultado

Após implementar, produzir este relatório:

```
MODELO IMPLEMENTADO: [nome]
PROJECTO:           [carteira / megasena]
FICHEIRO:           [caminho do ficheiro criado]
DEPENDÊNCIAS:       [adicionadas ao requirements.txt / nenhuma]
INTERFACE:          train() ✅ | predict() ✅
TESTES:             pytest tests/test_<modelo>.py ✅ | suite completa ✅
PRÓXIMO PASSO:      Correr /sync-check para confirmar e /eval para validar
```

Indicar que o utilizador deve actualizar o estado do modelo de ⬜ para ✅ no REGISTO MESTRE DE MODELOS em `.claude/commands/roadmap.md` após validar que funciona.

---

## RESTRIÇÕES

- Nunca implementar mais de um modelo por invocação
- Nunca alterar `ensemble.py` sem aprovação explícita do utilizador
- Nunca apagar código existente para adicionar o novo modelo
- Nunca fazer commit — aguardar instrução do utilizador
