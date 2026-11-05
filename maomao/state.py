"""
Persistência do estado do MaoMao entre execuções diárias do pipeline:
pesos aprendidos, apostas pendentes (maomao/learning.py), histórico curto de
alertas (pra CALM_STREAK em maomao/sensors.py) e o pequeno log de humor do
dia, pra o site/email mostrarem. Mesmo papel que memory/ no Pestinha, só que
um ficheiro JSON simples (escrita atómica), porque o estado aqui é muito
menor — pesos de ~9 canais x 2 pools, não ~16 mil sinapses.

Guarda também a última data já processada: o pipeline principal tem um
padrão conhecido de reexecuções redundantes no mesmo dia (ver
README_Errors.md ERR-010/ERR-013) — sem essa guarda, um segundo run no
mesmo dia duplicaria entradas no histórico e aplicaria a mesma aposta duas
vezes.

Toda função recebe o `path` explicitamente (não é um STATE_FILE global do
módulo): há dois humores independentes — carteira (output/maomao_state.json)
e mercado (output/maomao_market_state.json) — ver maomao/daily.py.
"""
import json
from pathlib import Path

PORTFOLIO_STATE_FILE = Path("output/maomao_state.json")
MARKET_STATE_FILE    = Path("output/maomao_market_state.json")

MAX_HISTORY = 30
MAX_RECENT_ALERTAS = 15


def _fresh_default() -> dict:
    # Função, não um dict de módulo partilhado de propósito: um
    # `{**_DEFAULT, **data}` só copia o dict de fora (shallow) — as listas e
    # dicts aninhados (mood_history, pending, learned...) continuariam sendo
    # os MESMOS objetos entre chamadas, e push_history() mutaria esse
    # default global pra sempre na primeira vez que o ficheiro não existisse.
    return {
        "last_processed_date": None,
        "learned": {},
        "pending": [],
        "recent_alertas": [],
        "mood_history": [],
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return _fresh_default()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _fresh_default()
    return {**_fresh_default(), **data}


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def push_history(state: dict, date: str, mood: dict, alerta: bool) -> None:
    state["mood_history"].append({"date": date, "mood": mood})
    state["mood_history"] = state["mood_history"][-MAX_HISTORY:]
    state["recent_alertas"].append(bool(alerta))
    state["recent_alertas"] = state["recent_alertas"][-MAX_RECENT_ALERTAS:]
