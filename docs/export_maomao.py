"""
Gera docs/assets/maomao_data.js a partir dos dois estados que
maomao/daily.py persiste a cada execução do pipeline principal:
output/maomao_state.json (humor de carteira) e
output/maomao_market_state.json (humor de mercado). Não recalcula nada, só
lê e publica, no mesmo espírito de docs/export_alerts.py.

Uso: python docs/export_maomao.py   (rodar a partir da raiz do repositório)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_STATE_FILE = ROOT / "output" / "maomao_state.json"
MARKET_STATE_FILE    = ROOT / "output" / "maomao_market_state.json"


def _read(path: Path) -> dict:
    if not path.exists():
        print(f"AVISO: {path} não existe ainda — main.py precisa rodar pelo menos uma vez antes.")
        return {"mood_history": [], "learned": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    return {"mood_history": state.get("mood_history", []), "learned": state.get("learned", {})}


def main():
    payload = {
        "portfolio": _read(PORTFOLIO_STATE_FILE),
        "market":    _read(MARKET_STATE_FILE),
    }

    out_path = Path(__file__).resolve().parent / "assets" / "maomao_data.js"
    out_path.write_text(
        "// Gerado por docs/export_maomao.py a partir de output/maomao_state.json e\n"
        "// output/maomao_market_state.json (o estado real dos dois humores do MaoMao,\n"
        "// persistidos a cada execução do pipeline).\n"
        f"const MAOMAO_DATA = {json.dumps(payload, indent=2, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    last_p = payload["portfolio"]["mood_history"][-1] if payload["portfolio"]["mood_history"] else None
    last_m = payload["market"]["mood_history"][-1] if payload["market"]["mood_history"] else None
    print(f"OK - {out_path} updated (carteira: {last_p} | mercado: {last_m})")


if __name__ == "__main__":
    main()
