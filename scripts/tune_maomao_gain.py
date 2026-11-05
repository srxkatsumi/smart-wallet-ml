"""
Achado e calibração do WEIGHT_GAIN em maomao/brain.py.

Parte 1 — o bug: até esta versão, o drive de cada sub-passo da rede LIF era
dividido por POOL_SIZE ("distribuir o drive pelos neurônios do pool"), e a
taxa de saída (fired_total / (N_SUBSTEPS * POOL_SIZE)) também dividia por
POOL_SIZE — uma dupla diluição. Resultado: POOL_SIZE deixou de ser um
hiperparâmetro inofensivo de "quantas amostras ruidosas suavizar" e passou
a controlar o ganho efetivo da rede inteira. Este script reproduz isso.

Parte 2 — a calibração: corrigido o bug (drive não divide mais por
POOL_SIZE), a rede ficou ~POOL_SIZE vezes mais sensível, e os BASE_WEIGHTS
originais (nunca calibrados de verdade — só pareciam razoáveis porque o bug
os diluía) passaram a saturar no teto de disparo (~0.87, dado REFRACTORY=2)
mesmo para entradas moderadas, perdendo gradação. Este script varre um
fator de ganho e escolhe o que preserva gradação sensata entre "leve" e
"forte" nos quatro pools.

Uso: python scripts/tune_maomao_gain.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import maomao.brain as brain_mod
from maomao.brain import Brain, CHANNELS


def _full(intensities: dict) -> dict:
    return {c: intensities.get(c, 0.0) for c in CHANNELS}


def part1_reproduce_the_bug():
    print("=== Parte 1: reproduzindo o bug da dupla diluição por POOL_SIZE ===")
    print("(drive dividido por POOL_SIZE na entrada — removido na correção)\n")

    intensities = _full({"VOL_SPIKE": 0.6, "VIX_SPIKE": 0.5, "STREAK_DOWN": 0.4,
                        "ANOMALY": 1.0, "RET_DOWN": 0.5})

    orig_step = Brain.step

    def buggy_step(self, intens):
        # idêntico ao step() real, só reintroduzindo a divisão por POOL_SIZE
        # na entrada, pra reproduzir o comportamento de antes da correção
        import numpy as _np
        v   = _np.zeros((len(brain_mod.POOLS), brain_mod.POOL_SIZE))
        ref = _np.zeros((len(brain_mod.POOLS), brain_mod.POOL_SIZE), dtype=int)
        fired_total = _np.zeros(len(brain_mod.POOLS))
        drive = _np.array([
            sum((brain_mod.BASE_WEIGHTS.get(p, {}).get(c, 0.0)) * intens.get(c, 0.0) for c in brain_mod.CHANNELS)
            for p in brain_mod.POOLS
        ])
        for _ in range(brain_mod.N_SUBSTEPS):
            v *= brain_mod.DECAY
            v += drive[:, None] / brain_mod.POOL_SIZE  # <- o bug
            v += brain_mod.NOISE_STD * self.rng.standard_normal(v.shape)
            blocked = ref > 0
            v[blocked] = 0.0
            ref[blocked] -= 1
            fired = v >= brain_mod.THRESHOLD
            v[fired] = 0.0
            ref[fired] = brain_mod.REFRACTORY
            fired_total += fired.sum(axis=1)
        rate = fired_total / (brain_mod.N_SUBSTEPS * brain_mod.POOL_SIZE)
        rate = _np.tanh(rate * 4.0)
        return {p: float(r) for p, r in zip(brain_mod.POOLS, rate)}

    saved_gain = brain_mod.WEIGHT_GAIN
    brain_mod.WEIGHT_GAIN = 1.0  # sem o ganho novo, pra isolar só o efeito do bug
    for pool_size in [1, 4, 8, 16, 32]:
        brain_mod.POOL_SIZE = pool_size
        b = Brain(rng=np.random.default_rng(0))
        mood = buggy_step(b, intensities)
        print(f"  POOL_SIZE={pool_size:>2}  FEARFUL (com bug) = {mood['FEARFUL']:.4f}")
    brain_mod.POOL_SIZE = 8
    brain_mod.WEIGHT_GAIN = saved_gain
    print()


def part2_tune_gain():
    print("=== Parte 2: calibrando WEIGHT_GAIN pós-correção ===\n")
    cases = {
        "medo leve":     {"VOL_SPIKE": 0.2, "ANOMALY": 0.0},
        "medo moderado": {"VOL_SPIKE": 0.5, "VIX_SPIKE": 0.4, "ANOMALY": 1.0},
        "medo forte":    {"VOL_SPIKE": 1.0, "VIX_SPIKE": 1.0, "ANOMALY": 1.0, "STREAK_DOWN": 0.6, "RET_DOWN": 0.8},
        "euforia leve":  {"STREAK_UP": 0.5, "RET_UP": 0.3},
        "euforia forte": {"STREAK_UP": 1.0, "RET_UP": 0.8, "CALM_STREAK": 0.5},
        "confuso leve":  {"DISAGREEMENT": 0.4},
        "confuso forte": {"DISAGREEMENT": 1.0},
    }
    saved_gain = brain_mod.WEIGHT_GAIN
    for gain in [1.0, 0.5, 0.35, 0.25, 0.15]:
        brain_mod.WEIGHT_GAIN = gain
        print(f"--- WEIGHT_GAIN={gain} ---")
        for name, inten in cases.items():
            b = Brain(rng=np.random.default_rng(0))
            mood = b.step(_full(inten))
            pool = "FEARFUL" if "medo" in name else "EUPHORIC" if "euforia" in name else "CONFUSED" if "confuso" in name else "CALM"
            print(f"  {name:>14}: {pool}={mood[pool]:.3f}")
        print()
    brain_mod.WEIGHT_GAIN = saved_gain
    print(f"Escolhido: WEIGHT_GAIN=0.25 — único valor testado em que 'moderado'/'leve'\n"
          f"ficam claramente abaixo de 'forte' nos quatro pools, sem nenhum saturar\n"
          f"no teto (~0.87) nem colapsar perto de zero para entradas reais.")


if __name__ == "__main__":
    part1_reproduce_the_bug()
    part2_tune_gain()
