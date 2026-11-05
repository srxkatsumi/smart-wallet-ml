"""
MaoMao: motor de rede neural pulsada (LIF) para o pet financeiro do site.

Inspirado na arquitetura descrita em github.com/mille-okws/pestinha
(canais sensoriais com habituação -> pools de interneurônios com pesos fixos
-> aprendizado de três fatores nas sinapses plásticas), mas é uma
implementação original, escrita do zero para este projeto — nenhuma linha
do código deles foi copiada. Ver maomao/README.md para a comparação completa
e a atribuição.

Diferença deliberada de cadência: o Pestinha simula continuamente, em passos
de 1 ms, porque reage em tempo real a mouse/teclado. O mercado só fecha uma
vez por dia — não há nada de real pra simular a cada milissegundo entre dois
fechamentos. Por isso o MaoMao chama a rede UMA vez por dia (um snapshot dos
canais sensoriais do dia), mas ainda roda a dinâmica LIF de verdade (vários
sub-passos, limiar, período refratário, ruído) dentro dessa única chamada,
até a taxa de disparo de cada pool se assentar — é a mesma técnica que
`core.step(dt_ms)` do Pestinha usa, só que invocada uma vez por dia em vez
de a cada frame.

O que é regra (pesos fixos, não aprende): a ligação de cada canal sensorial
a cada pool de humor (BASE_WEIGHTS). O que é aprendido: o ajuste fino dessas
ligações nos alvos plásticos (FEARFUL, EUPHORIC), por uma recompensa real e
tardia orquestrada em maomao/learning.py — ver maomao/README.md, "O que é
aprendido".
"""
import numpy as np

CHANNELS = [
    "RET_UP", "RET_DOWN", "VOL_SPIKE", "VIX_SPIKE",
    "STREAK_UP", "STREAK_DOWN", "ANOMALY", "DISAGREEMENT", "CALM_STREAK",
]
POOLS = ["CALM", "FEARFUL", "EUPHORIC", "CONFUSED"]
PLASTIC_POOLS = ["FEARFUL", "EUPHORIC"]  # únicos alvos que aprendem (ver README)

POOL_SIZE   = 8      # neurônios por pool — bem menor que o Pestinha (16-20): v1, dado só 1 amostra/dia
N_SUBSTEPS  = 300     # sub-passos por chamada diária, só pra deixar a taxa assentar (não é tempo real)
DECAY       = 0.92    # fator de vazamento do potencial de membrana por sub-passo
THRESHOLD   = 1.0
REFRACTORY  = 2        # sub-passos bloqueado após disparar
NOISE_STD   = 0.03

LEARN_MIN, LEARN_MAX = -0.5, 0.5

# Pesos fixos canal -> pool (a "regra" da rede, escrita à mão — igual ao
# espírito das tabelas de REDE_NEURAL.md do Pestinha, adaptado aos canais
# financeiros). Ajustáveis, mas não aprendidos. Magnitudes pensadas pra
# ranking relativo entre canais, não pro ganho absoluto — ver WEIGHT_GAIN.
BASE_WEIGHTS = {
    "CALM":     {"CALM_STREAK": 0.9, "VOL_SPIKE": -0.6, "ANOMALY": -0.7, "VIX_SPIKE": -0.4},
    "FEARFUL":  {"VOL_SPIKE": 0.8, "VIX_SPIKE": 0.7, "STREAK_DOWN": 0.6, "ANOMALY": 0.7, "RET_DOWN": 0.4},
    "EUPHORIC": {"STREAK_UP": 0.8, "RET_UP": 0.5, "CALM_STREAK": 0.3, "VOL_SPIKE": -0.3},
    "CONFUSED": {"DISAGREEMENT": 0.9, "VOL_SPIKE": 0.2},
}

# Ganho aplicado a BASE_WEIGHTS + learned antes de entrar na rede. Existe
# por um motivo concreto: até esta versão, o drive de cada substep era
# dividido por POOL_SIZE E a taxa de saída também era dividida por
# POOL_SIZE — uma dupla diluição que fazia o ganho efetivo da rede
# desmoronar conforme POOL_SIZE crescia (ex: POOL_SIZE=32 reduzia um humor
# de 0.87 para ~0.0015, só por trocar um hiperparâmetro que deveria só
# suavizar ruído). Corrigido para a entrada não dividir mais por
# POOL_SIZE — mas isso tornou a rede ~POOL_SIZE vezes mais sensível, e com
# os BASE_WEIGHTS originais (calibrados, sem querer, contra o bug) quase
# qualquer sinal real saturava no teto de disparo possível (~0.87, dado
# REFRACTORY=2), perdendo a gradação entre "medo moderado" e "medo forte".
# 0.25 foi escolhido testando contra cenários sintéticos (ver
# scripts/tune_maomao_gain.py) até recuperar gradação sensata nos quatro
# pools sem reintroduzir a diluição por POOL_SIZE.
WEIGHT_GAIN = 0.25


class Brain:
    """
    Rede LIF pequena: canais sensoriais -> pools de humor. Pesos fixos
    (BASE_WEIGHTS) mais uma matriz aprendida (`learned`), só para os pools em
    PLASTIC_POOLS, inicialmente zero.
    """

    def __init__(self, learned: dict | None = None, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.learned = {
            pool: dict(learned.get(pool, {})) if learned else {}
            for pool in PLASTIC_POOLS
        }

    def _effective_weight(self, channel: str, pool: str) -> float:
        base = BASE_WEIGHTS.get(pool, {}).get(channel, 0.0)
        extra = self.learned.get(pool, {}).get(channel, 0.0) if pool in PLASTIC_POOLS else 0.0
        return (base + extra) * WEIGHT_GAIN

    def step(self, intensities: dict) -> dict:
        """
        Um snapshot sensorial do dia -> taxa de disparo (0..1) de cada pool,
        assentada depois de N_SUBSTEPS de dinâmica LIF.
        """
        v   = np.zeros((len(POOLS), POOL_SIZE))
        ref = np.zeros((len(POOLS), POOL_SIZE), dtype=int)
        fired_total = np.zeros(len(POOLS))

        drive = np.array([
            sum(self._effective_weight(c, pool) * intensities.get(c, 0.0) for c in CHANNELS)
            for pool in POOLS
        ])

        for _ in range(N_SUBSTEPS):
            v *= DECAY
            v += drive[:, None]  # cada neurônio do pool recebe o drive inteiro (coding populacional:
                                  # POOL_SIZE só deve servir para suavizar ruído via a taxa agregada
                                  # abaixo, não para diluir o ganho — ver nota em POOL_SIZE)
            v += self.rng.normal(0.0, NOISE_STD, size=v.shape)

            blocked = ref > 0
            v[blocked] = 0.0
            ref[blocked] -= 1

            fired = v >= THRESHOLD
            v[fired] = 0.0
            ref[fired] = REFRACTORY
            fired_total += fired.sum(axis=1)

        rates = fired_total / (N_SUBSTEPS * POOL_SIZE)
        # normaliza pra 0..1 de forma suave (satura em vez de cortar) — taxas
        # cruas já ficam tipicamente entre 0 e ~0.3 com estes parâmetros
        rates = np.tanh(rates * 4.0)
        return {pool: float(r) for pool, r in zip(POOLS, rates)}

    def learn(self, pool: str, eligibility: dict, reward: float, rate: float = 0.05) -> None:
        """
        Regra de três fatores: elegibilidade (quais canais estavam ativos
        quando o pool disparou forte) x sinal do pool x recompensa real e
        tardia. `reward` é + ou - (ver ThreeFactorLearner — nunca vem da
        direção prevista, só de retorno realizado).
        """
        if pool not in PLASTIC_POOLS or reward == 0.0:
            return
        w = self.learned.setdefault(pool, {})
        for channel, elig in eligibility.items():
            delta = rate * reward * elig
            w[channel] = float(np.clip(w.get(channel, 0.0) + delta, LEARN_MIN, LEARN_MAX))

    def snapshot_learned(self) -> dict:
        return {pool: dict(w) for pool, w in self.learned.items()}
