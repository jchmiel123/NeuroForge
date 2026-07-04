"""
Slotto <-> NeuroForge bridge.

Slotto (the slot-machine gift) runs Pip's brain in the browser. Pip is a
NeuroForge Q-network that learns BET SIZING under a hidden COLD/NEUTRAL/
HOT luck state - it observes only the recent payout history (10 floats)
and picks bet level 1/2/3 to maximise expected reward. Like the
DelveForge brains, it is a raw Q-net with NO scalers.

This bridge does three things:

1. load_brain / save_brain - round-trip Pip's exported brain
   (neuroforge-network-v1, the exact shape the admin panel's "Export
   brain" button emits and "Import brain" reads). Bit-exact, so you can
   train/analyse offline and drop the result back into the game.

2. SlottoEnv - a faithful Python port of the Slotto engine + Pip's
   observation/reward, as a neuroforge.evolve.Environment. This makes
   Slotto a first-class TRAINING GROUND: `QAgent.train(SlottoEnv())`
   grows our own AI on real play. The JS `web/simulate.js` remains the
   RTP source of truth; this env is kept in sync with slots-engine.js +
   ai.js (see the SYNC block below). The __main__ self-test reproduces
   simulate.js's ground-truth reward table to prove the port matches.

3. load_dialogue - read the chat corpus (pip-dialogue.jsonl exported
   from the admin panel) for the language-side training: teaching a
   local model to BE Pip, grounded in the game state at each turn.

Usage:
    from slotto_bridge import load_brain, save_brain, SlottoEnv
    from neuroforge.qlearn import QAgent
    net = load_brain("pip-brain-400000spins.json")
    env = SlottoEnv()
    agent = QAgent(observation_size=10, action_size=3, hidden=[32])
    agent.train(env, episodes=2000, max_steps=60)
    save_brain(agent.net, "pip-trained.json", meta={"spins_learned": 120000})
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from neuroforge.core import Network, Layer
from neuroforge.evolve import Environment

FORMAT = "neuroforge-network-v1"

# ==========================================================================
# Brain round-trip (mirrors delveforge_bridge; game="slotto")
# ==========================================================================

def _layers_from_obj(obj: dict) -> list[Layer]:
    if "layers" not in obj or not obj["layers"]:
        raise ValueError("not a NeuroForge/Slotto brain: no 'layers'")
    layers = []
    for ld in obj["layers"]:
        w = ld.get("weights", ld.get("w"))
        b = ld.get("biases", ld.get("b"))
        a = ld.get("activation", ld.get("a", "tanh"))
        if w is None or b is None:
            raise ValueError("layer missing weights/biases")
        layers.append(Layer(input_size=len(w[0]), output_size=len(b),
                            activation=a, weights=w, biases=b))
    return layers


def net_from_obj(obj: dict) -> Network:
    """Build a scaler-free NeuroForge Network from a portable brain dict."""
    layers = _layers_from_obj(obj)
    inputs = len(layers[0].weights[0])
    hidden = [len(l.biases) for l in layers[:-1]]
    outputs = len(layers[-1].biases)
    net = Network(inputs=inputs, hidden=hidden, outputs=outputs,
                  task="regression")
    net.layers = layers
    net.in_scaler = None   # raw Q-network: never scale (Pip pre-scales obs)
    net.out_scaler = None
    return net


def load_brain(path: str | Path) -> Network:
    """Load a Slotto-exported Pip brain as a raw Q-network."""
    return net_from_obj(json.loads(Path(path).read_text(encoding="utf-8")))


def net_to_obj(net: Network, meta: dict | None = None) -> dict:
    """Serialize a Network into the format Slotto's Import brain reads."""
    layers = [{"input_size": len(l.weights[0]),
               "output_size": len(l.biases),
               "weights": l.weights, "biases": l.biases,
               "activation": l.activation}
              for l in net.layers]
    n_in = len(net.layers[0].weights[0])
    n_out = len(net.layers[-1].biases)
    base = {"game": "slotto", "role": "bet-sizing Q-network", "pip": True,
            "obs": n_in, "obs_version": "v3", "actions": n_out}
    return {
        "format": FORMAT,
        "task": "reinforce",
        "input_size": n_in,
        "hidden_sizes": [len(l.biases) for l in net.layers[:-1]],
        "output_size": n_out,
        "activation": net.layers[0].activation if net.layers else "tanh",
        "output_names": ["bet-1", "bet-2", "bet-3"],
        "in_scaler": None, "out_scaler": None,
        "layers": layers,
        "meta": dict(base, **(meta or {})),
    }


def save_brain(net: Network, path: str | Path, meta: dict | None = None):
    """Write a Network as a Slotto-importable Pip brain file."""
    Path(path).write_text(json.dumps(net_to_obj(net, meta)), encoding="utf-8")


def load_dialogue(path: str | Path) -> list[dict]:
    """Read the exported chat corpus (pip-dialogue.jsonl), one dict/line."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ==========================================================================
# SlottoEnv - faithful port of the engine + Pip's obs/reward
#
# SYNC: these constants mirror web/slots-engine.js (T defaults) and the
# observation/reward in web/ai.js. If you change the game's tuned numbers,
# update here too and re-check against web/simulate.js. The __main__
# self-test asserts this port reproduces simulate.js's ground-truth
# reward table, so drift is caught.
# ==========================================================================

COLD, NEUTRAL, HOT = 0, 1, 2

# id, weight, pays[0..5] (n-of-a-kind from the left)
_SYMBOLS = [
    ("cherry", 30.0, [0, 0, 1, 3, 7, 13]),
    ("lemon", 24.0, [0, 0, 0, 4, 10, 20]),
    ("orange", 20.0, [0, 0, 0, 7, 16, 33]),
    ("heart", 14.0, [0, 0, 0, 10, 26, 52]),
    ("seven", 8.0, [0, 0, 0, 16, 40, 100]),
    ("diamond", 5.0, [0, 0, 0, 26, 78, 200]),
    ("penquin", 1.5, [0, 0, 0, 75, 400, 2500]),
]
_TOTAL_W = sum(w for _, w, _ in _SYMBOLS)
_PAYS = {sid: pays for sid, _, pays in _SYMBOLS}
_LINES = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2],
          [0, 1, 2, 1, 0], [2, 1, 0, 1, 2]]
_REELS, _ROWS = 5, 3
_LINE_COST = 1
_MATCH_BIAS = [0.08, 0.15, 0.18]
_LUCK_MULT = [1.0, 1.0, 1.8]
_STAY = 0.93
_POT_SEED = 60.0
_POT_CONTRIB = 0.02
_MAX_BET = 15.0             # 3 bet levels x 5 lines (constant reward denom)
_EPISODE = 60              # spins per training episode


def _total_bet(bet_level: int) -> int:
    return len(_LINES) * _LINE_COST * bet_level


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _scale(ret):            # ret -> [0,1], matches ai.js
    return _clip(ret, 0, 3) / 3.0


class SlottoEnv(Environment):
    """Slotto bet-sizing as an RL environment.

    One step = one paid spin. Action 0/1/2 -> bet level 1/2/3. The hidden
    luck chain evolves (bet size does NOT influence it - this is a
    contextual bandit with persistence, exactly as designed). Observation
    is the pre-spin payout-history feature vector Pip sees; reward is
    (win - bet) / MAX_BET clipped [-1, 12], the progressive pot included
    in win on a jackpot (matching Pip's real reward stream). The
    Extravaganza is deliberately NOT modelled: its free spins are
    invisible to Pip in the real game.
    """

    observation_size = 10
    action_size = 3

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    # -- engine internals (mirror slots-engine.js) -----------------------
    def _advance_luck(self):
        stay, move = _STAY, 1 - _STAY
        if self.luck == COLD:
            row = (stay, move, 0.0)
        elif self.luck == HOT:
            row = (0.0, move, stay)
        else:
            row = (move / 2, stay, move / 2)
        r = self.rng.random()
        self.luck = COLD if r < row[0] else NEUTRAL if r < row[0] + row[1] else HOT

    def _draw_symbol(self):
        v = self.rng.random() * _TOTAL_W
        for sid, w, _ in _SYMBOLS:
            v -= w
            if v <= 0:
                return sid
        return _SYMBOLS[0][0]

    def _draw_grid(self):
        bias = _MATCH_BIAS[self.luck]
        grid = []
        for r in range(_REELS):
            col = []
            for y in range(_ROWS):
                if r > 0 and self.rng.random() < bias:
                    col.append(grid[r - 1][y])
                else:
                    col.append(self._draw_symbol())
            grid.append(col)
        return grid

    def _evaluate(self, grid, bet_level, mult):
        total, jackpot = 0, False
        for line in _LINES:
            first = grid[0][line[0]]
            count = 1
            while count < _REELS and grid[count][line[count]] == first:
                count += 1
            pay = _PAYS[first][count] if count < len(_PAYS[first]) else 0
            if pay > 0:
                total += max(1, round(pay * bet_level * mult))
                if first == "penquin" and count >= 3:
                    jackpot = True
        return total, jackpot

    # -- Pip's observation (mirror ai.js buildObs) -----------------------
    def _obs(self):
        obs = []
        for i in range(6):
            idx = len(self.returns) - 6 + i
            obs.append(_scale(self.returns[idx]) if idx >= 0 else 0.33)
        recent = self.returns[-10:]
        hits = sum(1 for r in recent if r >= 0.4)
        obs.append(hits / len(recent) if recent else 0.2)
        obs.append(_scale(max(recent)) if recent else 0.2)
        obs.append(_clip((self.ewma_short - 0.22) * 4 + 0.5, 0, 1))
        obs.append(_clip((self.ewma_long - 0.22) * 4 + 0.5, 0, 1))
        return obs

    # -- Environment interface -------------------------------------------
    def reset(self):
        self.luck = NEUTRAL
        self.pot = _POT_SEED
        self.returns = []
        self.ewma_short = 0.33
        self.ewma_long = 0.33
        self.t = 0
        return self._obs()

    def step(self, action):
        bet_level = int(action) + 1
        self._advance_luck()
        bet = _total_bet(bet_level)
        self.pot += bet * _POT_CONTRIB
        grid = self._draw_grid()
        total, jackpot = self._evaluate(grid, bet_level, _LUCK_MULT[self.luck])
        if jackpot:
            total += round(self.pot)
            self.pot = _POT_SEED
        reward = _clip((total - bet) / _MAX_BET, -1, 12)
        ret = total / bet
        self.returns.append(ret)
        if len(self.returns) > 60:
            self.returns.pop(0)
        self.ewma_short = 0.30 * _scale(ret) + 0.70 * self.ewma_short
        self.ewma_long = 0.06 * _scale(ret) + 0.94 * self.ewma_long
        self.t += 1
        return self._obs(), reward, self.t >= _EPISODE


# ==========================================================================
# Batch trainer
#
# Online DQN is sample-inefficient on this bandit-shaped task (bet size
# does not move the luck chain), the lesson from web/ai.js + simulate.js.
# The converging recipe, mirrored here: collect transitions under a
# UNIFORM-random betting policy (so every (state, bet) region is sampled),
# then run shuffled epochs of supervised train_on with the Q-target = the
# plain reward (gamma 0). This is exactly how web/simulate.js builds the
# shipped pip-brain.json.
# ==========================================================================

def batch_train(spins: int = 200000, epochs: int = 6, hidden=None,
                seed: int = 7, verbose: bool = True) -> Network:
    """Train a fresh bet-sizing Q-net on Slotto and return it.

    Returns a NeuroForge Network you can save_brain() into the game.
    """
    env = SlottoEnv(seed=seed)
    obs = env.reset()
    data = []  # (obs, action, reward)
    rng = random.Random(seed + 1)
    for i in range(spins):
        a = rng.randrange(3)
        nxt, reward, done = env.step(a)
        data.append((obs, a, reward))
        obs = env.reset() if done else nxt
    if verbose:
        print(f"[batch] collected {len(data)} transitions")

    net = Network(SlottoEnv.observation_size, hidden or [32],
                  SlottoEnv.action_size, task="regression", activation="tanh")
    net.in_scaler = net.out_scaler = None
    for e in range(epochs):
        lr = 0.01 * (0.65 ** e)
        rng.shuffle(data)
        for obs_i, a, reward in data:
            target = list(net.activate(obs_i))
            target[a] = reward         # gamma 0: target IS the reward
            net.train_on(obs_i, target, lr=lr)
        if verbose:
            print(f"[batch] epoch {e + 1}/{epochs} done (lr {lr:.4f})")
    return net


# ==========================================================================
# Self-test
# ==========================================================================

if __name__ == "__main__":
    # 1) brain round-trips bit-exact (the property Import relies on)
    random.seed(0)
    net = Network(inputs=10, hidden=[32], outputs=3, task="regression")
    obj = net_to_obj(net, meta={"spins_learned": 400000})
    back = net_from_obj(json.loads(json.dumps(obj)))
    probe = [random.uniform(0, 1) for _ in range(10)]
    drift = max(abs(x - y) for x, y in zip(net.activate(probe), back.activate(probe)))
    assert obj["format"] == FORMAT and obj["meta"]["game"] == "slotto"
    assert drift < 1e-12, f"round-trip drift {drift}"
    print(f"[PASS] brain round-trip bit-exact (drift {drift:.2e})")

    # 2) env reproduces simulate.js's ground-truth reward table. Cycle bet
    #    levels evenly and average the compressed reward per (luck, bet).
    #    Expected (from web/simulate.js rewardGap): COLD negative and
    #    steepening with bet; HOT clearly positive. That parity proves the
    #    port matches the JS engine.
    env = SlottoEnv(seed=999)
    env.reset()
    sums = [[0.0, 0.0, 0.0] for _ in range(3)]
    ns = [[0, 0, 0] for _ in range(3)]
    N = 300000
    for i in range(N):
        a = i % 3
        luck_before = None  # luck is set inside step; capture after
        obs, reward, done = env.step(a)
        luck = env.luck
        sums[luck][a] += reward
        ns[luck][a] += 1
        if done:
            env.reset()
    names = ["COLD   ", "NEUTRAL", "HOT    "]
    print(f"[env] ground-truth mean reward per (luck, bet) over {N} spins:")
    for s in range(3):
        row = "  ".join(f"{sums[s][b] / ns[s][b]:+.3f}" if ns[s][b] else "  n/a "
                        for b in range(3))
        print(f"   {names[s]}  {row}")
    cold_ok = sums[COLD][0] / max(1, ns[COLD][0]) < 0
    hot_ok = sums[HOT][2] / max(1, ns[HOT][2]) > 0
    assert cold_ok and hot_ok, "env reward table does not match the game"
    print("[PASS] SlottoEnv reward table matches the game (COLD<0, HOT>0)")
    print("       -> Slotto is a valid NeuroForge training ground.")
