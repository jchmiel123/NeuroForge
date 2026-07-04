"""
DelveForge <-> NeuroForge bridge.

DelveForge (the roguelike AI arena) runs brains in the browser and can
Export/Import them as JSON. Those brains are raw Q-networks: the game
feeds an unscaled sensor vector straight in and reads the output layer's
Q-values, picking the argmax. They carry NO input/output scalers.

This bridge lets a NeuroForge `Network` load a DelveForge-exported brain
(for offline training / analysis) and export one back in the same
portable format the game's Import button understands.

Portable format (game Export button emits this):
    {
      "format": "neuroforge-network-v1",
      "input_size": 27, "hidden_sizes": [32,16], "output_size": 7,
      "activation": "tanh",
      "layers": [ {input_size, output_size, weights, biases, activation}, ... ],
      "meta": {"game":"delveforge","sensors":27,"actions":7,"cls":...,"bestDepth":...}
    }

Usage:
    from delveforge_bridge import load_brain, save_brain, DelveEnv
    net = load_brain("hero.json")          # a NeuroForge Network
    action = net.act(sensor_vector)        # 0..6, raw (no scaler)
    save_brain(net, "hero.json", meta={"cls":"Ranger"})

Because these are Q-networks, ALWAYS use net.activate()/net.act() (raw),
never net.predict() (which expects scalers this bridge deliberately
leaves as None).
"""

from __future__ import annotations

import json
from pathlib import Path

from neuroforge.core import Network, Layer

FORMAT = "neuroforge-network-v1"


def _layers_from_obj(obj: dict) -> list[Layer]:
    if "layers" not in obj or not obj["layers"]:
        raise ValueError("not a NeuroForge/DelveForge brain: no 'layers'")
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
    net.in_scaler = None      # raw Q-network: never scale
    net.out_scaler = None
    return net


def load_brain(path: str | Path) -> Network:
    """Load a DelveForge-exported brain file as a raw Q-network."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return net_from_obj(obj)


def net_to_obj(net: Network, meta: dict | None = None) -> dict:
    """Serialize a Network into the portable format the game Imports."""
    layers = [{"input_size": len(l.weights[0]),
               "output_size": len(l.biases),
               "weights": l.weights, "biases": l.biases,
               "activation": l.activation}
              for l in net.layers]
    n_in = len(net.layers[0].weights[0])
    n_out = len(net.layers[-1].biases)
    return {
        "format": FORMAT,
        "task": "classification",
        "input_size": n_in,
        "hidden_sizes": [len(l.biases) for l in net.layers[:-1]],
        "output_size": n_out,
        "activation": net.layers[0].activation if net.layers else "tanh",
        "layers": layers,
        "meta": dict({"game": "delveforge", "sensors": n_in,
                      "actions": n_out}, **(meta or {})),
    }


def save_brain(net: Network, path: str | Path, meta: dict | None = None):
    """Write a Network as a DelveForge-importable brain file."""
    Path(path).write_text(json.dumps(net_to_obj(net, meta)), encoding="utf-8")


# --- Environment stub (N3 lands the real sim here) -------------------------

class DelveEnv:
    """Placeholder for the Python DelveForge arena (HANDOFF N3).

    When implemented it must match neuroforge.evolve.Environment
    (reset() -> obs, step(action) -> (obs, reward, done)) and mirror the
    sensor/action/reward contracts documented in DelveForge/HANDOFF.md
    section 2 EXACTLY (27 senses, 7 actions, the reward table). Until
    then this raises so nobody trains against a half-built arena.
    """

    observation_size = 27
    action_size = 7

    def reset(self):
        raise NotImplementedError(
            "DelveEnv is a stub - implement the Python arena (HANDOFF N3) "
            "before training. Sensor/action/reward contracts are in "
            "DelveForge/HANDOFF.md section 2.")

    def step(self, action):
        raise NotImplementedError("see reset()")


if __name__ == "__main__":
    # Self-test: a random Network round-trips through the portable format
    # with bit-exact activations (the property the game's Import relies on).
    import random
    random.seed(0)
    net = Network(inputs=27, hidden=[32, 16], outputs=7, task="regression")
    obj = net_to_obj(net, meta={"cls": "Ranger", "bestDepth": 3})
    txt = json.dumps(obj)
    back = net_from_obj(json.loads(txt))
    probe = [random.uniform(-1, 1) for _ in range(27)]
    a = net.activate(probe)
    b = back.activate(probe)
    drift = max(abs(x - y) for x, y in zip(a, b))
    assert obj["format"] == FORMAT
    assert obj["meta"]["sensors"] == 27 and obj["meta"]["actions"] == 7
    assert drift < 1e-12, f"round-trip drift {drift}"
    assert net.act(probe) == back.act(probe)
    print(f"[PASS] portable round-trip bit-exact (drift {drift:.2e}); "
          f"act={net.act(probe)}")
