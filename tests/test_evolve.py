"""NeuroForge evolution self-tests. Run: python tests/test_evolve.py

  1. Evolution beats random brains on a simple steering task
  2. mutate() changes weights, copy() does not
  3. Evolved brain survives a save/load round-trip (act() identical)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge import Network
from neuroforge.evolve import Evolution


def steering_fitness(net: Network) -> float:
    """Reward brains that pick action 1 when input is negative and
    action 2 when positive - a minimal 'turn toward the signal' task."""
    rng = random.Random(5)
    score = 0
    for _ in range(40):
        x = rng.uniform(-1, 1)
        want = 1 if x < 0 else 2
        if net.act([x, abs(x), 1.0, 0.0]) == want:
            score += 1
    return score


def test_evolution_learns():
    random.seed(3)
    base = sum(steering_fitness(Network(4, [6], 3, task="classification"))
               for _ in range(10)) / 10
    evo = Evolution(inputs=4, hidden=[6], outputs=3,
                    population=30, elite=4, seed=3)
    best, history = evo.run(steering_fitness, generations=15, verbose=False)
    final = steering_fitness(best)
    assert final >= 38, f"evolved brain only scored {final}/40"
    assert final > base + 10, f"no real improvement: {base:.1f} -> {final}"
    print(f"[PASS] evolution learns: random {base:.1f}/40 -> evolved {final}/40")


def test_mutate_and_copy():
    random.seed(3)
    net = Network(3, [4], 2, task="classification")
    clone = net.copy()
    child = net.mutate(rate=1.0, scale=0.5)
    same = net.layers[0].weights[0][0] == clone.layers[0].weights[0][0]
    changed = net.layers[0].weights[0][0] != child.layers[0].weights[0][0]
    assert same, "copy() must not change weights"
    assert changed, "mutate(rate=1.0) must change weights"
    probe = [0.1, -0.4, 0.7]
    assert net.activate(probe) == clone.activate(probe)
    print("[PASS] copy() identical, mutate() diverges")


def test_brain_save_load(tmp=Path(__file__).parent / "_tmp_brain.json"):
    random.seed(3)
    net = Network(4, [6], 3, task="classification").mutate(0.5, 0.5)
    probes = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(20)]
    before = [net.act(p) for p in probes]
    net.save(tmp)
    loaded = Network.load(tmp)
    tmp.unlink()
    after = [loaded.act(p) for p in probes]
    assert before == after, "actions differ after save/load"
    print("[PASS] evolved brain save/load: identical actions on 20 probes")


if __name__ == "__main__":
    test_evolution_learns()
    test_mutate_and_copy()
    test_brain_save_load()
    print("\nAll evolution tests passed.")
