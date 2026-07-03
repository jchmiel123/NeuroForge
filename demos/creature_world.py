"""Demo: a creature EVOLVES to hunt food. No training data, no gradients.

A 2D world with scattered food. The creature has simple senses (where is
the nearest food relative to my heading? how close am I to a wall?) and
three possible actions (forward / turn left / turn right). Nobody ever
tells it what the right move is - brains that eat more food get copied
with small random tweaks, brains that wander die out. Watch the average
generation score climb from "random stumbling" to "competent hunter".

Run:  python demos/creature_world.py            train + show best run
      python demos/creature_world.py --watch    animate the best creature
      python demos/creature_world.py --fast     shorter training (CI/tests)
"""

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge import Network
from neuroforge.evolve import Environment, Evolution, run_episode

W, H = 40.0, 20.0          # world size (also the ASCII render size)
FOOD_COUNT = 10
EAT_RADIUS = 1.2
TURN = math.radians(30.0)
SPEED = 0.8


class CreatureWorld(Environment):
    """Sensors (4): sin/cos of angle to nearest food relative to heading,
    distance to that food (0..1), distance to wall straight ahead (0..1).
    Actions (3): 0=forward, 1=turn left, 2=turn right.
    Reward: +10 per food eaten, tiny shaping reward for closing distance."""

    observation_size = 4
    action_size = 3

    def __init__(self, seed: int):
        self.seed = seed

    def reset(self) -> list[float]:
        rng = random.Random(self.seed)
        self.x, self.y = W / 2.0, H / 2.0
        self.heading = rng.uniform(0.0, 2.0 * math.pi)
        self.food = [(rng.uniform(2, W - 2), rng.uniform(2, H - 2))
                     for _ in range(FOOD_COUNT)]
        self.eaten = 0
        self._prev_dist = self._nearest()[0]
        return self._observe()

    def _nearest(self):
        best_d, best_f = float("inf"), None
        for fx, fy in self.food:
            d = math.hypot(fx - self.x, fy - self.y)
            if d < best_d:
                best_d, best_f = d, (fx, fy)
        return best_d, best_f

    def _wall_ahead(self) -> float:
        """Distance to the wall along current heading, normalized 0..1."""
        dx, dy = math.cos(self.heading), math.sin(self.heading)
        dists = []
        if dx > 1e-9:
            dists.append((W - self.x) / dx)
        elif dx < -1e-9:
            dists.append(-self.x / dx)
        if dy > 1e-9:
            dists.append((H - self.y) / dy)
        elif dy < -1e-9:
            dists.append(-self.y / dy)
        d = min(dists) if dists else max(W, H)
        return min(1.0, d / max(W, H))

    def _observe(self) -> list[float]:
        d, (fx, fy) = self._nearest()
        angle = math.atan2(fy - self.y, fx - self.x) - self.heading
        max_d = math.hypot(W, H)
        return [math.sin(angle), math.cos(angle),
                min(1.0, d / max_d), self._wall_ahead()]

    def step(self, action: int):
        if action == 1:
            self.heading -= TURN
        elif action == 2:
            self.heading += TURN
        else:
            self.x += SPEED * math.cos(self.heading)
            self.y += SPEED * math.sin(self.heading)
            self.x = max(0.5, min(W - 0.5, self.x))
            self.y = max(0.5, min(H - 0.5, self.y))

        reward = 0.0
        d, nearest = self._nearest()
        if d < EAT_RADIUS:
            self.food.remove(nearest)
            self.eaten += 1
            reward += 10.0
            if not self.food:
                return self._observe_or_zero(), reward, True
            d, _ = self._nearest()
        # Shaping: tiny reward for getting closer, so gen-1 brains that
        # merely drift foodward already out-score pure spinners.
        reward += 0.05 * (self._prev_dist - d)
        self._prev_dist = d
        return self._observe(), reward, False

    def _observe_or_zero(self):
        return self._observe() if self.food else [0.0] * 4


# --- rendering (ASCII, per house rules) ------------------------------------

def render(env: CreatureWorld, trail: set) -> str:
    rows = []
    for gy in range(int(H)):
        row = []
        for gx in range(int(W)):
            cell = " "
            if (gx, gy) in trail:
                cell = "."
            for fx, fy in env.food:
                if int(fx) == gx and int(fy) == gy:
                    cell = "*"
            if int(env.x) == gx and int(env.y) == gy:
                a = env.heading % (2 * math.pi)
                cell = (">" if a < 0.79 or a > 5.5 else
                        "v" if a < 2.36 else "<" if a < 3.93 else "^")
            row.append(cell)
        rows.append("|" + "".join(row) + "|")
    top = "+" + "-" * int(W) + "+"
    return "\n".join([top] + rows + [top])


def showcase(net: Network, seed: int, watch: bool, max_steps: int = 300):
    env = CreatureWorld(seed)
    obs = env.reset()
    trail = set()
    frames_at = {0, 60, 150, max_steps - 1}
    for step in range(max_steps):
        trail.add((int(env.x), int(env.y)))
        obs, _, done = env.step(net.act(obs))
        if watch:
            print("\033[H\033[J" + render(env, trail))
            print(f"step {step + 1:3d}   eaten {env.eaten}/{FOOD_COUNT}")
            time.sleep(0.04)
        elif step in frames_at or done:
            print(f"\n--- step {step + 1}, eaten {env.eaten}/{FOOD_COUNT} ---")
            print(render(env, trail))
        if done:
            break
    print(f"\nBest creature ate {env.eaten}/{FOOD_COUNT} food "
          f"in {step + 1} steps on an UNSEEN map (seed {seed}).")
    return env.eaten


def main():
    watch = "--watch" in sys.argv
    fast = "--fast" in sys.argv
    generations = 12 if fast else 35
    population = 40 if fast else 60

    random.seed(11)
    train_seeds = [101, 202, 303]  # every brain judged on the same 3 worlds

    def fitness(net: Network) -> float:
        return sum(run_episode(CreatureWorld(s), net, max_steps=250)
                   for s in train_seeds) / len(train_seeds)

    # Baseline: how does a random, unevolved brain do?
    baseline = sum(fitness(Network(4, [12], 3, task="classification",
                                   activation="tanh"))
                   for _ in range(5)) / 5
    print(f"Random-brain baseline fitness: {baseline:.2f}")
    print(f"Evolving {population} brains for {generations} generations...\n")

    evo = Evolution(inputs=4, hidden=[12], outputs=3,
                    population=population, elite=6, fresh=3,
                    mutation_rate=0.15, mutation_scale=0.4, seed=11)
    best, history = evo.run(fitness, generations=generations)

    print(f"\nFitness: baseline {baseline:.2f} -> evolved {history['best'][-1]:.2f}")
    eaten = showcase(best, seed=999, watch=watch)

    model_path = Path(__file__).parent / "creature_brain.json"
    best.save(model_path)
    print(f"Brain saved to {model_path.name} (reload with Network.load).")
    return history, baseline, eaten


if __name__ == "__main__":
    main()
