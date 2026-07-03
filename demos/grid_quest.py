"""Demo: Q-learning solves a two-stage quest with delayed rewards.

A grid world with a KEY and a DOOR. The door only pays out if the agent
already holds the key. So the right strategy is conditional:
    no key yet  -> go get the key
    have key    -> now go to the door
Nobody programs that rule in. The agent discovers it because Q-values
carry the door's payoff backwards through time, THROUGH the key pickup.

This is the mode for problems where evolution is overkill and supervised
learning is impossible (you can't label "the correct move" - you only
find out much later whether the moves paid off).

Run:  python demos/grid_quest.py           train + show solved runs
      python demos/grid_quest.py --fast    fewer episodes (smoke test)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge.evolve import Environment
from neuroforge.qlearn import QAgent

W, H = 8, 6
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right


class GridQuest(Environment):
    """Observation (9): SIGN of each delta to key and door, the raw
    normalized deltas, and have_key.

    Sensor-design lesson learned the hard way: with deltas alone the
    agent aced the long marches but fumbled the last cell - a delta of
    0.125 is too quiet for the net to hear, and greedy play oscillated.
    The sign features make "one cell left" exactly as loud as "seven
    cells left", which is all the POLICY needs; the magnitudes stay so
    Q-values can still estimate distance-to-payoff.
    Actions: up/down/left/right.
    Rewards: key +3, door WITH key +10 (ends episode), step -0.05,
    plus potential-based shaping toward the current objective."""

    observation_size = 9
    action_size = 4

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self):
        cells = [(x, y) for x in range(W) for y in range(H)]
        self.pos, self.key, self.door = self.rng.sample(cells, 3)
        self.has_key = False
        return self._observe()

    @staticmethod
    def _sgn(v):
        return 0.0 if v == 0 else (1.0 if v > 0 else -1.0)

    def _observe(self):
        x, y = self.pos
        kx, ky = self.key if not self.has_key else self.pos
        dx_k, dy_k = (kx - x) / W, (ky - y) / H
        dx_d, dy_d = (self.door[0] - x) / W, (self.door[1] - y) / H
        return [self._sgn(dx_k), self._sgn(dy_k),
                self._sgn(dx_d), self._sgn(dy_d),
                dx_k, dy_k, dx_d, dy_d,
                1.0 if self.has_key else 0.0]

    def _objective(self):
        return self.door if self.has_key else self.key

    @staticmethod
    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        # Potential-based shaping: a small bonus for closing distance to
        # the CURRENT objective. It never changes which policy is optimal
        # (Ng et al. 1999), but it turns the sparse end-reward into a
        # learning signal available at every single step - without it the
        # Q-gap between right and wrong moves (~0.05) drowns in
        # approximation noise and the greedy agent oscillates.
        obj = self._objective()
        d0 = self._dist(self.pos, obj)
        dx, dy = ACTIONS[action]
        x = max(0, min(W - 1, self.pos[0] + dx))
        y = max(0, min(H - 1, self.pos[1] + dy))
        self.pos = (x, y)
        reward = -0.05 + 0.3 * (d0 - self._dist(self.pos, obj))
        if not self.has_key and self.pos == self.key:
            self.has_key = True
            reward += 3.0
        if self.has_key and self.pos == self.door:
            return self._observe(), reward + 10.0, True
        return self._observe(), reward, False


def render(env: GridQuest, trail: dict) -> str:
    rows = []
    for y in range(H):
        row = []
        for x in range(W):
            cell = trail.get((x, y), ".")
            if (x, y) == env.door:
                cell = "D"
            if (x, y) == env.key and not env.has_key:
                cell = "K"
            if (x, y) == env.pos:
                cell = "@"
            row.append(cell)
        rows.append(" ".join(row))
    return "\n".join(rows)


def showcase(agent: QAgent, seed: int, max_steps: int = 60):
    env = GridQuest(seed)
    obs = env.reset()
    print(f"\n--- unseen quest (seed {seed}): @ start, K key, D door ---")
    print(render(env, {}))
    trail = {}
    for step in range(1, max_steps + 1):
        trail[env.pos] = "1" if not env.has_key else "2"
        obs, reward, done = env.step(agent.greedy(obs))
        if done:
            print(f"\nSolved in {step} steps "
                  f"(trail: 1 = hunting key, 2 = carrying key to door):")
            trail[env.pos] = "@"
            print(render(env, trail))
            return step
    print("\nFailed to solve within step limit:")
    print(render(env, trail))
    return None


def success_rate(agent: QAgent, seeds, max_steps: int = 60) -> float:
    wins = 0
    for s in seeds:
        env = GridQuest(s)
        obs = env.reset()
        for _ in range(max_steps):
            obs, _, done = env.step(agent.greedy(obs))
            if done:
                wins += 1
                break
    return wins / len(seeds)


def main():
    fast = "--fast" in sys.argv
    episodes = 150 if fast else 400

    random.seed(21)
    agent = QAgent(observation_size=9, action_size=4, hidden=[24],
                   gamma=0.9, lr=0.01, epsilon_decay=0.995,
                   batch_size=16, target_sync=300, seed=21)

    eval_seeds = list(range(5000, 5040))  # 40 quests never used in training
    base = success_rate(agent, eval_seeds)
    print(f"Untrained success rate on 40 unseen quests: {base:.0%}")
    print(f"Training for {episodes} episodes (randomized quests)...\n")

    agent.train(GridQuest(), episodes=episodes, max_steps=80)

    rate = success_rate(agent, eval_seeds)
    print(f"\nSuccess on 40 unseen quests: {base:.0%} -> {rate:.0%}")
    for s in (5001, 5017):
        showcase(agent, s)

    model_path = Path(__file__).parent / "quest_brain.json"
    agent.net.save(model_path)
    print(f"\nQ-network saved to {model_path.name}")


if __name__ == "__main__":
    main()
