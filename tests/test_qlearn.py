"""NeuroForge Q-learning self-tests. Run: python tests/test_qlearn.py

Corridor task: agent starts at the left end of a 7-cell corridor and the
ONLY reward sits at the far right end. Every step costs a little. Random
walks rarely cash in; a trained agent must march straight right - which
proves reward is propagating backwards from the goal through Q-values.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge.evolve import Environment
from neuroforge.qlearn import QAgent

LENGTH = 7


class Corridor(Environment):
    observation_size = 1
    action_size = 2  # 0=left, 1=right

    def reset(self):
        self.pos = 0
        return [self.pos / (LENGTH - 1)]

    def step(self, action):
        self.pos += 1 if action == 1 else -1
        self.pos = max(0, min(LENGTH - 1, self.pos))
        if self.pos == LENGTH - 1:
            return [1.0], 10.0, True
        return [self.pos / (LENGTH - 1)], -0.1, False


def run_greedy(agent, max_steps=30):
    env = Corridor()
    obs = env.reset()
    for step in range(1, max_steps + 1):
        obs, reward, done = env.step(agent.greedy(obs))
        if done:
            return step
    return None  # never reached the goal


def test_qlearning_solves_corridor():
    random.seed(9)
    agent = QAgent(observation_size=1, action_size=2, hidden=[8],
                   gamma=0.9, lr=0.02, epsilon_decay=0.97, seed=9)
    untrained = run_greedy(agent)
    agent.train(Corridor(), episodes=120, max_steps=30, verbose=False)
    steps = run_greedy(agent)
    assert steps is not None, "trained agent never reaches the goal"
    assert steps == LENGTH - 1, f"took {steps} steps, optimal is {LENGTH - 1}"
    print(f"[PASS] corridor: untrained {'never' if untrained is None else untrained} "
          f"-> trained reaches goal in optimal {steps} steps")


def test_q_values_reflect_distance():
    """After training, Q(right) should beat Q(left) everywhere, and states
    closer to the goal should promise more reward."""
    random.seed(9)
    agent = QAgent(observation_size=1, action_size=2, hidden=[8],
                   gamma=0.9, lr=0.02, epsilon_decay=0.97, seed=9)
    agent.train(Corridor(), episodes=120, max_steps=30, verbose=False)
    values = []
    for pos in range(LENGTH - 1):
        q = agent.net.activate([pos / (LENGTH - 1)])
        assert q[1] > q[0], f"at pos {pos}, left looks better than right"
        values.append(max(q))
    assert values[-1] > values[0], "goal-adjacent state should be worth more"
    print(f"[PASS] Q-values: right beats left at all {LENGTH - 1} states, "
          f"value rises toward goal ({values[0]:.2f} -> {values[-1]:.2f})")


if __name__ == "__main__":
    test_qlearning_solves_corridor()
    test_q_values_reflect_distance()
    print("\nAll Q-learning tests passed.")
