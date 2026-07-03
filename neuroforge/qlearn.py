"""
NeuroForge Q-Learning - learn from delayed rewards.

Evolution (Phase 2) scores a whole lifetime; Q-learning learns DURING
life, from individual moments - even when the payoff comes many moves
later. The trick: instead of predicting "what's the right action?", the
network predicts "how much total future reward does each action lead to
from here?" (its Q-values). Those predictions bootstrap off each other:

    Q(state, action) should equal  reward + gamma * best Q(next_state)

Each experience nudges one prediction toward that target, and the value
of a distant reward leaks backwards, one step per update, until moves
made long before the payoff know their worth.

Uses the same Environment interface as evolve.py:

    agent = QAgent(observation_size=7, action_size=4, hidden=[24])
    history = agent.train(env, episodes=300)
    agent.greedy(obs)          # -> best action index
    agent.net.save("q.json")   # the policy is just a Network
"""

from __future__ import annotations

import random

from .core import Network


class QAgent:
    """DQN-lite: neural Q-function + experience replay + target network.

    The two stabilizers matter even at this small scale:
    - replay buffer: learn from a shuffled memory of past moments, not
      just the last one (consecutive steps are too correlated)
    - target network: bootstrap targets come from a frozen copy that
      syncs occasionally, so the net isn't chasing its own moving output
    """

    def __init__(self, observation_size: int, action_size: int,
                 hidden: list[int] | None = None,
                 gamma: float = 0.95, lr: float = 0.01,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.99,
                 buffer_size: int = 5000, batch_size: int = 16,
                 target_sync: int = 250, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.action_size = action_size
        self.net = Network(observation_size, hidden or [24], action_size,
                           task="regression", activation="tanh")
        self.target = self.net.copy()
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer: list[tuple] = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_sync = target_sync
        self._steps = 0

    # -- policy ---------------------------------------------------------

    def act(self, obs: list[float]) -> int:
        """Epsilon-greedy: mostly exploit, sometimes explore."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return self.greedy(obs)

    def greedy(self, obs: list[float]) -> int:
        q = self.net.activate(obs)
        return max(range(len(q)), key=lambda i: q[i])

    # -- learning -------------------------------------------------------

    def remember(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def replay(self):
        """One batch of Q-updates from random past experiences."""
        if len(self.buffer) < self.batch_size:
            return
        for obs, action, reward, next_obs, done in \
                random.sample(self.buffer, self.batch_size):
            target_q = self.net.activate(obs)  # start from current beliefs
            if done:
                target_q[action] = reward
            else:
                future = max(self.target.activate(next_obs))
                target_q[action] = reward + self.gamma * future
            # Only the taken action's slot differs, so only it produces
            # gradient - the other outputs learn nothing from this moment.
            self.net.train_on(obs, target_q, self.lr)

    def train(self, env, episodes: int = 300, max_steps: int = 100,
              verbose: bool = True) -> dict:
        history = {"reward": [], "epsilon": []}
        report_every = max(1, episodes // 10)
        for ep in range(1, episodes + 1):
            obs = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.act(obs)
                next_obs, reward, done = env.step(action)
                self.remember(obs, action, reward, next_obs, done)
                self.replay()
                self._steps += 1
                if self._steps % self.target_sync == 0:
                    self.target = self.net.copy()
                obs = next_obs
                total += reward
                if done:
                    break
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)
            history["reward"].append(total)
            history["epsilon"].append(self.epsilon)
            if verbose and (ep % report_every == 0 or ep == 1):
                recent = history["reward"][-report_every:]
                print(f"episode {ep:4d}/{episodes}  "
                      f"avg reward {sum(recent) / len(recent):7.2f}  "
                      f"epsilon {self.epsilon:.2f}")
        return history
