"""
NeuroForge Evolution - learn by mutation and selection, no gradients.

Where backprop asks "which way should each weight move to reduce error?",
neuroevolution just tries random weight tweaks and keeps whatever scores
best. It needs no labeled data and no calculus - only a FITNESS number
("this brain ate 7 food") - which makes it the natural fit for game
creatures, controllers, and anything where you can score behavior but
can't say what the 'correct' output was at each moment.

    env = MyWorld()                       # implements Environment
    evo = Evolution(inputs=4, hidden=[12], outputs=3)
    best, history = evo.run(lambda net: run_episode(env, net), generations=40)
    best.act(sensors)                     # -> action index
"""

from __future__ import annotations

import random

from .core import Network


class Environment:
    """The plug-in point for anything a brain can live in: a game world,
    a simulator, a hardware rig. Phase 3 (Q-learning) reuses this.

    Contract:
        obs = env.reset()                    start an episode, first sensors
        obs, reward, done = env.step(action) apply action, advance one tick
    """

    observation_size: int = 0
    action_size: int = 0

    def reset(self) -> list[float]:
        raise NotImplementedError

    def step(self, action: int) -> tuple[list[float], float, bool]:
        raise NotImplementedError


def run_episode(env: Environment, net: Network, max_steps: int = 200) -> float:
    """Let one brain live one life; return total reward as its fitness."""
    obs = env.reset()
    total = 0.0
    for _ in range(max_steps):
        obs, reward, done = env.step(net.act(obs))
        total += reward
        if done:
            break
    return total


class Evolution:
    """Population-based trainer.

    Each generation:
      1. Score every brain with the fitness function.
      2. Keep the top `elite` unchanged (never lose your best).
      3. Refill the population with mutated copies of the elites,
         plus a few brand-new random brains (fresh ideas / anti-stagnation).
    """

    def __init__(self, inputs: int, hidden: list[int], outputs: int,
                 population: int = 50, elite: int = 5, fresh: int = 2,
                 mutation_rate: float = 0.15, mutation_scale: float = 0.4,
                 activation: str = "tanh", seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.inputs = inputs
        self.hidden = list(hidden)
        self.outputs = outputs
        self.population_size = population
        self.elite = elite
        self.fresh = fresh
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.activation = activation
        self.population = [self._new_brain() for _ in range(population)]

    def _new_brain(self) -> Network:
        return Network(self.inputs, self.hidden, self.outputs,
                       task="classification", activation=self.activation)

    def run(self, fitness, generations: int = 40,
            verbose: bool = True) -> tuple[Network, dict]:
        """fitness(net) -> float. Returns (best_network, history)."""
        history = {"best": [], "mean": []}
        best_ever: Network | None = None
        best_ever_score = float("-inf")
        report_every = max(1, generations // 10)

        for gen in range(1, generations + 1):
            scored = sorted(((fitness(net), net) for net in self.population),
                            key=lambda t: t[0], reverse=True)
            scores = [s for s, _ in scored]
            history["best"].append(scores[0])
            history["mean"].append(sum(scores) / len(scores))

            if scores[0] > best_ever_score:
                best_ever_score = scores[0]
                best_ever = scored[0][1].copy()

            if verbose and (gen % report_every == 0 or gen == 1):
                print(f"gen {gen:3d}/{generations}  best {scores[0]:8.2f}  "
                      f"mean {history['mean'][-1]:8.2f}")

            # Selection + reproduction
            elites = [net for _, net in scored[:self.elite]]
            next_pop = [e.copy() for e in elites]
            while len(next_pop) < self.population_size - self.fresh:
                parent = random.choice(elites)
                next_pop.append(parent.mutate(self.mutation_rate,
                                              self.mutation_scale))
            while len(next_pop) < self.population_size:
                next_pop.append(self._new_brain())
            self.population = next_pop

        return best_ever, history
