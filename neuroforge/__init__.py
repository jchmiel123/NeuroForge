"""NeuroForge - general-purpose neural network library for CodeLab.

Phase 1: supervised learning (regression + classification).
Phase 2: neuroevolution + Environment interface for game agents.
Phase 3 (planned): Q-learning decision agents.
"""

from .core import Network, Layer, Scaler, softmax
from .evolve import Environment, Evolution, run_episode

__version__ = "0.2.0"
__all__ = ["Network", "Layer", "Scaler", "softmax",
           "Environment", "Evolution", "run_episode"]
