"""NeuroForge - general-purpose neural network library for CodeLab.

Phase 1: supervised learning (regression + classification).
Phase 2: neuroevolution + Environment interface for game agents.
Phase 3: Q-learning decision agents (delayed rewards).
Phase 4: recurrence - RNN with memory for sequences (time series, text).
"""

from .core import Network, Layer, Scaler, softmax
from .evolve import Environment, Evolution, run_episode
from .qlearn import QAgent
from .recurrent import RNN

__version__ = "0.4.1"
__all__ = ["Network", "Layer", "Scaler", "softmax",
           "Environment", "Evolution", "run_episode", "QAgent", "RNN"]
