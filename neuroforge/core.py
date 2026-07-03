"""
NeuroForge Core - general-purpose neural network engine.

Pure Python, no dependencies. Descended from PixelForge's MiniNet
(PixelForge/visionmodel/network.py) but generalized:

- Regression AND classification (MiniNet was classification-only)
- Built-in input/target normalization (the #1 reason small nets fail
  to learn: raw features on wildly different scales)
- Clean softmax + cross-entropy gradient (probs - onehot fed to a
  LINEAR output layer, so no activation derivative is applied twice)
- Gradient clipping (borrowed from the QueueForge neural trainer)

The math, in one breath: forward() multiplies inputs through weighted
layers; the loss measures how wrong the output is; backward() walks the
layers in reverse applying the chain rule, so every weight learns how
much IT contributed to the error and nudges itself the other way.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x < -700.0:
        return 0.0
    if x > 700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_deriv(x: float) -> float:
    s = _sigmoid(x)
    return s * (1.0 - s)


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_deriv(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _tanh(x: float) -> float:
    return math.tanh(x)


def _tanh_deriv(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


def _linear(x: float) -> float:
    return x


def _linear_deriv(x: float) -> float:
    return 1.0


ACTIVATIONS = {
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "relu": (_relu, _relu_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "linear": (_linear, _linear_deriv),
}


def softmax(values: list[float]) -> list[float]:
    """Turn raw scores into probabilities that sum to 1."""
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


GRAD_CLIP = 5.0  # keep any single gradient from blowing up a weight


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """One fully-connected layer: outputs = activation(W * inputs + b)."""

    input_size: int
    output_size: int
    activation: str = "relu"
    weights: list[list[float]] = field(default_factory=list)
    biases: list[float] = field(default_factory=list)

    # Cached during forward() so backward() can compute gradients.
    _inputs: list[float] = field(default_factory=list, repr=False)
    _pre_activation: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.weights:
            # He initialization: random weights scaled so signals neither
            # explode nor vanish as they pass through many layers.
            scale = math.sqrt(2.0 / self.input_size)
            self.weights = [
                [random.gauss(0.0, scale) for _ in range(self.input_size)]
                for _ in range(self.output_size)
            ]
        if not self.biases:
            self.biases = [0.0] * self.output_size

    def forward(self, inputs: list[float]) -> list[float]:
        self._inputs = inputs
        self._pre_activation = []
        act, _ = ACTIVATIONS[self.activation]
        outputs = []
        for i in range(self.output_size):
            total = self.biases[i]
            row = self.weights[i]
            for j, x in enumerate(inputs):
                total += row[j] * x
            self._pre_activation.append(total)
            outputs.append(act(total))
        return outputs

    def backward(self, output_grads: list[float], lr: float,
                 skip_activation: bool = False) -> list[float]:
        """Chain rule step: given dLoss/dOutput, update weights and return
        dLoss/dInput for the layer before this one.

        skip_activation=True is used by the output layer under
        softmax+cross-entropy, where (probs - target) already IS the
        gradient at the pre-activation - applying the derivative again
        is the subtle bug that muddied MiniNet.
        """
        _, deriv = ACTIVATIONS[self.activation]
        if skip_activation:
            grads = list(output_grads)
        else:
            grads = [
                output_grads[i] * deriv(self._pre_activation[i])
                for i in range(self.output_size)
            ]
        # Clip so one bad sample cannot blow up the weights.
        grads = [max(-GRAD_CLIP, min(GRAD_CLIP, g)) for g in grads]

        # What the previous layer needs, computed BEFORE weights change.
        input_grads = [0.0] * self.input_size
        for j in range(self.input_size):
            s = 0.0
            for i in range(self.output_size):
                s += self.weights[i][j] * grads[i]
            input_grads[j] = s

        # Gradient descent: nudge every weight against its gradient.
        for i in range(self.output_size):
            g = grads[i]
            row = self.weights[i]
            for j, x in enumerate(self._inputs):
                row[j] -= lr * g * x
            self.biases[i] -= lr * g
        return input_grads

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation": self.activation,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Layer:
        return cls(**data)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

class Scaler:
    """Per-column z-score scaling. Networks learn far better when every
    feature is roughly mean 0 / std 1 instead of e.g. temperature in the
    hundreds next to a 0-1 catalyst fraction."""

    def __init__(self, means: list[float] | None = None,
                 stds: list[float] | None = None):
        self.means = means or []
        self.stds = stds or []

    def fit(self, rows: list[list[float]]) -> "Scaler":
        n = len(rows)
        cols = len(rows[0])
        self.means = [sum(r[c] for r in rows) / n for c in range(cols)]
        self.stds = []
        for c in range(cols):
            var = sum((r[c] - self.means[c]) ** 2 for r in rows) / n
            std = math.sqrt(var)
            self.stds.append(std if std > 1e-12 else 1.0)
        return self

    def transform(self, row: list[float]) -> list[float]:
        return [(x - m) / s for x, m, s in zip(row, self.means, self.stds)]

    def inverse(self, row: list[float]) -> list[float]:
        return [x * s + m for x, m, s in zip(row, self.means, self.stds)]

    def to_dict(self) -> dict:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        return cls(data["means"], data["stds"])


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """General-purpose feed-forward neural network.

    Regression (predict numbers from numbers):
        net = Network(inputs=3, hidden=[16, 8], outputs=2, task="regression")
        net.fit(X, y)                    # y: list of [out1, out2] rows
        net.predict([210.0, 4.5, 0.3])   # -> [pred1, pred2]

    Classification (pick a label):
        net = Network(inputs=4, hidden=[8], outputs=3, task="classification",
                      classes=["rock", "paper", "scissors"])
        net.fit(X, labels)               # labels: strings or class indices
        net.predict(x)                   # -> ("paper", 0.93)
    """

    def __init__(self, inputs: int, hidden: list[int], outputs: int,
                 task: str = "regression", activation: str = "relu",
                 classes: list[str] | None = None,
                 output_names: list[str] | None = None,
                 seed: int | None = None):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        if seed is not None:
            random.seed(seed)
        self.input_size = inputs
        self.hidden_sizes = list(hidden)
        self.output_size = outputs
        self.task = task
        self.activation = activation
        self.classes = list(classes) if classes else []
        self.output_names = list(output_names) if output_names else []
        self.in_scaler: Scaler | None = None
        self.out_scaler: Scaler | None = None

        self.layers: list[Layer] = []
        prev = inputs
        for h in hidden:
            self.layers.append(Layer(prev, h, activation))
            prev = h
        # Output layer is LINEAR for both tasks; classification applies
        # softmax on top at the network level.
        self.layers.append(Layer(prev, outputs, "linear"))

    # -- forward ------------------------------------------------------------

    def _forward_raw(self, scaled_inputs: list[float]) -> list[float]:
        current = scaled_inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    # -- training -----------------------------------------------------------

    def fit(self, X: list[list[float]], y, epochs: int = 300,
            lr: float = 0.01, validation_split: float = 0.0,
            verbose: bool = True, shuffle: bool = True) -> dict:
        """Train on rows X against targets y. Returns training history.

        y accepts, per row:
          regression      -> float or list of floats
          classification  -> class index (int) or label (str)
        """
        X = [list(map(float, row)) for row in X]
        targets = self._prepare_targets(y)

        # Fit scalers on the training data, then train in scaled space.
        self.in_scaler = Scaler().fit(X)
        Xs = [self.in_scaler.transform(r) for r in X]
        if self.task == "regression":
            self.out_scaler = Scaler().fit(targets)
            Ts = [self.out_scaler.transform(t) for t in targets]
        else:
            Ts = targets

        data = list(zip(Xs, Ts))
        n_val = int(len(data) * validation_split)
        if shuffle:
            random.shuffle(data)
        val, train = data[:n_val], data[n_val:]

        history = {"loss": [], "val_loss": []}
        report_every = max(1, epochs // 10)
        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(train)
            total = 0.0
            for xs, ts in train:
                out = self._forward_raw(xs)
                total += self._loss_and_backward(out, ts, lr)
            loss = total / max(1, len(train))
            history["loss"].append(loss)
            if val:
                vloss = sum(self._loss_only(self._forward_raw(xs), ts)
                            for xs, ts in val) / len(val)
                history["val_loss"].append(vloss)
            if verbose and (epoch % report_every == 0 or epoch == 1):
                msg = f"epoch {epoch:4d}/{epochs}  loss {loss:.5f}"
                if val:
                    msg += f"  val {history['val_loss'][-1]:.5f}"
                print(msg)
        return history

    def _prepare_targets(self, y) -> list[list[float]]:
        targets = []
        for t in y:
            if self.task == "classification":
                if isinstance(t, str):
                    if t not in self.classes:
                        self.classes.append(t)
                    idx = self.classes.index(t)
                else:
                    idx = int(t)
                onehot = [0.0] * self.output_size
                onehot[idx] = 1.0
                targets.append(onehot)
            else:
                targets.append([float(t)] if isinstance(t, (int, float))
                               else list(map(float, t)))
        return targets

    def _loss_only(self, out: list[float], target: list[float]) -> float:
        if self.task == "classification":
            probs = softmax(out)
            # Cross-entropy: penalize low probability on the true class.
            return -sum(t * math.log(max(p, 1e-12))
                        for p, t in zip(probs, target))
        return sum((o - t) ** 2 for o, t in zip(out, target)) / len(out)

    def _loss_and_backward(self, out: list[float], target: list[float],
                           lr: float) -> float:
        if self.task == "classification":
            probs = softmax(out)
            loss = -sum(t * math.log(max(p, 1e-12))
                        for p, t in zip(probs, target))
            # THE softmax+cross-entropy shortcut: the combined gradient at
            # the (linear) output is simply probs - onehot.
            grads = [p - t for p, t in zip(probs, target)]
        else:
            loss = sum((o - t) ** 2 for o, t in zip(out, target)) / len(out)
            grads = [2.0 * (o - t) / len(out) for o, t in zip(out, target)]

        skip = True  # gradient above is already at pre-activation (linear)
        for layer in reversed(self.layers):
            grads = layer.backward(grads, lr, skip_activation=skip)
            skip = False
        return loss

    # -- inference ----------------------------------------------------------

    def predict(self, inputs: list[float]):
        """Regression: list of outputs (or single float if one output).
        Classification: (label, confidence)."""
        if self.in_scaler is None:
            raise RuntimeError("Model is untrained - call fit() first")
        out = self._forward_raw(self.in_scaler.transform(list(map(float, inputs))))
        if self.task == "classification":
            probs = softmax(out)
            idx = max(range(len(probs)), key=lambda i: probs[i])
            label = self.classes[idx] if idx < len(self.classes) else str(idx)
            return label, probs[idx]
        real = self.out_scaler.inverse(out)
        return real[0] if self.output_size == 1 else real

    def predict_proba(self, inputs: list[float]) -> dict:
        """Classification only: {label: probability} for every class."""
        if self.task != "classification":
            raise RuntimeError("predict_proba is for classification models")
        out = self._forward_raw(self.in_scaler.transform(list(map(float, inputs))))
        probs = softmax(out)
        names = self.classes or [str(i) for i in range(self.output_size)]
        return dict(zip(names, probs))

    def predict_named(self, inputs: list[float]) -> dict:
        """Regression only: {output_name: value}."""
        if self.task != "regression":
            raise RuntimeError("predict_named is for regression models")
        vals = self.predict(inputs)
        if self.output_size == 1:
            vals = [vals]
        names = self.output_names or [f"out_{i}" for i in range(self.output_size)]
        return dict(zip(names, vals))

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "neuroforge-network-v1",
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "task": self.task,
            "activation": self.activation,
            "classes": self.classes,
            "output_names": self.output_names,
            "in_scaler": self.in_scaler.to_dict() if self.in_scaler else None,
            "out_scaler": self.out_scaler.to_dict() if self.out_scaler else None,
            "layers": [l.to_dict() for l in self.layers],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        net = cls(
            inputs=data["input_size"],
            hidden=data["hidden_sizes"],
            outputs=data["output_size"],
            task=data["task"],
            activation=data["activation"],
            classes=data["classes"],
            output_names=data["output_names"],
        )
        net.layers = [Layer.from_dict(ld) for ld in data["layers"]]
        if data["in_scaler"]:
            net.in_scaler = Scaler.from_dict(data["in_scaler"])
        if data["out_scaler"]:
            net.out_scaler = Scaler.from_dict(data["out_scaler"])
        return net

    def __repr__(self):
        arch = "-".join(str(s) for s in
                        [self.input_size] + self.hidden_sizes + [self.output_size])
        return f"Network({arch}, task={self.task})"
