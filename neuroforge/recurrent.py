"""
NeuroForge Recurrent - networks with MEMORY (Phase 4).

The feed-forward Network in core.py is an amnesiac: every predict() sees only
the current input. An RNN carries a hidden state forward through time, so it
can learn from ORDER and HISTORY - the next value in a time series, the next
character in a word, a sensor stream, a creature that remembers.

The whole idea in one line:

    h_t = tanh(W_xh . x_t  +  W_hh . h_{t-1}  +  b_h)      # memory update
    y_t = W_hy . h_t  +  b_y                               # readout

The SAME weights run at every timestep. h_{t-1} is last step's memory folded
back in - that feedback loop is the entire difference from a plain net.

Training uses BPTT (backprop through time): unroll the loop into a deep chain,
then walk gradients backward across timesteps. It is ordinary backprop applied
to time instead of layer depth. The math is commented step by step in
_bptt() because this is exactly where hand-rolled RNNs usually tip over.

Pure Python, no dependencies, ASCII-only - same house rules as core.py.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .core import Scaler, softmax

# Clip the per-timestep hidden gradient so a single step cannot spike. This is
# value clipping - cheap belt-and-suspenders on top of the global norm clip.
GRAD_CLIP = 5.0

# Global gradient-norm budget for the whole-sequence update. If the combined
# L2 norm of every gradient exceeds this, we scale them ALL down uniformly -
# preserving direction while taming magnitude. This is the single technique
# that makes RNNs trainable with momentum + a real learning rate instead of
# exploding to astronomically large losses (the classic pre-2013 RNN curse).
GRAD_NORM_MAX = 5.0


# ---------------------------------------------------------------------------
# Tiny matrix/vector helpers (list-of-lists, to stay numpy-free)
# ---------------------------------------------------------------------------

def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _zeros_mat(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _matvec(mat: list[list[float]], vec: list[float]) -> list[float]:
    """mat (rows x cols) times vec (cols) -> rows."""
    out = []
    for row in mat:
        s = 0.0
        for j, v in enumerate(vec):
            s += row[j] * v
        out.append(s)
    return out


def _matvec_T(mat: list[list[float]], vec: list[float], cols: int) -> list[float]:
    """mat^T (cols x rows) times vec (rows) -> cols, without building mat^T."""
    out = [0.0] * cols
    for i, v in enumerate(vec):
        row = mat[i]
        for j in range(cols):
            out[j] += row[j] * v
    return out


def _add_outer(acc: list[list[float]], a: list[float], b: list[float]):
    """acc += outer(a, b), i.e. acc[i][j] += a[i] * b[j]. In place."""
    for i, ai in enumerate(a):
        if ai == 0.0:
            continue
        row = acc[i]
        for j, bj in enumerate(b):
            row[j] += ai * bj


# ---------------------------------------------------------------------------
# RNN
# ---------------------------------------------------------------------------

class RNN:
    """Elman recurrent network - a plain net with a hidden-state feedback loop.

    Sequence regression (predict next number(s) from a stream):
        rnn = RNN(inputs=1, hidden=32, outputs=1, task="regression")
        rnn.fit(sequences, targets)         # each is a list of timestep vectors
        rnn.predict_sequence(seq)           # -> list of output vectors

    Sequence classification / generation (predict next token each step):
        rnn = RNN(inputs=V, hidden=64, outputs=V, task="classification")
        rnn.fit(sequences, targets)         # targets: class idx/label per step
        rnn.generate(seed, n=100)           # roll the model forward on itself

    A "sequence" is a list of timestep vectors: [[x0..], [x1..], ...].
    Targets align one-to-one with timesteps. A target of None at a step means
    "no loss here" - use that for sequence-to-one tasks (only the last step
    carries a target).
    """

    def __init__(self, inputs: int, hidden: int, outputs: int,
                 task: str = "regression",
                 classes: list[str] | None = None,
                 output_names: list[str] | None = None,
                 seed: int | None = None):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        if seed is not None:
            random.seed(seed)
        self.input_size = inputs
        self.hidden_size = hidden
        self.output_size = outputs
        self.task = task
        self.classes = list(classes) if classes else []
        self.output_names = list(output_names) if output_names else []
        self.in_scaler: Scaler | None = None
        self.out_scaler: Scaler | None = None

        # Input->hidden: He-style scaling on the input fan-in.
        sx = math.sqrt(2.0 / max(1, inputs))
        self.W_xh = [[random.gauss(0.0, sx) for _ in range(inputs)]
                     for _ in range(hidden)]
        # Hidden->hidden: SMALL init. A recurrent weight near identity scale
        # is what lets gradients survive many steps without blowing up; too
        # large here and the memory loop explodes.
        sh = math.sqrt(1.0 / max(1, hidden))
        self.W_hh = [[random.gauss(0.0, sh) for _ in range(hidden)]
                     for _ in range(hidden)]
        self.b_h = _zeros(hidden)
        # Hidden->output (linear readout; classification softmaxes on top).
        sy = math.sqrt(1.0 / max(1, hidden))
        self.W_hy = [[random.gauss(0.0, sy) for _ in range(hidden)]
                     for _ in range(outputs)]
        self.b_y = _zeros(outputs)

        # Momentum velocity buffers (plain SGD crawls on RNNs; momentum makes
        # BPTT actually converge in a reasonable number of epochs).
        self._init_velocity()

        # Live hidden state for step()/generate() streaming inference.
        self._h_live = _zeros(hidden)

    def _init_velocity(self):
        self._vW_xh = _zeros_mat(self.hidden_size, self.input_size)
        self._vW_hh = _zeros_mat(self.hidden_size, self.hidden_size)
        self._vb_h = _zeros(self.hidden_size)
        self._vW_hy = _zeros_mat(self.output_size, self.hidden_size)
        self._vb_y = _zeros(self.output_size)

    # -- forward over a whole sequence --------------------------------------

    def _forward_seq(self, seq: list[list[float]]):
        """Run the sequence, caching everything BPTT needs.

        Returns (outputs, hiddens) where hiddens[0] is the zero state h_{-1}
        and hiddens[t+1] is h_t, so hiddens has len(seq)+1 entries.
        """
        h = _zeros(self.hidden_size)
        hiddens = [h]
        outputs = []
        for x in seq:
            # a = W_xh x + W_hh h_prev + b_h
            a = _matvec(self.W_xh, x)
            rec = _matvec(self.W_hh, h)
            a = [a[i] + rec[i] + self.b_h[i] for i in range(self.hidden_size)]
            h = [math.tanh(v) for v in a]          # new memory
            o = _matvec(self.W_hy, h)
            o = [o[i] + self.b_y[i] for i in range(self.output_size)]
            outputs.append(o)
            hiddens.append(h)
        return outputs, hiddens

    # -- BPTT: the heart of it ----------------------------------------------

    def _bptt(self, seq, outputs, hiddens, targets, lr, momentum):
        """Backprop through time. Walk timesteps in REVERSE, accumulating the
        gradient of every weight (they are shared across all steps), then take
        one momentum step. Returns the summed loss over the sequence.
        """
        H, I, O = self.hidden_size, self.input_size, self.output_size

        # Gradient accumulators - one set for the whole sequence because the
        # weights are the same at every timestep.
        gW_xh = _zeros_mat(H, I)
        gW_hh = _zeros_mat(H, H)
        gb_h = _zeros(H)
        gW_hy = _zeros_mat(O, H)
        gb_y = _zeros(O)

        loss = 0.0
        dh_next = _zeros(H)          # gradient flowing back from future steps

        for t in reversed(range(len(seq))):
            h_t = hiddens[t + 1]     # hiddens[0] is h_{-1}
            h_prev = hiddens[t]
            target = targets[t]

            # --- gradient at the (linear) output pre-activation, do_t ---
            if target is None:
                do = _zeros(O)       # no supervision at this step
            elif self.task == "classification":
                probs = softmax(outputs[t])
                loss += -sum(target[k] * math.log(max(probs[k], 1e-12))
                             for k in range(O))
                # softmax+CE shortcut: gradient is simply probs - onehot.
                do = [probs[k] - target[k] for k in range(O)]
            else:
                diff = [outputs[t][k] - target[k] for k in range(O)]
                loss += sum(d * d for d in diff) / O
                do = [2.0 * d / O for d in diff]

            # --- output layer grads ---
            _add_outer(gW_hy, do, h_t)
            for k in range(O):
                gb_y[k] += do[k]

            # --- backprop into hidden state: from readout + from the future ---
            dh_from_out = _matvec_T(self.W_hy, do, H)
            dh = [dh_from_out[i] + dh_next[i] for i in range(H)]

            # through tanh: d/da tanh(a) = 1 - tanh(a)^2 = 1 - h^2
            da = [dh[i] * (1.0 - h_t[i] * h_t[i]) for i in range(H)]
            # Clip the recurrent gradient so long sequences stay stable.
            da = [max(-GRAD_CLIP, min(GRAD_CLIP, g)) for g in da]

            # --- input + recurrent + bias grads ---
            _add_outer(gW_xh, da, seq[t])
            _add_outer(gW_hh, da, h_prev)
            for i in range(H):
                gb_h[i] += da[i]

            # --- gradient handed to the PREVIOUS timestep's hidden state ---
            dh_next = _matvec_T(self.W_hh, da, H)

        self._clip_global_norm([gW_xh, gW_hh, gW_hy], [gb_h, gb_y])
        self._apply_grads(gW_xh, gW_hh, gb_h, gW_hy, gb_y, lr, momentum)
        return loss

    @staticmethod
    def _clip_global_norm(mats, vecs):
        """Rescale all gradients in place so their combined L2 norm <= budget."""
        total = 0.0
        for m in mats:
            for row in m:
                for g in row:
                    total += g * g
        for v in vecs:
            for g in v:
                total += g * g
        norm = math.sqrt(total)
        if norm <= GRAD_NORM_MAX or norm == 0.0:
            return
        scale = GRAD_NORM_MAX / norm
        for m in mats:
            for row in m:
                for j in range(len(row)):
                    row[j] *= scale
        for v in vecs:
            for j in range(len(v)):
                v[j] *= scale

    def _apply_grads(self, gW_xh, gW_hh, gb_h, gW_hy, gb_y, lr, momentum):
        """Momentum SGD update: v = momentum*v - lr*grad ; param += v."""
        def step_mat(param, vel, grad):
            for i in range(len(param)):
                p, ve, gr = param[i], vel[i], grad[i]
                for j in range(len(p)):
                    ve[j] = momentum * ve[j] - lr * gr[j]
                    p[j] += ve[j]

        def step_vec(param, vel, grad):
            for i in range(len(param)):
                vel[i] = momentum * vel[i] - lr * grad[i]
                param[i] += vel[i]

        step_mat(self.W_xh, self._vW_xh, gW_xh)
        step_mat(self.W_hh, self._vW_hh, gW_hh)
        step_vec(self.b_h, self._vb_h, gb_h)
        step_mat(self.W_hy, self._vW_hy, gW_hy)
        step_vec(self.b_y, self._vb_y, gb_y)

    # -- training -----------------------------------------------------------

    def fit(self, sequences: list[list[list[float]]], targets,
            epochs: int = 100, lr: float = 0.05, momentum: float = 0.9,
            verbose: bool = True, shuffle: bool = True) -> dict:
        """Train on a batch of sequences.

        sequences : list of sequences; each sequence is a list of timestep
                    input vectors [[x0..], [x1..], ...].
        targets   : list aligned with sequences; each is a list aligned with
                    that sequence's timesteps. Per element:
                      regression     -> float or list of floats (or None)
                      classification -> class index/label      (or None)
        """
        seqs = [[list(map(float, x)) for x in s] for s in sequences]
        tgts = [self._prepare_targets(t) for t in targets]

        # Fit scalers across ALL timesteps of ALL sequences (regression only).
        self.in_scaler = Scaler().fit([x for s in seqs for x in s])
        seqs = [[self.in_scaler.transform(x) for x in s] for s in seqs]
        if self.task == "regression":
            flat = [t for row in tgts for t in row if t is not None]
            self.out_scaler = Scaler().fit(flat)
            tgts = [[None if t is None else self.out_scaler.transform(t)
                     for t in row] for row in tgts]

        data = list(zip(seqs, tgts))
        history = {"loss": []}
        report_every = max(1, epochs // 10)
        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(data)
            total = 0.0
            steps = 0
            for seq, target in data:
                outputs, hiddens = self._forward_seq(seq)
                total += self._bptt(seq, outputs, hiddens, target, lr, momentum)
                steps += len(seq)
            loss = total / max(1, steps)
            history["loss"].append(loss)
            if verbose and (epoch % report_every == 0 or epoch == 1):
                print(f"epoch {epoch:4d}/{epochs}  loss {loss:.5f}")
        return history

    def _prepare_targets(self, row) -> list:
        """Turn one sequence's raw targets into model-space vectors (or None)."""
        out = []
        for t in row:
            if t is None:
                out.append(None)
            elif self.task == "classification":
                if isinstance(t, str):
                    if t not in self.classes:
                        self.classes.append(t)
                    idx = self.classes.index(t)
                else:
                    idx = int(t)
                onehot = [0.0] * self.output_size
                onehot[idx] = 1.0
                out.append(onehot)
            else:
                out.append([float(t)] if isinstance(t, (int, float))
                           else list(map(float, t)))
        return out

    # -- batched inference --------------------------------------------------

    def predict_sequence(self, seq: list[list[float]]):
        """Run a whole sequence and return one output per timestep.

        Regression     -> list of output vectors (real units, un-scaled).
        Classification -> list of (label, confidence) tuples.
        """
        if self.in_scaler is None:
            raise RuntimeError("Model is untrained - call fit() first")
        scaled = [self.in_scaler.transform(list(map(float, x))) for x in seq]
        outputs, _ = self._forward_seq(scaled)
        if self.task == "classification":
            result = []
            for o in outputs:
                probs = softmax(o)
                idx = max(range(len(probs)), key=lambda i: probs[i])
                label = self.classes[idx] if idx < len(self.classes) else str(idx)
                result.append((label, probs[idx]))
            return result
        return [self.out_scaler.inverse(o) for o in outputs]

    # -- streaming inference (memory persists across calls) -----------------

    def reset_state(self):
        """Clear the live hidden memory used by step()/generate()."""
        self._h_live = _zeros(self.hidden_size)

    def step(self, x: list[float]):
        """Feed ONE timestep, advancing the live hidden state. Use this for
        real-time streams or agents whose memory must persist between calls.
        Returns the same shape as predict_sequence's per-step output.
        """
        if self.in_scaler is None:
            raise RuntimeError("Model is untrained - call fit() first")
        xs = self.in_scaler.transform(list(map(float, x)))
        a = _matvec(self.W_xh, xs)
        rec = _matvec(self.W_hh, self._h_live)
        a = [a[i] + rec[i] + self.b_h[i] for i in range(self.hidden_size)]
        self._h_live = [math.tanh(v) for v in a]
        o = _matvec(self.W_hy, self._h_live)
        o = [o[i] + self.b_y[i] for i in range(self.output_size)]
        if self.task == "classification":
            probs = softmax(o)
            idx = max(range(len(probs)), key=lambda i: probs[i])
            label = self.classes[idx] if idx < len(self.classes) else str(idx)
            return label, probs[idx]
        return self.out_scaler.inverse(o)

    def generate(self, seed: list, n: int, temperature: float = 1.0,
                 seed_fn=None, encode=None) -> list:
        """Classification only: roll the model forward on its own predictions.

        seed        : list of initial tokens to prime the hidden state with.
        n           : how many new tokens to produce.
        temperature : >1 wilder, <1 safer, ->0 greedy argmax.
        encode      : optional fn(token)->input vector. Defaults to a one-hot
                      over class index (the natural char/token-model setup).

        Returns the list of generated tokens (labels), seed NOT included.
        """
        if self.task != "classification":
            raise RuntimeError("generate() is for classification models")
        if encode is None:
            def encode(tok):
                vec = [0.0] * self.input_size
                idx = self.classes.index(tok) if tok in self.classes else int(tok)
                vec[idx] = 1.0
                return vec

        self.reset_state()
        # Prime memory with the seed (advance state, ignore its predictions).
        last = None
        for tok in seed:
            self.step(encode(tok))
            last = tok

        out = []
        for _ in range(n):
            xs = self.in_scaler.transform(encode(last))
            a = _matvec(self.W_xh, xs)
            rec = _matvec(self.W_hh, self._h_live)
            a = [a[i] + rec[i] + self.b_h[i] for i in range(self.hidden_size)]
            self._h_live = [math.tanh(v) for v in a]
            o = _matvec(self.W_hy, self._h_live)
            o = [o[i] + self.b_y[i] for i in range(self.output_size)]
            idx = _sample(softmax([v / max(temperature, 1e-6) for v in o]))
            last = self.classes[idx] if idx < len(self.classes) else idx
            out.append(last)
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "neuroforge-rnn-v1",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "task": self.task,
            "classes": self.classes,
            "output_names": self.output_names,
            "in_scaler": self.in_scaler.to_dict() if self.in_scaler else None,
            "out_scaler": self.out_scaler.to_dict() if self.out_scaler else None,
            "W_xh": self.W_xh, "W_hh": self.W_hh, "b_h": self.b_h,
            "W_hy": self.W_hy, "b_y": self.b_y,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RNN":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        net = cls(inputs=data["input_size"], hidden=data["hidden_size"],
                  outputs=data["output_size"], task=data["task"],
                  classes=data["classes"], output_names=data["output_names"])
        net.W_xh = [row[:] for row in data["W_xh"]]
        net.W_hh = [row[:] for row in data["W_hh"]]
        net.b_h = list(data["b_h"])
        net.W_hy = [row[:] for row in data["W_hy"]]
        net.b_y = list(data["b_y"])
        if data["in_scaler"]:
            net.in_scaler = Scaler.from_dict(data["in_scaler"])
        if data["out_scaler"]:
            net.out_scaler = Scaler.from_dict(data["out_scaler"])
        return net

    def __repr__(self):
        return (f"RNN({self.input_size}->[{self.hidden_size} recurrent]->"
                f"{self.output_size}, task={self.task})")


def _sample(probs: list[float]) -> int:
    """Draw an index from a probability distribution (roulette wheel)."""
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1
