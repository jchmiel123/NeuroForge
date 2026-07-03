"""NeuroForge core self-tests. Run: python tests/test_core.py

Proves the engine actually learns:
  1. XOR classification (the classic non-linearly-separable problem -
     a network with no hidden layer CANNOT solve it, so passing proves
     backprop through hidden layers works)
  2. Nonlinear regression (sin + linear mix)
  3. Save/load round-trip produces identical predictions
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge import Network


def test_xor_classification():
    random.seed(7)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = ["same", "diff", "diff", "same"]
    net = Network(inputs=2, hidden=[8], outputs=2, task="classification")
    net.fit(X * 25, y * 25, epochs=200, lr=0.05, verbose=False)
    correct = sum(net.predict(x)[0] == t for x, t in zip(X, y))
    assert correct == 4, f"XOR: {correct}/4 correct"
    print(f"[PASS] XOR classification: {correct}/4")


def test_nonlinear_regression():
    random.seed(7)
    def truth(a, b):
        return [math.sin(a) * 2.0 + 0.5 * b, a * b * 0.1]
    X = [[random.uniform(-3, 3), random.uniform(-5, 5)] for _ in range(400)]
    y = [truth(a, b) for a, b in X]
    net = Network(inputs=2, hidden=[24, 12], outputs=2, task="regression",
                  activation="tanh")
    net.fit(X, y, epochs=400, lr=0.02, verbose=False)
    test_pts = [[random.uniform(-3, 3), random.uniform(-5, 5)] for _ in range(100)]
    mse = 0.0
    for a, b in test_pts:
        pred = net.predict([a, b])
        actual = truth(a, b)
        mse += sum((p - t) ** 2 for p, t in zip(pred, actual)) / 2
    mse /= len(test_pts)
    assert mse < 0.15, f"regression MSE too high: {mse:.4f}"
    print(f"[PASS] nonlinear regression: holdout MSE {mse:.4f}")


def test_save_load_roundtrip(tmp=Path(__file__).parent / "_tmp_model.json"):
    random.seed(7)
    X = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(50)]
    y = [[r[0] + r[1] * r[2]] for r in X]
    net = Network(inputs=3, hidden=[6], outputs=1, task="regression",
                  output_names=["result"])
    net.fit(X, y, epochs=50, verbose=False)
    probe = [0.3, -0.2, 0.9]
    before = net.predict(probe)
    net.save(tmp)
    loaded = Network.load(tmp)
    after = loaded.predict(probe)
    tmp.unlink()
    assert abs(before - after) < 1e-9, f"round-trip drift: {before} vs {after}"
    assert loaded.predict_named(probe)["result"] == after
    print(f"[PASS] save/load round-trip: prediction identical ({after:.5f})")


if __name__ == "__main__":
    test_xor_classification()
    test_nonlinear_regression()
    test_save_load_roundtrip()
    print("\nAll core tests passed.")
