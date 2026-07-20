"""NeuroForge recurrent self-tests. Run: python tests/test_recurrent.py

Three proofs:
  1. Delayed echo - target at step t is the input at step t-1. A memoryless
     net CANNOT do this; only a net that carries state can. This is the
     cleanest possible proof that recurrence works.
  2. Repeating pattern - learn "abcabc..." and generate it back, showing the
     hidden state tracks position in a cycle.
  3. Save/load round-trip is exact.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge.recurrent import RNN


def _rand_seq(n):
    return [[random.random()] for _ in range(n)]


def test_delayed_echo_needs_memory():
    """target[t] = input[t-1]. Unsolvable without memory."""
    random.seed(1)
    seqs, tgts = [], []
    for _ in range(60):
        s = _rand_seq(8)
        seqs.append(s)
        # None at t=0 (no previous input), then echo the prior value.
        tgts.append([None] + [s[t - 1][0] for t in range(1, len(s))])

    rnn = RNN(inputs=1, hidden=16, outputs=1, task="regression", seed=1)
    hist = rnn.fit(seqs, tgts, epochs=200, lr=0.02, momentum=0.9, verbose=False)
    assert hist["loss"][-1] < hist["loss"][0] * 0.2, "loss barely moved"

    # On a FRESH sequence, each prediction should recover the prior input.
    random.seed(999)
    test = _rand_seq(8)
    preds = rnn.predict_sequence(test)
    err = sum(abs(preds[t][0] - test[t - 1][0]) for t in range(1, len(test)))
    err /= (len(test) - 1)
    assert err < 0.08, f"echo error {err:.3f} too high - memory not learned"
    print(f"[PASS] delayed echo: final loss {hist['loss'][-1]:.4f}, "
          f"mean echo error on unseen seq {err:.3f}")


def test_repeating_pattern_and_generate():
    """Learn the cycle abcabc... and roll it forward from a seed."""
    random.seed(2)
    pattern = "abc" * 12
    classes = ["a", "b", "c"]
    idx = {c: i for i, c in enumerate(classes)}

    seq, tgt = [], []
    for t in range(len(pattern) - 1):
        onehot = [0.0, 0.0, 0.0]
        onehot[idx[pattern[t]]] = 1.0
        seq.append(onehot)
        tgt.append(pattern[t + 1])

    rnn = RNN(inputs=3, hidden=16, outputs=3, task="classification",
              classes=classes, seed=2)
    rnn.fit([seq], [tgt], epochs=300, lr=0.1, momentum=0.9, verbose=False)

    # Next-char accuracy on the training pattern should be ~perfect.
    preds = rnn.predict_sequence(seq)
    correct = sum(1 for t in range(len(tgt)) if preds[t][0] == tgt[t])
    acc = correct / len(tgt)
    assert acc > 0.95, f"next-char accuracy only {acc:.2f}"

    # Generation from a seed should continue the cycle deterministically.
    gen = rnn.generate(["a"], n=6, temperature=0.01)
    assert gen == ["b", "c", "a", "b", "c", "a"], f"generated {gen}"
    print(f"[PASS] repeating pattern: next-char acc {acc:.2f}, "
          f"generate('a') -> {''.join(gen)}")


def test_save_load_roundtrip(tmp_path=None):
    random.seed(3)
    seqs = [_rand_seq(6) for _ in range(20)]
    tgts = [[None] + [s[t - 1][0] for t in range(1, len(s))] for s in seqs]
    rnn = RNN(inputs=1, hidden=8, outputs=1, task="regression", seed=3)
    rnn.fit(seqs, tgts, epochs=50, verbose=False)

    path = Path(__file__).resolve().parent / "_tmp_rnn.json"
    rnn.save(path)
    loaded = RNN.load(path)
    probe = _rand_seq(6)
    a = rnn.predict_sequence(probe)
    b = loaded.predict_sequence(probe)
    for pa, pb in zip(a, b):
        assert abs(pa[0] - pb[0]) < 1e-9, "save/load changed predictions"
    path.unlink()
    print("[PASS] save/load: predictions identical after round-trip")


if __name__ == "__main__":
    test_delayed_echo_needs_memory()
    test_repeating_pattern_and_generate()
    test_save_load_roundtrip()
    print("\nAll recurrent tests passed.")
