"""
NeuroForge recurrent demo - two faces of memory.

  python demos/sequence_predict.py            # both demos
  python demos/sequence_predict.py --fast     # quick smoke run
  python demos/sequence_predict.py --text     # char-generation only
  python demos/sequence_predict.py --wave     # sine-prediction only

CHAR demo : learn a short phrase one character at a time, then GENERATE it
            back from a single seed letter. The only way the net can produce
            the next letter is by remembering where it is in the phrase -
            that is recurrence doing its job.

WAVE demo : learn a sine wave and predict the next value from the last one.
            A feed-forward net sees a single x and cannot know if the wave is
            rising or falling there; the RNN's hidden state carries the phase.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge.recurrent import RNN


def char_demo(fast: bool):
    phrase = "neuroforge remembers "
    text = phrase * (2 if fast else 6)
    classes = sorted(set(text))
    idx = {c: i for i, c in enumerate(classes)}
    V = len(classes)

    # Input at step t = one-hot(char_t); target = char_{t+1}.
    seq, tgt = [], []
    for t in range(len(text) - 1):
        oh = [0.0] * V
        oh[idx[text[t]]] = 1.0
        seq.append(oh)
        tgt.append(text[t + 1])

    print(f"CHAR demo: {V} distinct chars, {len(seq)} training steps")
    rnn = RNN(inputs=V, hidden=32, outputs=V, task="classification",
              classes=classes, seed=7)
    rnn.fit([seq], [tgt], epochs=80 if fast else 400, lr=0.1, verbose=True)

    # Prime the hidden state with the phrase (a WARM seed - a cold zero-state
    # seed lands the net in a context it saw only once, at char 0), then let
    # it free-run and watch it hold the cycle on its own.
    for temp in (0.01, 0.5):
        gen = rnn.generate(list(phrase), n=len(phrase) * 2, temperature=temp)
        print(f"  primed, temp {temp}, continues: {''.join(gen)!r}")
    print()


def wave_demo(fast: bool):
    # One long sine sequence: input = value now, target = value next step.
    steps = 60 if fast else 120
    xs = [math.sin(t * 0.35) for t in range(steps + 1)]
    seq = [[xs[t]] for t in range(steps)]
    tgt = [xs[t + 1] for t in range(steps)]

    print(f"WAVE demo: predict next sine value over {steps} steps")
    rnn = RNN(inputs=1, hidden=24, outputs=1, task="regression", seed=7)
    rnn.fit([seq], [tgt], epochs=150 if fast else 600, lr=0.02, verbose=True)

    preds = rnn.predict_sequence(seq)
    err = sum(abs(preds[t][0] - tgt[t]) for t in range(steps)) / steps
    print(f"  mean abs error over the wave: {err:.4f}")

    # Free-run: feed the model its OWN prediction and watch it trace a wave.
    rnn.reset_state()
    val = xs[0]
    trace = []
    for _ in range(24):
        out = rnn.step([val])
        val = out[0]
        trace.append(val)
    spark = "".join(_spark(v) for v in trace)
    print(f"  free-run (model feeding itself): {spark}")
    print()


def _spark(v: float) -> str:
    bars = " .:-=+*#%@"
    i = int((max(-1.0, min(1.0, v)) + 1.0) / 2.0 * (len(bars) - 1))
    return bars[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="quick smoke run")
    ap.add_argument("--text", action="store_true", help="char demo only")
    ap.add_argument("--wave", action="store_true", help="sine demo only")
    args = ap.parse_args()

    do_text = args.text or not args.wave
    do_wave = args.wave or not args.text
    if do_text:
        char_demo(args.fast)
    if do_wave:
        wave_demo(args.fast)


if __name__ == "__main__":
    main()
