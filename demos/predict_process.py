"""Demo: predict process outputs from inputs, then ask what-if questions.

Simulates the exact workflow Justin described: you have historical rows of
"process settings in -> results out", you train on them, then you ask
"what happens to the output if I change one input?"

The fake process here is a chemical reactor:
  inputs:  temperature (C), pressure (bar), catalyst fraction (0-1)
  outputs: yield (%), purity (%)
Swap in your own CSV rows and the same code works for anything.

Run: python demos/predict_process.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroforge import Network


# --- The "real" process we pretend not to know (generates training data) ---

def real_process(temp, pressure, catalyst):
    """Hidden ground truth with interactions and sweet spots."""
    yield_pct = (
        55.0
        + 25.0 * pow(2.71828, -((temp - 210.0) / 40.0) ** 2)  # sweet spot ~210C
        + 3.0 * pressure
        - 0.25 * pressure ** 2                                 # too much hurts
        + 18.0 * catalyst * (temp / 250.0)                     # interaction term
    )
    purity_pct = 99.0 - 0.05 * max(0.0, temp - 200.0) - 1.5 * catalyst + 0.4 * pressure
    return [min(100.0, yield_pct), min(100.0, purity_pct)]


def noisy(vals, sd=0.5):
    return [v + random.gauss(0.0, sd) for v in vals]


def main():
    random.seed(42)

    # 1. "Historical data": 300 past production runs with sensor noise.
    print("Generating 300 historical process runs...")
    X, y = [], []
    for _ in range(300):
        row = [random.uniform(150, 250),   # temp C
               random.uniform(1, 10),      # pressure bar
               random.uniform(0, 1)]       # catalyst fraction
        X.append(row)
        y.append(noisy(real_process(*row)))

    # 2. Train.
    net = Network(inputs=3, hidden=[24, 12], outputs=2, task="regression",
                  activation="tanh", output_names=["yield_pct", "purity_pct"])
    print("Training...")
    net.fit(X, y, epochs=300, lr=0.02, validation_split=0.15, verbose=True)

    # 3. How good is it on runs it never saw?
    print("\n--- Accuracy on 5 unseen settings ---")
    print(f"{'temp':>6} {'bar':>5} {'cat':>5} | {'pred yield':>10} {'true':>6} | {'pred purity':>11} {'true':>6}")
    for _ in range(5):
        row = [random.uniform(150, 250), random.uniform(1, 10), random.uniform(0, 1)]
        pred = net.predict(row)
        true = real_process(*row)
        print(f"{row[0]:6.1f} {row[1]:5.2f} {row[2]:5.2f} | "
              f"{pred[0]:10.2f} {true[0]:6.2f} | {pred[1]:11.2f} {true[1]:6.2f}")

    # 4. The what-if question: hold pressure/catalyst, sweep temperature.
    print("\n--- What-if: sweep temperature at pressure=6.0, catalyst=0.5 ---")
    print(f"{'temp C':>7} | {'pred yield %':>12} | {'pred purity %':>13}")
    best = (None, -1.0)
    for temp in range(150, 251, 10):
        p = net.predict_named([float(temp), 6.0, 0.5])
        marker = ""
        if p["yield_pct"] > best[1]:
            best = (temp, p["yield_pct"])
        print(f"{temp:7d} | {p['yield_pct']:12.2f} | {p['purity_pct']:13.2f}")
    print(f"\nModel's advice: run at ~{best[0]}C for peak yield "
          f"(true sweet spot is 210C - did it find it?)")

    # 5. Persist for later use.
    model_path = Path(__file__).parent / "process_model.json"
    net.save(model_path)
    print(f"Model saved to {model_path.name} - reload with Network.load()")


if __name__ == "__main__":
    main()
