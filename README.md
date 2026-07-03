# NeuroForge

General-purpose neural network library for CodeLab. Pure Python, zero
dependencies, models saved as JSON. One engine, multiple learning modes.

Descended from PixelForge's MiniNet (which hit 99.5% on vision), but
generalized: regression AND classification, built-in normalization,
clean softmax/cross-entropy gradients, gradient clipping.

## Quick start

Predict outputs from inputs (any "numbers in -> numbers out" problem):

```python
from neuroforge import Network

# X: rows of inputs, y: rows of outputs (your historical data)
net = Network(inputs=3, hidden=[24, 12], outputs=2, task="regression",
              output_names=["yield_pct", "purity_pct"])
net.fit(X, y, epochs=300, lr=0.02, validation_split=0.15)

net.predict([210.0, 6.0, 0.5])        # -> [96.3, 99.8]
net.predict_named([210.0, 6.0, 0.5])  # -> {"yield_pct": 96.3, "purity_pct": 99.8}

net.save("model.json")
net = Network.load("model.json")
```

Pick a label instead of predicting numbers:

```python
net = Network(inputs=4, hidden=[8], outputs=3, task="classification")
net.fit(X, ["walk", "run", "hide", ...])   # string labels or class indices
label, confidence = net.predict(x)
net.predict_proba(x)                        # {"walk": 0.1, "run": 0.85, ...}
```

Evolve a brain instead of training it (games, controllers - anything you
can score but can't label):

```python
from neuroforge import Evolution, Environment, run_episode

class MyWorld(Environment):          # implement reset() and step(action)
    ...

evo = Evolution(inputs=4, hidden=[12], outputs=3, population=60)
best, history = evo.run(lambda net: run_episode(MyWorld(), net),
                        generations=35)
best.act(sensor_values)              # -> action index
best.save("brain.json")
```

## Demos

```bash
python demos/predict_process.py     # train on process data, ask what-if questions
python demos/creature_world.py      # evolve a food-hunting creature (ASCII world)
python demos/creature_world.py --watch   # animate the best creature live
python tests/test_core.py           # XOR + regression + save/load self-tests
python tests/test_evolve.py         # evolution + mutation + brain persistence
```

## Why it learns when from-scratch attempts don't

Three things silently kill hand-rolled networks, all handled here:

1. **Normalization** - features on different scales (temperature ~200 vs
   catalyst ~0.5) wreck gradient descent. `fit()` z-scores inputs and
   regression targets automatically; `predict()` un-scales for you.
2. **Softmax + cross-entropy gradient** - the combined gradient at a linear
   output layer is simply `probs - onehot`. Applying the activation
   derivative on top of that (easy mistake) breaks or slows learning.
3. **Gradient clipping + He init** - one outlier sample can't blow up the
   weights, and signals don't vanish/explode through deep layers.

## Roadmap

- [x] Phase 1: supervised core (regression + classification), save/load, demos
- [x] Phase 2: `Environment` interface + neuroevolution trainer + creature demo
      (evolved hunter eats 10/10 food on unseen maps; random baseline eats 0)
- [ ] Phase 3: Q-learning decision agents (learn from delayed rewards)
- [ ] Phase 4: NumPy fast path, QueueForge `neural_train` offload,
      network visualizer (port from PixelForge)
