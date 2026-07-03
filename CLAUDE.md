# NeuroForge - Claude Instructions

General-purpose neural network library. Pure Python, zero dependencies,
ASCII-only output everywhere.

## Layout

- `neuroforge/core.py` - Layer, Network, Scaler. Phase 1 engine + the
  agent interface (activate/act/copy/mutate) used by evolution.
- `neuroforge/evolve.py` - Environment interface, run_episode, Evolution
  trainer (elitism + mutation + fresh blood). Phase 3 Q-learning will
  reuse Environment.
- `tests/test_core.py`, `tests/test_evolve.py` - run BOTH after any
  change to core: `python tests/test_core.py && python tests/test_evolve.py`
- `demos/predict_process.py` - input->output prediction + what-if sweeps.
- `demos/creature_world.py` - evolved food-hunter in an ASCII world
  (`--watch` animates, `--fast` for quick smoke runs).

## Design rules

1. **Pure Python stays pure.** No numpy import in `neuroforge/` (Phase 4 may
   add an OPTIONAL fast path, never a hard dependency).
2. **Output layer is linear.** Classification applies softmax at the Network
   level and backprop uses the `probs - onehot` shortcut with
   `skip_activation=True`. Do not "fix" this by giving the output layer a
   nonlinear activation - that reintroduces MiniNet's double-derivative bug.
3. **Scalers are part of the model.** Anything that fits state during
   training must serialize in `save()` and restore in `load()`, and the
   round-trip test must stay exact.
4. **Models are JSON.** Human-inspectable, diffable, no pickle.
5. **No aliased weights.** `Layer.to_dict()` MUST return fresh lists -
   copy()/mutate() rely on it. Sharing references between clones corrupts
   elites during evolution (bug found and fixed 2026-07-03; the
   monotonicity of `history["best"]` in test_evolve is the canary).

## Relationship to other CodeLab projects

- Descended from `PixelForge/visionmodel/network.py` (MiniNet). PixelForge
  still uses its own copy - do NOT try to swap it out casually.
- QueueForge server (192.168.1.180:5555) has a `neural_train` task type for
  heavy training (Phase 4 target).
- Phase 2 creature demo may later target ESP32 boards; keep the core free of
  desktop-only assumptions where cheap.

## Versioning

`VERSION` file is the source of truth; keep `neuroforge/__init__.py`
`__version__` in sync. SemVer.
