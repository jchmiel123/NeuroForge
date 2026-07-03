# NeuroForge - Claude Instructions

General-purpose neural network library. Pure Python, zero dependencies,
ASCII-only output everywhere.

## Layout

- `neuroforge/core.py` - Layer, Network, Scaler. The entire Phase 1 engine.
- `tests/test_core.py` - self-tests (XOR, nonlinear regression, save/load).
  Run after ANY change to core: `python tests/test_core.py`
- `demos/predict_process.py` - input->output prediction + what-if sweeps.

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
