# NeuroForge - Claude Instructions

General-purpose neural network library. Pure Python, zero dependencies,
ASCII-only output everywhere.

## Layout

- `neuroforge/core.py` - Layer, Network, Scaler. Phase 1 engine + the
  agent interface (activate/act/copy/mutate) used by evolution.
- `neuroforge/evolve.py` - Environment interface, run_episode, Evolution
  trainer (elitism + mutation + fresh blood).
- `neuroforge/qlearn.py` - QAgent (DQN-lite: replay buffer + target
  network + epsilon-greedy). Uses Network.train_on() raw updates and the
  same Environment interface as evolve.
- `neuroforge/recurrent.py` - RNN (Phase 4): Elman recurrent net with
  memory, trained by BPTT. Sequences of timestep vectors; regression or
  classification per step; step()/generate() for streaming + text; JSON
  save/load. Numpy-free, own tiny matvec helpers.
- `tests/` - test_core, test_evolve, test_qlearn, test_recurrent. Run ALL
  FOUR after any change to core (recurrent imports Scaler+softmax from it).
- `demos/predict_process.py` - input->output prediction + what-if sweeps.
- `demos/creature_world.py` - evolved food-hunter in an ASCII world
  (`--watch` animates, `--fast` for quick smoke runs).
- `demos/grid_quest.py` - Q-learning key-then-door quest (`--fast` for
  smoke runs).

## RL tuning lessons (learned on grid_quest, 2026-07-03)

- Sparse terminal rewards drown in function-approximation noise: the
  Q-gap between right and wrong moves was ~0.05 against values of ~10,
  and greedy play oscillated while training reward looked great. Fixes
  that worked, in order of impact: (1) SIGN features in the observation
  so near-target states are as loud as far ones, (2) potential-based
  shaping toward the current objective (does not change the optimal
  policy), (3) gamma 0.9 instead of 0.95 to widen adjacent-state gaps.
- Evaluate policies GREEDY on unseen seeds. Training reward includes
  exploration noise that hides broken greedy behavior.

## RNN / BPTT lessons (learned building recurrent.py, 2026-07-04)

- Global gradient-NORM clipping (GRAD_NORM_MAX, rescale the whole gradient
  when its L2 norm exceeds the budget) is what makes BPTT trainable with
  momentum + a real lr. Per-element value clipping alone is not enough and
  distorts direction; without the norm clip, momentum 0.9 + lr 0.02
  exploded loss to ~1e165 within one epoch. This is THE RNN fix.
- Init W_hh small (scale sqrt(1/hidden)) so the recurrent loop does not
  amplify signals/gradients across timesteps. W_xh uses He on input fan-in.
- Cold-start vs warm-state: generation from a zero hidden state lands the
  net in a context it saw only at char 0 of the corpus, so the first step
  or two look wrong then re-sync. PRIME generate() with real context (a
  warm seed) for clean output - it is not a training bug.
- Delayed-echo (target[t] = input[t-1]) is the canary test: it is
  impossible without memory, so if it converges, recurrence genuinely works.

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
