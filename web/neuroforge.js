/* NeuroForge browser runtime - run and train NeuroForge brains in JS.
   Mirrors neuroforge/core.py (forward, train_on) and qlearn.py (QAgent).
   Brain format: array of layers {w:[[...]], b:[...], a:"tanh"|"linear"|
   "relu"|"sigmoid"} - the same layers array found inside Network.save()
   JSON (use NF.fromNetworkJSON to convert a saved model file). */

const NF = (() => {
  const CLIP = 5;
  const ACT = {
    tanh: Math.tanh,
    linear: x => x,
    relu: x => (x > 0 ? x : 0),
    sigmoid: x => 1 / (1 + Math.exp(-x)),
  };
  const clip = g => (g > CLIP ? CLIP : g < -CLIP ? -CLIP : g);

  function gauss() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randomBrain(sizes, hiddenAct) {
    hiddenAct = hiddenAct || "tanh";
    const layers = [];
    for (let li = 0; li < sizes.length - 1; li++) {
      const nin = sizes[li], nout = sizes[li + 1];
      const s = Math.sqrt(2 / nin);
      layers.push({
        w: Array.from({ length: nout }, () =>
          Array.from({ length: nin }, () => gauss() * s)),
        b: new Array(nout).fill(0),
        a: li === sizes.length - 2 ? "linear" : hiddenAct,
      });
    }
    return layers;
  }

  function copyBrain(b) { return JSON.parse(JSON.stringify(b)); }

  /* Mutated copy, mirroring core.py Network.mutate(): each weight/bias has
     `rate` chance of a gaussian nudge of size `scale`. This is the entire
     'learning' mechanism of neuroevolution - no gradients, no calculus. */
  function mutateBrain(brain, rate, scale) {
    rate = rate ?? 0.1;
    scale = scale ?? 0.3;
    const child = copyBrain(brain);
    for (const L of child) {
      for (const row of L.w)
        for (let j = 0; j < row.length; j++)
          if (Math.random() < rate) row[j] += gauss() * scale;
      for (let i = 0; i < L.b.length; i++)
        if (Math.random() < rate) L.b[i] += gauss() * scale;
    }
    return child;
  }

  function fromNetworkJSON(data) {
    return data.layers.map(l => ({
      w: l.weights, b: l.biases, a: l.activation,
    }));
  }

  function forwardAll(brain, x) {
    const acts = [x];
    let cur = x;
    for (const L of brain) {
      const act = ACT[L.a], out = [];
      for (let i = 0; i < L.w.length; i++) {
        let t = L.b[i];
        const row = L.w[i];
        for (let j = 0; j < row.length; j++) t += row[j] * cur[j];
        out.push(act(t));
      }
      acts.push(out);
      cur = out;
    }
    return acts;
  }

  function activate(brain, x) { const a = forwardAll(brain, x); return a[a.length - 1]; }
  function argmax(a) {
    let bi = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
    return bi;
  }
  const act = (brain, x) => argmax(activate(brain, x));

  /* One raw MSE gradient step, matching core.py order exactly:
     input gradients computed BEFORE weights change. 2-layer brains only
     (the shape every NeuroForge demo uses). */
  function trainOn(brain, x, target, lr) {
    const A = forwardAll(brain, x), a1 = A[1], out = A[2];
    const nH = a1.length, nO = out.length, nI = x.length;
    const g2 = out.map((o, i) => clip(2 * (o - target[i]) / nO));
    const g1raw = new Array(nH).fill(0);
    for (let j = 0; j < nH; j++) {
      let s = 0;
      for (let i = 0; i < nO; i++) s += brain[1].w[i][j] * g2[i];
      g1raw[j] = s;
    }
    for (let i = 0; i < nO; i++) {
      const g = g2[i], row = brain[1].w[i];
      for (let j = 0; j < nH; j++) row[j] -= lr * g * a1[j];
      brain[1].b[i] -= lr * g;
    }
    for (let i = 0; i < nH; i++) {
      const g = clip(g1raw[i] * (brain[0].a === "tanh" ? 1 - a1[i] * a1[i]
        : brain[0].a === "linear" ? 1
        : brain[0].a === "relu" ? (a1[i] > 0 ? 1 : 0)
        : a1[i] * (1 - a1[i])));
      const row = brain[0].w[i];
      for (let j = 0; j < nI; j++) row[j] -= lr * g * x[j];
      brain[0].b[i] -= lr * g;
    }
  }

  /* DQN-lite mirroring qlearn.py: replay buffer + target network. */
  class QAgent {
    constructor(opts) {
      this.brain = opts.brain || randomBrain([opts.inputs, opts.hidden || 24, opts.actions]);
      this.target = copyBrain(this.brain);
      this.actions = this.brain[this.brain.length - 1].b.length;
      this.gamma = opts.gamma ?? 0.9;
      this.lr = opts.lr ?? 0.01;
      this.epsilon = opts.epsilon ?? 1.0;
      this.epsilonMin = opts.epsilonMin ?? 0.05;
      this.epsilonDecay = opts.epsilonDecay ?? 0.985;
      this.batch = opts.batch ?? 16;
      this.bufMax = opts.bufMax ?? 4000;
      this.sync = opts.sync ?? 250;
      this.buffer = [];
      this.steps = 0;
    }
    act(obs) {
      if (Math.random() < this.epsilon) return Math.floor(Math.random() * this.actions);
      return this.greedy(obs);
    }
    greedy(obs) { return argmax(activate(this.brain, obs)); }
    learn(s0, a, reward, s1, done) {
      this.buffer.push([s0, a, reward, s1, done]);
      if (this.buffer.length > this.bufMax) this.buffer.shift();
      if (this.buffer.length >= this.batch) {
        for (let n = 0; n < this.batch; n++) {
          const e = this.buffer[Math.floor(Math.random() * this.buffer.length)];
          const tq = activate(this.brain, e[0]).slice();
          tq[e[1]] = e[4] ? e[2]
            : e[2] + this.gamma * Math.max.apply(null, activate(this.target, e[3]));
          trainOn(this.brain, e[0], tq, this.lr);
        }
      }
      this.steps++;
      if (this.steps % this.sync === 0) this.target = copyBrain(this.brain);
    }
    endEpisode() {
      this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
    }
  }

  return { randomBrain, copyBrain, mutateBrain, fromNetworkJSON, forwardAll, activate, act, argmax, trainOn, QAgent };
})();
if (typeof module !== "undefined") module.exports = NF;
