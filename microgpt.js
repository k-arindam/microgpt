/**
 * The most atomic way to train and inference a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * @karpathy (original Python), converted to JavaScript (Node.js)
 */

const fs = require("fs");

// --- RNG (xoshiro256** seeded with 42, using BigInt for 64-bit) ---
class Rng {
  constructor(seed) {
    // SplitMix64 to seed xoshiro
    let sm = BigInt(seed);
    const mask = (1n << 64n) - 1n;
    const smNext = () => {
      sm = (sm + 0x9e3779b97f4a7c15n) & mask;
      let z = sm;
      z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
      z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & mask;
      return (z ^ (z >> 31n)) & mask;
    };
    this.s = [smNext(), smNext(), smNext(), smNext()];
    this._mask = mask;
  }

  _rotl(x, k) {
    const m = this._mask;
    return ((x << k) | (x >> (64n - k))) & m;
  }

  nextU64() {
    const m = this._mask;
    const s = this.s;
    const result = (this._rotl((s[1] * 5n) & m, 7n) * 9n) & m;
    const t = (s[1] << 17n) & m;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = this._rotl(s[3], 45n);
    return result;
  }

  nextF64() {
    return Number(this.nextU64() >> 11n) / 2 ** 53;
  }

  gauss(mean, std) {
    const u1 = Math.max(this.nextF64(), 1e-30);
    const u2 = this.nextF64();
    return mean + std * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Number(this.nextU64() % BigInt(i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  weightedChoice(weights) {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.nextF64() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}

// --- Let there be Autograd ---
class Value {
  constructor(data, children = [], localGrads = []) {
    this.data = data;           // scalar value of this node calculated during forward pass
    this.grad = 0.0;            // derivative of the loss w.r.t. this node
    this.children = children;   // children of this node in the computation graph
    this.localGrads = localGrads; // local derivative of this node w.r.t. its children
  }

  add(other) {
    other = other instanceof Value ? other : new Value(other);
    return new Value(this.data + other.data, [this, other], [1.0, 1.0]);
  }

  mul(other) {
    other = other instanceof Value ? other : new Value(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }

  pow(exp) {
    return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
  }

  log() {
    return new Value(Math.log(this.data), [this], [1.0 / this.data]);
  }

  exp() {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu() {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1.0 : 0.0]);
  }

  neg() { return this.mul(-1); }
  sub(other) { other = other instanceof Value ? other : new Value(other); return this.add(other.neg()); }
  div(other) { other = other instanceof Value ? other : new Value(other); return this.mul(other.pow(-1)); }

  backward() {
    const topo = [];
    const visited = new Set();
    const buildTopo = (v) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.children) buildTopo(child);
      topo.push(v);
    };
    buildTopo(this);
    this.grad = 1.0;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v.children.length; j++) {
        v.children[j].grad += v.localGrads[j] * v.grad;
      }
    }
  }
}

// --- Utility ---
function sumVals(arr) {
  let s = new Value(0);
  for (const x of arr) s = s.add(x);
  return s;
}

// --- Model functions ---
function linear(x, w) {
  return w.map((wo) => {
    let s = new Value(0);
    for (let i = 0; i < wo.length; i++) s = s.add(wo[i].mul(x[i]));
    return s;
  });
}

function softmax(logits) {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = sumVals(exps);
  return exps.map((e) => e.div(total));
}

function rmsnorm(x) {
  let ms = new Value(0);
  for (const xi of x) ms = ms.add(xi.mul(xi));
  ms = ms.div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// --- Main ---
(function main() {
  const rng = new Rng(42);

  // Let there be an input dataset
  if (!fs.existsSync("input.txt")) {
    console.error("Error: input.txt not found. Please download it first.");
    process.exit(1);
  }
  const content = fs.readFileSync("input.txt", "utf-8");
  const docs = content.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  rng.shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  // Let there be a Tokenizer
  const charSet = new Set();
  for (const d of docs) for (const c of d) charSet.add(c);
  const uchars = [...charSet].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  console.log(`vocab size: ${vocabSize}`);

  const charIndex = (c) => uchars.indexOf(c);

  // Initialize the parameters
  const nEmbd = 16;
  const nHead = 4;
  const nLayer = 1;
  const blockSize = 16;
  const headDim = nEmbd / nHead;

  const makeMatrix = (nout, nin) =>
    Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => new Value(rng.gauss(0, 0.08)))
    );

  const wte = makeMatrix(vocabSize, nEmbd);
  const wpe = makeMatrix(blockSize, nEmbd);
  const lmHead = makeMatrix(vocabSize, nEmbd);

  const layers = [];
  for (let i = 0; i < nLayer; i++) {
    layers.push({
      attnWq: makeMatrix(nEmbd, nEmbd),
      attnWk: makeMatrix(nEmbd, nEmbd),
      attnWv: makeMatrix(nEmbd, nEmbd),
      attnWo: makeMatrix(nEmbd, nEmbd),
      mlpFc1: makeMatrix(4 * nEmbd, nEmbd),
      mlpFc2: makeMatrix(nEmbd, 4 * nEmbd),
    });
  }

  // Flatten params
  const params = [];
  for (const row of wte) for (const p of row) params.push(p);
  for (const row of wpe) for (const p of row) params.push(p);
  for (const row of lmHead) for (const p of row) params.push(p);
  for (const lw of layers) {
    for (const mat of [lw.attnWq, lw.attnWk, lw.attnWv, lw.attnWo, lw.mlpFc1, lw.mlpFc2]) {
      for (const row of mat) for (const p of row) params.push(p);
    }
  }
  console.log(`num params: ${params.length}`);

  // GPT function
  function gpt(tokenId, posId, keys, values) {
    const tokEmb = wte[tokenId];
    const posEmb = wpe[posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < nLayer; li++) {
      // 1) Multi-head attention block
      const xResidual = [...x];
      x = rmsnorm(x);
      const q = linear(x, layers[li].attnWq);
      const k = linear(x, layers[li].attnWk);
      const v = linear(x, layers[li].attnWv);
      keys[li].push(k);
      values[li].push(v);
      const xAttn = [];
      for (let h = 0; h < nHead; h++) {
        const hs = h * headDim;
        const qH = q.slice(hs, hs + headDim);
        const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
        const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));
        const scale = Math.sqrt(headDim);
        const attnLogits = kH.map((kt) => {
          let dot = new Value(0);
          for (let j = 0; j < headDim; j++) dot = dot.add(qH[j].mul(kt[j]));
          return dot.div(scale);
        });
        const attnWeights = softmax(attnLogits);
        for (let j = 0; j < headDim; j++) {
          let s = new Value(0);
          for (let t = 0; t < vH.length; t++) s = s.add(attnWeights[t].mul(vH[t][j]));
          xAttn.push(s);
        }
      }
      x = linear(xAttn, layers[li].attnWo);
      x = x.map((xi, i) => xi.add(xResidual[i]));
      // 2) MLP block
      const xResidual2 = [...x];
      x = rmsnorm(x);
      x = linear(x, layers[li].mlpFc1);
      x = x.map((xi) => xi.relu());
      x = linear(x, layers[li].mlpFc2);
      x = x.map((xi, i) => xi.add(xResidual2[i]));
    }

    return linear(x, lmHead);
  }

  // Let there be Adam
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;
  const mBuf = new Float64Array(params.length);
  const vBuf = new Float64Array(params.length);

  // Repeat in sequence
  const numSteps = 1000;
  for (let step = 0; step < numSteps; step++) {
    // Take single document, tokenize it
    const doc = docs[step % docs.length];
    const tokens = [BOS];
    for (const c of doc) tokens.push(charIndex(c));
    tokens.push(BOS);
    const n = Math.min(blockSize, tokens.length - 1);

    // Forward
    const keys = Array.from({ length: nLayer }, () => []);
    const vals = Array.from({ length: nLayer }, () => []);
    const losses = [];
    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(tokenId, posId, keys, vals);
      const probs = softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);
    }
    const loss = sumVals(losses).div(n);

    // Backward
    loss.backward();

    // Adam optimizer update
    const lrT = learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad;
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad;
      const mHat = mBuf[i] / (1 - Math.pow(beta1, step + 1));
      const vHat = vBuf[i] / (1 - Math.pow(beta2, step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      p.grad = 0;
    }

    const s1 = String(step + 1).padStart(4);
    const s2 = String(numSteps).padStart(4);
    console.log(`step ${s1} / ${s2} | loss ${loss.data.toFixed(4)}`);
  }

  // Inference
  const temperature = 0.5;
  console.log("\n--- inference (new, hallucinated names) ---");
  for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    const keys = Array.from({ length: nLayer }, () => []);
    const vals = Array.from({ length: nLayer }, () => []);
    let tokenId = BOS;
    const sample = [];
    for (let posId = 0; posId < blockSize; posId++) {
      const logits = gpt(tokenId, posId, keys, vals);
      const tempLogits = logits.map((l) => l.div(temperature));
      const probs = softmax(tempLogits);
      const weights = probs.map((p) => p.data);
      tokenId = rng.weightedChoice(weights);
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }
    const idx = String(sampleIdx + 1).padStart(2);
    console.log(`sample ${idx}: ${sample.join("")}`);
  }
})();
