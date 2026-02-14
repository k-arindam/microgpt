/// The most atomic way to train and inference a GPT in pure, dependency-free Dart.
/// This file is the complete algorithm.
/// Everything else is just efficiency.
///
/// @karpathy (original Python), converted to Dart

import 'dart:io';
import 'dart:math';

// --- RNG (xoshiro256** seeded with 42) ---
class Rng {
  late int s0, s1, s2, s3; // 64-bit state

  Rng(int seed) {
    // SplitMix64 to seed xoshiro
    int sm = seed;
    int smNext() {
      sm = (sm + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF;
      int z = sm;
      z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF;
      z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF;
      return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF;
    }

    s0 = smNext();
    s1 = smNext();
    s2 = smNext();
    s3 = smNext();
  }

  int _rotl(int x, int k) =>
      ((x << k) | ((x & 0xFFFFFFFFFFFFFFFF) >>> (64 - k))) & 0xFFFFFFFFFFFFFFFF;

  int nextU64() {
    final result =
        (_rotl((s1 * 5) & 0xFFFFFFFFFFFFFFFF, 7) * 9) & 0xFFFFFFFFFFFFFFFF;
    final t = (s1 << 17) & 0xFFFFFFFFFFFFFFFF;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = _rotl(s3, 45);
    return result;
  }

  double nextF64() => (nextU64() >>> 11) / (1 << 53);

  double gauss(double mean, double std) {
    final u1 = max(nextF64(), 1e-30);
    final u2 = nextF64();
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }

  void shuffle<T>(List<T> list) {
    for (int i = list.length - 1; i > 0; i--) {
      final j = nextU64() % (i + 1);
      final tmp = list[i];
      list[i] = list[j];
      list[j] = tmp;
    }
  }

  int weightedChoice(List<double> weights) {
    final total = weights.reduce((a, b) => a + b);
    double r = nextF64() * total;
    for (int i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}

// --- Let there be Autograd ---
class Value {
  double data; // scalar value of this node calculated during forward pass
  double grad = 0.0; // derivative of the loss w.r.t. this node
  final List<Value> children; // children of this node in the computation graph
  final List<double>
  localGrads; // local derivative of this node w.r.t. its children

  Value(this.data, {List<Value>? children, List<double>? localGrads})
    : children = children ?? [],
      localGrads = localGrads ?? [];

  Value operator +(Value other) =>
      Value(data + other.data, children: [this, other], localGrads: [1.0, 1.0]);

  Value addScalar(double s) => this + Value(s);

  Value operator *(Value other) => Value(
    data * other.data,
    children: [this, other],
    localGrads: [other.data, data],
  );

  Value mulScalar(double s) => this * Value(s);

  Value vpow(double exp) => Value(
    pow(data, exp).toDouble(),
    children: [this],
    localGrads: [exp * pow(data, exp - 1).toDouble()],
  );

  Value vlog() => Value(log(data), children: [this], localGrads: [1.0 / data]);

  Value vexp() {
    final e = exp(data);
    return Value(e, children: [this], localGrads: [e]);
  }

  Value relu() =>
      Value(max(0, data), children: [this], localGrads: [data > 0 ? 1.0 : 0.0]);

  Value operator -() => mulScalar(-1.0);

  Value operator -(Value other) => this + (-other);

  Value operator /(Value other) => this * other.vpow(-1.0);

  Value divScalar(double s) => mulScalar(1.0 / s);

  void backward() {
    final topo = <Value>[];
    final visited = <Value>{};

    void buildTopo(Value v) {
      if (visited.contains(v)) return;
      visited.add(v);
      for (final child in v.children) {
        buildTopo(child);
      }
      topo.add(v);
    }

    buildTopo(this);
    grad = 1.0;
    for (final v in topo.reversed) {
      for (int i = 0; i < v.children.length; i++) {
        v.children[i].grad += v.localGrads[i] * v.grad;
      }
    }
  }
}

// --- Utility ---
Value sumVals(List<Value> v) {
  var s = Value(0.0);
  for (final x in v) s = s + x;
  return s;
}

// --- Model functions ---
List<Value> linear(List<Value> x, List<List<Value>> w) {
  return w.map((wo) {
    var s = Value(0.0);
    for (int i = 0; i < wo.length; i++) {
      s = s + wo[i] * x[i];
    }
    return s;
  }).toList();
}

List<Value> softmax(List<Value> logits) {
  final maxVal = logits.map((v) => v.data).reduce(max);
  final exps = logits.map((v) => (v - Value(maxVal)).vexp()).toList();
  final total = sumVals(exps);
  return exps.map((e) => e / total).toList();
}

List<Value> rmsnorm(List<Value> x) {
  var ms = Value(0.0);
  for (final xi in x) ms = ms + xi * xi;
  ms = ms.divScalar(x.length.toDouble());
  final scale = (ms.addScalar(1e-5)).vpow(-0.5);
  return x.map((xi) => xi * scale).toList();
}

// --- Main ---
void main() {
  final rng = Rng(42);

  // Let there be an input dataset
  final file = File('input.txt');
  if (!file.existsSync()) {
    print('Error: input.txt not found. Please download it first.');
    exit(1);
  }
  final content = file.readAsStringSync();
  final docs = content
      .split('\n')
      .map((l) => l.trim())
      .where((l) => l.isNotEmpty)
      .toList();
  rng.shuffle(docs);
  print('num docs: ${docs.length}');

  // Let there be a Tokenizer
  final charSet = <String>{};
  for (final d in docs) {
    for (int i = 0; i < d.length; i++) {
      charSet.add(d[i]);
    }
  }
  final uchars = charSet.toList()..sort();
  final bos = uchars.length;
  final vocabSize = uchars.length + 1;
  print('vocab size: $vocabSize');

  int charIndex(String c) => uchars.indexOf(c);

  // Initialize the parameters
  const nEmbd = 16;
  const nHead = 4;
  const nLayer = 1;
  const blockSize = 16;
  const headDim = nEmbd ~/ nHead;

  List<List<Value>> makeMatrix(int nout, int nin) {
    return List.generate(
      nout,
      (_) => List.generate(nin, (_) => Value(rng.gauss(0, 0.08))),
    );
  }

  final wte = makeMatrix(vocabSize, nEmbd);
  final wpe = makeMatrix(blockSize, nEmbd);
  final lmHead = makeMatrix(vocabSize, nEmbd);

  // Per-layer weights
  final layerAttnWq = <List<List<Value>>>[];
  final layerAttnWk = <List<List<Value>>>[];
  final layerAttnWv = <List<List<Value>>>[];
  final layerAttnWo = <List<List<Value>>>[];
  final layerMlpFc1 = <List<List<Value>>>[];
  final layerMlpFc2 = <List<List<Value>>>[];

  for (int i = 0; i < nLayer; i++) {
    layerAttnWq.add(makeMatrix(nEmbd, nEmbd));
    layerAttnWk.add(makeMatrix(nEmbd, nEmbd));
    layerAttnWv.add(makeMatrix(nEmbd, nEmbd));
    layerAttnWo.add(makeMatrix(nEmbd, nEmbd));
    layerMlpFc1.add(makeMatrix(4 * nEmbd, nEmbd));
    layerMlpFc2.add(makeMatrix(nEmbd, 4 * nEmbd));
  }

  // Flatten params
  final params = <Value>[];
  for (final row in wte) for (final p in row) params.add(p);
  for (final row in wpe) for (final p in row) params.add(p);
  for (final row in lmHead) for (final p in row) params.add(p);
  for (int i = 0; i < nLayer; i++) {
    for (final mat in [
      layerAttnWq[i],
      layerAttnWk[i],
      layerAttnWv[i],
      layerAttnWo[i],
      layerMlpFc1[i],
      layerMlpFc2[i],
    ]) {
      for (final row in mat) for (final p in row) params.add(p);
    }
  }
  print('num params: ${params.length}');

  // GPT function
  List<Value> gpt(
    int tokenId,
    int posId,
    List<List<List<Value>>> keys,
    List<List<List<Value>>> values,
  ) {
    final tokEmb = wte[tokenId];
    final posEmb = wpe[posId];
    var x = List.generate(nEmbd, (i) => tokEmb[i] + posEmb[i]);
    x = rmsnorm(x);

    for (int li = 0; li < nLayer; li++) {
      // 1) Multi-head attention block
      final xResidual = List<Value>.from(x);
      x = rmsnorm(x);
      final q = linear(x, layerAttnWq[li]);
      final k = linear(x, layerAttnWk[li]);
      final v = linear(x, layerAttnWv[li]);
      keys[li].add(k);
      values[li].add(v);
      final xAttn = <Value>[];
      for (int h = 0; h < nHead; h++) {
        final hs = h * headDim;
        final qH = q.sublist(hs, hs + headDim);
        final kH = keys[li].map((ki) => ki.sublist(hs, hs + headDim)).toList();
        final vH = values[li]
            .map((vi) => vi.sublist(hs, hs + headDim))
            .toList();
        final scale = sqrt(headDim.toDouble());
        final attnLogits = List.generate(kH.length, (t) {
          var dot = Value(0.0);
          for (int j = 0; j < headDim; j++) dot = dot + qH[j] * kH[t][j];
          return dot.divScalar(scale);
        });
        final attnWeights = softmax(attnLogits);
        for (int j = 0; j < headDim; j++) {
          var s = Value(0.0);
          for (int t = 0; t < vH.length; t++) s = s + attnWeights[t] * vH[t][j];
          xAttn.add(s);
        }
      }
      x = linear(xAttn, layerAttnWo[li]);
      for (int i = 0; i < nEmbd; i++) x[i] = x[i] + xResidual[i];
      // 2) MLP block
      final xResidual2 = List<Value>.from(x);
      x = rmsnorm(x);
      x = linear(x, layerMlpFc1[li]);
      x = x.map((xi) => xi.relu()).toList();
      x = linear(x, layerMlpFc2[li]);
      for (int i = 0; i < nEmbd; i++) x[i] = x[i] + xResidual2[i];
    }

    return linear(x, lmHead);
  }

  // Let there be Adam
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;
  final mBuf = List.filled(params.length, 0.0);
  final vBuf = List.filled(params.length, 0.0);

  // Repeat in sequence
  const numSteps = 1000;
  for (int step = 0; step < numSteps; step++) {
    // Take single document, tokenize it
    final doc = docs[step % docs.length];
    final tokens = <int>[bos];
    for (int i = 0; i < doc.length; i++) tokens.add(charIndex(doc[i]));
    tokens.add(bos);
    final n = min(blockSize, tokens.length - 1);

    // Forward
    final keys = List.generate(nLayer, (_) => <List<Value>>[]);
    final vals = List.generate(nLayer, (_) => <List<Value>>[]);
    final losses = <Value>[];
    for (int posId = 0; posId < n; posId++) {
      final tokenId = tokens[posId];
      final targetId = tokens[posId + 1];
      final logits = gpt(tokenId, posId, keys, vals);
      final probs = softmax(logits);
      final lossT = -(probs[targetId].vlog());
      losses.add(lossT);
    }
    final loss = sumVals(losses).divScalar(n.toDouble());

    // Backward
    loss.backward();

    // Adam optimizer update
    final lrT = learningRate * (1.0 - step / numSteps);
    for (int i = 0; i < params.length; i++) {
      final p = params[i];
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad;
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad;
      final mHat = mBuf[i] / (1.0 - pow(beta1, step + 1));
      final vHat = vBuf[i] / (1.0 - pow(beta2, step + 1));
      p.data -= lrT * mHat / (sqrt(vHat) + epsAdam);
      p.grad = 0;
    }

    final s1 = (step + 1).toString().padLeft(4);
    final s2 = numSteps.toString().padLeft(4);
    print('step $s1 / $s2 | loss ${loss.data.toStringAsFixed(4)}');
  }

  // Inference
  const temperature = 0.5;
  print('\n--- inference (new, hallucinated names) ---');
  for (int sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    final keys = List.generate(nLayer, (_) => <List<Value>>[]);
    final vals = List.generate(nLayer, (_) => <List<Value>>[]);
    var tokenId = bos;
    final sample = <String>[];
    for (int posId = 0; posId < blockSize; posId++) {
      final logits = gpt(tokenId, posId, keys, vals);
      final tempLogits = logits.map((l) => l.divScalar(temperature)).toList();
      final probs = softmax(tempLogits);
      final weights = probs.map((p) => p.data).toList();
      tokenId = rng.weightedChoice(weights);
      if (tokenId == bos) break;
      sample.add(uchars[tokenId]);
    }
    final idx = (sampleIdx + 1).toString().padLeft(2);
    print('sample $idx: ${sample.join()}');
  }
}
