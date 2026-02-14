/**
 * The most atomic way to train and inference a GPT in pure, dependency-free Swift.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * @karpathy (original Python), converted to Swift
 */

import Foundation

// --- RNG (xoshiro256** seeded with 42) ---
class Rng {
    var s: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // SplitMix64 to seed xoshiro
        var sm = seed
        func next() -> UInt64 {
            sm = sm &+ 0x9e3779b97f4a7c15
            var z = sm
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return z ^ (z >> 31)
        }
        s = (next(), next(), next(), next())
    }

    func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }

    func nextU64() -> UInt64 {
        let result = rotl(s.1 &* 5, 7) &* 9
        let t = s.1 << 17
        s.2 ^= s.0
        s.3 ^= s.1
        s.1 ^= s.2
        s.0 ^= s.3
        s.2 ^= t
        s.3 = rotl(s.3, 45)
        return result
    }

    func nextF64() -> Double {
        Double(nextU64() >> 11) / Double(1 << 53)
    }

    func gauss(_ mean: Double, _ std: Double) -> Double {
        let u1 = max(nextF64(), 1e-30)
        let u2 = nextF64()
        return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }

    func shuffle<T>(_ array: inout [T]) {
        for i in stride(from: array.count - 1, through: 1, by: -1) {
            let j = Int(nextU64() % UInt64(i + 1))
            array.swapAt(i, j)
        }
    }

    func weightedChoice(_ weights: [Double]) -> Int {
        let total = weights.reduce(0, +)
        var r = nextF64() * total
        for (i, w) in weights.enumerated() {
            r -= w
            if r <= 0 { return i }
        }
        return weights.count - 1
    }
}

// --- Let there be Autograd ---
class Value {
    var data: Double                     // scalar value of this node calculated during forward pass
    var grad: Double = 0.0              // derivative of the loss w.r.t. this node
    var children: [Value]               // children of this node in the computation graph
    var localGrads: [Double]            // local derivative of this node w.r.t. its children

    init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
        self.data = data
        self.children = children
        self.localGrads = localGrads
    }

    func vpow(_ exp: Double) -> Value {
        Value(pow(data, exp), children: [self], localGrads: [exp * pow(data, exp - 1)])
    }

    func vlog() -> Value {
        Value(log(data), children: [self], localGrads: [1.0 / data])
    }

    func vexp() -> Value {
        let e = exp(data)
        return Value(e, children: [self], localGrads: [e])
    }

    func relu() -> Value {
        Value(max(0, data), children: [self], localGrads: [data > 0 ? 1.0 : 0.0])
    }

    func backward() {
        var topo: [Value] = []
        var visited = Set<ObjectIdentifier>()

        func buildTopo(_ v: Value) {
            let id = ObjectIdentifier(v)
            if visited.contains(id) { return }
            visited.insert(id)
            for child in v.children {
                buildTopo(child)
            }
            topo.append(v)
        }

        buildTopo(self)
        self.grad = 1.0
        for v in topo.reversed() {
            for (child, lg) in zip(v.children, v.localGrads) {
                child.grad += lg * v.grad
            }
        }
    }
}

func + (lhs: Value, rhs: Value) -> Value {
    Value(lhs.data + rhs.data, children: [lhs, rhs], localGrads: [1.0, 1.0])
}
func + (lhs: Value, rhs: Double) -> Value { lhs + Value(rhs) }
func + (lhs: Double, rhs: Value) -> Value { Value(lhs) + rhs }

func * (lhs: Value, rhs: Value) -> Value {
    Value(lhs.data * rhs.data, children: [lhs, rhs], localGrads: [rhs.data, lhs.data])
}
func * (lhs: Value, rhs: Double) -> Value { lhs * Value(rhs) }
func * (lhs: Double, rhs: Value) -> Value { Value(lhs) * rhs }

prefix func - (v: Value) -> Value { v * (-1.0) }
func - (lhs: Value, rhs: Value) -> Value { lhs + (-rhs) }
func / (lhs: Value, rhs: Value) -> Value { lhs * rhs.vpow(-1.0) }
func / (lhs: Value, rhs: Double) -> Value { lhs * (1.0 / rhs) }

// --- Utility ---
typealias Vec1 = [Value]
typealias Mat = [Vec1]

func sumVals(_ v: [Value]) -> Value {
    var s = Value(0.0)
    for x in v { s = s + x }
    return s
}

// --- Model functions ---
func linear(_ x: Vec1, _ w: Mat) -> Vec1 {
    w.map { wo in
        var s = Value(0.0)
        for (wi, xi) in zip(wo, x) { s = s + wi * xi }
        return s
    }
}

func softmax(_ logits: Vec1) -> Vec1 {
    let maxVal = logits.map { $0.data }.max()!
    let exps = logits.map { ($0 - maxVal).vexp() }
    let total = sumVals(exps)
    return exps.map { $0 / total }
}

func rmsnorm(_ x: Vec1) -> Vec1 {
    var ms = Value(0.0)
    for xi in x { ms = ms + xi * xi }
    ms = ms / Double(x.count)
    let scale = (ms + 1e-5).vpow(-0.5)
    return x.map { $0 * scale }
}

// --- Main execution ---
let rng = Rng(seed: 42)

// Let there be an input dataset
guard let content = try? String(contentsOfFile: "input.txt", encoding: .utf8) else {
    print("Error: input.txt not found. Please download it first.")
    exit(1)
}
var docs = content.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
rng.shuffle(&docs)
print("num docs: \(docs.count)")

// Let there be a Tokenizer
let uchars = Array(Set(docs.joined()).sorted())
let BOS = uchars.count
let vocabSize = uchars.count + 1
print("vocab size: \(vocabSize)")

func charIndex(_ c: Character) -> Int {
    uchars.firstIndex(of: c)!
}

// Initialize the parameters
let nEmbd = 16
let nHead = 4
let nLayer = 1
let blockSize = 16
let headDim = nEmbd / nHead

func makeMatrix(_ nout: Int, _ nin: Int) -> Mat {
    (0..<nout).map { _ in (0..<nin).map { _ in Value(rng.gauss(0, 0.08)) } }
}

let wte = makeMatrix(vocabSize, nEmbd)
let wpe = makeMatrix(blockSize, nEmbd)
let lmHead = makeMatrix(vocabSize, nEmbd)

struct LayerWeights {
    let attnWq: Mat
    let attnWk: Mat
    let attnWv: Mat
    let attnWo: Mat
    let mlpFc1: Mat
    let mlpFc2: Mat
}

var layers: [LayerWeights] = []
for _ in 0..<nLayer {
    layers.append(LayerWeights(
        attnWq: makeMatrix(nEmbd, nEmbd),
        attnWk: makeMatrix(nEmbd, nEmbd),
        attnWv: makeMatrix(nEmbd, nEmbd),
        attnWo: makeMatrix(nEmbd, nEmbd),
        mlpFc1: makeMatrix(4 * nEmbd, nEmbd),
        mlpFc2: makeMatrix(nEmbd, 4 * nEmbd)
    ))
}

// Flatten params
var params: [Value] = []
for row in wte { for p in row { params.append(p) } }
for row in wpe { for p in row { params.append(p) } }
for row in lmHead { for p in row { params.append(p) } }
for lw in layers {
    for mat in [lw.attnWq, lw.attnWk, lw.attnWv, lw.attnWo, lw.mlpFc1, lw.mlpFc2] {
        for row in mat { for p in row { params.append(p) } }
    }
}
print("num params: \(params.count)")

// GPT function
func gpt(_ tokenId: Int, _ posId: Int,
         _ keys: inout [[Vec1]], _ values: inout [[Vec1]]) -> Vec1 {
    let tokEmb = wte[tokenId]
    let posEmb = wpe[posId]
    var x: Vec1 = zip(tokEmb, posEmb).map { $0 + $1 }
    x = rmsnorm(x)

    for li in 0..<nLayer {
        // 1) Multi-head attention block
        let xResidual = x
        x = rmsnorm(x)
        let q = linear(x, layers[li].attnWq)
        let k = linear(x, layers[li].attnWk)
        let v = linear(x, layers[li].attnWv)
        keys[li].append(k)
        values[li].append(v)
        var xAttn: Vec1 = []
        for h in 0..<nHead {
            let hs = h * headDim
            let qH = Array(q[hs..<hs + headDim])
            let kH = keys[li].map { Array($0[hs..<hs + headDim]) }
            let vH = values[li].map { Array($0[hs..<hs + headDim]) }
            let scale = sqrt(Double(headDim))
            let attnLogits: Vec1 = kH.map { kt in
                var dot = Value(0.0)
                for j in 0..<headDim { dot = dot + qH[j] * kt[j] }
                return dot / scale
            }
            let attnWeights = softmax(attnLogits)
            for j in 0..<headDim {
                var s = Value(0.0)
                for t in 0..<vH.count { s = s + attnWeights[t] * vH[t][j] }
                xAttn.append(s)
            }
        }
        x = linear(xAttn, layers[li].attnWo)
        for i in 0..<nEmbd { x[i] = x[i] + xResidual[i] }
        // 2) MLP block
        let xResidual2 = x
        x = rmsnorm(x)
        x = linear(x, layers[li].mlpFc1)
        x = x.map { $0.relu() }
        x = linear(x, layers[li].mlpFc2)
        for i in 0..<nEmbd { x[i] = x[i] + xResidual2[i] }
    }

    return linear(x, lmHead)
}

// Let there be Adam
let learningRate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let epsAdam = 1e-8
var mBuf = [Double](repeating: 0.0, count: params.count)
var vBuf = [Double](repeating: 0.0, count: params.count)

// Repeat in sequence
let numSteps = 1000
for step in 0..<numSteps {
    // Take single document, tokenize it
    let doc = docs[step % docs.count]
    var tokens = [BOS]
    for c in doc { tokens.append(charIndex(c)) }
    tokens.append(BOS)
    let n = min(blockSize, tokens.count - 1)

    // Forward
    var keys: [[Vec1]] = (0..<nLayer).map { _ in [] }
    var vals: [[Vec1]] = (0..<nLayer).map { _ in [] }
    var losses: Vec1 = []
    for posId in 0..<n {
        let tokenId = tokens[posId]
        let targetId = tokens[posId + 1]
        let logits = gpt(tokenId, posId, &keys, &vals)
        let probs = softmax(logits)
        let lossT = -(probs[targetId].vlog())
        losses.append(lossT)
    }
    let loss = sumVals(losses) / Double(n)

    // Backward
    loss.backward()

    // Adam optimizer update
    let lrT = learningRate * (1.0 - Double(step) / Double(numSteps))
    for (i, p) in params.enumerated() {
        mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad
        vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad
        let mHat = mBuf[i] / (1.0 - pow(beta1, Double(step + 1)))
        let vHat = vBuf[i] / (1.0 - pow(beta2, Double(step + 1)))
        p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
        p.grad = 0
    }

    print(String(format: "step %4d / %4d | loss %.4f", step + 1, numSteps, loss.data))
}

// Inference
let temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sampleIdx in 0..<20 {
    var keys: [[Vec1]] = (0..<nLayer).map { _ in [] }
    var vals: [[Vec1]] = (0..<nLayer).map { _ in [] }
    var tokenId = BOS
    var sample: [Character] = []
    for posId in 0..<blockSize {
        let logits = gpt(tokenId, posId, &keys, &vals)
        let tempLogits = logits.map { $0 / temperature }
        let probs = softmax(tempLogits)
        let weights = probs.map { $0.data }
        tokenId = rng.weightedChoice(weights)
        if tokenId == BOS { break }
        sample.append(uchars[tokenId])
    }
    print(String(format: "sample %2d: %@", sampleIdx + 1, String(sample)))
}
